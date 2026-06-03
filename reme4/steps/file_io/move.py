"""``file_move`` — relocate / rename a file within the vault by copy → retarget → delete.

Three-step ordering keeps vault_dir referentially consistent at every
intermediate point — no window in which inbound ``[[src_path]]`` wikilinks
dangle:

  1. ``shutil.copyfile(src_path, dst_path)``. Both files now exist on disk;
     inbound ``[[src_path]]`` still resolves (to the original location).
  2. ``retarget_links(src_path, dst_path)``. Rewrites every inbound
     ``[[src_path]]`` → ``[[dst_path]]`` across the vault. Both files
     exist throughout, so rewrites can land in any order without
     breaking resolution.
  3. ``src_abs.unlink()``. References now point at ``dst_path``; the
     original is an orphan and is safely removed.

If retargeting fails (raises or returns an error payload), step 3 is
skipped — both files remain on disk so the caller can diagnose and
retry; vault_dir stays consistent (references still resolve to the
original). The ``src_removed`` boolean in the payload distinguishes
the two cases.

``src_path`` must resolve inside vault_dir as a path relative to the
vault; ``dst_path`` must be relative to the vault with a directory
component (same rule as ``file_upload``). For cross-realm transfer
(vault_dir ↔ local fs) use ``file_upload`` / ``file_download``.

Opt out via ``retarget=False`` for the rare case where you intentionally
want to leave references stale (e.g. moving aside before delete) — the
original is still removed in that case (move semantics, not copy).
"""

import shutil
from pathlib import Path

from ..base_step import BaseStep
from ...components import R
from ...utils.wikilink_handler import WikilinkHandler


@R.register("move_step")
class MoveStep(BaseStep):
    """Move ``src_path`` to ``dst_path`` within the vault (copy → retarget → unlink)."""

    async def execute(self):
        assert self.context is not None
        src_path: str = self.context.get("src_path", "")
        dst_path: str = self.context.get("dst_path", "")
        overwrite: bool = bool(self.context.get("overwrite", False))
        retarget: bool = bool(self.context.get("retarget", True))
        assert src_path and dst_path, "src_path and dst_path are required"
        payload = await self._move(src_path, dst_path, overwrite, retarget)
        if "error" in payload:
            self.context.response.success = False
            self.context.response.answer = f"Error: {payload['error']}"
            self.logger.info(f"[{self.name}] move failed src={src_path} dst={dst_path} error={payload['error']!r}")
        else:
            self.context.response.success = True
            self.context.response.answer = f"Moved {src_path} → {dst_path}"
            retarget_info = payload.get("retarget") or {}
            self.logger.info(
                f"[{self.name}] moved src={src_path} dst={dst_path} src_removed={payload.get('src_removed')} "
                f"retarget_files={retarget_info.get('files_touched', 0) if isinstance(retarget_info, dict) else '-'} "
                f"retarget_links={retarget_info.get('links_changed', 0) if isinstance(retarget_info, dict) else '-'}",
            )
        self.context.response.metadata.update(payload)

    async def _move(self, src_path: str, dst_path: str, overwrite: bool, retarget: bool) -> dict:
        vault_dir = Path(self.file_store.vault_path or ".")
        src_abs = (vault_dir / src_path).resolve() if src_path else None
        dst_abs = (vault_dir / dst_path).resolve() if not Path(dst_path).is_absolute() else None
        precheck_error = _precheck_move(src_path, dst_path, src_abs, dst_abs, overwrite)
        if precheck_error:
            return precheck_error
        assert src_abs is not None and dst_abs is not None  # narrowed by precheck
        dst_abs.parent.mkdir(parents=True, exist_ok=True)

        # Step 1 — copy. Both files exist; inbound [[src_path]] still resolves.
        shutil.copyfile(str(src_abs), str(dst_abs))
        payload: dict = {"src_path": src_path, "dst_path": dst_path, "size": dst_abs.stat().st_size}

        # Step 2 — retarget. vault_dir stays consistent throughout: refs still
        # at [[src_path]] resolve to the original; refs already rewritten to
        # [[dst_path]] resolve to the new location. On error, bail before
        # unlinking so the caller can retry; vault_dir is still consistent.
        if retarget:
            try:
                report = await WikilinkHandler.retarget_links(self.file_store, src=src_path, dst=dst_path)
            except Exception as exc:
                payload["retarget"] = {"error": f"retarget raised: {exc!r}"}
                payload["src_removed"] = False
                return payload
            if "error" in report:
                payload["retarget"] = report
                payload["src_removed"] = False
                return payload
            payload["retarget"] = {
                "files_touched": report.get("files_touched", 0),
                "links_changed": report.get("links_changed", 0),
                "by_file": report.get("by_file", []),
                "ambiguous": report.get("ambiguous", []),
            }
        else:
            payload["retarget"] = None

        # Step 3 — unlink the original. Refs all point at dst_path now; the
        # original is an orphan. If unlink fails, vault_dir is still
        # consistent (refs resolve to dst_path); the original just lingers
        # as an orphan that the caller can clean up.
        try:
            src_abs.unlink()
            payload["src_removed"] = True
        except Exception as exc:
            payload["src_removed"] = False
            payload["src_remove_error"] = f"unlink raised: {exc!r}"

        return payload


def _precheck_move(
    src_path: str,
    dst_path: str,
    src_abs: Path | None,
    dst_abs: Path | None,
    overwrite: bool,
) -> dict | None:
    """Validate inputs for ``_move``; return an error payload or ``None`` when OK."""
    if src_abs is None or not src_abs.is_file():
        return {"src_path": src_path, "error": "not found"}
    if dst_abs is None or "/" not in dst_path:
        return {
            "dst_path": dst_path,
            "error": "dst_path must be relative to the vault with a directory component",
        }
    if dst_abs == src_abs:
        return {"src_path": src_path, "dst_path": dst_path, "error": "src_path and dst_path are the same"}
    if dst_abs.exists() and not overwrite:
        return {"dst_path": dst_path, "error": "destination exists; pass overwrite=True"}
    return None
