"""``file_move`` — relocate / rename a file within the vault by copy → retarget → delete.

Three-step ordering keeps vault_dir referentially consistent at every
intermediate point — no window in which inbound ``[[path]]`` wikilinks
dangle:

  1. ``shutil.copyfile(path, new_path)``. Both files now exist on disk;
     inbound ``[[path]]`` still resolves (to the original location).
  2. ``retarget_links(path, new_path)``. Rewrites every inbound
     ``[[path]]`` → ``[[new_path]]`` across the vault. Both files
     exist throughout, so rewrites can land in any order without
     breaking resolution.
  3. ``path_abs.unlink()``. References now point at ``new_path``; the
     original is an orphan and is safely removed.

If retargeting fails (raises or returns an error payload), step 3 is
skipped — both files remain on disk so the caller can diagnose and
retry; vault_dir stays consistent (references still resolve to the
original). The ``src_removed`` boolean in the payload distinguishes
the two cases.

Source (``path``) must resolve inside vault_dir as a path relative to
the vault; ``new_path`` must be relative to the vault with a directory
component (same rule as ``file_upload``). For cross-realm transfer
(vault_dir ↔ local fs) use ``file_upload`` / ``file_download``.

Opt out via ``retarget=False`` for the rare case where you intentionally
want to leave references stale (e.g. moving aside before delete) — the
original is still removed in that case (move semantics, not copy).
"""

from __future__ import annotations

import shutil
from pathlib import Path

from ..base_step import BaseStep
from ...utils import set_answer
from ...utils.wikilink_utils import retarget_links

from ...components import R
from ...enumeration import ComponentEnum


@R.register("move_step")
class MoveStep(BaseStep):
    """Move ``path`` to ``new_path`` within the vault (copy → retarget → unlink)."""

    component_type = ComponentEnum.STEP

    async def execute(self):
        assert self.context is not None
        path: str = self.context.get("path", "") or ""
        new_path: str = self.context.get("new_path", "") or ""
        overwrite: bool = bool(self.context.get("overwrite", False))
        retarget: bool = bool(self.context.get("retarget", True))
        assert path and new_path, "path and new_path are required"
        payload = await self._move(path, new_path, overwrite, retarget)
        self.context.response.success = "error" not in payload
        set_answer(self.context, payload)

    async def _move(self, path: str, new_path: str, overwrite: bool, retarget: bool) -> dict:
        vault_dir = Path(self.file_store.vault_path or ".")
        path_abs = (vault_dir / path).resolve() if path else None
        new_abs = (vault_dir / new_path).resolve() if not Path(new_path).is_absolute() else None
        precheck_error = _precheck_move(path, new_path, path_abs, new_abs, overwrite)
        if precheck_error:
            return precheck_error
        assert path_abs is not None and new_abs is not None  # narrowed by precheck
        new_abs.parent.mkdir(parents=True, exist_ok=True)

        # Step 1 — copy. Both files exist; inbound [[path]] still resolves.
        shutil.copyfile(str(path_abs), str(new_abs))
        payload: dict = {"path": path, "new_path": new_path, "size": new_abs.stat().st_size}

        # Step 2 — retarget. vault_dir stays consistent throughout: refs still
        # at [[path]] resolve to the original; refs already rewritten to
        # [[new_path]] resolve to the new location. On error, bail before
        # unlinking so the caller can retry; vault_dir is still consistent.
        if retarget:
            try:
                report = await retarget_links(self.file_store, src=path, dst=new_path)
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

        # Step 3 — unlink the original. Refs all point at new_path now; the
        # original is an orphan. If unlink fails, vault_dir is still
        # consistent (refs resolve to new_path); the original just lingers
        # as an orphan that the caller can clean up.
        try:
            path_abs.unlink()
            payload["src_removed"] = True
        except Exception as exc:
            payload["src_removed"] = False
            payload["src_remove_error"] = f"unlink raised: {exc!r}"

        return payload


def _precheck_move(
    path: str,
    new_path: str,
    path_abs: Path | None,
    new_abs: Path | None,
    overwrite: bool,
) -> dict | None:
    """Validate inputs for ``_move``; return an error payload or ``None`` when OK."""
    if path_abs is None or not path_abs.is_file():
        return {"path": path, "error": "not found"}
    if new_abs is None or "/" not in new_path:
        return {
            "new_path": new_path,
            "error": "new_path must be relative to the vault with a directory component",
        }
    if new_abs == path_abs:
        return {"path": path, "new_path": new_path, "error": "path and new_path are the same"}
    if new_abs.exists() and not overwrite:
        return {"new_path": new_path, "error": "destination exists; pass overwrite=True"}
    return None
