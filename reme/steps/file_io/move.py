"""``file_move`` — relocate / rename a file within the workspace by copy → retarget → delete.

Three-step ordering keeps workspace_dir referentially consistent at every
intermediate point — no window in which inbound ``[[src_path]]`` wikilinks
dangle:

  1. ``shutil.copyfile(src_path, dst_path)``. Both files now exist on disk;
     inbound ``[[src_path]]`` still resolves (to the original location).
  2. ``retarget_links(src_path, dst_path)``. Rewrites every inbound
     ``[[src_path]]`` → ``[[dst_path]]`` across the workspace. Both files
     exist throughout, so rewrites can land in any order without
     breaking resolution.
  3. ``src_abs.unlink()``. References now point at ``dst_path``; the
     original is an orphan and is safely removed.

If retargeting fails (raises or returns an error payload), step 3 is
skipped — both files remain on disk so the caller can diagnose and
retry; workspace_dir stays consistent (references still resolve to the
original). The ``src_removed`` boolean in the payload distinguishes
the two cases.

``src_path`` must resolve inside workspace_dir as a path relative to the
workspace; ``dst_path`` must be relative to the workspace with a directory
component (same rule as ``file_upload``). For cross-realm transfer
(workspace_dir ↔ local fs) use ``file_upload`` / ``file_download``.

Opt out via ``retarget=False`` for the rare case where you intentionally
want to leave references stale (e.g. moving aside before delete) — the
original is still removed in that case (move semantics, not copy).

Permission: honors the request-scoped ``_allowed_paths`` constraint
injected by the server into the RuntimeContext; both the source (which is
unlinked) and the destination (which is written) must be inside the scope.
Automatic inbound-link retargeting counts as a write too: with a scope
injected, a dry run rejects the move up front when retargeting would
rewrite a referencing file outside it, and the same constraint is passed
into the real rewrite pass so every file retargeting is about to modify is
re-checked at write time — a referencing file that gains a matching link
between the two passes (concurrent save, lagging graph backend) cannot
bypass the scope.
"""

import shutil
from pathlib import Path

from ._path import check_path_permission, resolve_path
from ..base_step import BaseStep
from ...core.registry import R
from ...utils.documents.wikilink_handler import WikilinkHandler


@R.register("move_step")
class MoveStep(BaseStep):
    """Move ``src_path`` to ``dst_path`` within the workspace (copy → retarget → unlink)."""

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

    # pylint: disable=too-many-return-statements
    async def _move(self, src_path: str, dst_path: str, overwrite: bool, retarget: bool) -> dict:
        workspace_dir = Path(self.file_store.workspace_path or ".").resolve()
        src_abs, src_err = resolve_path(workspace_dir, src_path) if src_path else (None, "src_path is required")
        dst_abs, dst_err = resolve_path(workspace_dir, dst_path) if dst_path else (None, "dst_path is required")
        if src_err:
            return {"src_path": src_path, "error": src_err}
        if dst_err:
            return {"dst_path": dst_path, "error": dst_err}
        precheck_error = _precheck_move(src_path, dst_path, src_abs, dst_abs, overwrite)
        if precheck_error:
            return precheck_error
        assert src_abs is not None and dst_abs is not None  # narrowed by precheck
        allowed_paths = self.context.get("_allowed_paths")
        if not check_path_permission(workspace_dir, src_abs, allowed_paths):
            return {"src_path": src_path, "dst_path": dst_path, "error": "no permission to move this file"}
        if not check_path_permission(workspace_dir, dst_abs, allowed_paths):
            return {"src_path": src_path, "dst_path": dst_path, "error": "no permission to move this file"}
        retarget_guard = None
        if retarget and allowed_paths is not None:
            retarget_guard = _retarget_write_guard(workspace_dir, allowed_paths)
            retarget_precheck_error = await _precheck_retarget_permission(
                self.file_store,
                src_path=src_path,
                dst_path=dst_path,
                write_guard=retarget_guard,
            )
            if retarget_precheck_error:
                retarget_precheck_error.setdefault("src_path", src_path)
                retarget_precheck_error.setdefault("dst_path", dst_path)
                return retarget_precheck_error
        dst_abs.parent.mkdir(parents=True, exist_ok=True)

        # Step 1 — copy. Wikilinks use workspace-relative paths, so outgoing
        # targets do not change when their containing document moves.
        shutil.copyfile(str(src_abs), str(dst_abs))
        payload: dict = {"src_path": src_path, "dst_path": dst_path, "size": dst_abs.stat().st_size}

        # Step 2 — retarget. workspace_dir stays consistent throughout: refs still
        # at [[src_path]] resolve to the original; refs already rewritten to
        # [[dst_path]] resolve to the new location. On error, bail before
        # unlinking so the caller can retry; workspace_dir is still consistent.
        if retarget:
            try:
                report = await WikilinkHandler.retarget_links(
                    self.file_store,
                    src=src_path,
                    dst=dst_path,
                    write_guard=retarget_guard,
                )
            except Exception as exc:
                error = f"retarget raised: {exc!r}"
                payload["error"] = error
                payload["retarget"] = {"error": error}
                payload["src_removed"] = False
                return payload
            if "error" in report:
                payload["error"] = report["error"]
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
        # original is an orphan. If unlink fails, workspace_dir is still
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
            "error": "dst_path must be relative to the workspace with a directory component",
        }
    if dst_abs == src_abs:
        return {"src_path": src_path, "dst_path": dst_path, "error": "src_path and dst_path are the same"}
    if dst_abs.exists() and not overwrite:
        return {"dst_path": dst_path, "error": "destination exists; pass overwrite=True"}
    return None


def _retarget_write_guard(workspace_dir: Path, allowed_paths):
    """Build the per-target permission gate for automatic link retargeting.

    The gate maps a workspace-relative referencing file to ``None`` (allowed)
    or a denial reason, using the same ``resolve_path`` +
    ``check_path_permission`` helpers as the endpoint checks
    (component-bounded containment, fail-closed on invalid scopes). It is
    passed to ``retarget_links`` for both the dry-run precheck and the real
    rewrite pass, so the constraint is re-evaluated on the files the rewrite
    actually touches — not just on the snapshot the precheck saw.

    The gate authorizes the same literal file identity the writer uses:
    ``retarget_links`` reads and writes ``workspace_dir / rel`` verbatim,
    while ``resolve_path`` strips surrounding whitespace before resolving.
    Authorizing the trimmed form would vet a different file than the one
    modified — a graph-registered ``" allowed/ref.md"`` (sibling directory
    named with a leading space, distinct from ``allowed/``) would pass as
    the trimmed, in-scope ``allowed/ref.md``. Paths whose identity would
    change under trimming are therefore denied outright (fail closed), so
    the vetted path and the written path always coincide.
    """

    def guard(rel: str) -> str | None:
        if rel != rel.strip():
            return f"no permission to retarget links in {rel!r}: path identity would change under trimming"
        resolved, err = resolve_path(workspace_dir, rel)
        if err or resolved is None or not check_path_permission(workspace_dir, resolved, allowed_paths):
            return f"no permission to retarget links in {rel}"
        return None

    return guard


async def _precheck_retarget_permission(
    file_store,
    src_path: str,
    dst_path: str,
    write_guard,
) -> dict | None:
    """Reject the move when automatic retargeting would rewrite a file outside the scope.

    ``retarget_links`` rewrites inbound ``[[src_path]]`` references wherever the
    file_graph finds them — including files outside ``_allowed_paths`` — so the
    endpoints alone don't cover every write this step performs. A dry run with
    the write guard attached enumerates exactly the files retargeting would
    modify (candidates the graph reports AND that still contain a matching
    link) and vets each one; any violation fails the move before the copy, the
    rewrites, or the unlink happen. Only runs when a scope is injected; without
    ``_allowed_paths`` retargeting stays unrestricted. The same guard then runs
    again in the real rewrite pass, because files are re-read at rewrite time
    and may have changed since the dry run.
    """
    try:
        report = await WikilinkHandler.retarget_links(
            file_store,
            src=src_path,
            dst=dst_path,
            dry_run=True,
            write_guard=write_guard,
        )
    except Exception as exc:
        return {"error": f"retarget precheck raised: {exc!r}"}
    if "error" in report:
        return {
            "error": f"retarget precheck failed: {report['error']}",
            "retarget_blocked_path": report.get("retarget_blocked_path"),
        }
    return None
