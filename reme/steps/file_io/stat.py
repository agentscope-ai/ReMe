"""``file_stat`` — peek at file metadata under the workspace without copying it.

Cheap inspection alternative to ``file_download``. The answer carries path,
type, size and mtime; MCP returns only this answer. MIME type and Markdown
frontmatter remain in ``metadata`` for direct ``Response`` consumers.

Returns a uniform envelope:

    {exists, type, size, mtime, ctime, mime, frontmatter}

``exists=False`` short-circuits everything else to ``None``. ``type``
is ``"file"`` / ``"dir"`` (covers event workspace probes too).
``frontmatter`` is populated only for ``.md`` files and only when
parsing succeeds — schema validity is a lint concern.

``path`` accepts a file path or directory path relative to the workspace.
Joined with ``file_store.workspace_path`` and inspected on disk.
"""

import mimetypes
from datetime import datetime
from pathlib import Path

import frontmatter

from ._path import check_path_permission, display_path, gate_md, resolve_path
from ..base_step import BaseStep
from ...core.registry import R


@R.register("stat_step")
class StatStep(BaseStep):
    """Return metadata for a file or directory under the workspace."""

    async def execute(self):
        assert self.context is not None
        path: str = self.context.get("path", "") or ""
        assert path, "path is required"

        workspace_dir = Path(self.file_store.workspace_path or ".").resolve()
        target, err = resolve_path(workspace_dir, path)
        if err or target is None:
            self.context.response.success = False
            self.context.response.answer = f"Error: {err or 'invalid path'}"
            self.context.response.metadata.update({"path": path, "exists": False, "error": err or "invalid path"})
            self.logger.info(f"[{self.name}] path={path} error={err!r}")
            return
        # Auto-append .md for suffix-less paths (consistent with edit). An
        # existing directory wins top priority: the daily layout keeps
        # <daily_dir>/<date>/ and its <daily_dir>/<date>.md index side by
        # side, and directory probing is a supported use of this step.
        # Otherwise fall back to the original when the .md candidate is
        # missing but the original exists (e.g., an extension-less file).
        #
        # Honor the request-scoped ``_allowed_paths`` constraint like ``read``
        # (stat also returns the frontmatter of Markdown files). Candidates
        # outside the scope are dropped before the precedence above applies,
        # so an out-of-scope directory or file can neither shadow an
        # authorized note nor reveal that it exists.
        original_target = target
        md_target, _is_md = gate_md(target)
        allowed_paths = self.context.get("_allowed_paths")

        def allowed(candidate: Path) -> bool:
            return check_path_permission(workspace_dir, candidate, allowed_paths)

        if original_target.is_dir() and allowed(original_target):
            target = original_target
        elif md_target.exists() and allowed(md_target):
            target = md_target
        elif original_target.exists() and allowed(original_target):
            target = original_target
        elif allowed(md_target):
            target = md_target
        elif allowed(original_target):
            target = original_target
        else:
            self.context.response.success = False
            self.context.response.answer = "Error: no permission to access this path"
            self.context.response.metadata.update({"path": path, "error": "no permission to access this path"})
            self.logger.info(f"[{self.name}] path={path} permission denied")
            return
        if not target.exists():
            probed = display_path(workspace_dir, target)
            self.context.response.success = False
            self.context.response.answer = f"stat: {probed} not found"
            not_found: dict = {"path": path, "exists": False}
            if target != original_target:
                not_found["resolved_path"] = probed
            self.context.response.metadata.update(not_found)
            self.logger.info(f"[{self.name}] path={path} exists=False")
            return

        st = target.stat()
        mtime = datetime.fromtimestamp(st.st_mtime).isoformat()
        payload: dict = {
            "path": path,
            "absolute_path": str(target),
            "exists": True,
            "type": "dir" if target.is_dir() else "file",
            "mtime": mtime,
            "ctime": datetime.fromtimestamp(st.st_ctime).isoformat(),
        }
        if target != original_target:
            payload["resolved_path"] = display_path(workspace_dir, target)
        if target.is_file():
            payload["size"] = st.st_size
            is_md = target.suffix.lower() == ".md"
            payload["mime"] = (
                "text/markdown" if is_md else (mimetypes.guess_type(target.name)[0] or "application/octet-stream")
            )
            if is_md:
                try:
                    meta = dict(frontmatter.loads(target.read_text(encoding="utf-8")).metadata)
                except Exception:
                    meta = {}
                payload["frontmatter"] = meta
            answer = f"stat: {path} (file, {st.st_size} bytes, mtime {mtime})"
        else:
            answer = f"stat: {path} (dir, mtime {mtime})"

        self.context.response.success = True
        self.context.response.answer = answer
        self.context.response.metadata.update(payload)
        self.logger.info(
            f"[{self.name}] path={path} type={payload['type']} "
            f"size={payload.get('size', '-')} mime={payload.get('mime', '-')}",
        )
