"""``file_stat`` — peek at file metadata under the workspace without copying it.

Cheap inspection alternative to ``file_download``: the agent gets
size, mtime, mime type, and (for markdown files) the parsed
frontmatter — enough to decide whether to download / parse / skip
without paying the copy cost.

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

from ._path import display_path, gate_md, resolve_path
from ..base_step import BaseStep
from ...components import R


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
        original_target = target
        target, _is_md = gate_md(target)
        if original_target.is_dir() or (not target.exists() and original_target.exists()):
            target = original_target
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
        payload: dict = {
            "path": path,
            "absolute_path": str(target),
            "exists": True,
            "type": "dir" if target.is_dir() else "file",
            "mtime": datetime.fromtimestamp(st.st_mtime).isoformat(),
            "ctime": datetime.fromtimestamp(st.st_ctime).isoformat(),
        }
        if target != original_target:
            payload["resolved_path"] = display_path(workspace_dir, target)
        if target.is_file():
            payload["size"] = st.st_size
            payload["mime"] = mimetypes.guess_type(target.name)[0] or "application/octet-stream"
            if target.suffix == ".md":
                try:
                    meta = dict(frontmatter.loads(target.read_text(encoding="utf-8")).metadata)
                except Exception:
                    meta = {}
                payload["frontmatter"] = meta
            answer = f"stat: {path} (file, {st.st_size} bytes)"
        else:
            answer = f"stat: {path} (dir)"

        self.context.response.success = True
        self.context.response.answer = answer
        self.context.response.metadata.update(payload)
        self.logger.info(
            f"[{self.name}] path={path} type={payload['type']} "
            f"size={payload.get('size', '-')} mime={payload.get('mime', '-')}",
        )
