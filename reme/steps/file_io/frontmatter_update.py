"""``frontmatter_update_step`` — set frontmatter keys on a markdown file.

Read-modify-write the YAML frontmatter; body content is untouched.
The watcher / parser pick up the change asynchronously.

Input shape: ``frontmatter_update_step path=foo.md metadata={"x": "y", "z": "w"}``
— ``metadata`` is an explicit dict whose entries are merged into the
file's frontmatter (existing keys overwritten, missing keys inserted).

``path`` is a path relative to the workspace. Non-markdown targets return
``error="not markdown"``. An empty or missing ``metadata`` returns
``error="no fields to update"``.
"""

from pathlib import Path

import frontmatter

from ._file_io import get_path_lock
from ._path import resolve_path
from ..base_step import BaseStep
from ...components import R


@R.register("frontmatter_update_step")
class FrontmatterUpdateStep(BaseStep):
    """Set frontmatter keys on a markdown file from a ``metadata`` dict."""

    async def execute(self):
        assert self.context is not None
        path: str = self.context.get("path") or ""
        assert path, "path is required"
        metadata = self.context.get("metadata") or {}
        assert isinstance(metadata, dict), "metadata must be a dict"

        workspace_dir = Path(self.file_store.workspace_path or ".").resolve()
        target, err = resolve_path(workspace_dir, path)
        if err or target is None:
            payload: dict = {"path": path, "error": err or "invalid path"}
        else:
            lock = await get_path_lock(target)
            async with lock:
                if not target.is_file():
                    payload = {"path": path, "error": "not found"}
                elif target.suffix != ".md":
                    payload = {"path": path, "error": "not markdown"}
                elif not metadata:
                    payload = {"path": path, "error": "no fields to update"}
                else:
                    post = frontmatter.loads(target.read_text(encoding="utf-8"))
                    post.metadata.update(metadata)
                    target.write_text(frontmatter.dumps(post), encoding="utf-8")
                    payload = {"path": path, "updated": metadata}

        if "error" in payload:
            self.context.response.success = False
            self.context.response.answer = f"Error: {payload['error']}"
            self.logger.info(f"[{self.name}] update failed path={path} error={payload['error']!r}")
        else:
            self.context.response.success = True
            self.context.response.answer = f"Updated {len(metadata)} key(s) on {path}"
            self.logger.info(f"[{self.name}] path={path} keys={list(metadata.keys())}")
        self.context.response.metadata.update(payload)
