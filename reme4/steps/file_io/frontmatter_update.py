"""``frontmatter_update_step`` — set frontmatter keys on a markdown file.

Read-modify-write the YAML frontmatter; body content is untouched.
The watcher / parser pick up the change asynchronously.

Input shape: ``frontmatter_update_step path=foo.md metadata={"x": "y", "z": "w"}``
— ``metadata`` is an explicit dict whose entries are merged into the
file's frontmatter (existing keys overwritten, missing keys inserted).

``path`` is a path relative to the vault. Non-markdown targets return
``error="not markdown"``. An empty or missing ``metadata`` returns
``error="no fields to update"``.
"""

from pathlib import Path

import frontmatter

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

        target = (Path(self.file_store.vault_path or ".") / path).resolve()
        if not target.is_file():
            payload: dict = {"path": path, "error": "not found"}
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
