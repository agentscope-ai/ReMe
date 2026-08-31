"""``frontmatter_read_step`` — return the frontmatter dict of a markdown file.

Cheap structured read — frontmatter only, no body. Use ``body:read``
for the post-frontmatter content slice, or whole-file ``read`` when
you want both at once. Returns ``{exists: false}`` when the target
doesn't exist; otherwise ``{exists: true, frontmatter: {...}}``.

``path`` is a path relative to the workspace.
"""

from pathlib import Path

import frontmatter
import yaml

from ._path import resolve_path
from ..base_step import BaseStep
from ...components import R


@R.register("frontmatter_read_step")
class FrontmatterReadStep(BaseStep):
    """Read a Markdown file's frontmatter (YAML metadata only)."""

    async def execute(self):
        assert self.context is not None
        path: str = self.context.get("path") or ""
        assert path, "path is required"

        workspace_dir = Path(self.file_store.workspace_path or ".").resolve()
        target, err = resolve_path(workspace_dir, path)
        if err or target is None:
            self.context.response.success = False
            self.context.response.answer = f"Error: {err or 'invalid path'}"
            self.context.response.metadata.update({"path": path, "exists": False, "error": err or "invalid path"})
            self.logger.info(f"[{self.name}] path={path} error={err!r}")
            return
        if not target.is_file():
            self.context.response.success = False
            self.context.response.answer = f"Error: {path} not found"
            self.context.response.metadata.update({"path": path, "exists": False})
            self.logger.info(f"[{self.name}] path={path} exists=False")
            return
        if target.suffix != ".md":
            self.context.response.success = False
            self.context.response.answer = "Error: not markdown"
            self.context.response.metadata.update({"path": path, "error": "not markdown"})
            self.logger.info(f"[{self.name}] path={path} error=not_markdown")
            return

        try:
            meta = dict(frontmatter.loads(target.read_text(encoding="utf-8")).metadata)
        except yaml.YAMLError as exc:
            self.context.response.success = False
            self.context.response.answer = f"Error: failed to parse frontmatter in {path}: {exc}"
            self.context.response.metadata.update({"path": path, "exists": True, "error": str(exc)})
            self.logger.info(f"[{self.name}] path={path} parse_error={exc!r}")
            return
        self.context.response.success = True
        self.context.response.answer = f"Read frontmatter from {path} ({len(meta)} key(s))"
        self.context.response.metadata.update({"path": path, "exists": True, "frontmatter": meta})
        self.logger.info(f"[{self.name}] path={path} keys={len(meta)}")
