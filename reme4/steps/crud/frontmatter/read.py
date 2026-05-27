"""``frontmatter_read_step`` — return the frontmatter dict of a markdown file.

Cheap structured read — frontmatter only, no body. Use ``body:read``
for the post-frontmatter content slice, or whole-file ``read`` when
you want both at once. Returns ``{exists: false}`` when the target
doesn't exist; otherwise ``{exists: true, frontmatter: {...}}``.

``path`` is a path relative to the vault.
"""

from pathlib import Path

import frontmatter

from ...base_step import BaseStep

from ....components import R


@R.register("frontmatter_read_step")
class FrontmatterReadStep(BaseStep):
    """Read a markdown file's frontmatter (YAML metadata only)."""

    async def execute(self):
        assert self.context is not None
        path: str = self.context.get("path") or ""
        assert path, "path is required"

        target = (Path(self.file_store.vault_path or ".") / path).resolve()
        if not target.is_file():
            self.context.response.success = False
            self.context.response.answer = f"Error: {path} not found"
            self.context.response.metadata.update({"path": path, "exists": False})
            return
        if target.suffix != ".md":
            self.context.response.success = False
            self.context.response.answer = "Error: not markdown"
            self.context.response.metadata.update({"path": path, "error": "not markdown"})
            return

        meta = dict(frontmatter.loads(target.read_text(encoding="utf-8")).metadata)
        self.context.response.success = True
        self.context.response.answer = f"Read frontmatter from {path} ({len(meta)} key(s))"
        self.context.response.metadata.update({"path": path, "exists": True, "frontmatter": meta})
