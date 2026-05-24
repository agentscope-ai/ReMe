"""``frontmatter_read_step`` — return the frontmatter dict of a markdown file.

Cheap structured read — frontmatter only, no body. Use ``body:read``
for the post-frontmatter content slice, or whole-file ``read`` when
you want both at once. Returns ``{exists: false}`` when the target
doesn't exist; otherwise ``{exists: true, frontmatter: {...}}``.

``path`` is a path relative to the vault.
"""

from __future__ import annotations

from pathlib import Path

import frontmatter

from ..base_step import BaseStep
from ...utils import set_answer

from ...components import R
from ...enumeration import ComponentEnum


async def _read(file_store, path: str) -> dict:
    if not path:
        return {"path": path, "exists": False}
    target = (Path(file_store.vault_path or ".") / path).resolve()
    if not target.is_file():
        return {"path": path, "exists": False}
    if target.suffix != ".md":
        return {"path": path, "error": "not markdown"}
    raw = target.read_text(encoding="utf-8")
    meta = dict(frontmatter.loads(raw).metadata)
    return {"path": path, "exists": True, "frontmatter": meta}


@R.register("frontmatter_read_step")
class FrontmatterReadStep(BaseStep):
    """Read a markdown file's frontmatter (YAML metadata only)."""

    component_type = ComponentEnum.STEP

    async def execute(self):
        assert self.context is not None
        path: str = self.context.get("path") or ""
        assert path, "path is required"
        payload = await _read(self.file_store, path)
        self.context.response.success = payload.get("exists", False)
        set_answer(self.context, payload)
