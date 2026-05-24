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

from __future__ import annotations

from pathlib import Path

import frontmatter

from ..base_step import BaseStep
from ...utils import set_answer

from ...components import R
from ...enumeration import ComponentEnum


async def _update(file_store, path: str, fields: dict) -> dict:
    if not path:
        return {"path": path, "error": "not found"}
    target = (Path(file_store.vault_path or ".") / path).resolve()
    if not target.is_file():
        return {"path": path, "error": "not found"}
    if target.suffix != ".md":
        return {"path": path, "error": "not markdown"}
    if not fields:
        return {"path": path, "error": "no fields to update"}
    raw = target.read_text(encoding="utf-8")
    post = frontmatter.loads(raw)
    post.metadata.update(fields)
    target.write_text(frontmatter.dumps(post), encoding="utf-8")
    return {"path": path, "updated": fields}


@R.register("frontmatter_update_step")
class FrontmatterUpdateStep(BaseStep):
    """Set frontmatter keys on a markdown file from a ``metadata`` dict."""

    component_type = ComponentEnum.STEP

    async def execute(self):
        assert self.context is not None
        path: str = self.context.get("path") or ""
        assert path, "path is required"
        metadata = self.context.get("metadata") or {}
        assert isinstance(metadata, dict), "metadata must be a dict"
        payload = await _update(self.file_store, path, metadata)
        self.context.response.success = "error" not in payload
        set_answer(self.context, payload)
