"""``frontmatter_delete_step`` — remove keys from a markdown file's frontmatter.

Returns both ``deleted`` (keys that were present and removed) and
``missing`` (keys that weren't there) so the agent can tell whether
a no-op happened. The file is rewritten only when at least one key
is actually removed — calling delete with all-missing keys is a
zero-side-effect read.

``path`` is a path relative to the vault.
"""

from __future__ import annotations

from pathlib import Path

import frontmatter

from ..base_step import BaseStep
from ...utils import set_answer

from ...components import R
from ...enumeration import ComponentEnum


async def _delete(file_store, path: str, keys: list[str]) -> dict:
    if not path:
        return {"path": path, "error": "not found"}
    target = (Path(file_store.vault_path or ".") / path).resolve()
    if not target.is_file():
        return {"path": path, "error": "not found"}
    if target.suffix != ".md":
        return {"path": path, "error": "not markdown"}
    if not keys:
        return {"path": path, "error": "keys is empty"}
    raw = target.read_text(encoding="utf-8")
    post = frontmatter.loads(raw)
    deleted: list[str] = []
    missing: list[str] = []
    for k in keys:
        if k in post.metadata:
            del post.metadata[k]
            deleted.append(k)
        else:
            missing.append(k)
    if deleted:
        target.write_text(frontmatter.dumps(post), encoding="utf-8")
    return {
        "path": path,
        "deleted": deleted,
        "missing": missing,
        "frontmatter": dict(post.metadata),
    }


@R.register("frontmatter_delete_step")
class FrontmatterDeleteStep(BaseStep):
    """Remove keys from a markdown file's frontmatter."""

    component_type = ComponentEnum.STEP

    async def execute(self):
        assert self.context is not None
        path: str = self.context.get("path") or ""
        assert path, "path is required"
        keys = self.context.get("keys") or []
        if isinstance(keys, str):
            keys = [keys]
        payload = await _delete(self.file_store, path, list(keys))
        self.context.response.success = "error" not in payload
        set_answer(self.context, payload)
