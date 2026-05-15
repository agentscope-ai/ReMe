"""``property:update`` — merge a patch into a memory file's frontmatter.

Patch semantics are merge-write: keys in ``patch`` overwrite existing
keys, keys not in ``patch`` are left untouched. To remove a key,
use ``property:delete`` — passing ``None`` here just sets the literal
None (which is rarely what you want).

Per the obsidian config convention the agent calls this with
arbitrary keyword arguments — ``path="My Note" status=done xx=xxx``.
The Step packs everything except ``path`` into the patch dict, so
the tool method accepts ``**fields`` directly.
"""

from __future__ import annotations

import frontmatter
from agentscope.tool import ToolResponse

from ..download import resolve_path

from ...base_step import BaseStep
from ...runtime_response import _set_answer, _tool_response

from ....component import R
from ....enumeration import ComponentEnum


# Context keys that belong to plumbing (request envelope, response
# slot, the path itself) and must never be promoted into a free-form
# frontmatter patch.
_RESERVED_CONTEXT_KEYS = {
    "path", "path", "patch", "response", "request", "data",
}


def _update(file_store, path: str, patch: dict) -> dict:
    target = resolve_path(file_store, path)
    if not target.is_file():
        return {"path": path, "error": "not found"}
    if not patch:
        return {"path": path, "error": "patch is empty"}
    raw = target.read_text(encoding="utf-8")
    post = frontmatter.loads(raw)
    post.metadata.update(patch)
    target.write_text(frontmatter.dumps(post), encoding="utf-8")
    return {
        "path": path,
        "applied": dict(patch),
        "frontmatter": dict(post.metadata),
    }


@R.register("property:update")
class PropertyUpdate(BaseStep):
    """Merge a patch into a memory file's frontmatter."""

    component_type = ComponentEnum.STEP

    audit: list[dict] | None = None

    async def execute(self):
        assert self.context is not None
        path: str = self.context.get("path") or self.context.get("path") or ""
        assert path, "path is required"
        patch = self.context.get("patch")
        if patch is None:
            # Free-form mode — every other context key is treated as a
            # patch entry, matching the obsidian convention
            # `path=... key=val key=val`.
            patch = {
                k: v for k, v in self.context.data.items()
                if k not in _RESERVED_CONTEXT_KEYS
            }
        payload = _update(self.file_store, path, dict(patch or {}))
        self.context.response.success = "error" not in payload
        _set_answer(self.context, payload)

    async def property_update(
        self,
        path: str,
        patch: dict | None = None,
        **fields,
    ) -> ToolResponse:
        """Merge ``patch`` (or free-form ``**fields``) into the
        frontmatter at ``path``."""
        merged: dict = {}
        if patch:
            merged.update(patch)
        if fields:
            merged.update(fields)
        payload = _update(self.file_store, path, merged)
        ok = "error" not in payload
        return _tool_response("property:update", ok, payload, audit=self.audit)
