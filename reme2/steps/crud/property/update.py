"""``property:update`` — merge a patch into a memory file's frontmatter.

Read-modify-write the YAML frontmatter; body content is untouched.
The watcher / parser pick up the change asynchronously.

Two input modes (mutually compatible — they merge in order):
    * ``patch`` — explicit ``dict`` of frontmatter updates.
    * ``**fields`` — free-form key/value pairs (obsidian convention
      ``path=... key=val key=val``); merged on top of ``patch``.

``path`` accepts a short link or vault-relative path. Short-link
ambiguity is reported via ``error="ambiguous"`` with the candidate
list — the update is **not** executed in that case.
"""

from __future__ import annotations

import frontmatter
from agentscope.tool import ToolResponse

from ...base_step import BaseStep
from ...runtime_response import _set_answer, _tool_response

from ....component import R
from ....enumeration import ComponentEnum
from ....utils import path_resolver


# Context keys that belong to plumbing (request envelope, response
# slot, the path itself) and must never be promoted into a free-form
# frontmatter patch.
_RESERVED_CONTEXT_KEYS = {"path", "patch", "response", "request", "data"}


async def _update(file_store, path: str, patch: dict) -> dict:
    try:
        target = await path_resolver.resolve_to_absolute(file_store, path)
    except path_resolver.PathAmbiguous as e:
        return {"path": path, "error": "ambiguous", "candidates": e.candidates}
    except path_resolver.PathNotFound:
        return {"path": path, "error": "not found"}
    if not target.is_file():
        return {"path": path, "error": "not found"}
    if not patch:
        return {"path": path, "error": "patch is empty"}
    raw = target.read_text(encoding="utf-8")
    post = frontmatter.loads(raw)
    post.metadata.update(patch)
    target.write_text(frontmatter.dumps(post), encoding="utf-8")
    return {"path": path, "updated": list(patch.keys())}


@R.register("property:update")
class PropertyUpdate(BaseStep):
    """Merge a patch into a memory file's frontmatter."""

    component_type = ComponentEnum.STEP

    audit: list[dict] | None = None

    async def execute(self):
        assert self.context is not None
        path: str = self.context.get("path") or ""
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
        payload = await _update(self.file_store, path, dict(patch or {}))
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
        payload = await _update(self.file_store, path, merged)
        ok = "error" not in payload
        return _tool_response("property:update", ok, payload, audit=self.audit)
