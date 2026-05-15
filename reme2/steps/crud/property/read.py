"""``property:read`` — return the frontmatter dict of a memory file.

Cheap structured read — frontmatter only, no body. Use ``crud:read``
when you need the body too. Returns ``{exists: false}`` when the
target doesn't exist; otherwise ``{exists: true, frontmatter: {...}}``.

``path`` accepts a short link or vault-relative path. Short-link
ambiguity is reported via ``error="ambiguous"`` with the candidate
list — the read is **not** executed in that case.
"""

from __future__ import annotations

import frontmatter
from agentscope.tool import ToolResponse

from ...base_step import BaseStep
from ...runtime_response import _set_answer, _tool_response

from ....component import R
from ....enumeration import ComponentEnum
from ....utils import path_resolver


async def _read(file_store, path: str) -> dict:
    try:
        target = await path_resolver.resolve_to_absolute(file_store, path)
    except path_resolver.PathAmbiguous as e:
        return {"path": path, "error": "ambiguous", "candidates": e.candidates}
    except path_resolver.PathNotFound:
        return {"path": path, "exists": False}
    if not target.is_file():
        return {"path": path, "exists": False}
    raw = target.read_text(encoding="utf-8")
    meta = dict(frontmatter.loads(raw).metadata)
    return {"path": path, "exists": True, "frontmatter": meta}


@R.register("property:read")
class PropertyRead(BaseStep):
    """Read a memory file's frontmatter (YAML metadata only)."""

    component_type = ComponentEnum.STEP

    audit: list[dict] | None = None

    async def execute(self):
        assert self.context is not None
        path: str = self.context.get("path") or ""
        assert path, "path is required"
        payload = await _read(self.file_store, path)
        self.context.response.success = payload.get("exists", False)
        _set_answer(self.context, payload)

    async def property_read(self, path: str) -> ToolResponse:
        """Return the frontmatter dict of the memory file at ``path``."""
        payload = await _read(self.file_store, path)
        ok = payload.get("exists", False)
        return _tool_response("property:read", ok, payload, audit=self.audit)
