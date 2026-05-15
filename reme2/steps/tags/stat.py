"""``tags:stat`` — usage statistics for a single tag.

Reuses ``_iter_tagged`` from ``tags:list`` so node discovery and tag
extraction follow the same rule. Filters the stream by the requested
tag and collects matching paths.

Returns:

    {tag, exists, count, paths, truncated}

``exists=False`` short-circuits to ``count=0`` and ``paths=[]`` —
useful for "does the agent need to introduce this tag?" probes.
``limit`` caps the path list (default 100); ``count`` is always
the true total.
"""

from __future__ import annotations

from agentscope.tool import ToolResponse

from .list import _iter_tagged

from ..base_step import BaseStep
from ..runtime_response import _set_answer, _tool_response

from ...component import R
from ...enumeration import ComponentEnum


async def _stat(file_store, tag: str, limit: int) -> dict:
    matches: list[str] = []
    count = 0
    async for path, tags in _iter_tagged(file_store):
        if tag in tags:
            count += 1
            if len(matches) < limit:
                matches.append(path)
    return {
        "tag": tag,
        "exists": count > 0,
        "count": count,
        "paths": matches,
        "truncated": count > len(matches),
    }


@R.register("tags:stat")
class TagsStat(BaseStep):
    """Usage statistics + file list for a single tag."""

    component_type = ComponentEnum.STEP

    audit: list[dict] | None = None

    async def execute(self):
        assert self.context is not None
        tag: str = self.context.get("tag", "") or ""
        assert tag, "tag is required"
        limit = int(self.context.get("limit") or 100)
        payload = await _stat(self.file_store, tag, limit)
        self.context.response.success = True
        _set_answer(self.context, payload)

    async def tags_stat(
        self,
        tag: str,
        limit: int = 100,
    ) -> ToolResponse:
        """Return usage statistics for ``tag``: count + file list.

        Args:
            tag: the tag to look up.
            limit: cap the number of returned paths (count is always
                the true total). Default 100.
        """
        payload = await _stat(self.file_store, tag, limit)
        return _tool_response("tags:stat", True, payload, audit=self.audit)
