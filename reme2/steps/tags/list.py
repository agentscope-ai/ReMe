"""``tags:list`` — enumerate every tag declared in the vault.

Single ``file_graph.get_nodes(None)`` call streams every real node;
we accumulate tag counts in one pass. No filesystem walk, no
per-file frontmatter parse.

Returns:

    {tags: [{tag, count}, ...], total: int}

Per-tag file lists live on ``tags:stat`` so this listing stays cheap.
"""

from __future__ import annotations

from collections import Counter

from agentscope.tool import ToolResponse

from ..base_step import BaseStep
from ..runtime_response import _set_answer, _tool_response

from ...component import R
from ...enumeration import ComponentEnum


async def _iter_tagged(file_store):
    """Yield ``(path, tags_list)`` for every indexed node that has tags.

    Owns the tag-extraction rule so ``tags:stat`` shares it via import —
    single source of truth for "what counts as tagged".
    """
    if not file_store.file_graph:
        return
    for node in await file_store.file_graph.get_nodes():
        tags = node.front_matter.tags or []
        if tags:
            yield node.path, [str(t) for t in tags if t]


async def _list(file_store, sort: str, limit: int | None) -> dict:
    counter: Counter[str] = Counter()
    async for _, tags in _iter_tagged(file_store):
        counter.update(tags)
    if sort == "alpha":
        items = [{"tag": t, "count": c} for t, c in sorted(counter.items())]
    else:  # "count" (default) — desc count then alpha tiebreak
        items = [
            {"tag": t, "count": c}
            for t, c in sorted(counter.items(), key=lambda kv: (-kv[1], kv[0]))
        ]
    if limit is not None and limit > 0:
        items = items[:limit]
    return {"tags": items, "total": len(counter)}


@R.register("tags:list")
class TagsList(BaseStep):
    """List every distinct tag in the vault with its document count."""

    component_type = ComponentEnum.STEP

    audit: list[dict] | None = None

    async def execute(self):
        assert self.context is not None
        sort = (self.context.get("sort") or "count").lower()
        limit = self.context.get("limit")
        payload = await _list(self.file_store, sort, int(limit) if limit else None)
        _set_answer(self.context, payload)

    async def tags_list(
        self,
        sort: str = "count",
        limit: int | None = None,
    ) -> ToolResponse:
        """List every tag in the vault with its document count.

        Args:
            sort: ``count`` (default, descending) or ``alpha``.
            limit: cap the number of returned tags; ``None`` = all.
        """
        payload = await _list(self.file_store, sort.lower(), limit)
        return _tool_response("tags:list", True, payload, audit=self.audit)
