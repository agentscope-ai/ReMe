"""``bm25_search_step`` — plain BM25 keyword search with tool_context dedup."""

import datetime
from typing import Final

from ..base_step import BaseStep
from ._source_format import render_with_source
from ...components import R
from ...schema import FileChunk

_MAX_CANDIDATES: Final = 200


@R.register("bm25_search_step")
class Bm25SearchStep(BaseStep):
    """BM25-only search: retrieve, filter by min_score, dedup by tool_context, truncate."""

    TOOL_CONTEXTS_KEY: Final[str] = "tool_contexts"
    SEARCH_SEEN_KEY: Final[str] = "search_seen_chunk_ids"

    def __init__(self, *args, seen_ttl_hours: float = 24, include_source: bool = True, **kwargs):
        super().__init__(*args, **kwargs)
        self.seen_ttl_hours = seen_ttl_hours
        self.include_source = include_source

    def _tool_context_store(self, tool_context_id: str) -> dict:
        """Return the mutable state bucket for a tool context."""
        if self.app_context is not None:
            contexts = self.app_context.metadata.setdefault(self.TOOL_CONTEXTS_KEY, {})
        else:
            contexts = self.kwargs.setdefault(self.TOOL_CONTEXTS_KEY, {})
        return contexts.setdefault(tool_context_id, {})

    def _dedupe_tool_context(self, chunks: list[FileChunk], tool_context_id: str, limit: int) -> list[FileChunk]:
        """Drop chunks already returned for this tool_context within the TTL window."""
        now = datetime.datetime.now().timestamp()
        ttl = float(self.seen_ttl_hours) * 60 * 60
        store = self._tool_context_store(tool_context_id)
        seen: dict = store.get(self.SEARCH_SEEN_KEY, {})
        seen = {cid: ts for cid, ts in seen.items() if now - float(ts) < ttl}

        returned = [c for c in chunks if c.id not in seen][:limit]
        for c in returned:
            seen[c.id] = now
        store[self.SEARCH_SEEN_KEY] = seen
        return returned

    async def execute(self):
        assert self.context is not None
        query: str = (self.context.get("query", "") or "").strip()
        limit: int = int(self.context.get("limit") or 5)
        tool_context_id: str = (self.context.get("tool_context_id", "") or "").strip()

        if not query:
            self.context.response.success = False
            self.context.response.answer = "Error: query cannot be empty"
            return self.context.response
        assert limit > 0, f"limit must be positive, got {limit}"

        candidates = min(_MAX_CANDIDATES, max(1, limit * 5))
        results = await self.file_store.keyword_search(query, candidates, {})
        self.logger.info(f"[{self.name}] query={query!r} candidates={candidates} hits={len(results)}")

        if tool_context_id:
            results = self._dedupe_tool_context(results, tool_context_id, limit)
        else:
            results = results[:limit]

        if self.include_source:
            self.context.response.answer = render_with_source(results, self.workspace_path)
        else:
            self.context.response.answer = "\n\n".join(c.text for c in results)
        self.context.response.metadata["results"] = [
            c.model_dump(exclude_none=True, exclude={"embedding"}) for c in results
        ]
        return self.context.response
