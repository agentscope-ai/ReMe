"""``bm25_search_step`` — plain BM25 keyword search with tool_context dedup."""

from typing import Final

from ._dedup import _ToolContextDedupMixin
from ._source_format import ALL_RETURNED_MESSAGE, NO_RESULTS_MESSAGE, format_chunks_answer
from ..base_step import BaseStep
from ...components import R

_MAX_CANDIDATES: Final = 200
_CANDIDATE_MULTIPLIER: Final = 10


@R.register("bm25_search_step")
class Bm25SearchStep(_ToolContextDedupMixin, BaseStep):
    """BM25-only search: retrieve, filter by min_score, dedup by tool_context, truncate."""

    def __init__(self, *args, seen_ttl_hours: float = 24, include_source: bool = True, **kwargs):
        super().__init__(*args, **kwargs)
        self.seen_ttl_hours = seen_ttl_hours
        self.include_source = include_source

    async def execute(self):
        assert self.context is not None
        query: str = (self.context.get("query", "") or "").strip()
        limit: int = int(self.context.get("limit") or 5)
        min_score: float = float(self.context.get("min_score") or 0.0)
        tool_context_id: str = (self.context.get("tool_context_id", "") or "").strip()

        if not query:
            self.context.response.success = False
            self.context.response.answer = "Error: query cannot be empty"
            return self.context.response
        assert limit > 0, f"limit must be positive, got {limit}"

        candidates = min(_MAX_CANDIDATES, max(1, limit * _CANDIDATE_MULTIPLIER))
        results = await self.file_store.keyword_search(query, candidates, {})
        self.logger.info(f"[{self.name}] query={query!r} candidates={candidates} hits={len(results)}")

        if min_score > 0.0:
            results = [chunk for chunk in results if chunk.score >= min_score]

        pre_dedup_count = 0
        if tool_context_id:
            pre_dedup_count = len(results)
            results, _ = self._dedupe_tool_context(results, tool_context_id, limit)
        else:
            results = results[:limit]

        dialog_dir = self.config_value("dialog_dir")
        self.context.response.answer = format_chunks_answer(
            results,
            dialog_dir,
            include_source=self.include_source,
        )
        if not results:
            self.context.response.answer = ALL_RETURNED_MESSAGE if pre_dedup_count > 0 else NO_RESULTS_MESSAGE
        self.context.response.metadata["results"] = [
            c.model_dump(exclude_none=True, exclude={"embedding"}) for c in results
        ]
        return self.context.response
