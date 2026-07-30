"""Hybrid search (v2) over file_store using RRF fusion of vector + keyword results.

This is the local fork of the upstream search step. It uses
:class:`_ToolContextDedupMixin` for subset-aware interval-merging dedup and
:func:`format_chunks_answer` for session-aware chunk formatting with
:data:`ALL_RETURNED_MESSAGE` / :data:`NO_RESULTS_MESSAGE` notices.
"""

import asyncio
import datetime
import os
from typing import Final

from ._dedup import _ToolContextDedupMixin
from ._source_format import ALL_RETURNED_MESSAGE, NO_RESULTS_MESSAGE, format_chunks_answer
from ..base_step import BaseStep
from ..file_io import extract_daily_date
from ...components import R
from ...schema import FileChunk
from ...utils import expand_links

_RRF_K: Final = 60
_MAX_CANDIDATES: Final = 200
_DEFAULT_LIMIT_ENV: Final = "REME_SEARCH_LIMIT"
_DEFAULT_LIMIT: Final = 5


def _default_limit() -> int:
    value = os.getenv(_DEFAULT_LIMIT_ENV)
    if value is None:
        return _DEFAULT_LIMIT
    try:
        return int(value)
    except ValueError:
        return _DEFAULT_LIMIT


@R.register("search_v2_step")
class SearchV2Step(_ToolContextDedupMixin, BaseStep):
    """Hybrid search: run vector + keyword in parallel, fuse via RRF, filter, truncate."""

    def __init__(
        self,
        *args,
        seen_ttl_hours: float = 24,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.seen_ttl_hours = seen_ttl_hours

    @staticmethod
    def _rrf_merge(
        vector: list[FileChunk],
        keyword: list[FileChunk],
        vector_weight: float,
    ) -> list[FileChunk]:
        """Fuse two ranked lists with Reciprocal Rank Fusion, keyed by chunk.id."""
        text_weight = 1.0 - vector_weight
        merged: dict[str, FileChunk] = {}

        for rank, chunk in enumerate(vector, start=1):
            contrib = vector_weight / (_RRF_K + rank)
            c = chunk.model_copy(deep=False)
            c.scores = {**chunk.scores, "vector": chunk.scores.get("vector", chunk.score), "score": contrib}
            merged[c.id] = c

        for rank, chunk in enumerate(keyword, start=1):
            contrib = text_weight / (_RRF_K + rank)
            existing = merged.get(chunk.id)
            if existing is not None:
                existing.scores = {
                    **existing.scores,
                    "keyword": chunk.scores.get("keyword", chunk.score),
                    "score": existing.scores["score"] + contrib,
                }
            else:
                c = chunk.model_copy(deep=False)
                c.scores = {**chunk.scores, "keyword": chunk.scores.get("keyword", chunk.score), "score": contrib}
                merged[c.id] = c

        results = list(merged.values())
        results.sort(key=lambda r: r.score, reverse=True)
        return results

    @staticmethod
    def _format_scores(scores: dict[str, float], hybrid: bool) -> str:
        """Format scores for the answer line: always show fused; show per-branch when hybrid."""
        parts = [f"score={scores.get('score', 0.0):.4f}"]
        if hybrid:
            for k in ("vector", "keyword"):
                v = scores.get(k)
                parts.append(f"{k}={v:.4f}" if v is not None else f"{k}=-")
        return " ".join(parts)

    async def execute(self):
        assert self.context is not None
        query: str = (self.context.get("query", "") or "").strip()
        limit: int = int(self.context.get("limit") or _default_limit())
        min_score: float = float(self.context.get("min_score") or 0.0)
        # vector_weight: prefer agent-supplied context value; fallback to YAML kwargs / default 0.7.
        # Convertible numeric inputs are clipped to [0.0, 1.0]; non-numeric inputs are silently ignored.
        raw_vw = self.context.get("vector_weight")
        vector_weight: float | None = None
        if raw_vw is not None:
            try:
                vector_weight = float(raw_vw)
            except (TypeError, ValueError):
                self.logger.warning(
                    f"[{self.name}] non-numeric vector_weight={raw_vw!r}; ignoring and using default 0.7",
                )
                vector_weight = None
        if vector_weight is None:
            vector_weight = float(self.kwargs.get("vector_weight", 0.7))
        vector_weight = max(0.0, min(1.0, vector_weight))
        candidate_multiplier: float = float(self.kwargs.get("candidate_multiplier", 5.0))
        expand_links_enabled: bool = bool(self.kwargs.get("expand_links", True))
        max_links_per_direction: int = int(self.kwargs.get("max_links_per_direction", 10))
        tool_context_id: str = (self.context.get("tool_context_id", "") or "").strip()
        strict_date_filter: bool = bool(
            self.context.get("strict_date_filter") or self.kwargs.get("strict_date_filter", False),
        )

        if not query:
            self.context.response.success = False
            self.context.response.answer = "Error: query cannot be empty"
            return self.context.response
        assert limit > 0, f"limit must be positive, got {limit}"

        candidates = min(_MAX_CANDIDATES, max(1, int(limit * candidate_multiplier)))
        search_filter: dict = dict(self.context.get("search_filter", {}) or {})

        # Promote top-level date parameters into search_filter for file_store.
        for date_key in ("start_date", "end_date"):
            value = self.context.get(date_key)
            if value and date_key not in search_filter:
                search_filter[date_key] = value

        # Validate and normalize date filters before they reach file_store.
        # _matches_search_filter does lexicographic string comparison against
        # path_date (always a canonical YYYY-MM-DD), so raw caller values like
        # "2026-2-28" or "abc" would produce silently wrong results.
        for date_key in ("start_date", "end_date"):
            raw = search_filter.get(date_key)
            if raw is None:
                continue
            normalized = extract_daily_date(raw)
            if normalized is None:
                # Fallback: accept non-zero-padded dates like "2024-1-5".
                try:
                    normalized = (
                        datetime.datetime.strptime(
                            str(raw).strip(),
                            "%Y-%m-%d",
                        )
                        .date()
                        .isoformat()
                    )
                except ValueError:
                    self.logger.warning(
                        f"Ignoring invalid {date_key}={raw!r}; " f"expected a valid YYYY-MM-DD date.",
                    )
                    del search_filter[date_key]
                    continue
            search_filter[date_key] = normalized

        if strict_date_filter:
            search_filter["strict_date_filter"] = True

        vector_results, keyword_results = await asyncio.gather(
            self.file_store.vector_search(query, candidates, search_filter),
            self.file_store.keyword_search(query, candidates, search_filter),
        )

        self.logger.info(
            f"[{self.name}] query={query!r} candidates={candidates} "
            f"vector_hits={len(vector_results)} keyword_hits={len(keyword_results)}",
        )

        hybrid = bool(vector_results) and bool(keyword_results)
        if not vector_results and not keyword_results:
            fused: list[FileChunk] = []
        elif not keyword_results:
            fused = vector_results
        elif not vector_results:
            fused = keyword_results
        else:
            fused = self._rrf_merge(vector_results, keyword_results, vector_weight)

        if min_score > 0.0:
            fused = [c for c in fused if c.score >= min_score]

        pre_dedup_count = 0
        dedup: dict | None = None
        if tool_context_id:
            pre_dedup_count = len(fused)
            fused, dedup = self._dedupe_tool_context(
                fused,
                tool_context_id,
                limit,
                clock=self.kwargs.get("clock"),
                ttl_override=self.kwargs.get("tool_context_chunk_ttl_seconds"),
            )
        else:
            fused = fused[:limit]

        unique_paths = list(dict.fromkeys(c.path for c in fused))
        link_expansion: dict[str, dict] = (
            await expand_links(self.file_store, unique_paths, max_links_per_direction) if expand_links_enabled else {}
        )

        dialog_dir = self.config_value("dialog_dir")
        self.context.response.answer = format_chunks_answer(
            fused,
            dialog_dir,
            score_fn=lambda c: self._format_scores(c.scores, hybrid),
            link_expansion=link_expansion,
        )
        if not fused:
            self.context.response.answer = ALL_RETURNED_MESSAGE if pre_dedup_count > 0 else NO_RESULTS_MESSAGE
        self.context.response.metadata["results"] = [
            c.model_dump(exclude_none=True, exclude={"embedding"}) for c in fused
        ]
        self.context.response.metadata["link_expansion"] = link_expansion
        self.context.response.metadata["counts"] = {
            "vector": len(vector_results),
            "keyword": len(keyword_results),
            "returned": len(fused),
            "hybrid": hybrid,
        }
        if dedup is not None:
            self.context.response.metadata["dedup"] = dedup
        return self.context.response
