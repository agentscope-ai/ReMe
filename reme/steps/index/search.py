"""Hybrid search over file_store using RRF fusion of vector + keyword results."""

import asyncio
import datetime
import re

from ..base_step import BaseStep
from ..file_io import extract_daily_date
from ...components import R
from ...schema import FileChunk
from ...utils import expand_links, render_expansion_lines

_RRF_K = 60
_MAX_CANDIDATES = 200
_ISO_DATE_RE = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")
_DAY_MONTH_RE = re.compile(
    r"\b(?P<day>\d{1,2})\s+(?P<month>Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|"
    r"Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Sept(?:ember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)"
    r",?\s+(?P<year>\d{4})\b",
    re.IGNORECASE,
)
_MONTH_DAY_RE = re.compile(
    r"\b(?P<month>Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|"
    r"Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Sept(?:ember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)"
    r"\s+(?P<day>\d{1,2}),?\s+(?P<year>\d{4})\b",
    re.IGNORECASE,
)
_MONTHS = {
    "jan": 1,
    "january": 1,
    "feb": 2,
    "february": 2,
    "mar": 3,
    "march": 3,
    "apr": 4,
    "april": 4,
    "may": 5,
    "jun": 6,
    "june": 6,
    "jul": 7,
    "july": 7,
    "aug": 8,
    "august": 8,
    "sep": 9,
    "sept": 9,
    "september": 9,
    "oct": 10,
    "october": 10,
    "nov": 11,
    "november": 11,
    "dec": 12,
    "december": 12,
}


@R.register("search_step")
class SearchStep(BaseStep):
    """Hybrid search: run vector + keyword in parallel, fuse via RRF, filter, truncate."""

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
        if scores.get("temporal"):
            parts.append(f"temporal={scores['temporal']:.4f}")
        return " ".join(parts)

    @staticmethod
    def _date_from_parts(year: str, month: str, day: str) -> str | None:
        month_num = _MONTHS.get(month.lower())
        if month_num is None:
            return None
        try:
            return datetime.date(int(year), month_num, int(day)).isoformat()
        except ValueError:
            return None

    @classmethod
    def _query_dates(cls, query: str) -> set[str]:
        """Extract explicit date constraints from the query."""
        dates = set(_ISO_DATE_RE.findall(query))
        for pattern in (_DAY_MONTH_RE, _MONTH_DAY_RE):
            for match in pattern.finditer(query):
                date = cls._date_from_parts(match.group("year"), match.group("month"), match.group("day"))
                if date:
                    dates.add(date)
        return dates

    @staticmethod
    def _chunk_temporal_text(chunk: FileChunk) -> str:
        metadata = " ".join(str(value) for value in chunk.metadata.values())
        return " ".join([chunk.path, metadata, chunk.text])

    @classmethod
    def _apply_temporal_boost(
        cls,
        chunks: list[FileChunk],
        query: str,
        temporal_boost: float,
    ) -> list[FileChunk]:
        """Optionally boost chunks that mention explicit dates from the query."""
        if temporal_boost <= 0.0:
            return chunks
        query_dates = cls._query_dates(query)
        if not query_dates:
            return chunks

        boosted: list[FileChunk] = []
        for chunk in chunks:
            temporal_text = cls._chunk_temporal_text(chunk)
            matched = [date for date in query_dates if date in temporal_text]
            if not matched:
                boosted.append(chunk)
                continue
            c = chunk.model_copy(deep=False)
            score = chunk.score * (1.0 + temporal_boost)
            c.scores = {**chunk.scores, "score": score, "temporal": temporal_boost}
            boosted.append(c)
        boosted.sort(key=lambda r: r.score, reverse=True)
        return boosted

    async def execute(self):
        assert self.context is not None
        query: str = (self.context.get("query", "") or "").strip()
        limit: int = int(self.context.get("limit") or 5)
        min_score: float = float(self.context.get("min_score") or 0.0)
        vector_weight: float = float(self.kwargs.get("vector_weight", 0.7))
        candidate_multiplier: float = float(self.kwargs.get("candidate_multiplier", 3.0))
        temporal_boost: float = float(self.kwargs.get("temporal_boost", 0.0))
        expand_links_enabled: bool = bool(self.kwargs.get("expand_links", True))
        max_links_per_direction: int = int(self.kwargs.get("max_links_per_direction", 10))
        strict_date_filter: bool = bool(
            self.context.get("strict_date_filter") or self.kwargs.get("strict_date_filter", False),
        )

        if not query:
            self.context.response.success = False
            self.context.response.answer = "Error: query cannot be empty"
            return self.context.response
        assert 0.0 <= vector_weight <= 1.0, f"vector_weight must be in [0, 1], got {vector_weight}"
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

        fused = self._apply_temporal_boost(fused, query, temporal_boost)

        if min_score > 0.0:
            fused = [c for c in fused if c.score >= min_score]
        fused = fused[:limit]

        unique_paths = list(dict.fromkeys(c.path for c in fused))
        link_expansion: dict[str, dict] = (
            await expand_links(self.file_store, unique_paths, max_links_per_direction) if expand_links_enabled else {}
        )

        answer_lines: list[str] = []
        for c in fused:
            answer_lines.append(
                f"========== {c.path}:{c.start_line}-{c.end_line} "
                f"[{self._format_scores(c.scores, hybrid)}] ==========\n{c.text}",
            )
            answer_lines.extend(render_expansion_lines(link_expansion.get(c.path, {})))

        self.context.response.answer = "\n".join(answer_lines)
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
        return self.context.response
