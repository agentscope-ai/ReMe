"""Hybrid search over file_store using RRF fusion of vector + keyword results."""

import asyncio

from .link_expansion import get_first_order_neighbors_batch
from ..base_step import BaseStep
from ...components import R
from ...schema import FileChunk, FileNode

_RRF_K = 60
_MAX_CANDIDATES = 200


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
        return " ".join(parts)

    @staticmethod
    def _node_meta(node: FileNode | None) -> dict:
        """Extract a compact meta dict (title/description/tags) from a FileNode."""
        if node is None:
            return {}
        fm = node.front_matter
        meta: dict = {}
        if fm.title:
            meta["title"] = fm.title
        if fm.description:
            meta["description"] = fm.description
        if fm.tags:
            meta["tags"] = list(fm.tags)
        return meta

    @staticmethod
    def _format_meta_inline(meta: dict) -> str:
        """One-line render of node meta for the answer; '(no meta)' when empty."""
        parts = []
        if "title" in meta:
            parts.append(f'title="{meta["title"]}"')
        if "tags" in meta:
            parts.append(f"tags={meta['tags']}")
        return "  ".join(parts) if parts else "(no meta)"

    @staticmethod
    def _format_via(edge: dict) -> str:
        """Render a single (predicate, anchor) edge as a 'via ...' descriptor."""
        bits = []
        if edge.get("predicate"):
            bits.append(f"predicate={edge['predicate']}")
        if edge.get("anchor"):
            bits.append(f"anchor=#{edge['anchor']}")
        return ", ".join(bits) if bits else "plain"

    async def _expand_for_chunks(
        self,
        chunk_paths: list[str],
        max_per_direction: int,
    ) -> dict[str, dict]:
        """Project shared neighbor expansion into search's compact display structure."""
        raw = await get_first_order_neighbors_batch(
            self.file_store,
            chunk_paths,
            max_per_direction=max_per_direction,
        )
        return {
            cp: {
                "outlinks": [
                    {"path": e["path"], "meta": self._node_meta(e["node"]), "edges": e["edges"]}
                    for e in entry.get("outlinks", [])
                ],
                "inlinks": [
                    {"path": e["path"], "meta": self._node_meta(e["node"]), "edges": e["edges"]}
                    for e in entry.get("inlinks", [])
                ],
            }
            for cp, entry in raw.items()
        }

    @classmethod
    def _render_expansion_lines(cls, expansion: dict) -> list[str]:
        """Render outlinks/inlinks blocks for one chunk path; return zero or more indented lines."""
        lines: list[str] = []
        for direction, arrow, items in (
            ("outlinks", "→", expansion.get("outlinks") or []),
            ("inlinks", "←", expansion.get("inlinks") or []),
        ):
            if not items:
                continue
            lines.append(f"  {direction} ({len(items)}):")
            for item in items:
                lines.append(f"    {arrow} {item['path']}  {cls._format_meta_inline(item['meta'])}")
                for edge in item["edges"]:
                    lines.append(f"        via {cls._format_via(edge)}")
        return lines

    async def execute(self):
        assert self.context is not None
        query: str = (self.context.get("query", "") or "").strip()
        limit: int = int(self.context.get("limit", 5))
        min_score: float = float(self.context.get("min_score", 0.0))
        vector_weight: float = float(self.context.get("vector_weight", 0.7))
        candidate_multiplier: float = float(self.context.get("candidate_multiplier", 3.0))
        expand_links: bool = bool(self.context.get("expand_links", True))
        max_links_per_direction: int = int(self.context.get("max_links_per_direction", 10))

        assert query, "query cannot be empty"
        assert 0.0 <= vector_weight <= 1.0, f"vector_weight must be in [0, 1], got {vector_weight}"
        assert limit > 0, f"limit must be positive, got {limit}"

        candidates = min(_MAX_CANDIDATES, max(1, int(limit * candidate_multiplier)))
        search_filter: dict = self.context.get("search_filter", {}) or {}

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
        fused = fused[:limit]

        unique_paths = list(dict.fromkeys(c.path for c in fused))
        link_expansion: dict[str, dict] = (
            await self._expand_for_chunks(unique_paths, max_links_per_direction) if expand_links else {}
        )

        answer_lines: list[str] = []
        for c in fused:
            answer_lines.append(
                f"========== {c.path}:{c.start_line}-{c.end_line} "
                f"[{self._format_scores(c.scores, hybrid)}] ==========\n{c.text}",
            )
            answer_lines.extend(self._render_expansion_lines(link_expansion.get(c.path, {})))

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
