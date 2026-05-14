"""Retriever — read service. The Memory subsystem's read-side facade.

Per the architecture blueprint, retrieval policy lives here (not in the
file_store): query understanding (intent routing) + multi-channel
fusion (V + K + graph BFS) + ranking. The file_store exposes only
single-channel index primitives; this module composes them with
strategy.

Two surfaces:

    `search(query, ...)`        — pure relevance retrieval (V + K hybrid).
    `graph_search(query, ...)`  — V + K + graph fusion (context expansion
                                  through wikilinks).

`BaseRetriever` inherits `BaseStep` so concrete retrievers get the
standard Step lifecycle + `self.file_store` property + per-instance
configuration via constructor kwargs. The default `execute()` dispatch
just runs `graph_search` so the retriever can also be used directly as
a step inside a job pipeline; the search step shells in
`reme2.memory.memory_search` instead bypass `execute` and call
`search` / `graph_search` directly so they own the result-serialization
shape.
"""

from __future__ import annotations

import asyncio
from abc import abstractmethod
from collections import defaultdict

from ..component import R
from ..component.base_step import BaseStep
from ..schema import ChunkFilter, FileChunk
from . import memory_io


class BaseRetriever(BaseStep):
    """Pluggable retrieval strategy.

    Inherits `BaseStep`, so:
        - `component_type = ComponentEnum.STEP` (for registry lookup).
        - `self.file_store` resolves the configured file_store via the
          standard kwargs / app_context route — no manual `_start` lookup.
        - kwargs (`vector_weight`, `graph_weight`, …) flow through the
          standard Step kwargs handshake.

    Concrete subclasses implement `search` and `graph_search`.
    """

    @abstractmethod
    async def search(
        self,
        query: str,
        *,
        max_results: int = 5,
        min_score: float = 0.0,
        chunk_filter: ChunkFilter | None = None,
    ) -> list[FileChunk]:
        """Pure-relevance retrieval (V + K hybrid by convention)."""

    @abstractmethod
    async def graph_search(
        self,
        query: str = "",
        *,
        seeds: list[str] | None = None,
        max_results: int = 5,
        min_score: float = 0.0,
        chunk_filter: ChunkFilter | None = None,
        # Per-call overrides; None = use this retriever's own defaults.
        vector_weight: float | None = None,
        graph_weight: float | None = None,
        graph_depth: int | None = None,
        graph_decay: float | None = None,
        graph_direction: str | None = None,
        graph_mode: str | None = None,
        graph_per_path_cap: int | None = None,
        anchor_expand: bool | None = None,
    ) -> tuple[list[FileChunk], dict[str, int]]:
        """Three-way fusion search (V + K + graph BFS over wikilinks).

        Returns:
            (chunks, hops) — `hops[path]` is the BFS distance from the
            seed set to that path (0 = seed, 1 = 1-hop neighbor, …).
        """

    async def execute(self):
        """Default Step entry point — dispatches to `graph_search`.

        The retriever is normally invoked via `search` / `graph_search`
        directly by the MCP step shells, but this lets the retriever
        also be wired as a step inside a job pipeline (e.g. for
        debugging / scripts). Reads RuntimeContext for the standard
        query / max_results / min_score / seeds + filter args.
        """
        assert self.context is not None
        ctx = self.context
        chunk_filter = memory_io.make_filter(
            self.file_store,
            paths=ctx.get("paths") or None,
            tags=ctx.get("tags") or None,
            exclude_paths=ctx.get("exclude_paths") or None,
        )
        results, _hops = await self.graph_search(
            query=ctx.get("query", "").strip(),
            seeds=list(ctx.get("seeds") or []),
            max_results=int(ctx.get("max_results", 5)),
            min_score=float(ctx.get("min_score", 0.0)),
            chunk_filter=chunk_filter,
        )
        ctx.response.success = True
        ctx.response.answer = [c.model_dump(exclude_none=True, exclude={"embedding"}) for c in results]


@R.register("hybrid")
class HybridRetriever(BaseRetriever):
    """V + K (+ optional graph BFS) fusion retriever.

    Composes the engine API's projection primitives (`search_vector`,
    `search_keyword`, `expand_neighbors`, `extract_anchors`,
    `get_chunks`). Knobs (`vector_weight`, `graph_weight`,
    `graph_depth`, `graph_decay`, `graph_direction`, `graph_mode`,
    `graph_per_path_cap`, `anchor_expand`, `candidate_multiplier`) are
    constructor defaults; `graph_search` accepts per-call overrides for
    the graph fusion knobs (any None → fall back to constructor default).
    """

    def __init__(
        self,
        vector_weight: float = 0.7,
        graph_weight: float = 0.3,
        graph_depth: int = 1,
        graph_decay: float = 0.5,
        graph_direction: str = "both",
        graph_mode: str = "additive",
        graph_per_path_cap: int = 3,
        candidate_multiplier: float = 3.0,
        anchor_expand: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if graph_mode not in ("additive", "boost"):
            raise ValueError(f"graph_mode must be 'additive' or 'boost', got {graph_mode!r}")
        self.vector_weight = vector_weight
        self.graph_weight = graph_weight
        self.graph_depth = graph_depth
        self.graph_decay = graph_decay
        self.graph_direction = graph_direction
        self.graph_mode = graph_mode
        self.graph_per_path_cap = graph_per_path_cap
        self.candidate_multiplier = candidate_multiplier
        self.anchor_expand = anchor_expand

    async def search(
        self,
        query: str,
        *,
        max_results: int = 5,
        min_score: float = 0.0,
        chunk_filter: ChunkFilter | None = None,
    ) -> list[FileChunk]:
        """V + K hybrid retrieval (no graph). Owns the fusion policy.

        Pipeline:
            1. If both channels are enabled → run V + K in parallel,
               merge by unique_key with weighted score.
            2. If only one is enabled → return that channel's results
               directly (no fusion to do).
            3. If neither → empty.
        """
        fs = self.file_store
        candidates = min(200, max(1, int(max_results * self.candidate_multiplier)))
        text_weight = 1.0 - self.vector_weight

        if fs.embedding_model and fs.keyword_index:
            v_task = memory_io.search_vector(fs, query, limit=candidates, chunk_filter=chunk_filter)
            k_task = memory_io.search_keyword(fs, query, limit=candidates, chunk_filter=chunk_filter)
            v_results, k_results = await asyncio.gather(v_task, k_task)

            if not k_results:
                results = v_results[:max_results]
            elif not v_results:
                results = k_results[:max_results]
            else:
                results = self._merge_vk(
                    v_results,
                    k_results,
                    self.vector_weight,
                    text_weight,
                )[:max_results]
        elif fs.embedding_model:
            results = await memory_io.search_vector(fs, query, limit=max_results, chunk_filter=chunk_filter)
        elif fs.keyword_index:
            results = await memory_io.search_keyword(fs, query, limit=max_results, chunk_filter=chunk_filter)
        else:
            results = []

        if min_score > 0:
            results = [r for r in results if r.score >= min_score]
        return results

    @staticmethod
    def _merge_vk(
        vector: list[FileChunk],
        keyword: list[FileChunk],
        vector_weight: float,
        text_weight: float,
    ) -> list[FileChunk]:
        """Weighted V+K merge by unique_key. Vector first (canonical),
        keyword adds its weighted subscore on collision or seeds fresh."""
        merged: dict[str, FileChunk] = {}
        for r in vector:
            r.scores["score"] = r.scores.get("vector", 0.0) * vector_weight
            merged[r.unique_key] = r
        for r in keyword:
            key = r.unique_key
            k = r.scores.get("keyword", 0.0)
            if key in merged:
                merged[key].scores["score"] += k * text_weight
            else:
                r.scores["score"] = k * text_weight
                merged[key] = r
        results = list(merged.values())
        results.sort(key=lambda c: c.score, reverse=True)
        return results

    async def graph_search(
        self,
        query: str = "",
        *,
        seeds: list[str] | None = None,
        max_results: int = 5,
        min_score: float = 0.0,
        chunk_filter: ChunkFilter | None = None,
        vector_weight: float | None = None,
        graph_weight: float | None = None,
        graph_depth: int | None = None,
        graph_decay: float | None = None,
        graph_direction: str | None = None,
        graph_mode: str | None = None,
        graph_per_path_cap: int | None = None,
        anchor_expand: bool | None = None,
    ) -> tuple[list[FileChunk], dict[str, int]]:
        # Resolve per-call overrides → constructor defaults.
        vw = self.vector_weight if vector_weight is None else float(vector_weight)
        gw = self.graph_weight if graph_weight is None else float(graph_weight)
        gd = self.graph_depth if graph_depth is None else int(graph_depth)
        gdc = self.graph_decay if graph_decay is None else float(graph_decay)
        gdr = self.graph_direction if graph_direction is None else graph_direction
        gm = self.graph_mode if graph_mode is None else graph_mode
        gpc = self.graph_per_path_cap if graph_per_path_cap is None else int(graph_per_path_cap)
        ae = self.anchor_expand if anchor_expand is None else bool(anchor_expand)

        if gm not in ("additive", "boost"):
            raise ValueError(f"graph_mode must be 'additive' or 'boost', got {gm!r}")
        if not (0.0 <= vw <= 1.0):
            raise ValueError(f"vector_weight must be in [0,1], got {vw}")
        if not (0.0 <= gw <= 1.0):
            raise ValueError(f"graph_weight must be in [0,1], got {gw}")

        explicit_seeds = list(seeds or [])
        if not query and not explicit_seeds:
            raise ValueError("graph_search: query or seeds must be provided")

        candidate_count = max(max_results, int(max_results * self.candidate_multiplier))
        fs = self.file_store

        # 1. V + K in parallel (each is a no-op when its backend is disabled).
        if query:
            v_task = memory_io.search_vector(fs, query, limit=candidate_count, chunk_filter=chunk_filter)
            k_task = memory_io.search_keyword(fs, query, limit=candidate_count, chunk_filter=chunk_filter)
            v_results, k_results = await asyncio.gather(v_task, k_task)
        else:
            v_results, k_results = [], []

        # 2. Build seed set.
        seed_paths: set[str] = set()
        for c in v_results:
            seed_paths.add(c.path)
        for c in k_results:
            seed_paths.add(c.path)
        for p in explicit_seeds:
            if p in fs:
                seed_paths.add(p)
        if ae and query:
            seed_paths.update(memory_io.extract_anchors(fs, query))

        # 3. Graph expansion.
        if gw > 0 and seed_paths and gd >= 0:
            hops = memory_io.expand_neighbors(fs, seed_paths, depth=gd, direction=gdr)
        else:
            hops = {}
        graph_scores: dict[str, float] = {p: gdc**h for p, h in hops.items()}

        # 4. Pull graph-only chunks (paths in expansion but not in V∪K).
        # Skip in 'boost' mode — boost only re-ranks V∪K, never adds candidates.
        vk_paths = {c.path for c in v_results} | {c.path for c in k_results}
        if gm == "additive" and hops:
            extra_paths = set(hops) - vk_paths
            if chunk_filter is not None and chunk_filter.resolved_paths is not None:
                extra_paths &= chunk_filter.resolved_paths
            extra_chunks = await memory_io.get_chunks(fs, extra_paths)
            # Per-path cap so a hub topic with N chunks doesn't flood results.
            by_path: dict[str, list] = defaultdict(list)
            for c in extra_chunks:
                by_path[c.path].append(c)
            graph_only_chunks: list[FileChunk] = []
            for path_chunks in by_path.values():
                path_chunks.sort(key=lambda c: c.start_line)
                graph_only_chunks.extend(path_chunks[:gpc])
        else:
            graph_only_chunks = []

        # 5. Merge by unique_key. Vector first (canonical), keyword fills its
        # subscore on existing entries, graph-only chunks come in fresh.
        pool: dict = {}
        for c in v_results:
            pool[c.unique_key] = c
        for c in k_results:
            existing = pool.get(c.unique_key)
            if existing is None:
                pool[c.unique_key] = c
            else:
                existing.scores["keyword"] = c.scores.get("keyword", 0.0)
        for c in graph_only_chunks:
            pool.setdefault(c.unique_key, c)

        # 6. Final scoring.
        for c in pool.values():
            v = c.scores.get("vector", 0.0)
            k = c.scores.get("keyword", 0.0)
            g = graph_scores.get(c.path, 0.0)
            vk = vw * v + (1.0 - vw) * k
            if gm == "boost":
                final = vk * (1.0 + gw * g)
            else:
                final = (1.0 - gw) * vk + gw * g
            c.scores["graph"] = g
            c.scores["score"] = final

        # 7. Rank + filter + slice.
        results = sorted(pool.values(), key=lambda c: c.score, reverse=True)
        if min_score > 0:
            results = [r for r in results if r.score >= min_score]
        results = results[:max_results]

        return results, hops
