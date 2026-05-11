"""Memory retriever steps — thin MCP-facing wrappers over `BaseRetriever`.

Per the architecture blueprint, retrieval policy (V+K hybrid + graph
BFS fusion + ranking + intent routing) is the **Retriever service**
(`reme2.memory.retriever`), registered as `ComponentEnum.RETRIEVER`.
These steps are the MCP projection: they translate `RuntimeContext`
(paths/tags/exclude_paths filter, per-call knob overrides) into the
retriever's call surface, then serialize results for the API response
(joining file metadata onto each chunk so callers don't have to fire a
`memory_get` per hit).

Direct primary-key file ops (read/list/links/backlinks) live in
`memory_io` — those aren't retrieval.
"""

from __future__ import annotations

from ...component import R
from ...component.base_step import BaseStep
from ...component.runtime_response import _set_answer
from ...enumeration import ComponentEnum
from ...memory import memory_io
from ...memory.retriever import BaseRetriever, HybridRetriever


# Per-shell singleton cache: instantiating the retriever is cheap (it
# just stashes constructor knobs), but doing it once per call would
# still allocate every request. Keyed by step instance so each MCP
# job's overrides stay isolated.
_RETRIEVER_CACHE: dict[int, BaseRetriever] = {}


def _resolve_retriever(step: BaseStep) -> BaseRetriever:
    """Get (or build) the retriever instance for this MCP step.

    The retriever is a registered Step (`@R.register("hybrid")`), but
    it isn't pre-instantiated as a singleton component — there's no
    RETRIEVER enum slot. Instead, each MCP shell builds its own
    HybridRetriever the first time it's called, sharing the calling
    step's `app_context` (so the lookup of `file_store`/`as_llm` works)
    and forwarding any retriever knobs (`vector_weight`, `graph_*`, …)
    from the step's kwargs as constructor defaults.

    Callers can also pass a pre-built `BaseRetriever` instance via
    `kwargs["retriever"]` — useful for tests / Python callers that want
    to inject a stub.
    """
    injected = step.kwargs.get("retriever")
    if isinstance(injected, BaseRetriever):
        return injected

    cached = _RETRIEVER_CACHE.get(id(step))
    if cached is not None:
        return cached

    backend = injected if isinstance(injected, str) else "hybrid"
    cls = R.get(ComponentEnum.STEP, backend)
    if cls is None or not (isinstance(cls, type) and issubclass(cls, BaseRetriever)):
        # Fall back to the canonical implementation; lets configs that
        # don't override `retriever` work out of the box.
        cls = HybridRetriever

    knob_keys = (
        "vector_weight", "graph_weight", "graph_depth", "graph_decay",
        "graph_direction", "graph_mode", "graph_per_path_cap",
        "candidate_multiplier", "anchor_expand", "file_store",
    )
    init_kwargs = {k: step.kwargs[k] for k in knob_keys if k in step.kwargs}
    init_kwargs["app_context"] = step.app_context

    instance = cls(**init_kwargs)
    _RETRIEVER_CACHE[id(step)] = instance
    return instance


def _serialize_chunk(chunk, file_store, extras: dict | None = None) -> dict:
    """Serialize a FileChunk for search-result payloads.

    Joins the owning file's metadata (frontmatter + st_mtime) so callers
    don't have to fire a `memory_get` per result just to read `category`,
    `status`, dates, etc. The join is a single in-memory dict lookup —
    cost is negligible vs. the round-trip we save.

    `extras` lets the caller attach step-specific fields (e.g. `graph_hop`).
    """
    item = chunk.model_dump(exclude_none=True, exclude={"embedding"})
    meta = file_store.get_file_meta(chunk.path)
    if meta is not None:
        item["file_metadata"] = meta.metadata
        item["file_st_mtime"] = meta.st_mtime
    else:
        item["file_metadata"] = None
        item["file_st_mtime"] = None
    if extras:
        item.update(extras)
    return item


@R.register("memory_search")
class MemorySearch(BaseStep):
    """Pure-relevance retrieval (V + K hybrid). Delegates to the Retriever service."""

    async def execute(self):
        assert self.context is not None, "Context is not set"
        query: str = self.context.get("query", "").strip()
        min_score: float = self.context.get("min_score", 0.1)
        max_results: int = self.context.get("max_results", 5)

        assert query, "Query cannot be empty"
        assert (
            isinstance(min_score, float | int) and 0.0 <= min_score <= 1.0
        ), f"min_score must be between 0 and 1, got {min_score}"
        assert (
            isinstance(max_results, int) and max_results > 0
        ), f"max_results must be a positive integer, got {max_results}"

        chunk_filter = memory_io.make_filter(
            self.file_store,
            paths=self.context.get("paths") or None,
            tags=self.context.get("tags") or None,
            exclude_paths=self.context.get("exclude_paths") or None,
        )

        retriever = _resolve_retriever(self)
        results = await retriever.search(
            query=query,
            max_results=max_results,
            min_score=min_score,
            chunk_filter=chunk_filter,
        )

        payload = [_serialize_chunk(r, self.file_store) for r in results]
        _set_answer(self.context, payload)


@R.register("memory_graph_search")
class MemoryGraphSearch(BaseStep):
    """V + K + graph BFS fusion. Delegates to the Retriever service.

    Per-call overrides for fusion knobs (vector_weight, graph_weight,
    graph_depth, graph_decay, graph_direction, graph_mode,
    graph_per_path_cap, anchor_expand) are forwarded if present in the
    RuntimeContext; otherwise the retriever falls back to its own
    constructor defaults. These knobs are intentionally NOT exposed via
    MCP — they're internal-Python-caller tuning.
    """

    _OVERRIDE_KEYS = (
        "vector_weight",
        "graph_weight",
        "graph_depth",
        "graph_decay",
        "graph_direction",
        "graph_mode",
        "graph_per_path_cap",
        "anchor_expand",
    )

    async def execute(self):
        assert self.context is not None
        ctx = self.context

        query: str = ctx.get("query", "").strip()
        max_results: int = int(ctx.get("max_results", 5))
        min_score: float = float(ctx.get("min_score", 0.0))
        explicit_seeds: list[str] = list(ctx.get("seeds") or [])

        assert query or explicit_seeds, "query or seeds must be provided"
        assert max_results > 0

        chunk_filter = memory_io.make_filter(
            self.file_store,
            paths=ctx.get("paths") or None,
            tags=ctx.get("tags") or None,
            exclude_paths=ctx.get("exclude_paths") or None,
        )

        # Forward per-call overrides only if the caller actually set them;
        # the retriever falls back to its own defaults for missing keys.
        overrides = {k: ctx.get(k) for k in self._OVERRIDE_KEYS if ctx.get(k) is not None}

        retriever = _resolve_retriever(self)
        results, hops = await retriever.graph_search(
            query=query,
            seeds=explicit_seeds,
            max_results=max_results,
            min_score=min_score,
            chunk_filter=chunk_filter,
            **overrides,
        )

        payload = []
        for c in results:
            extras = {}
            hop = hops.get(c.path)
            if hop is not None:
                extras["graph_hop"] = hop
            payload.append(_serialize_chunk(c, self.file_store, extras))
        _set_answer(ctx, payload)
