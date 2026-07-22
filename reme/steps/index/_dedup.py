"""Shared tool_context-scoped chunk dedup with TTL.

Used by ``search``/``vector_search``/``bm25_search`` to avoid returning the
same content twice within one agent tool_context. Per-context state lives at
``app_context.metadata["tool_contexts"][tool_context_id]["search_seen_chunk_ids"]``
as ``{path: [(start_line, end_line, timestamp), ...]}``; a chunk is skipped when
its ``[start_line, end_line]`` is fully covered by the union of seen entries
(merged overlapping/adjacent intervals) for the same ``path``. Entries older
than ``seen_ttl_hours`` are expired on each call.
When ``app_context`` is absent the same structure is mirrored under
``self.kwargs["tool_contexts"][tool_context_id]`` for unit tests.
"""

import datetime
from typing import TYPE_CHECKING, Any, Callable, Final

from ...schema import FileChunk

if TYPE_CHECKING:
    from ...components import ApplicationContext


class _ToolContextDedupMixin:
    """Mixin providing tool_context-scoped chunk dedup with TTL.

    Must be mixed into a ``BaseStep`` subclass (e.g. ``SearchStep``); it
    cannot be instantiated or subclassed on its own. The mixin relies on
    ``app_context``/``kwargs`` from ``BaseStep`` and on ``seen_ttl_hours``
    set by the host step's ``__init__`` (default 24h).
    """

    TOOL_CONTEXTS_KEY: Final[str] = "tool_contexts"
    SEARCH_SEEN_KEY: Final[str] = "search_seen_chunk_ids"

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        # Deferred import avoids a circular dependency at module load time.
        from ..base_step import BaseStep

        if not issubclass(cls, BaseStep):
            raise TypeError(
                f"{cls.__name__!r} mixes in _ToolContextDedupMixin but does not "
                f"inherit from BaseStep. Mix it in alongside BaseStep, e.g. "
                f"class {cls.__name__}(_ToolContextDedupMixin, BaseStep)."
            )

    def __new__(cls, *args, **kwargs):
        if cls is _ToolContextDedupMixin:
            raise TypeError(
                "_ToolContextDedupMixin is a mixin and cannot be instantiated "
                "directly. Mix it into a BaseStep subclass, e.g. "
                "class SearchStep(_ToolContextDedupMixin, BaseStep)."
            )
        return super().__new__(cls, *args, **kwargs)

    if TYPE_CHECKING:
        # Declared by BaseStep; repeated here so static analysis resolves
        # attribute access on the mixin without inheriting BaseStep.
        app_context: "ApplicationContext | None"
        kwargs: dict[str, Any]
        seen_ttl_hours: float

    def _tool_context_store(self, tool_context_id: str) -> dict:
        """Return the mutable state bucket for a tool context.

        The bucket is created lazily on first access and lives at
        ``metadata["tool_contexts"][tool_context_id]`` (or the same path under
        ``kwargs`` when no ``app_context`` is available, e.g. in unit tests).
        """
        if self.app_context is not None:
            contexts = self.app_context.metadata.setdefault(self.TOOL_CONTEXTS_KEY, {})
        else:
            contexts = self.kwargs.setdefault(self.TOOL_CONTEXTS_KEY, {})
        return contexts.setdefault(tool_context_id, {})

    @staticmethod
    def _now_ts() -> float:
        return datetime.datetime.now().timestamp()

    def _dedupe_tool_context(
        self,
        chunks: list[FileChunk],
        tool_context_id: str,
        limit: int,
        *,
        clock: Callable[[], float] | None = None,
        ttl_override: float | None = None,
    ) -> tuple[list[FileChunk], dict]:
        """Drop chunks whose line range is already covered by a previously
        returned chunk for this tool_context within the TTL window.

        Seen intervals per path are merged (overlapping or adjacent) into a
        minimal set of disjoint ranges. A chunk is skipped when its
        ``[start_line, end_line]`` is a subset of any merged range for the
        same ``path`` — multiple previously returned chunks can jointly cover
        a new chunk even if no single entry does. Partial overlap (superset or
        straddle) is NOT skipped — the chunk carries lines not yet returned,
        so it is kept.

        ``clock`` (a zero-arg callable returning a float timestamp) and
        ``ttl_override`` (seconds) allow tests to inject deterministic time.
        Returns ``(returned, stats)``; callers that don't need the stats may
        discard the second element.
        """
        now = (clock or self._now_ts)()
        ttl = ttl_override if ttl_override is not None else float(self.seen_ttl_hours) * 60 * 60
        store = self._tool_context_store(tool_context_id)
        seen = store.get(self.SEARCH_SEEN_KEY, {})
        # Normalize legacy in-memory formats to {path: [(s, e, t), ...]}.
        # Older shapes ({chunk_id: timestamp} or a plain list of ids) cannot
        # be migrated because chunk_id is an opaque hash; seen is a transient
        # per-Application cache, so dropping it is safe.
        if not isinstance(seen, dict) or (
            seen and all(not isinstance(v, list) for v in seen.values())
        ):
            seen = {}

        before_expire = sum(len(v) for v in seen.values())
        # Expire stale tuples across all paths.
        seen = {
            path: [(s, e, t) for (s, e, t) in entries if now - t < ttl]
            for path, entries in seen.items()
        }
        seen = {path: entries for path, entries in seen.items() if entries}
        store[self.SEARCH_SEEN_KEY] = seen

        seen_before = sum(len(v) for v in seen.values())

        def _is_covered(chunk: FileChunk) -> bool:
            entries = seen.get(chunk.path)
            if not entries:
                return False
            # Merge overlapping/adjacent intervals so that multiple seen
            # entries can jointly cover a new chunk (e.g. (1,10)+(11,20)
            # merge into (1,20) and cover (5,15)).
            intervals = sorted((s, e) for s, e, _ in entries)
            merged: list[tuple[int, int]] = []
            for s, e in intervals:
                if merged and s <= merged[-1][1] + 1:
                    merged[-1] = (merged[-1][0], max(merged[-1][1], e))
                else:
                    merged.append((s, e))
            return any(s <= chunk.start_line and chunk.end_line <= e for s, e in merged)

        unvisited = [chunk for chunk in chunks if not _is_covered(chunk)]
        returned = unvisited[:limit]
        for chunk in returned:
            seen.setdefault(chunk.path, []).append((chunk.start_line, chunk.end_line, now))

        # Reorder for readability: keep chunks of the same path adjacent and sorted by
        # ascending start_line; order paths by where each first appears in the original
        # sequence (the path owning the earliest-ranked chunk comes first).
        path_order: dict[str, int] = {}
        for idx, chunk in enumerate(returned):
            path_order.setdefault(chunk.path, idx)
        returned = sorted(returned, key=lambda c: (path_order[c.path], c.start_line))

        return returned, {
            "tool_context_id": tool_context_id,
            "seen_before": seen_before,
            "skipped_seen": len(chunks) - len(unvisited),
            "seen_after": sum(len(v) for v in seen.values()),
            "expired": before_expire - seen_before,
            "ttl_seconds": ttl,
        }
