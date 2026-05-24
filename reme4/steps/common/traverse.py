"""``traverse_step`` — BFS over wikilink edges from a seed file.

Single tool for relationship browsing. ``depth=1`` covers the trivial
"what does this link to / what links here" lookups (set ``direction``
accordingly); higher depth opens up multi-hop exploration.

Output is one record per edge traversed (not per node), so the same
target can appear multiple times if reached via different predicates
or paths — agents dedupe at the call site if they want a flat node
set. Each record carries ``via`` (the predecessor) and the link's
``predicate`` / ``anchor`` so the agent can reconstruct the path.

Adjacency is loaded once via ``file_graph.get_nodes(None)`` — every
real node arrives with its full ``links`` payload, and we build both
the outbound and the inbound index in a single pass. The BFS then
runs purely in memory: no per-frontier-node graph round-trips, no
filesystem walk. The ``get_inlinks`` / ``get_outlinks`` contract
methods stay unused here because they'd add network round-trips for
data we already have.

Direction vocabulary accepts both the standard convention
(``forward`` / ``backward`` / ``both``) and the engine convention
(``out`` / ``in`` / ``both``).

Seeds are paths relative to the vault used as-is — short-form resolution
is no longer attempted. Seeds that don't match any graph node yield
empty BFS results (no error).
"""

from __future__ import annotations

from collections import deque
from pathlib import Path

from ..base_step import BaseStep
from ...utils import set_answer

from ...components import R
from ...enumeration import ComponentEnum
from ...schema import FileLink


_FORWARD = {"out", "forward"}
_BACKWARD = {"in", "backward"}
_BOTH = {"both"}
_VALID_DIRECTIONS = _FORWARD | _BACKWARD | _BOTH


async def _build_indexes(
    file_store,
) -> tuple[
    dict[str, list[tuple[str, FileLink]]],
    dict[str, list[tuple[str, FileLink]]],
]:
    """One ``get_nodes(None)`` call → (outbound, inbound) adjacency dicts.

    Each dict is keyed by node path; values are ``(neighbor_path, link)``
    tuples. Source paths land in the inbound index alongside the link
    object — solving the contract gap where ``get_inlinks`` returns
    target-shaped FileLinks without source attribution.
    """
    outbound: dict[str, list[tuple[str, FileLink]]] = {}
    inbound: dict[str, list[tuple[str, FileLink]]] = {}
    if not file_store.file_graph:
        return outbound, inbound
    for node in await file_store.file_graph.get_nodes():
        for link in node.links:
            if not link.target_path:
                continue
            outbound.setdefault(node.path, []).append((link.target_path, link))
            inbound.setdefault(link.target_path, []).append((node.path, link))
    return outbound, inbound


def _bfs(
    seeds: list[str],
    max_depth: int,
    direction: str,
    outbound: dict[str, list[tuple[str, FileLink]]],
    inbound: dict[str, list[tuple[str, FileLink]]],
) -> list[dict]:
    """In-memory BFS. One record per edge traversed."""
    walk_out = direction in _FORWARD or direction in _BOTH
    walk_in = direction in _BACKWARD or direction in _BOTH

    visited_edges: set[tuple[str, str, str | None]] = set()
    results: list[dict] = []
    queue: deque[tuple[str, int]] = deque((s, 0) for s in seeds)

    while queue:
        current, depth = queue.popleft()
        if depth >= max_depth:
            continue

        edges: list[tuple[str, str | None, str | None]] = []
        if walk_out:
            for tgt, link in outbound.get(current, ()):
                edges.append((tgt, link.predicate, link.target_anchor))
        if walk_in:
            for src, link in inbound.get(current, ()):
                edges.append((src, link.predicate, link.target_anchor))

        for next_path, pred, anchor in edges:
            edge_key = (current, next_path, pred)
            if edge_key in visited_edges:
                continue
            visited_edges.add(edge_key)
            results.append(
                {
                    "path": next_path,
                    "depth": depth + 1,
                    "via": current,
                    "predicate": pred,
                    "anchor": anchor,
                },
            )
            if depth + 1 < max_depth:
                queue.append((next_path, depth + 1))

    return results


def _normalize_seeds(raw) -> list[str]:
    """Coerce raw seed input to a non-empty list of strings relative to the vault."""
    if isinstance(raw, (str, Path)):
        items = [raw]
    else:
        items = list(raw or [])
    return [str(p) for p in items if p]


@R.register("traverse_step")
class TraverseStep(BaseStep):
    """BFS from a seed file to explore wikilink relationships.

    Parameters:
        path       — single seed (str) or a list of seeds.
        direction  — ``forward`` / ``backward`` / ``both`` (or ``out`` / ``in`` / ``both``).
        depth      — hop limit (default 1 = immediate neighbors).
    """

    component_type = ComponentEnum.STEP

    async def execute(self):
        assert self.context is not None
        seeds_raw = self.context.get("path")
        depth = int(self.context.get("depth") or 1)
        direction = (self.context.get("direction") or "forward").lower()
        assert (
            direction in _VALID_DIRECTIONS
        ), f"direction must be one of {sorted(_VALID_DIRECTIONS)}, got {direction!r}"
        seeds = _normalize_seeds(seeds_raw)
        assert seeds, "path is required"
        outbound, inbound = await _build_indexes(self.file_store)
        results = _bfs(seeds, depth, direction, outbound, inbound)
        set_answer(self.context, results)
