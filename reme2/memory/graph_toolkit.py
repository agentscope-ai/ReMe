"""Graph category — relationship exploration via BFS.

Single tool: ``graph_traverse``. depth=1 covers the trivial outlinks /
inlinks lookups (just set direction); multi-hop covers exploration.
Output is one record per edge traversed (not per node), so the same
node may appear multiple times if reached via different predicates or
paths — agents dedupe at the call site if they want a flat node set.

Adjacency source: walks ``file_store.file_nodes`` directly (each
FileNode carries its ``links`` list). Inbound is a linear scan over
all nodes; cheap for vault sizes. Swap for a precomputed reverse
index on the file_graph component if it ever becomes a hot path.
"""

from __future__ import annotations

from collections import deque

from agentscope.tool import ToolResponse

from ..component import R
from ..component.base_step import BaseStep
from .runtime_response import _set_answer, _tool_response


# ===========================================================================
# Section 1 — Adjacency lookups
# ===========================================================================


def _outlinks(file_store, path: str) -> list[tuple[str, str | None, str | None]]:
    """Outgoing edges from ``path`` — [(target_path, predicate, anchor)]."""
    node = file_store.file_nodes.get(path)
    if node is None:
        return []
    return [(link.path, link.predicate, link.anchor) for link in node.links if link.path]


def _inlinks(file_store, path: str) -> list[tuple[str, str | None, str | None]]:
    """Incoming edges to ``path`` — linear scan over all nodes' links."""
    out: list[tuple[str, str | None, str | None]] = []
    for src_path, src_node in file_store.file_nodes.items():
        if src_path == path:
            continue
        for link in src_node.links:
            if link.path == path:
                out.append((src_path, link.predicate, link.anchor))
    return out


def _bfs(
    file_store,
    seeds: list[str],
    max_depth: int,
    direction: str,
    predicate: str | None,
) -> list[dict]:
    """BFS from each seed. One record per edge traversed."""
    visited_edges: set[tuple[str, str, str | None]] = set()
    results: list[dict] = []
    queue: deque[tuple[str, int]] = deque((s, 0) for s in seeds)
    while queue:
        current, depth = queue.popleft()
        if depth >= max_depth:
            continue
        edges: list[tuple[str, str | None, str | None]] = []
        if direction in ("out", "both"):
            for tgt, pred, anchor in _outlinks(file_store, current):
                if predicate is not None and pred != predicate:
                    continue
                edges.append((tgt, pred, anchor))
        if direction in ("in", "both"):
            for src, pred, anchor in _inlinks(file_store, current):
                if predicate is not None and pred != predicate:
                    continue
                edges.append((src, pred, anchor))
        for next_path, pred, anchor in edges:
            edge_key = (current, next_path, pred)
            if edge_key in visited_edges:
                continue
            visited_edges.add(edge_key)
            results.append({
                "path": next_path,
                "depth": depth + 1,
                "via": current,
                "predicate": pred,
                "anchor": anchor,
            })
            if depth + 1 < max_depth:
                queue.append((next_path, depth + 1))
    return results


# ===========================================================================
# Section 2 — Graph tool
# ===========================================================================


@R.register("graph_traverse")
class GraphTraverse(BaseStep):
    """BFS from seed(s) to explore relationships in the memory graph."""

    audit: list[dict] | None = None

    async def execute(self):
        assert self.context is not None
        seeds_raw = self.context.get("seeds") or []
        if isinstance(seeds_raw, str):
            seeds = [seeds_raw]
        else:
            seeds = list(seeds_raw)
        max_depth: int = int(self.context.get("max_depth") or 1)
        direction: str = self.context.get("direction", "out") or "out"
        predicate = self.context.get("predicate")
        assert seeds, "seeds is required (single path or list of paths)"
        assert direction in ("out", "in", "both"), \
            f"direction must be 'out' | 'in' | 'both', got {direction!r}"
        results = _bfs(self.file_store, seeds, max_depth, direction, predicate)
        _set_answer(self.context, results)

    async def graph_traverse(
        self,
        seeds: str | list[str],
        max_depth: int = 1,
        direction: str = "out",
        predicate: str | None = None,
    ) -> ToolResponse:
        """BFS from seed(s) to explore relationships.

        Args:
            seeds: single path or list of paths to start from.
            max_depth: hops to expand (default 1 = immediate neighbors).
            direction: "out" / "in" / "both".
            predicate: filter edges by predicate (None = no filter).
        """
        if isinstance(seeds, str):
            seeds_list = [seeds]
        else:
            seeds_list = list(seeds)
        assert direction in ("out", "in", "both"), \
            f"direction must be 'out' | 'in' | 'both', got {direction!r}"
        results = _bfs(self.file_store, seeds_list, max_depth, direction, predicate)
        return _tool_response("graph_traverse", True, results, audit=self.audit)


GRAPH_TOOL_NAMES: tuple[str, ...] = (
    "graph_traverse",
)
