"""``graph_traverse_step`` — BFS over wikilink edges from a seed file.

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

The seed ``path`` is taken as-is (vault-relative). A seed that doesn't
match any graph node yields an empty result (no error).
"""

from collections import deque

from ..base_step import BaseStep
from ...components import R
from ...schema import FileLink


_FORWARD = {"out", "forward"}
_BACKWARD = {"in", "backward"}
_BOTH = {"both"}
_VALID_DIRECTIONS = _FORWARD | _BACKWARD | _BOTH


@R.register("graph_traverse_step")
class GraphTraverseStep(BaseStep):
    """BFS from a seed file to explore wikilink relationships.

    Parameters:
        path       — seed path (vault-relative).
        direction  — ``forward`` / ``backward`` / ``both`` (or ``out`` / ``in`` / ``both``).
        depth      — hop limit (default 1 = immediate neighbors).
        predicate  — optional edge-type filter; ``None`` = no filter.
    """

    async def execute(self):
        """BFS from ``path`` and emit one record per traversed edge."""
        assert self.context is not None
        seed = str(self.context.get("path") or "").strip()
        assert seed, "path is required"
        max_depth = int(self.context.get("depth") or 1)
        direction = (self.context.get("direction") or "both").lower()
        predicate = self.context.get("predicate")
        assert (
            direction in _VALID_DIRECTIONS
        ), f"direction must be one of {sorted(_VALID_DIRECTIONS)}, got {direction!r}"

        # Build outbound / inbound adjacency in one pass over all nodes.
        outbound: dict[str, list[tuple[str, FileLink]]] = {}
        inbound: dict[str, list[tuple[str, FileLink]]] = {}
        if self.file_store.file_graph:
            for node in await self.file_store.file_graph.get_nodes():
                for link in node.links:
                    if not link.target_path:
                        continue
                    outbound.setdefault(node.path, []).append((link.target_path, link))
                    inbound.setdefault(link.target_path, []).append((node.path, link))

        walk_out = direction in _FORWARD or direction in _BOTH
        walk_in = direction in _BACKWARD or direction in _BOTH

        visited_edges: set[tuple[str, str, str | None]] = set()
        results: list[dict] = []
        queue: deque[tuple[str, int]] = deque([(seed, 0)])

        while queue:
            current, depth = queue.popleft()
            if depth >= max_depth:
                continue

            edges: list[tuple[str, str | None, str | None]] = []
            if walk_out:
                for tgt, link in outbound.get(current, ()):
                    if predicate is not None and link.predicate != predicate:
                        continue
                    edges.append((tgt, link.predicate, link.target_anchor))
            if walk_in:
                for src, link in inbound.get(current, ()):
                    if predicate is not None and link.predicate != predicate:
                        continue
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

        self.context.response.success = True
        self.context.response.answer = f"Traversed {len(results)} edge(s) from {seed}"
        self.context.response.metadata.update({"edges": results, "count": len(results)})
