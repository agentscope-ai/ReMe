"""BFS over wikilink edges from one or more seed files.

One record per traversed *edge* (not per node): the same target can repeat
if reached via different predicates or paths. Each record carries the
predecessor plus the link's predicate/anchor so callers can reconstruct
the path. Adjacency is built once via a single ``file_store.get_nodes()``
call — BFS then runs purely in memory with no per-frontier round-trips.
"""

from collections import deque
from pathlib import Path

from ..base_step import BaseStep
from ...components import R
from ...schema import FileLink

_OUT = {"out", "forward", "both"}
_IN = {"in", "backward", "both"}
_VALID = _OUT | _IN

# source path -> list of (neighbor path, link)
Adjacency = dict[str, list[tuple[str, FileLink]]]


async def _build_adjacency(file_store) -> tuple[Adjacency, Adjacency]:
    """Single ``get_nodes()`` pass → (outbound, inbound) adjacency maps.

    Inbound stores the source path next to each link so BFS can attribute
    inbound edges back to their origin — ``get_inlinks`` alone returns
    target-shaped FileLinks without source attribution.
    """
    outbound: Adjacency = {}
    inbound: Adjacency = {}
    for node in await file_store.get_nodes():
        for link in node.links:
            if link.target_path:
                outbound.setdefault(node.path, []).append((link.target_path, link))
                inbound.setdefault(link.target_path, []).append((node.path, link))
    return outbound, inbound


def _bfs(
    seeds: list[str],
    max_depth: int,
    direction: str,
    outbound: Adjacency,
    inbound: Adjacency,
) -> list[dict]:
    """In-memory BFS; emits one record per unique (src, dst, predicate) edge."""
    sources: list[Adjacency] = []
    if direction in _OUT:
        sources.append(outbound)
    if direction in _IN:
        sources.append(inbound)

    visited: set[tuple[str, str, str | None]] = set()
    results: list[dict] = []
    queue: deque[tuple[str, int]] = deque((s, 0) for s in seeds)

    while queue:
        current, depth = queue.popleft()
        if depth >= max_depth:
            continue
        for src in sources:
            for next_path, link in src.get(current, ()):
                key = (current, next_path, link.predicate)
                if key in visited:
                    continue
                visited.add(key)
                results.append(
                    {
                        "path": next_path,
                        "depth": depth + 1,
                        "via": current,
                        "predicate": link.predicate,
                        "anchor": link.target_anchor,
                    },
                )
                if depth + 1 < max_depth:
                    queue.append((next_path, depth + 1))
    return results


@R.register("traverse_step")
class TraverseStep(BaseStep):
    """BFS from one or more seed files to explore wikilink relationships.

    Parameters:
        path       — single seed (str) or list of seeds (vault-relative).
        direction  — ``forward`` / ``backward`` / ``both`` (or ``out`` / ``in`` / ``both``).
        depth      — hop limit (default 1 = immediate neighbors).
    """

    async def execute(self):
        assert self.context is not None
        raw = self.context.get("path")
        items = [raw] if isinstance(raw, (str, Path)) else list(raw or [])
        seeds = [str(p) for p in items if p]
        assert seeds, "path is required"
        depth = int(self.context.get("depth") or 1)
        direction = (self.context.get("direction") or "both").lower()
        assert direction in _VALID, f"direction must be one of {sorted(_VALID)}, got {direction!r}"

        outbound, inbound = await _build_adjacency(self.file_store)
        results = _bfs(seeds, depth, direction, outbound, inbound)

        self.logger.info(
            f"[{self.name}] seeds={seeds!r} depth={depth} direction={direction} "
            f"nodes={len(outbound) + len(inbound)} edges={len(results)}",
        )

        label = seeds[0] if len(seeds) == 1 else f"{len(seeds)} seeds"
        if not results:
            answer = f"No edges found from {label}"
        else:
            header = f"Traversed {len(results)} edge(s) from {label}"
            lines = [header, ""]
            for r in results:
                target = r["path"]
                if r["anchor"]:
                    target = f"{target}#{r['anchor']}"
                predicate = r["predicate"] or "-"
                lines.append(f"[depth={r['depth']}] {r['via']} --{predicate}--> {target}")
            answer = "\n".join(lines)

        self.context.response.success = True
        self.context.response.answer = answer
        self.context.response.metadata.update({"edges": results, "count": len(results)})
        return self.context.response
