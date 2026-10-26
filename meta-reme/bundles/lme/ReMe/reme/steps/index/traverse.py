"""Build a bounded, frontend-ready graph from indexed Markdown wikilinks."""

from collections import deque
from dataclasses import dataclass
from pathlib import Path

from ..base_step import BaseStep
from ...components import R
from ...schema import FileNode, TraverseGraph, TraverseGraphEdge, TraverseGraphNode

_DIRECTION_ALIASES = {
    "out": "forward",
    "forward": "forward",
    "in": "backward",
    "backward": "backward",
    "both": "both",
}


@dataclass(frozen=True, slots=True)
class _TraversalLink:
    """An adjacency entry carrying both traversal and original edge direction."""

    neighbor: str
    source: str
    target: str
    target_anchor: str | None


Adjacency = dict[str, list[_TraversalLink]]
EdgeKey = tuple[str, str, str | None]


def _build_adjacency(nodes: list[FileNode]) -> tuple[Adjacency, Adjacency]:
    """Build forward/backward adjacency while preserving actual wikilink direction."""
    forward: Adjacency = {}
    backward: Adjacency = {}
    for node in nodes:
        for link in node.links:
            if not link.target_path:
                continue
            item = _TraversalLink(
                neighbor=link.target_path,
                source=node.path,
                target=link.target_path,
                target_anchor=link.target_anchor,
            )
            forward.setdefault(node.path, []).append(item)
            backward.setdefault(link.target_path, []).append(
                _TraversalLink(
                    neighbor=node.path,
                    source=item.source,
                    target=item.target,
                    target_anchor=item.target_anchor,
                ),
            )
    return forward, backward


def _traverse(
    seeds: list[str],
    max_depth: int,
    direction: str,
    forward: Adjacency,
    backward: Adjacency,
) -> tuple[dict[str, int], dict[EdgeKey, int]]:
    """Return shortest node depths and directed edges reached by bounded BFS."""
    adjacency_maps = []
    if direction in {"forward", "both"}:
        adjacency_maps.append(forward)
    if direction in {"backward", "both"}:
        adjacency_maps.append(backward)

    node_depths = {seed: 0 for seed in seeds}
    edge_depths: dict[EdgeKey, int] = {}
    queue = deque(seeds)

    while queue:
        current = queue.popleft()
        current_depth = node_depths[current]
        if current_depth >= max_depth:
            continue

        next_depth = current_depth + 1
        for adjacency in adjacency_maps:
            for link in adjacency.get(current, ()):
                edge_key = (link.source, link.target, link.target_anchor)
                previous_edge_depth = edge_depths.get(edge_key)
                if previous_edge_depth is None or next_depth < previous_edge_depth:
                    edge_depths[edge_key] = next_depth

                previous_node_depth = node_depths.get(link.neighbor)
                if previous_node_depth is None or next_depth < previous_node_depth:
                    node_depths[link.neighbor] = next_depth
                    queue.append(link.neighbor)

    return node_depths, edge_depths


def _build_graph(
    *,
    seeds: list[str],
    max_depth: int,
    direction: str,
    indexed_nodes: list[FileNode],
    node_depths: dict[str, int],
    edge_depths: dict[EdgeKey, int],
) -> TraverseGraph:
    """Materialize deterministic public graph schemas from traversal state."""
    node_by_path = {node.path: node for node in indexed_nodes}
    graph_nodes = []
    for path, node_depth in sorted(node_depths.items(), key=lambda item: (item[1], item[0])):
        node = node_by_path.get(path)
        graph_nodes.append(
            TraverseGraphNode(
                id=path,
                path=path,
                name=node.front_matter.name if node is not None else "",
                description=node.front_matter.description if node is not None else "",
                depth=node_depth,
                indexed=node is not None,
            ),
        )

    graph_edges = [
        TraverseGraphEdge(source=source, target=target, target_anchor=anchor, depth=edge_depth)
        for (source, target, anchor), edge_depth in sorted(
            edge_depths.items(),
            key=lambda item: (item[1], item[0][0], item[0][1], item[0][2] or ""),
        )
    ]
    return TraverseGraph(
        seeds=seeds,
        depth=max_depth,
        direction=direction,
        nodes=graph_nodes,
        edges=graph_edges,
    )


@R.register("traverse_step")
class TraverseStep(BaseStep):
    """Return a bounded wikilink graph rooted at one or more workspace paths."""

    async def execute(self):
        assert self.context is not None
        raw_paths = self.context.get("path")
        items = [raw_paths] if isinstance(raw_paths, (str, Path)) else list(raw_paths or [])
        seeds = list(dict.fromkeys(str(path).replace("\\", "/") for path in items if path))
        if not seeds:
            raise ValueError("path is required")

        raw_depth = self.context.get("depth")
        max_depth = 1 if raw_depth is None else int(raw_depth)
        if max_depth < 0:
            raise ValueError("depth must be greater than or equal to 0")

        raw_direction = str(self.context.get("direction") or "both").lower()
        direction = _DIRECTION_ALIASES.get(raw_direction)
        if direction is None:
            raise ValueError(f"direction must be one of {sorted(_DIRECTION_ALIASES)}, got {raw_direction!r}")

        indexed_nodes = await self.file_store.get_nodes()
        forward, backward = _build_adjacency(indexed_nodes)
        node_depths, edge_depths = _traverse(seeds, max_depth, direction, forward, backward)
        graph = _build_graph(
            seeds=seeds,
            max_depth=max_depth,
            direction=direction,
            indexed_nodes=indexed_nodes,
            node_depths=node_depths,
            edge_depths=edge_depths,
        )

        self.context.response.success = True
        self.context.response.answer = graph.model_dump()
        self.logger.info(
            f"[{self.name}] seeds={seeds!r} depth={max_depth} direction={direction} "
            f"nodes={len(graph.nodes)} edges={len(graph.edges)}",
        )
        return self.context.response
