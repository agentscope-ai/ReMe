"""Networkx file-graph backend."""

import pickle
from pathlib import Path

try:
    import networkx as nx
except ImportError:
    nx = None

from .base_file_graph import BaseFileGraph
from ..component_registry import R
from ...schema import FileLink, FileNode


@R.register("nx")
class NxFileGraph(BaseFileGraph):
    """Networkx-backed file graph; uses FileLink.target_path for adjacency."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if nx is None:
            raise ImportError("NxFileGraph requires networkx — pip install networkx")
        self._graph: nx.MultiDiGraph = nx.MultiDiGraph()
        self._graph_file: Path = self.graph_path / f"{self.graph_name}_{self.graph_version}.pkl"

    # -- Lifecycle ---------------------------------------------------------

    async def load(self) -> None:
        """Load graph from pickle file; keep current graph on failure."""
        if not self._graph_file.exists():
            return
        try:
            with open(self._graph_file, "rb") as f:
                self._graph = pickle.load(f)
            n_real = sum(1 for _, d in self._graph.nodes(data=True) if "node" in d)
            self.logger.info(f"Loaded {n_real} nodes from {self._graph_file}")
        except Exception as e:
            self.logger.exception(f"Failed to load {self._graph_file}: {e}")

    async def dump(self) -> None:
        """Persist graph to pickle via atomic rename."""
        try:
            tmp = self._graph_file.with_suffix(".tmp")
            with open(tmp, "wb") as f:
                pickle.dump(self._graph, f, protocol=pickle.HIGHEST_PROTOCOL)
            tmp.replace(self._graph_file)
            n_real = sum(1 for _, d in self._graph.nodes(data=True) if "node" in d)
            self.logger.info(f"Saved {n_real} nodes to {self._graph_file}")
        except Exception as e:
            self.logger.exception(f"Failed to write {self._graph_file}: {e}")

    # -- Node CRUD ---------------------------------------------------------

    async def upsert_nodes(self, nodes: list[FileNode]) -> None:
        for node in nodes:
            path = node.path
            if self._graph.has_node(path):
                # Drop outgoing edges; inbound stay intact.
                self._graph.remove_edges_from(list(self._graph.out_edges(path, keys=True)))
            self._graph.add_node(path, node=node)  # promotes virtual node if present
            # Missing targets become attr-less virtual nodes.
            self._graph.add_edges_from((path, lnk.target_path, {"link": lnk}) for lnk in node.links if lnk.target_path)

    async def delete_nodes(self, paths: list[str]) -> None:
        for path in paths:
            if not self._graph.has_node(path):
                continue
            self._graph.remove_edges_from(list(self._graph.out_edges(path, keys=True)))
            # Demote to virtual: keep inbound edges, drop node payload.
            self._graph.nodes[path].pop("node", None)
            if self._graph.in_degree(path) == 0:
                self._graph.remove_node(path)  # remove orphan virtual node

    async def get_nodes(self, paths: list[str] | None = None) -> list[FileNode]:
        nodes_view = self._graph.nodes
        if paths is None:
            return [d["node"] for _, d in nodes_view(data=True) if "node" in d]
        return [nodes_view[path]["node"] for path in paths if path in nodes_view and "node" in nodes_view[path]]

    async def rebuild_links(self) -> None:
        """Rebuild all edges from real node payloads; drop virtual nodes."""
        self._graph.remove_edges_from(list(self._graph.edges(keys=True)))
        virtual = [n for n, d in self._graph.nodes(data=True) if "node" not in d]
        self._graph.remove_nodes_from(virtual)
        self._graph.add_edges_from(
            (path, lnk.target_path, {"link": lnk})
            for path, data in self._graph.nodes(data=True)
            for lnk in data["node"].links
            if lnk.target_path
        )

    async def clear(self):
        """Remove all nodes and edges, and remove persisted file."""
        self._graph.clear()
        self._graph_file.unlink(missing_ok=True)

    # -- Link access -------------------------------------------------------

    async def get_outlinks(self, path: str) -> list[FileLink]:
        nodes_view = self._graph.nodes
        if path not in nodes_view or "node" not in nodes_view[path]:
            return []
        return [
            d["link"]
            for _, target, d in self._graph.out_edges(path, data=True)
            if "link" in d and "node" in nodes_view[target]
        ]

    async def get_inlinks(self, path: str) -> list[FileLink]:
        nodes_view = self._graph.nodes
        if path not in nodes_view or "node" not in nodes_view[path]:
            return []
        return [d["link"] for _, _, d in self._graph.in_edges(path, data=True) if "link" in d]
