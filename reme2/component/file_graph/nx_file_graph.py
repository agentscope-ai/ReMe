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
    """Networkx-backed file graph. Trusts ``FileLink.path`` for adjacency."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        if nx is None:
            raise ImportError("NxFileGraph requires networkx — install it with `pip install networkx`")
        self._graph: nx.MultiDiGraph = nx.MultiDiGraph()
        self._graph_file: Path = self.graph_path / f"{self.graph_name}.pkl"

    # -- Lifecycle ---------------------------------------------------------

    async def _start(self) -> None:
        await super()._start()
        loaded = self._load()
        if loaded is not None:
            self._graph = loaded
        real = sum(1 for _, d in self._graph.nodes(data=True) if "node" in d)
        virtual = self._graph.number_of_nodes() - real
        self.logger.info(
            f"NxFileGraph '{self.graph_name}' ready: "
            f"{real} nodes, {self._graph.number_of_edges()} edges, "
            f"{virtual} virtual",
        )

    async def _close(self) -> None:
        self._dump()
        await super()._close()

    def _load(self) -> nx.MultiDiGraph | None:
        if not self._graph_file.exists():
            return None
        try:
            with open(self._graph_file, "rb") as f:
                return pickle.load(f)
        except Exception as e:
            self.logger.exception(f"Failed to load {self._graph_file}: {e}")
            return None

    def _dump(self) -> None:
        try:
            tmp = self._graph_file.with_suffix(".tmp")
            with open(tmp, "wb") as f:
                pickle.dump(self._graph, f, protocol=pickle.HIGHEST_PROTOCOL)
            tmp.replace(self._graph_file)
        except Exception as e:
            self.logger.exception(f"Failed to write {self._graph_file}: {e}")

    # -- Node CRUD ---------------------------------------------------------

    async def upsert_nodes(self, nodes: list[FileNode]) -> None:
        for node in nodes:
            path = node.path
            if self._graph.has_node(path):
                # Drop only outgoing edges; inbound real/virtual edges stay.
                self._graph.remove_edges_from(
                    list(self._graph.out_edges(path, keys=True)),
                )
            # Merges attrs: writes a real node, or promotes a virtual one in place.
            self._graph.add_node(path, node=node)
            # Missing targets become attr-less virtual nodes automatically.
            self._graph.add_edges_from(
                (path, link.path, {"link": link})
                for link in node.links
                if link.path
            )

    async def delete_nodes(self, paths: list[str]) -> None:
        for path in paths:
            if not self._graph.has_node(path):
                continue
            self._graph.remove_edges_from(
                list(self._graph.out_edges(path, keys=True)),
            )
            # Demote to virtual: keep inbound edges, drop ``node`` attr.
            self._graph.nodes[path].pop("node", None)
            # Fully drop the now-orphan virtual node if nothing points here.
            if self._graph.in_degree(path) == 0:
                self._graph.remove_node(path)

    async def get_nodes(self, paths: list[str]) -> list[FileNode]:
        nodes_view = self._graph.nodes
        return [
            nodes_view[path]["node"]
            for path in paths
            if path in nodes_view and "node" in nodes_view[path]
        ]

    async def rebuild_links(self) -> None:
        # Defensive full rebuild from each real node's payload.
        self._graph.remove_edges_from(list(self._graph.edges(keys=True)))
        virtual = [n for n, d in self._graph.nodes(data=True) if "node" not in d]
        self._graph.remove_nodes_from(virtual)
        real = [(path, data["node"]) for path, data in self._graph.nodes(data=True)]
        self._graph.add_edges_from(
            (path, link.path, {"link": link})
            for path, node in real
            for link in node.links
            if link.path
        )

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
