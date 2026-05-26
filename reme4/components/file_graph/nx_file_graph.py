"""Networkx file-graph backend."""

import pickle
from pathlib import Path

try:
    import networkx as nx
except ImportError:
    nx = None

from .base_file_graph import BaseFileGraph
from ..component_registry import R
from ...enumeration import LinkScopeEnum
from ...schema import FileLink, FileNode


@R.register("nx")
class NxFileGraph(BaseFileGraph):
    """Networkx-backed file graph; uses FileLink.target_path for adjacency.

    Real node carries ``node`` attr; virtual (dangling target) does not.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if nx is None:
            raise ImportError("NxFileGraph requires networkx — pip install networkx")
        self._graph: nx.MultiDiGraph = nx.MultiDiGraph()
        self.component_metadata_path.mkdir(parents=True, exist_ok=True)
        self._graph_file: Path = self.component_metadata_path / f"{self.name}.pkl"

    # -- Lifecycle ---------------------------------------------------------

    async def load(self) -> None:
        if not self._graph_file.exists():
            return
        try:
            with open(self._graph_file, "rb") as f:
                self._graph = pickle.load(f)
            self.logger.info(f"Loaded {self._real_count()} nodes from {self._graph_file}")
        except Exception as e:
            self.logger.exception(f"Failed to load {self._graph_file}: {e}")

    async def dump(self) -> None:
        try:
            tmp = self._graph_file.with_suffix(".tmp")
            with open(tmp, "wb") as f:
                pickle.dump(self._graph, f, protocol=pickle.HIGHEST_PROTOCOL)
            tmp.replace(self._graph_file)
            self.logger.info(f"Saved {self._real_count()} nodes to {self._graph_file}")
        except Exception as e:
            self.logger.exception(f"Failed to write {self._graph_file}: {e}")

    # -- Internals ---------------------------------------------------------

    def _real_count(self) -> int:
        return sum(1 for _, d in self._graph.nodes(data=True) if "node" in d)

    def _is_real(self, key: str) -> bool:
        return "node" in self._graph.nodes[key]

    @staticmethod
    def _edges_from(src: str, node: FileNode):
        return ((src, lnk.target_path, {"link": lnk}) for lnk in node.links if lnk.target_path)

    def _scope_match(self, key: str, scope: LinkScopeEnum) -> bool:
        if scope is LinkScopeEnum.ALL:
            return True
        is_real = self._is_real(key)
        return is_real if scope is LinkScopeEnum.REAL else not is_real

    # -- Node CRUD ---------------------------------------------------------

    async def upsert_nodes(self, nodes: list[FileNode]) -> None:
        for node in nodes:
            path = node.path
            if self._graph.has_node(path):
                self._graph.remove_edges_from(list(self._graph.out_edges(path, keys=True)))
            self._graph.add_node(path, node=node)  # promotes virtual placeholder
            self._graph.add_edges_from(self._edges_from(path, node))

    async def delete_nodes(self, paths: list[str]) -> None:
        for path in paths:
            if not self._graph.has_node(path):
                continue
            self._graph.remove_edges_from(list(self._graph.out_edges(path, keys=True)))
            self._graph.nodes[path].pop("node", None)  # demote to virtual
            if self._graph.in_degree(path) == 0:
                self._graph.remove_node(path)

    async def get_nodes(self, paths: list[str] | None = None) -> list[FileNode]:
        view = self._graph.nodes
        if paths is None:
            return [d["node"] for _, d in view(data=True) if "node" in d]
        return [view[p]["node"] for p in paths if p in view and "node" in view[p]]

    async def rebuild_links(self) -> None:
        self._graph.remove_edges_from(list(self._graph.edges(keys=True)))
        virtual = [n for n, d in self._graph.nodes(data=True) if "node" not in d]
        self._graph.remove_nodes_from(virtual)
        for path, data in list(self._graph.nodes(data=True)):
            self._graph.add_edges_from(self._edges_from(path, data["node"]))

    async def clear(self):
        self._graph.clear()
        self._graph_file.unlink(missing_ok=True)

    # -- Link access -------------------------------------------------------

    async def get_outlinks(self, path: str, scope: LinkScopeEnum = LinkScopeEnum.REAL) -> list[FileLink]:
        view = self._graph.nodes
        if path not in view or "node" not in view[path]:
            return []
        return [
            d["link"]
            for _, tgt, d in self._graph.out_edges(path, data=True)
            if "link" in d and self._scope_match(tgt, scope)
        ]

    async def get_inlinks(self, path: str, scope: LinkScopeEnum = LinkScopeEnum.REAL) -> list[FileLink]:
        view = self._graph.nodes
        if path not in view or not self._scope_match(path, scope):
            return []
        return [d["link"] for _, _, d in self._graph.in_edges(path, data=True) if "link" in d]
