"""Networkx file-graph backend."""

import asyncio
import pickle
from pathlib import Path
from uuid import uuid4

from .base_file_graph import BaseFileGraph
from ..component_registry import R
from ...enumeration import LinkScopeEnum
from ...schema import FileLink, FileNode
from ...utils.async_utils import complete_in_thread


@R.register("nx")
class NxFileGraph(BaseFileGraph):
    """Networkx-backed file graph; uses FileLink.target_path for adjacency.

    Real node carries ``node`` attr; virtual (dangling target) does not.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        try:
            import networkx as nx  # pylint: disable=import-outside-toplevel
        except ImportError as exc:
            raise ImportError("NxFileGraph requires networkx — pip install networkx") from exc
        self._graph = nx.MultiDiGraph()
        self._io_lock = asyncio.Lock()
        self.component_metadata_path.mkdir(parents=True, exist_ok=True)
        self._graph_file: Path = self.component_metadata_path / f"{self.name}.pkl"

    # -- Lifecycle ---------------------------------------------------------

    async def load(self) -> None:
        async with self._io_lock:
            if not self._graph_file.exists():
                return
            try:
                graph, real_count = await complete_in_thread(self._load_sync)
                self._graph = graph
                self.logger.info(f"Loaded {real_count} nodes from {self._graph_file}")
            except Exception as e:
                self.logger.exception(f"Failed to load {self._graph_file}: {e}")

    async def dump(self) -> None:
        async with self._io_lock:
            try:
                real_count = await complete_in_thread(self._dump_sync)
                self.logger.info(f"Saved {real_count} nodes to {self._graph_file}")
            except Exception as e:
                self.logger.exception(f"Failed to write {self._graph_file}: {e}")

    def _load_sync(self):
        """Load and count a NetworkX checkpoint outside the event loop."""
        with open(self._graph_file, "rb") as file:
            graph = pickle.load(file)
        real_count = sum(1 for _, data in graph.nodes(data=True) if "node" in data)
        return graph, real_count

    def _dump_sync(self) -> int:
        """Serialize and atomically publish the locked graph off-loop."""
        tmp = self._graph_file.with_name(f".{self._graph_file.name}.{uuid4().hex}.tmp")
        try:
            with open(tmp, "wb") as file:
                pickle.dump(self._graph, file, protocol=pickle.HIGHEST_PROTOCOL)
            tmp.replace(self._graph_file)
            return self._real_count()
        finally:
            tmp.unlink(missing_ok=True)

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
        async with self._io_lock:
            for node in nodes:
                path = node.path
                if self._graph.has_node(path):
                    self._graph.remove_edges_from(list(self._graph.out_edges(path, keys=True)))
                self._graph.add_node(path, node=node)  # promotes virtual placeholder
                self._graph.add_edges_from(self._edges_from(path, node))

    async def delete_nodes(self, paths: list[str]) -> None:
        async with self._io_lock:
            for path in paths:
                if not self._graph.has_node(path):
                    continue
                self._graph.remove_edges_from(list(self._graph.out_edges(path, keys=True)))
                self._graph.nodes[path].pop("node", None)  # demote to virtual
                if self._graph.in_degree(path) == 0:
                    self._graph.remove_node(path)

    async def get_nodes(self, paths: list[str] | None = None) -> list[FileNode]:
        async with self._io_lock:
            view = self._graph.nodes
            if paths is None:
                return [d["node"] for _, d in view(data=True) if "node" in d]
            return [view[p]["node"] for p in paths if p in view and "node" in view[p]]

    async def rebuild_links(self) -> None:
        async with self._io_lock:
            await complete_in_thread(self._rebuild_links_sync)

    def _rebuild_links_sync(self) -> None:
        """Rebuild NetworkX edges outside the event-loop thread."""
        self._graph.remove_edges_from(list(self._graph.edges(keys=True)))
        virtual = [n for n, d in self._graph.nodes(data=True) if "node" not in d]
        self._graph.remove_nodes_from(virtual)
        for path, data in list(self._graph.nodes(data=True)):
            self._graph.add_edges_from(self._edges_from(path, data["node"]))

    async def clear(self):
        async with self._io_lock:
            self._graph.clear()
            self._graph_file.unlink(missing_ok=True)

    # -- Link access -------------------------------------------------------

    async def get_outlinks(self, path: str, scope: LinkScopeEnum = LinkScopeEnum.REAL) -> list[FileLink]:
        async with self._io_lock:
            view = self._graph.nodes
            if path not in view or "node" not in view[path]:
                return []
            return [
                d["link"]
                for _, tgt, d in self._graph.out_edges(path, data=True)
                if "link" in d and self._scope_match(tgt, scope)
            ]

    async def get_inlinks(self, path: str, scope: LinkScopeEnum = LinkScopeEnum.REAL) -> list[FileLink]:
        async with self._io_lock:
            view = self._graph.nodes
            if path not in view or not self._scope_match(path, scope):
                return []
            return [d["link"] for _, _, d in self._graph.in_edges(path, data=True) if "link" in d]
