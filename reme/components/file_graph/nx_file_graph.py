"""Networkx file-graph backend."""

import pickle
from pathlib import Path

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
        try:
            import networkx as nx  # pylint: disable=import-outside-toplevel
        except ImportError as exc:
            raise ImportError("NxFileGraph requires networkx — pip install networkx") from exc
        self._graph = nx.MultiDiGraph()
        self.component_metadata_path.mkdir(parents=True, exist_ok=True)
        self._graph_file: Path = self.component_metadata_path / f"{self.name}.pkl"

    # -- Lifecycle ---------------------------------------------------------

    async def load(self) -> None:
        if not self._graph_file.exists():
            return
        try:
            with open(self._graph_file, "rb") as f:
                self._graph = pickle.load(f)
            migrated = self._canonicalize_graph()
            if migrated:
                self.logger.info(f"Canonicalized {migrated} legacy path spelling(s) in {self._graph_file}")
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

    def _has_legacy_keys(self) -> bool:
        """Report whether any persisted node key is not canonical.

        Every edge endpoint is itself a node key, so canonical nodes imply
        canonical edges.  Iterating ``MultiDiGraph.edges`` here yields
        ``(source, target, key)`` and raised instead of answering, which made
        every load of a graph with edges fail before migration could run.
        """
        return any(self.canonical_path(key) != key for key in self._graph.nodes)

    def _canonicalize_graph(self) -> int:
        """Rewrite persisted node keys and rebuild edges from winning records."""
        if not self._has_legacy_keys():
            return 0
        graph = self._graph.__class__()
        migrated = 0
        for key, data in self._graph.nodes(data=True):
            node = data.get("node")
            if node is None:
                # Placeholders are recreated from the winning records below.
                continue
            path = self.canonical_path(key)
            migrated += path != key
            node = self.canonical_node(node)
            existing = graph.nodes[path].get("node") if graph.has_node(path) else None
            if existing is not None:
                keep, extra = (node, existing) if node.st_mtime >= existing.st_mtime else (existing, node)
                graph.add_node(path, node=self.merge_chunk_ids(keep, extra))
                continue
            graph.add_node(path, node=node)
        # Copying the persisted edges verbatim would keep the dropped spelling's
        # links as outlinks of the merged node, in addition to the winner's.
        for path, data in list(graph.nodes(data=True)):
            graph.add_edges_from(self._edges_from(path, data["node"]))
        self._graph = graph
        return migrated

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
            node = self.canonical_node(node)
            path = node.path
            if self._graph.has_node(path):
                self._graph.remove_edges_from(list(self._graph.out_edges(path, keys=True)))
            self._graph.add_node(path, node=node)  # promotes virtual placeholder
            self._graph.add_edges_from(self._edges_from(path, node))

    async def delete_nodes(self, paths: list[str]) -> None:
        for path in paths:
            path = self.canonical_path(path)
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
        canonical = [self.canonical_path(p) for p in paths]
        return [view[p]["node"] for p in canonical if p in view and "node" in view[p]]

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
        path = self.canonical_path(path)
        view = self._graph.nodes
        if path not in view or "node" not in view[path]:
            return []
        return [
            d["link"]
            for _, tgt, d in self._graph.out_edges(path, data=True)
            if "link" in d and self._scope_match(tgt, scope)
        ]

    async def get_inlinks(self, path: str, scope: LinkScopeEnum = LinkScopeEnum.REAL) -> list[FileLink]:
        path = self.canonical_path(path)
        view = self._graph.nodes
        if path not in view or not self._scope_match(path, scope):
            return []
        return [d["link"] for _, _, d in self._graph.in_edges(path, data=True) if "link" in d]
