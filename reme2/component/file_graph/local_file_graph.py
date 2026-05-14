"""Local file-graph backend — networkx ``MultiDiGraph``.

Each ``FileNode`` lives as ``data['node']`` on the graph; each
``FileLink`` whose ``link.path`` exists in the graph lives as
``data['link']`` on a directed graph edge between the two file nodes.

file_graph trusts ``link.path`` directly — there is no internal
wikilink resolution. The parser pipeline (with the external resolver)
is responsible for producing safe ``FileLink`` records where
``link.path`` is a real vault-relative target path. Stem ambiguity is
already handled there by emitting one link per candidate.

Late-arriving target: when ``upsert_node(B)`` runs, other nodes whose
links have ``link.path == B.path`` get those in-edges restored
(one O(N×L) sweep per upsert; cheap for vault sizes).

Persistence: pickle to ``graph.pkl`` on close, restored on _start.
"""

from __future__ import annotations

import pickle
from collections.abc import AsyncIterator
from pathlib import Path

from .base_file_graph import BaseFileGraph
from ..component_registry import R
from ...schema import FileLink, FileNode
import networkx as nx

@R.register("local")
class LocalFileGraph(BaseFileGraph):
    """Networkx-backed file graph. Trusts ``FileLink.path`` for adjacency."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._graph = None  # networkx.MultiDiGraph; set in _start
        self._graph_file: Path = self.graph_path / f"{self.graph_name}.pkl"

    # -- Lifecycle ---------------------------------------------------------

    async def _start(self) -> None:
        await super()._start()
        self._graph = self._load_graph(nx) or nx.MultiDiGraph()
        self.logger.info(
            f"LocalFileGraph '{self.graph_name}' ready: "
            f"{self._graph.number_of_nodes()} nodes, "
            f"{self._graph.number_of_edges()} edges",
        )

    async def _close(self) -> None:
        self._save_graph()
        await super()._close()

    def _load_graph(self, nx):
        if not self._graph_file.exists():
            return None
        try:
            with open(self._graph_file, "rb") as f:
                graph = pickle.load(f)
            if not isinstance(graph, nx.MultiDiGraph):
                self.logger.warning(
                    f"{self._graph_file} is not a MultiDiGraph; ignoring",
                )
                return None
            return graph
        except Exception as e:
            self.logger.exception(f"Failed to load {self._graph_file}: {e}")
            return None

    def _save_graph(self) -> None:
        try:
            tmp = self._graph_file.with_suffix(".tmp")
            with open(tmp, "wb") as f:
                pickle.dump(self._graph, f, protocol=pickle.HIGHEST_PROTOCOL)
            tmp.replace(self._graph_file)
        except Exception as e:
            self.logger.exception(f"Failed to write {self._graph_file}: {e}")

    # -- Node CRUD ---------------------------------------------------------

    async def upsert_node(self, node: FileNode) -> None:
        path = node.path
        if self._graph.has_node(path):
            self._graph.remove_node(path)
        self._graph.add_node(path, node=node)

        # Out-links from this node — trust link.path.
        for link in node.links:
            if link.path and self._graph.has_node(link.path):
                self._graph.add_edge(path, link.path, link=link)

        # Late-arriving target: restore in-links from other nodes whose
        # links resolve here. Scan all other nodes' links; cheap for
        # vault sizes (O(N×L_avg) per upsert).
        for src_path, src_data in self._graph.nodes(data=True):
            if src_path == path:
                continue
            src_node = src_data.get("node")
            if src_node is None:
                continue
            for link in src_node.links:
                if link.path == path:
                    self._graph.add_edge(src_path, path, link=link)

    async def delete_node(self, path: str) -> FileNode | None:
        if not self._graph.has_node(path):
            return None
        node = self._graph.nodes[path].get("node")
        self._graph.remove_node(path)
        return node

    async def get_node(self, path: str) -> FileNode | None:
        if not self._graph.has_node(path):
            return None
        return self._graph.nodes[path].get("node")

    # -- Link access -------------------------------------------------------

    async def get_outlinks(self, path: str) -> list[tuple[FileNode, FileLink]]:
        if not self._graph.has_node(path):
            return []
        out: list[tuple[FileNode, FileLink]] = []
        for _src, dst, data in self._graph.out_edges(path, data=True):
            target = self._graph.nodes[dst].get("node")
            link = data.get("link")
            if target is not None and isinstance(link, FileLink):
                out.append((target, link))
        return out

    async def get_inlinks(self, path: str) -> list[tuple[FileNode, FileLink]]:
        if not self._graph.has_node(path):
            return []
        out: list[tuple[FileNode, FileLink]] = []
        for src, _dst, data in self._graph.in_edges(path, data=True):
            source = self._graph.nodes[src].get("node")
            link = data.get("link")
            if source is not None and isinstance(link, FileLink):
                out.append((source, link))
        return out
