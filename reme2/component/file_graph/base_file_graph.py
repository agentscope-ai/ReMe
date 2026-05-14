"""Abstract base for the file-graph engine.

The file-graph owns ``FileNode`` records keyed by vault-relative path
and serves graph traversal over wikilink links. file_graph trusts
``FileLink.path`` directly — there is no internal wikilink resolution.
Pre-resolution forms (raw stem-form wikilinks like ``[[Foo]]``) are a
vault convention resolved by the external ``utils.wikilink_resolver``
before nodes are upserted.

Contract — six abstract methods, two blocks:

    Node CRUD     upsert_node, delete_node, get_node, iter_nodes
    Link access   get_outlinks, get_inlinks
"""

from __future__ import annotations

from abc import abstractmethod
from pathlib import Path

from ..base_component import BaseComponent
from ...enumeration import ComponentEnum
from ...schema import FileLink, FileNode


class BaseFileGraph(BaseComponent):
    """Pluggable file-graph backend — node CRUD + link adjacency."""

    component_type = ComponentEnum.FILE_GRAPH

    def __init__(self, graph_name: str = "default", **kwargs):
        super().__init__(**kwargs)
        self.graph_name: str = graph_name or self.name
        self.graph_path: Path = self.working_path / self.component_type.value / graph_name
        self.graph_path.mkdir(parents=True, exist_ok=True)

    # -- Node CRUD ---------------------------------------------------------

    @abstractmethod
    async def upsert_node(self, node: FileNode) -> None:
        """Add or replace a node and its outgoing links."""

    @abstractmethod
    async def delete_node(self, path: str) -> FileNode | None:
        """Remove a node + its incident links. Returns the removed node, or None."""

    @abstractmethod
    async def get_node(self, path: str) -> FileNode | None:
        """Single-node lookup."""

    # -- Link access -------------------------------------------------------

    @abstractmethod
    async def get_outlinks(self, path: str) -> list[FileLink]:
        """Resolved outgoing links from ``path``."""

    @abstractmethod
    async def get_inlinks(self, path: str) -> list[FileLink]:
        """Resolved incoming links to ``path``."""
