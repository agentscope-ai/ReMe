from abc import abstractmethod
from pathlib import Path

from ..base_component import BaseComponent
from ...enumeration import ComponentEnum
from ...schema import FileLink, FileNode


class BaseFileGraph(BaseComponent):
    component_type = ComponentEnum.FILE_GRAPH

    def __init__(self, graph_name: str = "default", **kwargs):
        super().__init__(**kwargs)
        self.graph_name: str = graph_name or self.name
        self.graph_path: Path = self.working_path / self.component_type.value
        self.graph_path.mkdir(parents=True, exist_ok=True)

    # -- Node CRUD ---------------------------------------------------------

    @abstractmethod
    async def upsert_nodes(self, nodes: list[FileNode]) -> None:
        """Upsert nodes into the graph."""

    @abstractmethod
    async def delete_nodes(self, paths: list[str]) -> None:
        """Delete nodes from the graph."""

    @abstractmethod
    async def get_nodes(self, paths: list[str]) -> list[FileNode]:
        """Get nodes from the graph."""

    @abstractmethod
    async def rebuild_links(self) -> None:
        """Rebuild all links in the graph from each node's payload."""

    # -- Link access -------------------------------------------------------

    @abstractmethod
    async def get_outlinks(self, path: str) -> list[FileLink]:
        """Get outlinks from the graph."""

    @abstractmethod
    async def get_inlinks(self, path: str) -> list[FileLink]:
        """Get inlinks from the graph."""
