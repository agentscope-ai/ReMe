"""Abstract base for file-graph backends."""

from abc import abstractmethod
from pathlib import Path

from ..base_component import BaseComponent
from ...enumeration import ComponentEnum
from ...schema import FileLink, FileNode


class BaseFileGraph(BaseComponent):
    """Abstract base for file-graph backends."""

    component_type = ComponentEnum.FILE_GRAPH

    def __init__(self, graph_name: str = "default", graph_version: str = "v1", **kwargs):
        super().__init__(**kwargs)
        self.graph_name: str = graph_name or self.name
        self.graph_version: str = graph_version
        self.graph_path: Path = self.vault_metadata_path / self.component_type.value
        self.graph_path.mkdir(parents=True, exist_ok=True)

    # -- Lifecycle ---------------------------------------------------------

    async def _start(self) -> None:
        await super()._start()
        await self.load()

    async def _close(self) -> None:
        await self.dump()
        await super()._close()

    async def load(self) -> None:
        """Load persisted state. No-op for backends without local files.

        Called at the end of ``_start()`` after base resources are ready
        but before subclass-specific resources are initialised. Backends
        that need their own resources for loading should override
        ``_start()`` instead of this hook.
        """

    async def dump(self) -> None:
        """Persist state. No-op for backends without local files."""

    # -- Node CRUD ---------------------------------------------------------

    @abstractmethod
    async def upsert_nodes(self, nodes: list[FileNode]) -> None:
        """Insert or update nodes in the graph."""

    @abstractmethod
    async def delete_nodes(self, paths: list[str]) -> None:
        """Delete nodes by path."""

    @abstractmethod
    async def get_nodes(self, paths: list[str] | None = None) -> list[FileNode]:
        """Return nodes by paths; None = all real nodes; [] = []."""

    @abstractmethod
    async def rebuild_links(self) -> None:
        """Rebuild all edges from each node's link payload."""

    @abstractmethod
    async def clear(self):
        """Remove all nodes and edges."""

    # -- Link access -------------------------------------------------------

    @abstractmethod
    async def get_outlinks(self, path: str) -> list[FileLink]:
        """Return outgoing links for *path*."""

    @abstractmethod
    async def get_inlinks(self, path: str) -> list[FileLink]:
        """Return incoming links for *path*."""
