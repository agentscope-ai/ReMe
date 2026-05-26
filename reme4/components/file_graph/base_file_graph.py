"""Abstract base for file-graph backends."""

from abc import abstractmethod

from ..base_component import BaseComponent
from ...enumeration import ComponentEnum, LinkScopeEnum
from ...schema import FileLink, FileNode


class BaseFileGraph(BaseComponent):
    """Abstract base for file-graph backends.

    Link scope (``get_outlinks`` / ``get_inlinks``):
        REAL    edges touching an indexed node
        VIRTUAL edges touching a dangling placeholder
                (referenced but never upserted, or already deleted)
        ALL     both
    """

    component_type = ComponentEnum.FILE_GRAPH

    # -- Lifecycle ---------------------------------------------------------

    async def _start(self) -> None:
        await super()._start()
        await self.load()

    async def _close(self) -> None:
        await self.dump()
        await super()._close()

    async def load(self) -> None:
        """Restore persisted state. No-op for backends without local files."""

    async def dump(self) -> None:
        """Persist state. No-op for backends without local files."""

    # -- Node CRUD ---------------------------------------------------------

    @abstractmethod
    async def upsert_nodes(self, nodes: list[FileNode]) -> None:
        """Insert or update nodes."""

    @abstractmethod
    async def delete_nodes(self, paths: list[str]) -> None:
        """Remove nodes by path."""

    @abstractmethod
    async def get_nodes(self, paths: list[str] | None = None) -> list[FileNode]:
        """Return nodes by paths; ``None`` = all real nodes."""

    @abstractmethod
    async def rebuild_links(self) -> None:
        """Rebuild all edges from each node's link payload."""

    @abstractmethod
    async def clear(self):
        """Remove all nodes and edges."""

    # -- Link access -------------------------------------------------------

    @abstractmethod
    async def get_outlinks(self, path: str, scope: LinkScopeEnum = LinkScopeEnum.REAL) -> list[FileLink]:
        """Outgoing links from *path*."""

    @abstractmethod
    async def get_inlinks(self, path: str, scope: LinkScopeEnum = LinkScopeEnum.REAL) -> list[FileLink]:
        """Inbound links to *path*."""
