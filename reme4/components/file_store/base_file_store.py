"""Abstract base for file store backends."""

from abc import abstractmethod

from ..base_component import BaseComponent
from ...enumeration import ComponentEnum, LinkScopeEnum
from ...schema import FileChunk, FileLink, FileNode


class BaseFileStore(BaseComponent):
    """Abstract base for file store backends.

    Defines the *semantic* contract a file store must offer: write (upsert / delete / clear),
    retrieve (vector / keyword), and graph queries (nodes / links). Sub-component composition
    (embedding model, keyword index, file graph) is each backend's implementation choice and
    is not part of the base contract.
    """

    component_type = ComponentEnum.FILE_STORE

    # -- CRUD ------------------------------------------------------------

    @abstractmethod
    async def upsert(self, files: list[tuple[FileNode, list[FileChunk]]]) -> None:
        """Upsert files and their chunks into the store."""

    @abstractmethod
    async def delete(self, path: str | list[str]) -> None:
        """Delete files by path from the store."""

    @abstractmethod
    async def get_nodes(self, paths: list[str] | None = None) -> list[FileNode]:
        """Return file nodes; None = all nodes; missing paths are skipped."""

    @abstractmethod
    async def get_outlinks(
        self,
        path: str,
        scope: LinkScopeEnum = LinkScopeEnum.REAL,
    ) -> list[FileLink]:
        """Return outgoing links for *path*. See ``BaseFileGraph.get_outlinks`` for scope semantics."""

    @abstractmethod
    async def get_inlinks(
        self,
        path: str,
        scope: LinkScopeEnum = LinkScopeEnum.REAL,
    ) -> list[FileLink]:
        """Return incoming links for *path*. See ``BaseFileGraph.get_inlinks`` for scope semantics."""

    @abstractmethod
    async def clear(self) -> None:
        """Clear the store of all files and chunks."""

    # -- Search -----------------------------------------------------------

    @abstractmethod
    async def vector_search(self, query: str, limit: int, search_filter: dict) -> list[FileChunk]:
        """Perform vector similarity search."""

    @abstractmethod
    async def keyword_search(self, query: str, limit: int, search_filter: dict) -> list[FileChunk]:
        """Perform full-text keyword search."""
