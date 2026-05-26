"""Abstract base for file store backends."""

from abc import abstractmethod

from ..base_component import BaseComponent
from ...enumeration import ComponentEnum, LinkScopeEnum
from ...schema import FileChunk, FileLink, FileNode


class BaseFileStore(BaseComponent):
    """Abstract base for file store backends.

    Defines the *semantic* contract a file store must offer: write (upsert / delete /
    clear), retrieve (vector / keyword), and graph queries (nodes / links). How the
    backend composes sub-components (embedding model, keyword index, file graph) is
    an implementation detail outside this contract.
    """

    component_type = ComponentEnum.FILE_STORE

    # -- CRUD -----------------------------------------------------------------

    @abstractmethod
    async def upsert(self, files: list[tuple[FileNode, list[FileChunk]]]) -> None:
        """Upsert files and their chunks; existing chunks for the same path are replaced."""

    @abstractmethod
    async def delete(self, path: str | list[str]) -> None:
        """Delete the given path(s) and all their chunks; unknown paths are skipped."""

    @abstractmethod
    async def clear(self) -> None:
        """Drop every file and chunk in the store."""

    # -- graph queries --------------------------------------------------------

    @abstractmethod
    async def get_nodes(self, paths: list[str] | None = None) -> list[FileNode]:
        """Return file nodes; ``None`` = all; missing paths are skipped."""

    @abstractmethod
    async def get_outlinks(
        self,
        path: str,
        scope: LinkScopeEnum = LinkScopeEnum.REAL,
    ) -> list[FileLink]:
        """Outgoing links for *path*; scope semantics match ``BaseFileGraph.get_outlinks``."""

    @abstractmethod
    async def get_inlinks(
        self,
        path: str,
        scope: LinkScopeEnum = LinkScopeEnum.REAL,
    ) -> list[FileLink]:
        """Incoming links for *path*; scope semantics match ``BaseFileGraph.get_inlinks``."""

    # -- search ---------------------------------------------------------------

    @abstractmethod
    async def vector_search(self, query: str, limit: int, search_filter: dict) -> list[FileChunk]:
        """Vector similarity search over chunk embeddings."""

    @abstractmethod
    async def keyword_search(self, query: str, limit: int, search_filter: dict) -> list[FileChunk]:
        """Full-text keyword search over chunk text."""
