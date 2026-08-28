"""Abstract base for file store backends."""

import asyncio
from abc import abstractmethod
from contextlib import asynccontextmanager
from functools import wraps

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

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._maintenance_lock = asyncio.Lock()
        self._maintenance_lock_owner = None

    @asynccontextmanager
    async def _maintenance_guard(self):
        """Serialize maintenance and mutations, allowing nested backend overrides."""
        task = asyncio.current_task()
        if self._maintenance_lock_owner is task:
            yield
            return
        async with self._maintenance_lock:
            self._maintenance_lock_owner = task
            try:
                yield
            finally:
                self._maintenance_lock_owner = None

    @staticmethod
    def serialized(method):
        """Mark a mutation or maintenance method as mutually exclusive."""

        @wraps(method)
        async def wrapped(self, *args, **kwargs):
            async with self._maintenance_guard():  # pylint: disable=protected-access
                return await method(self, *args, **kwargs)

        return wrapped

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

    # -- maintenance ------------------------------------------------------------

    async def optimize_index(self) -> None:
        """Optional idle-time maintenance hook (e.g. compacting derived indexes).

        Meant to be invoked off the request path (cron / idle schedulers).
        Backends without derived index state keep the default no-op.
        """

    async def require_embedding_rebuild(self) -> None:
        """Disable vector reads and writes until a full manual rebuild."""
        raise NotImplementedError

    async def reindex(self, scope: str) -> dict:
        """Synchronously rebuild ``all``, ``bm25``, or ``embedding`` state."""
        raise NotImplementedError
