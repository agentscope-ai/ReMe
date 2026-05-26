"""Abstract base class for file catalog backends."""

from abc import abstractmethod

from ..base_component import BaseComponent
from ...enumeration import ComponentEnum
from ...schema import FileNode


class BaseFileCatalog(BaseComponent):
    """File-catalog backend recording FileNode entries keyed by path."""

    component_type = ComponentEnum.FILE_CATALOG

    async def _start(self) -> None:
        await super()._start()
        await self.load()

    async def _close(self) -> None:
        await self.dump()
        await super()._close()

    async def load(self) -> None:
        """Load persisted state. No-op without local files."""

    async def dump(self) -> None:
        """Persist state. No-op without local files."""

    @abstractmethod
    async def upsert(self, nodes: list[FileNode]) -> None:
        """Insert or update nodes keyed by path."""

    @abstractmethod
    async def delete(self, path: str | list[str]) -> None:
        """Delete nodes by path; missing paths are skipped."""

    @abstractmethod
    async def get_nodes(self, paths: list[str] | None = None) -> list[FileNode]:
        """Return nodes by paths; None = all; missing paths are skipped."""
