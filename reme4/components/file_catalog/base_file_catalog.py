"""Abstract base for file-catalog backends."""

from abc import abstractmethod
from pathlib import Path

from ..base_component import BaseComponent
from ...enumeration import ComponentEnum
from ...schema import FileNode


class BaseFileCatalog(BaseComponent):
    """Abstract base for file-catalog backends.

    A catalog records FileNode entries keyed by path — a lightweight
    counterpart to FileStore that drops chunk/embedding/keyword/link
    machinery and exposes only node upsert / delete / lookup.
    """

    component_type = ComponentEnum.FILE_CATALOG

    def __init__(self, catalog_name: str = "default", catalog_version: str = "v1", **kwargs):
        super().__init__(**kwargs)
        self.catalog_name: str = catalog_name or self.name
        self.catalog_version: str = catalog_version
        self.catalog_path: Path = self.vault_metadata_path / self.component_type.value / self.catalog_name
        self.catalog_path.mkdir(parents=True, exist_ok=True)

    # -- Lifecycle ---------------------------------------------------------

    async def _start(self) -> None:
        await super()._start()
        await self.load()

    async def _close(self) -> None:
        await self.dump()
        await super()._close()

    async def load(self) -> None:
        """Load persisted state. No-op for backends without local files."""

    async def dump(self) -> None:
        """Persist state. No-op for backends without local files."""

    # -- CRUD --------------------------------------------------------------

    @abstractmethod
    async def upsert(self, nodes: list[FileNode]) -> None:
        """Insert or update nodes keyed by path."""

    @abstractmethod
    async def delete(self, path: str | list[str]) -> None:
        """Delete nodes by path; missing paths are skipped."""

    @abstractmethod
    async def get_nodes(self, paths: list[str] | None = None) -> list[FileNode]:
        """Return nodes by paths; None = all nodes; [] = []; missing paths are skipped."""
