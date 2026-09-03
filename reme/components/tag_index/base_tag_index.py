"""Abstract maintenance interface for file-level tag indexes."""

from abc import abstractmethod

from ..base_component import BaseComponent
from ...enumeration import ComponentEnum
from ...schema import TagSourceRecord


class BaseTagIndex(BaseComponent):
    """A rebuildable index of normalized frontmatter tags."""

    component_type = ComponentEnum.TAG_INDEX

    @abstractmethod
    def normalize_tags(self, value: object) -> list[str]:
        """Return canonical tags according to this index's configured limits."""

    @abstractmethod
    async def get_records(self) -> list[TagSourceRecord]:
        """Return a detached view of indexed source records."""

    @abstractmethod
    async def upsert(self, records: list[TagSourceRecord]) -> None:
        """Insert or replace source records."""

    @abstractmethod
    async def delete(self, paths: list[str]) -> None:
        """Delete source records by workspace-relative path."""

    @abstractmethod
    async def reconcile(
        self,
        records: list[TagSourceRecord],
        deleted_paths: list[str],
        *,
        rebuild: bool = False,
        persist: bool = True,
    ) -> None:
        """Apply one complete prepared reconciliation batch."""

    @abstractmethod
    async def clear(self) -> None:
        """Clear memory and persisted state."""
