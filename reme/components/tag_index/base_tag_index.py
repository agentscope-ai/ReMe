"""Abstract interface for file-level tag indexes derived from graph nodes."""

from abc import abstractmethod

from ..base_component import BaseComponent
from ...enumeration import ComponentEnum
from ...schema import FileNode


class BaseTagIndex(BaseComponent):
    """A rebuildable index of normalized ``FileNode`` frontmatter tags."""

    component_type = ComponentEnum.TAG_INDEX

    @abstractmethod
    def normalize_tags(self, value: object) -> list[str]:
        """Return canonical tags according to this index's configured limits."""

    @abstractmethod
    async def rebuild(self, nodes: list[FileNode]) -> None:
        """Replace the complete index with relationships derived from ``nodes``."""

    @abstractmethod
    async def upsert_nodes(self, nodes: list[FileNode]) -> None:
        """Insert or replace relationships derived from ``nodes``."""

    @abstractmethod
    async def delete(self, paths: list[str]) -> None:
        """Delete relationships by workspace-relative path."""

    @abstractmethod
    async def paths_for_tags(self, tags: object, *, match_all: bool = True) -> list[str]:
        """Return sorted paths matching all or any normalized tags."""

    @abstractmethod
    async def tags_for_path(self, path: str) -> list[str]:
        """Return normalized tags for one workspace-relative path."""

    @abstractmethod
    async def clear(self) -> None:
        """Clear memory and persisted state."""
