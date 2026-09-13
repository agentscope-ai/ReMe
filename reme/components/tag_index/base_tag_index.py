"""Abstract interface for file-level tag indexes derived from graph nodes."""

from abc import abstractmethod
from typing import ClassVar, Literal, TypedDict

from ..base_component import BaseComponent
from ...constants import DEFAULT_MAX_MEMORY_TAG_LENGTH, DEFAULT_MAX_MEMORY_TAGS, DEFAULT_MEMORY_TAG_KEY
from ...enumeration import ComponentEnum
from ...schema import FileNode

TagOrderBy = Literal["tag", "file_count"]
TagOrder = Literal["asc", "desc"]
TagListItem = tuple[str, int]


class TagListResult(TypedDict):
    """One page of active tags and their indexed-file counts."""

    total_tags: int
    total_pages: int
    page: int
    range: tuple[int, int]
    items: list[TagListItem]


class BaseTagIndex(BaseComponent):
    """A rebuildable index of normalized ``FileNode`` frontmatter tags."""

    component_type = ComponentEnum.TAG_INDEX
    reserved_tag_keys: ClassVar[frozenset[str]] = frozenset()

    def __init__(
        self,
        tag_key: object = DEFAULT_MEMORY_TAG_KEY,
        max_tags_per_file: int = DEFAULT_MAX_MEMORY_TAGS,
        max_tag_length: int = DEFAULT_MAX_MEMORY_TAG_LENGTH,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._tag_key = self._validate_tag_key(tag_key)
        self.max_tags_per_file = self._positive_int("max_tags_per_file", max_tags_per_file)
        self.max_tag_length = self._positive_int("max_tag_length", max_tag_length)
        self.is_healthy = True

    @staticmethod
    def _positive_int(name: str, value: object) -> int:
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
        return value

    @property
    def tag_key(self) -> str:
        """Frontmatter field from which this index derives tags."""
        return self._tag_key

    @tag_key.setter
    def tag_key(self, value: object) -> None:
        tag_key = self._validate_tag_key(value)
        if tag_key != self._tag_key:
            self._tag_key = tag_key
            self.set_healthy(False)

    def _validate_tag_key(self, value: object) -> str:
        """Validate and normalize the configured frontmatter tag field."""
        if not isinstance(value, str) or not value.strip():
            raise ValueError("tag_key must be a non-empty string")
        tag_key = value.strip()
        if tag_key in self.reserved_tag_keys:
            raise ValueError(f"tag_key must not be a reserved frontmatter key: {tag_key!r}")
        return tag_key

    def set_healthy(self, healthy: bool) -> None:
        """Mark whether lookups can safely use the current derived state."""
        self.is_healthy = healthy

    @property
    @abstractmethod
    def n_files(self) -> int:
        """Return the number of files that currently have indexed tags."""

    @abstractmethod
    def normalize_tags(self, value: object) -> list[str]:
        """Return canonical tags according to this index's configured limits."""

    @abstractmethod
    def normalize_query_tags(self, value: object) -> list[str]:
        """Return canonical lookup tags without per-file count limits."""

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
    async def list_tags(
        self,
        *,
        page: int = 1,
        order_by: TagOrderBy = "tag",
        order: TagOrder | None = None,
        page_size: int = 100,
    ) -> TagListResult:
        """Return a page of active tags with counts and a 1-based result range."""

    @abstractmethod
    async def clear(self) -> None:
        """Clear memory and persisted state."""
