"""Search filter schema for constraining search results."""

from pydantic import BaseModel, Field


class SearchFilter(BaseModel):
    """Filter conditions for search operations.

    All specified conditions are combined with AND logic.
    Within paths/exclude_paths, items are combined with OR logic.
    Within tags, items are combined with AND logic (all must match).
    """

    paths: list[str] | None = Field(
        default=None,
        description="Include only chunks whose path starts with any of these prefixes",
    )
    tags: list[str] | None = Field(default=None, description="Include only chunks containing ALL specified tags")
    exclude_paths: list[str] | None = Field(
        default=None,
        description="Exclude chunks whose path starts with any of these prefixes",
    )

    def is_empty(self) -> bool:
        return not self.paths and not self.tags and not self.exclude_paths

    def match(self, path: str, metadata: dict | None = None) -> bool:
        if self.paths and not any(path.startswith(p) for p in self.paths):
            return False
        if self.exclude_paths and any(path.startswith(p) for p in self.exclude_paths):
            return False
        if self.tags:
            chunk_tags = set((metadata or {}).get("tags", []))
            if not all(t in chunk_tags for t in self.tags):
                return False
        return True
