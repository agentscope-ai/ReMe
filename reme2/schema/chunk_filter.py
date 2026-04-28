"""Filter for chunk-store search.

User-facing fields (paths/tags/exclude_paths) describe metadata-level intent.
FileGraph compiles them into `resolved_paths` (the concrete path set that
ChunkStore actually consumes for filtering).
"""

from pydantic import BaseModel, Field


class ChunkFilter(BaseModel):
    """User-facing search filter, resolved by FileGraph into a path set.

    User input fields:
        paths: include only paths starting with any of these prefixes
        tags: include only files whose metadata contains ALL these tags
        exclude_paths: exclude paths starting with any of these prefixes

    Compiled field (set by FileGraph.filter):
        resolved_paths: concrete path set to restrict ChunkStore search.
            None = no restriction (all chunks).
            Empty set = no chunks match (search returns empty).
    """

    paths: list[str] | None = Field(default=None)
    tags: list[str] | None = Field(default=None)
    exclude_paths: list[str] | None = Field(default=None)
    resolved_paths: set[str] | None = Field(default=None)

    def is_empty(self) -> bool:
        return not self.paths and not self.tags and not self.exclude_paths

    def match_metadata(self, path: str, metadata: dict | None = None) -> bool:
        """Match a single path+metadata against user-input conditions."""
        if self.paths and not any(path.startswith(p) for p in self.paths):
            return False
        if self.exclude_paths and any(path.startswith(p) for p in self.exclude_paths):
            return False
        if self.tags:
            file_tags = set((metadata or {}).get("tags", []))
            if not all(t in file_tags for t in self.tags):
                return False
        return True

    def match_path(self, path: str) -> bool:
        """Match a path against the resolved path set (used by ChunkStore)."""
        if self.resolved_paths is None:
            return True
        return path in self.resolved_paths
