"""FileNode — engine-level file index entry.

Structural fields only: `path`, `st_mtime`, `edges`. Frontmatter fields
ride along as Pydantic extras (model_config has `extra="allow"`), so a
markdown file's parsed frontmatter dict is flattened into the node:

    FileNode(path=..., st_mtime=..., edges=[...], **frontmatter_dict)

Two convenience properties:

  * `file`     — `Path(self.path).stem`, the filename without extension.
  * `metadata` — dict view of the extras (the original frontmatter).

Domain-aware code that wants typed access to memory-schema fields
(lifecycle / scope / source / role / status / confidence …) wraps with
`reme2.memory.schema.MemoryFileNode.model_validate(node.model_dump())`.

The engine layer (file_store, file_parser, file_watcher) never imports
the memory schema — it stays domain-agnostic and just shuttles
`FileNode` instances around with their extras intact.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from reme2.schema.file_edge import FileEdge


class FileNode(BaseModel):
    model_config = ConfigDict(extra="allow")

    path: str = Field(...)
    st_mtime: float = Field(...)
    edges: list["FileEdge"] = Field(default_factory=list)

    @property
    def file(self) -> str:
        """Filename stem derived from `path` (no extension)."""
        return Path(self.path).stem

    @property
    def metadata(self) -> dict:
        """Frontmatter dict view — extras stored via `extra="allow"`.

        Returns a fresh copy so callers can't mutate the node's internal
        state by writing into the dict.
        """
        return dict(self.__pydantic_extra__ or {})
