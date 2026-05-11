from __future__ import annotations

from pydantic import BaseModel, Field, ConfigDict


class FileNode(BaseModel):
    model_config = ConfigDict(extra="allow")

    path: str = Field(...)
    st_mtime: float = Field(...)
    edges: list["FileEdge"] = Field(default_factory=list)

    @property
    def file(self):
        ...


class FileMetadata(BaseModel):
    # FileNode -> FileMetadata
    path: str = Field(...)

    edges: list["FileEdge"] = Field(default_factory=list)

    title: str = Field(default="")
    description: str = Field(default="")
    tags: list[str] = Field(default_factory=list)


# Rebuild models after FileEdge is defined
from .file_edge import FileEdge

FileNode.model_rebuild()
FileMetadata.model_rebuild()