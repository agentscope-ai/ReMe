"""File node — a file's metadata, links, and chunk references in the graph."""

from pydantic import BaseModel, Field

from .file_front_matter import FileFrontMatter
from .file_link import FileLink


class FileNode(BaseModel):
    """A vault file as a graph node."""

    path: str = Field(default=..., description="Path relative to the vault")
    st_mtime: float = Field(default=..., description="Filesystem mtime (seconds)")
    links: list[FileLink] = Field(default_factory=list, description="Outgoing wikilinks")
    chunk_ids: list[str] = Field(default_factory=list, description="Owned FileChunk ids")
    front_matter: FileFrontMatter = Field(default_factory=FileFrontMatter, description="Parsed front matter")
