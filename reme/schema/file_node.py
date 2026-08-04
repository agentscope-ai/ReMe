"""File node — a file's metadata, links, and chunk references in the graph."""

from typing import Any

from pydantic import BaseModel, Field, model_validator

from .file_front_matter import FileFrontMatter
from .file_link import FileLink


class FileNode(BaseModel):
    """A workspace file as a graph node."""

    path: str = Field(default=..., description="Path relative to the workspace")
    st_mtime: float = Field(default=..., description="Filesystem mtime (seconds)")
    links: list[FileLink] = Field(default_factory=list, description="Outgoing wikilinks")
    chunk_ids: list[str] = Field(default_factory=list, description="Owned FileChunk ids")
    front_matter: FileFrontMatter = Field(default_factory=FileFrontMatter, description="Parsed front matter")

    @model_validator(mode="before")
    @classmethod
    def discard_legacy_link_predicates(cls, value: Any) -> Any:
        """Load derived indexes written before typed-link predicates were removed."""
        if not isinstance(value, dict) or not isinstance(value.get("links"), list):
            return value
        cleaned = dict(value)
        cleaned["links"] = [
            {key: item for key, item in link.items() if key != "predicate"} if isinstance(link, dict) else link
            for link in value["links"]
        ]
        return cleaned
