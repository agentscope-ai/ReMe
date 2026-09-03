"""Serializable records used by the file-level tag index."""

from pydantic import BaseModel, ConfigDict, Field


class TagFileRecord(BaseModel):
    """One indexed workspace file and its tag relationships."""

    model_config = ConfigDict(extra="forbid", strict=True)

    id: int
    path: str
    mtime_ns: int
    tag_ids: list[int] = Field(default_factory=list)


class TagRecord(BaseModel):
    """One canonical tag and its posting list."""

    model_config = ConfigDict(extra="forbid", strict=True)

    id: int
    name: str
    file_ids: list[int] = Field(default_factory=list)


class TagIndexSnapshot(BaseModel):
    """Complete on-disk tag-index snapshot."""

    model_config = ConfigDict(extra="forbid", strict=True)

    next_file_id: int
    next_tag_id: int
    files: list[TagFileRecord]
    tags: list[TagRecord]
    max_tag_length: int
    max_tags_per_file: int
    max_frontmatter_bytes: int


class TagSourceRecord(BaseModel):
    """Prepared source-file state accepted by index mutations."""

    model_config = ConfigDict(extra="forbid", strict=True)

    path: str
    mtime_ns: int
    tags: list[str] = Field(default_factory=list)
