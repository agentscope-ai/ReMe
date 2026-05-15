"""FileLink"""

from pydantic import BaseModel, ConfigDict, Field


class FileLink(BaseModel):
    """A parsed wikilink with optional anchor and predicate."""

    model_config = ConfigDict(extra="forbid")

    path: str = Field(
        default=...,
        description="Wikilink target — raw text pre-resolution, vault-relative path after.",
    )
    anchor: str | None = Field(
        default=None,
        description="Heading or block anchor (text after '#'); None if absent.",
    )
    predicate: str | None = Field(
        default=None,
        description="Dataview-style typed-link predicate; None for bare [[X]].",
    )
