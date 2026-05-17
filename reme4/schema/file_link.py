"""FileLink"""

from pydantic import BaseModel, ConfigDict, Field


class FileLink(BaseModel):
    """file link
    [[target_path]]
    [[target_path#target_anchor]]
    predicate:: [[target_*]]
    [predicate:: [[target_*]]]
    """

    model_config = ConfigDict(extra="forbid")
    source_path: str = Field(default=..., description="source file path relative to working dir")
    target_path: str = Field(default=..., description="target file path relative to working dir")
    target_anchor: str | None = Field(default=None, description="Heading or block anchor (text after '#')")
    predicate: str | None = Field(default=None, description="Dataview-style typed-link predicate")
