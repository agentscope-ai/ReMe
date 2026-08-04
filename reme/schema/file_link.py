"""FileLink"""

from pydantic import BaseModel, ConfigDict, Field


class FileLink(BaseModel):
    """file link
    [[target_path]]
    [[target_path#target_anchor]]
    [label](target_path)
    [label](target_path#L9-L10,L15-L20)
    """

    model_config = ConfigDict(extra="forbid")
    source_path: str = Field(default=..., description="source file path relative to working dir")
    target_path: str = Field(default=..., description="target file path relative to working dir")
    target_anchor: str | None = Field(default=None, description="Heading, block, or line anchor (text after '#')")
    predicate: str | None = Field(
        default=None,
        exclude=True,
        description="Deprecated compatibility field; accepted when loading legacy indexes but otherwise unused",
    )
