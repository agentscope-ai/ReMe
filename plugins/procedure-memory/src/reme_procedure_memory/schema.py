"""Durable experience contract and internal processing schemas."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

IndexView = Literal["when_to_use", "when_to_use_content"]
VIEWS = ("when_to_use", "when_to_use_content")


class ProcedureMemory(BaseModel):
    """One complete experience; its filename is its identity."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True, strict=True)

    description: str = Field(min_length=1)
    when_to_use: str = Field(alias="when-to-use", min_length=1)
    content: str = Field(min_length=1)
    sources: list[str] = Field(min_length=1)

    @field_validator("description", "when_to_use", "content")
    @classmethod
    def nonblank(cls, value: str) -> str:
        """Reject blank fields without rewriting user content."""
        if not value.strip():
            raise ValueError("Experience fields must not be blank")
        return value

    def document(self) -> dict:
        """Return the four public JSON fields."""
        return self.model_dump(by_alias=True)

    def projection(self, view: IndexView) -> str:
        """Return exactly the text belonging to one search view."""
        if view == "when_to_use":
            return self.when_to_use
        if view == "when_to_use_content":
            return self.when_to_use + "\n\n" + self.content
        raise ValueError(f"Unknown index: {view}")


class SegmentResult(BaseModel):
    """Agent's bounded rolling summary, after all tool writes."""

    model_config = ConfigDict(extra="forbid", strict=True)
    summary: str
    complete: bool


class ProcessingCheckpoint(BaseModel):
    """Rebuildable progress; original files and experiences remain authoritative."""

    model_config = ConfigDict(extra="forbid")
    source: str
    sha256: str
    fingerprint: str
    next_segment: int = 0
    summary: str = ""
    complete: bool = False
    active: dict = Field(default_factory=dict)
