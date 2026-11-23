"""One-sentence ideas that can be referenced by tasks."""

from uuid import uuid4

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field

from .notebook import DEFAULT_PROJECT, NotebookProject


class Idea(BaseModel):
    """A stable project note with an optional archive timestamp."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    id: str = Field(default_factory=lambda: uuid4().hex, min_length=1)
    project: NotebookProject = DEFAULT_PROJECT
    text: str = Field(min_length=1)
    closed_at: AwareDatetime | None = None
    event_ids: list[str] | None = None
