"""One-sentence tasks and their explicitly reported progress."""

from datetime import date, datetime, timezone
from typing import Literal
from uuid import uuid4

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field

from .notebook import DEFAULT_PROJECT, NotebookProject


class TodoProgress(BaseModel):
    """One progress note, timestamped by the system unless explicitly supplied."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    time: AwareDatetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    text: str = Field(min_length=1)


class Todo(BaseModel):
    """A task with a stable identity, current status and optional planning details."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    id: str = Field(default_factory=lambda: uuid4().hex, min_length=1)
    project: NotebookProject = DEFAULT_PROJECT
    text: str = Field(min_length=1)
    status: Literal["open", "partial", "done", "cancelled"] = "open"
    progress: list[TodoProgress] = Field(default_factory=list)
    due_date: date | None = None
    closed_at: AwareDatetime | None = None
    source_idea: str | None = Field(default=None, min_length=1)
