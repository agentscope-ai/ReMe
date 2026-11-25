"""One-sentence tasks and their explicitly reported progress."""

from datetime import date
from uuid import uuid4

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field

from .notebook import DEFAULT_PROJECT, NotebookProject, TodoProgress, TodoStatus


class Todo(BaseModel):
    """A task with a stable identity, current status and optional planning details."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    id: str = Field(default_factory=lambda: uuid4().hex, min_length=1)
    project: NotebookProject = DEFAULT_PROJECT
    text: str = Field(min_length=1)
    status: TodoStatus = "pending"
    progress: list[TodoProgress] = Field(default_factory=list)
    due_date: date | None = None
    created_at: AwareDatetime | None = None
    closed_at: AwareDatetime | None = None
    close_note: str | None = Field(default=None, min_length=1)
    source_idea: str | None = Field(default=None, min_length=1)
