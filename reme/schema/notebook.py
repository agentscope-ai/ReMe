"""Action-specific requests for the Todo and Idea jobs."""

from datetime import date, datetime, timezone
from typing import Annotated, ClassVar, Literal

from pydantic import AfterValidator, AwareDatetime, BaseModel, ConfigDict, Field, StringConstraints, model_validator

DEFAULT_PROJECT = "随笔"
NotebookProject = Annotated[
    str,
    StringConstraints(strip_whitespace=True),
    AfterValidator(lambda value: value or DEFAULT_PROJECT),
]
TodoStatus = Literal["pending", "in_progress", "closed"]


class TodoProgress(BaseModel):
    """One progress note, timestamped by the system unless explicitly supplied."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    time: AwareDatetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    text: str = Field(min_length=1)


class _RecordRequest(BaseModel):
    """Share pagination and reject arguments that an action would ignore."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    action_fields: ClassVar[dict[str, set[str]]]

    action: str
    id: str | None = Field(default=None, min_length=1)
    project: NotebookProject = DEFAULT_PROJECT
    text: str | None = Field(default=None, min_length=1)
    scope: Literal["active", "archive", "all"] = "active"
    query: str = ""
    offset: int = Field(default=0, ge=0)
    limit: int = Field(default=20, ge=1, le=100)

    @model_validator(mode="after")
    def validate_action(self):
        """Validate public arguments before any workspace files are accessed."""
        supplied = self.model_fields_set - {"action"}
        allowed = self.action_fields[self.action]
        if supplied - allowed:
            raise ValueError(f"unsupported fields for {self.action}: {', '.join(sorted(supplied - allowed))}")
        if self.action in {"get", "update", "archive", "restore"} and not self.id:
            raise ValueError("id is required")
        if "text" in supplied and self.text is None:
            raise ValueError("text cannot be null")
        if self.action == "create" and self.text is None and not getattr(self, "source_idea", None):
            raise ValueError("text is required unless source_idea supplies it")
        if self.action == "update" and supplied == {"id"}:
            raise ValueError("update requires at least one changed field")
        return self


class TodoRequest(_RecordRequest):
    """Create, query and patch a task and its progress."""

    _write_fields: ClassVar[set[str]] = {
        "project",
        "text",
        "status",
        "due_date",
        "source_idea",
        "progress_text",
        "progress_time",
        "close_note",
    }
    action_fields = {
        "create": _write_fields,
        "list": {"project", "status", "source_idea", "scope", "query", "offset", "limit", "order"},
        "list_projects": {"scope"},
        "get": {"id"},
        "update": _write_fields | {"id", "progress_match"},
    }

    action: Literal["create", "list", "list_projects", "get", "update"]
    order: Literal["due", "created"] = "due"
    status: TodoStatus = "pending"
    due_date: date | None = None
    source_idea: str | None = Field(default=None, min_length=1)
    progress_text: str | None = Field(default=None, min_length=1)
    progress_time: AwareDatetime | None = None
    progress_match: TodoProgress | None = None
    close_note: str | None = Field(default=None, min_length=1)

    @model_validator(mode="after")
    def validate_notes(self):
        """Keep progress timestamps and closure notes attached to their respective updates."""
        if "progress_text" in self.model_fields_set and self.progress_text is None:
            raise ValueError("progress_text cannot be null")
        if "progress_time" in self.model_fields_set and not self.progress_text:
            raise ValueError("progress_time requires progress_text")
        if "progress_match" in self.model_fields_set and (
            self.progress_match is None or "time" not in self.progress_match.model_fields_set
        ):
            raise ValueError("progress_match requires the existing progress text and time")
        if "close_note" in self.model_fields_set and self.status != "closed":
            raise ValueError("close_note requires status=closed")
        return self


class IdeaRequest(_RecordRequest):
    """Keep idea closure explicit without adding a redundant status field."""

    action_fields = {
        "create": {"project", "text"},
        "list": {"project", "scope", "query", "offset", "limit"},
        "get": {"id"},
        "update": {"id", "project", "text"},
        "archive": {"id"},
        "restore": {"id"},
    }

    action: Literal["create", "list", "get", "update", "archive", "restore"]
