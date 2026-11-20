"""Action-specific requests for the Todo and Idea jobs."""

from datetime import date
from typing import Annotated, ClassVar, Literal

from pydantic import AfterValidator, AwareDatetime, BaseModel, ConfigDict, Field, StringConstraints, model_validator

DEFAULT_PROJECT = "随笔"
NotebookProject = Annotated[
    str,
    StringConstraints(strip_whitespace=True),
    AfterValidator(lambda value: value or DEFAULT_PROJECT),
]


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
        if self.action == "update" and not supplied - {"id"}:
            raise ValueError("update requires at least one changed field")
        return self


class TodoRequest(_RecordRequest):
    """Create, query and patch a task, optionally appending one progress note."""

    action_fields = {
        "create": {"project", "text", "status", "due_date", "source_idea", "progress_text", "progress_time"},
        "list": {"project", "status", "source_idea", "scope", "query", "offset", "limit"},
        "get": {"id"},
        "update": {"id", "project", "text", "status", "due_date", "source_idea", "progress_text", "progress_time"},
    }

    action: Literal["create", "list", "get", "update"]
    status: Literal["open", "partial", "done", "cancelled"] = "open"
    due_date: date | None = None
    source_idea: str | None = Field(default=None, min_length=1)
    progress_text: str | None = Field(default=None, min_length=1)
    progress_time: AwareDatetime | None = None

    @model_validator(mode="after")
    def validate_progress(self):
        """A supplied progress timestamp must belong to a nonempty progress note."""
        if "progress_text" in self.model_fields_set and self.progress_text is None:
            raise ValueError("progress_text cannot be null")
        if "progress_time" in self.model_fields_set and not self.progress_text:
            raise ValueError("progress_time requires progress_text")
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
