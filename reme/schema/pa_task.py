"""File-native scheduled tasks and append-only execution events."""

from typing import Literal

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, StrictBool, field_validator, model_validator

from .token_usage import TokenUsage


class PaTask(BaseModel):
    """A stable task identity with a five-field cron schedule."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(pattern=r"^[A-Za-z0-9_-]{1,128}$")
    cron: str
    query: str = Field(min_length=1)
    enabled: StrictBool

    @field_validator("cron")
    @classmethod
    def validate_cron(cls, value: str) -> str:
        """Keep schedule syntax consistent with minute-based task definitions."""
        from croniter import croniter

        if len(value.split()) != 5 or not croniter.is_valid(value):
            raise ValueError("cron must be a valid five-field expression")
        return value

    @field_validator("query")
    @classmethod
    def validate_query(cls, value: str) -> str:
        """Reject whitespace-only tasks."""
        if not value.strip():
            raise ValueError("query must not be blank")
        return value.strip()


class PaTaskRun(BaseModel):
    """Task identity and start time associate events; Markdown owns successful results."""

    model_config = ConfigDict(extra="forbid")

    task_id: str
    query: str
    started_at: AwareDatetime
    finished_at: AwareDatetime | None = None
    status: Literal["running", "succeeded", "failed", "interrupted", "skipped"] = "running"
    reason: str | None = None
    result_path: str | None = None
    error: str | None = None
    agent_session_id: str | None = None
    target_session_id: str | None = None
    message_id: str | None = None
    queried: bool = False


class PaTaskCompletion(BaseModel):
    """A durable submission intent, published before a run becomes complete."""

    model_config = ConfigDict(extra="forbid")

    summary: str = ""
    reminder: str = ""
    history_version: tuple[int, int, int] | None = None


class PaTaskState(BaseModel):
    """Native sessions own query context; checkpoints retain progress and publication state."""

    model_config = ConfigDict(extra="forbid")

    task: PaTask
    session_id: str | None = None
    run: PaTaskRun | None = None
    completion: PaTaskCompletion | None = None
    usage: TokenUsage = Field(default_factory=TokenUsage)

    @model_validator(mode="after")
    def validate_run(self):
        """A checkpoint always belongs to its task and submitted execution."""
        if self.run is not None and (self.run.task_id != self.task.id or self.run.query != self.task.query):
            raise ValueError("Task recovery identity does not match its definition")
        if self.completion is not None and self.run is None:
            raise ValueError("Task completion requires an execution")
        return self
