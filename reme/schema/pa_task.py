"""File-native scheduled tasks and append-only execution events."""

from typing import Literal
from uuid import uuid4

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, StrictBool, field_validator, model_validator

from .connector_result import ConnectorResult
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
    """Start and finish events share a run_id; Markdown owns successful results."""

    model_config = ConfigDict(extra="forbid")

    run_id: str = Field(default_factory=lambda: uuid4().hex)
    task_id: str
    query: str
    started_at: AwareDatetime
    finished_at: AwareDatetime | None = None
    status: Literal["running", "succeeded", "failed", "interrupted", "skipped"] = "running"
    reason: str | None = None
    result_path: str | None = None
    error: str | None = None
    agent_session_id: str | None = None
    phase: Literal["query", "summary", "message", "handoff"] = "query"
    target_session_id: str | None = None
    handoff_path: str | None = None
    message_id: str | None = None


class PaTaskRecovery(BaseModel):
    """Durable progress for resuming one task without repeating a publication."""

    model_config = ConfigDict(extra="forbid")

    task: PaTask
    run: PaTaskRun
    target_path: str
    usage: TokenUsage = Field(default_factory=TokenUsage)

    @model_validator(mode="after")
    def validate_run(self):
        """Only a completed query can be resumed into summary or handoff."""
        if self.run.task_id != self.task.id or self.run.query != self.task.query:
            raise ValueError("Task recovery identity does not match its definition")
        if self.run.phase == "query" or not self.run.agent_session_id:
            raise ValueError("Task recovery requires a completed query and native session")
        return self

    @field_validator("target_path")
    @classmethod
    def validate_target(cls, value):
        """Keep stored output references workspace-relative."""
        ConnectorResult(context="", paths=[value])
        return value
