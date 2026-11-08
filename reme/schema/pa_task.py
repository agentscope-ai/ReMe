"""File-native scheduled tasks and append-only execution events."""

from typing import Literal
from uuid import uuid4

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, StrictBool, field_validator, model_validator

from .connector_result import ConnectorResult
from .screen_capture import CapturedDisplay
from .token_usage import TokenUsage


class PaTask(BaseModel):
    """A stable task identity with a five-field cron schedule."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(pattern=r"^[A-Za-z0-9_-]{1,128}$")
    cron: str
    query: str | None = Field(default=None, min_length=1)
    message: str | None = Field(default=None, min_length=1, max_length=4000)
    delivery: Literal["inbox", "immediate"] = "inbox"
    source_session_id: str | None = Field(default=None, min_length=1)
    enabled: StrictBool = True
    preflight: Literal["screen_capture"] | None = None
    created_at: AwareDatetime | None = None

    @field_validator("cron")
    @classmethod
    def validate_cron(cls, value: str) -> str:
        """Keep schedule syntax consistent with minute-based task definitions."""
        from croniter import croniter

        if len(value.split()) != 5 or not croniter.is_valid(value):
            raise ValueError("cron must be a valid five-field expression")
        return value

    @field_validator("query", "message")
    @classmethod
    def validate_query(cls, value: str) -> str:
        """Reject whitespace-only tasks."""
        if value is None:
            return None
        if not value.strip():
            raise ValueError("query must not be blank")
        return value.strip()

    @model_validator(mode="after")
    def validate_delivery(self):
        """Bind punctual messages to their source conversation and reject ambiguous work."""
        if (self.query is None) == (self.message is None):
            raise ValueError("Specify exactly one of query or fixed message")
        if self.delivery == "inbox" and self.message is not None:
            raise ValueError("Fixed messages require immediate delivery")
        if self.delivery == "immediate" and not self.source_session_id:
            raise ValueError("Immediate tasks require their source Main session")
        if self.delivery == "immediate" and self.preflight is not None:
            raise ValueError("Immediate tasks cannot skip execution through preflight")
        return self


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
    phase: Literal["query", "summary", "handoff"] = "query"
    handoff_path: str | None = None
    message_id: str | None = None


class PaTaskRecovery(BaseModel):
    """Durable progress for resuming one task without repeating a publication."""

    model_config = ConfigDict(extra="forbid")

    task: PaTask
    run: PaTaskRun
    target_path: str
    prefetched: dict | None = None
    query_result: str = ""
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

    @field_validator("prefetched")
    @classmethod
    def validate_preflight(cls, value):
        """Restore the exact validated tool binding used by the native session."""
        if value is None:
            return None
        if set(value) != {"result", "frames"} or not isinstance(value["frames"], list) or len(value["frames"]) > 32:
            raise ValueError("Invalid saved screenshot preflight")
        return {
            "result": ConnectorResult.model_validate(value["result"]).model_dump(mode="json"),
            "frames": [CapturedDisplay.model_validate(frame).model_dump(mode="json") for frame in value["frames"]],
        }
