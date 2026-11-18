"""Proactive task definitions and structured model results."""

from datetime import datetime
from typing import Literal

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, StrictBool, field_validator, model_validator

from .agent_runtime import AgentName, AgentNotification


class NotificationWindow(BaseModel):
    """An application-local notification window; equal endpoints mean all day."""

    model_config = ConfigDict(extra="forbid")

    start: str = Field(pattern=r"^(?:[01][0-9]|2[0-3]):[0-5][0-9]$")
    end: str = Field(pattern=r"^(?:[01][0-9]|2[0-3]):[0-5][0-9]$")

    def allows(self, now: datetime) -> bool:
        """Include the opening minute and exclude the closing minute."""
        current = now.strftime("%H:%M")
        if self.start == self.end:
            return True
        if self.start < self.end:
            return self.start <= current < self.end
        return current >= self.start or current < self.end


class ProactiveAgentConfig(BaseModel):
    """User-owned execution settings. Task requirements live in TASK.md."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    agent_name: AgentName
    agent_type: Literal["proactive", "cron"]
    enabled: StrictBool = True
    temporary: StrictBool = False
    cron: str | None = None
    min_trigger_interval_seconds: float = Field(default=0, ge=0)
    max_batch_events: int = Field(default=20, ge=1, strict=True)
    notify_window: NotificationWindow | None = None

    @field_validator("agent_name")
    @classmethod
    def safe_name(cls, value: str) -> str:
        """Names also identify workspace files, including non-English names."""
        if value.casefold() in {"remy", "tmp"}:
            raise ValueError("Remy and tmp are reserved agent names")
        return value

    @field_validator("min_trigger_interval_seconds", mode="before")
    @classmethod
    def numeric_interval(cls, value):
        """Do not interpret booleans or numeric strings as delays."""
        if type(value) not in (int, float):
            raise ValueError("min_trigger_interval_seconds must be a number")
        return value

    @model_validator(mode="after")
    def validate_trigger(self):
        """Validate the final task after creation or a partial update."""
        if self.agent_type == "cron":
            from croniter import croniter

            if not self.cron or not croniter.is_valid(self.cron):
                raise ValueError("cron tasks require a valid cron expression")
            if self.min_trigger_interval_seconds:
                raise ValueError("cron tasks do not accept an event trigger interval")
        elif self.cron is not None:
            raise ValueError("event tasks do not accept cron")
        return self


class CreateProactiveAgentConfig(ProactiveAgentConfig):
    """Accept initial task text only when creating an Agent."""

    task_description: str = Field(min_length=1, max_length=100000)

    @field_validator("task_description")
    @classmethod
    def nonblank_task_description(cls, value: str) -> str:
        """Reject blank task descriptions and preserve the original Markdown."""
        if not value.strip():
            raise ValueError("task_description must not be blank")
        return value


class ProactiveAgent(ProactiveAgentConfig):
    """The creation name is the immutable application identity, including after restoration."""

    model_config = ConfigDict(extra="forbid")

    created_at: AwareDatetime
    deleted_at: AwareDatetime | None = None


class ProactiveRunResult(BaseModel):
    """One successful round can intentionally produce no notifications."""

    model_config = ConfigDict(extra="forbid")

    notifications: list[AgentNotification]


class ManageProactiveAgentRequest(BaseModel):
    """Named task operations expose only the fields relevant to each action."""

    model_config = ConfigDict(extra="forbid")

    action: Literal["list", "create", "status", "update", "delete", "query"]
    agent_name: AgentName | None = None
    task_description: str | None = Field(default=None, min_length=1, max_length=100000)
    agent_type: Literal["proactive", "cron"] | None = None
    enabled: StrictBool = True
    temporary: StrictBool = False
    cron: str | None = None
    min_trigger_interval_seconds: float = Field(default=0, ge=0, strict=True)
    notify_window: NotificationWindow | None = None
    question: str | None = Field(default=None, min_length=1, max_length=100000)

    @field_validator("task_description", "question")
    @classmethod
    def nonblank_text(cls, value):
        """Keep task text verbatim while rejecting whitespace-only requests."""
        if value is not None and not value.strip():
            raise ValueError("must not be blank")
        return value

    @model_validator(mode="after")
    def action_arguments(self):
        """Reject unused fields instead of silently ignoring misspecified calls."""
        settings = {"agent_type", "enabled", "cron", "min_trigger_interval_seconds", "notify_window"}
        required, allowed = {
            "list": (set(), set()),
            "create": (
                {"agent_name", "task_description", "agent_type"},
                settings | {"agent_name", "task_description", "temporary"},
            ),
            "status": ({"agent_name"}, {"agent_name"}),
            "update": ({"agent_name"}, settings | {"agent_name"}),
            "delete": ({"agent_name"}, {"agent_name"}),
            "query": ({"agent_name", "question"}, {"agent_name", "question"}),
        }[self.action]
        supplied = self.model_fields_set - {"action"}
        if supplied - allowed:
            raise ValueError(f"{self.action} does not accept: {', '.join(sorted(supplied - allowed))}")
        missing = {key for key in required if getattr(self, key) is None}
        if missing:
            raise ValueError(f"{self.action} requires: {', '.join(sorted(missing))}")
        if self.action == "update" and not supplied & settings:
            raise ValueError("update requires at least one setting")
        return self
