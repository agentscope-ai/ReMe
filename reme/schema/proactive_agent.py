"""Proactive task definitions and structured model results."""

from datetime import datetime
from typing import Any, Literal

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, StrictBool, field_validator, model_validator

from .agent_runtime import AgentId, AgentNotification


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
    """User-owned execution settings; task instructions live in MEMORY.md."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    name: AgentId
    trigger_type: Literal["event", "cron"]
    enabled: StrictBool = True
    temporary: StrictBool = False
    cron: str | None = None
    min_trigger_interval_seconds: float = Field(default=0, ge=0)
    notify_window: NotificationWindow | None = None

    @field_validator("name")
    @classmethod
    def safe_name(cls, value: str) -> str:
        """Names also identify workspace files, including non-English names."""
        if value.casefold() in {"main", "tmp"}:
            raise ValueError("main and tmp are reserved agent names")
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
        if self.trigger_type == "cron":
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

    description: str = Field(min_length=1, max_length=100000)

    @field_validator("description")
    @classmethod
    def nonempty_description(cls, value: str) -> str:
        """Reject whitespace-only tasks without rewriting user instructions."""
        if not value.strip():
            raise ValueError("description must not be blank")
        return value


class ProactiveAgent(ProactiveAgentConfig):
    """The creation name is the immutable application identity, including after restoration."""

    model_config = ConfigDict(extra="forbid")

    created_at: AwareDatetime
    deleted_at: AwareDatetime | None = None

    @property
    def agent_id(self) -> str:
        """Avoid storing two identities that can disagree."""
        return self.name


class ProactiveRunResult(BaseModel):
    """One successful round can intentionally produce no notifications."""

    model_config = ConfigDict(extra="forbid")

    notifications: list[AgentNotification]


class ManageProactiveAgentRequest(BaseModel):
    """The six management actions have explicit, disjoint argument sets."""

    model_config = ConfigDict(extra="forbid")

    action: Literal["list", "create", "status", "update", "delete", "query"]
    agent_id: AgentId | None = None
    config: dict[str, Any] | None = None
    query: str | None = Field(default=None, min_length=1, max_length=100000)

    @model_validator(mode="after")
    def action_arguments(self):
        """Reject unused fields instead of silently ignoring misspecified calls."""
        allowed = {
            "list": set(),
            "create": {"config"},
            "status": {"agent_id"},
            "update": {"agent_id", "config"},
            "delete": {"agent_id"},
            "query": {"agent_id", "query"},
        }[self.action]
        supplied = {key for key in ("agent_id", "config", "query") if getattr(self, key) is not None}
        if supplied - allowed:
            raise ValueError(f"{self.action} does not accept: {', '.join(sorted(supplied - allowed))}")
        if self.action not in {"list", "create"} and self.agent_id is None:
            raise ValueError(f"{self.action} requires agent_id")
        if self.action in {"create", "update"} and self.config is None:
            raise ValueError(f"{self.action} requires config")
        if self.action == "query" and (self.query is None or not self.query.strip()):
            raise ValueError("query requires a non-empty question")
        return self
