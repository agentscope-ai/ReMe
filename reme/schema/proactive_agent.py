"""Proactive task definitions and structured model results."""

from datetime import datetime
from typing import Literal

from pydantic import (
    AwareDatetime,
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    ValidationInfo,
    field_validator,
    model_validator,
)

from ..constants import CHAT_AGENT_NAME
from .agent_message import ClientModel
from .event import Event
from .proactive_runtime import AgentName, AgentReminder, InboxPriority, validate_agent_name

CRON_EVENT_SOURCE = "cron"


def validate_task_names(names):
    """Validate recipient identities and preserve their configured order."""
    if not isinstance(names, list) or not names:
        raise ValueError("task_names must be a nonempty list")
    targets = [validate_agent_name(name) for name in names]
    if len({name.casefold() for name in targets}) != len(targets):
        raise ValueError("task_names must be unique")
    return targets


class AppendTaskEventsRequest(BaseModel):
    """Append a nonempty batch of observations to existing background tasks."""

    model_config = ConfigDict(extra="forbid")

    task_names: list[AgentName] = Field(min_length=1)
    events: list[Event] = Field(min_length=1)

    @field_validator("task_names")
    @classmethod
    def validate_targets(cls, names):
        """Require one or more distinct recipients."""
        return validate_task_names(names)

    @field_validator("events")
    @classmethod
    def validate_sources(cls, events):
        """Keep scheduled triggers private to the runtime."""
        if any(event.source == CRON_EVENT_SOURCE for event in events):
            raise ValueError("cron is a reserved framework source")
        return events


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


class ProactiveRuntimeSettings(ClientModel):
    """One set of editable settings shared by task definitions and client views."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    enabled: StrictBool = True
    cron: str | None = None
    min_trigger_interval_seconds: float = Field(default=0, ge=0)
    notify_window: NotificationWindow | None = None

    @field_validator("min_trigger_interval_seconds", mode="before")
    @classmethod
    def numeric_interval(cls, value):
        """Do not interpret booleans or numeric strings as delays."""
        if type(value) not in (int, float):
            raise ValueError("min_trigger_interval_seconds must be a number")
        return value

    @field_validator("cron")
    @classmethod
    def validate_cron(cls, expression):
        """Validate the optional schedule independently of observation processing."""
        if expression is not None:
            from croniter import croniter

            if not expression or not croniter.is_valid(expression):
                raise ValueError("cron must be a valid expression")
        return expression


class ProactiveRuntimeConfig(ProactiveRuntimeSettings):
    """A named task with internal batching limits; requirements live in TASK.md."""

    agent_name: AgentName
    max_batch_events: int = Field(default=20, ge=1, strict=True)
    step_backend: Literal["proactive_agent_step", "proactive_notebook_step"] = "proactive_agent_step"

    @field_validator("agent_name")
    @classmethod
    def safe_name(cls, value: str) -> str:
        """Task names identify workspace files; Remy owns the main conversation."""
        if value.casefold() == CHAT_AGENT_NAME.casefold():
            raise ValueError(f"{CHAT_AGENT_NAME} is a reserved agent name")
        return value

    @model_validator(mode="after")
    def validate_notebook_trigger(self):
        """Run the Notebook step only on observed events."""
        if self.step_backend == "proactive_notebook_step" and self.cron is not None:
            raise ValueError("Notebook runs only on events and cannot have a cron schedule")
        return self


class CreateProactiveRuntimeConfig(ProactiveRuntimeConfig):
    """Accept initial task text only when creating an Agent."""

    task_description: str = Field(min_length=1, max_length=100000)

    @field_validator("task_description")
    @classmethod
    def nonblank_task_description(cls, value: str) -> str:
        """Reject blank task descriptions and preserve the original Markdown."""
        if not value.strip():
            raise ValueError("task_description must not be blank")
        return value


class ProactiveRuntimeDefinition(ProactiveRuntimeConfig):
    """The creation name is the immutable application identity, including after restoration."""

    model_config = ConfigDict(extra="forbid")

    created_at: AwareDatetime
    deleted_at: AwareDatetime | None = None


class ProactiveRecordDecision(BaseModel):
    """The inbox delivery decision in a daily record's frontmatter."""

    model_config = ConfigDict(extra="forbid")

    reminder_priority: Literal[InboxPriority, "skip"]
    reminder_message: str = Field(max_length=100000)

    @classmethod
    def parse(cls, value):
        """Read only the fields used by the round, including after structured generation."""
        return cls.model_validate(value, extra="ignore")

    @field_validator("reminder_message", mode="before")
    @classmethod
    def validate_reminder_text(cls, value, info: ValidationInfo):
        """Accept an empty YAML value for a skipped reminder."""
        if info.data.get("reminder_priority") == "skip":
            return "" if value is None else value
        if isinstance(value, str) and not value.strip():
            raise ValueError("reminder_message must not be blank")
        return value

    def reminder(self, path: str) -> AgentReminder | None:
        """Build the reminder identified by its source daily record."""
        if self.reminder_priority == "skip":
            return None
        return AgentReminder(priority=self.reminder_priority, message=self.reminder_message, id=path)


class ManageProactiveAgentRequest(ProactiveRuntimeSettings):
    """Named task operations expose only the fields relevant to each action."""

    model_config = ConfigDict(extra="forbid")

    action: Literal["list", "create", "status", "update", "delete", "query"]
    agent_name: AgentName | None = None
    task_description: str | None = Field(default=None, min_length=1, max_length=100000)
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
        settings = {"enabled", "cron", "min_trigger_interval_seconds", "notify_window"}
        required, allowed = {
            "list": (set(), set()),
            "create": (
                {"agent_name", "task_description"},
                settings | {"agent_name", "task_description"},
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
