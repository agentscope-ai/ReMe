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
from .agent_skill import SkillName
from .event import Event
from .proactive_runtime import (
    AgentName,
    AgentReminder,
    AgentRuntimeBinding,
    InboxPriority,
    validate_agent_name,
    validate_binding_names,
)

TIME_EVENT_SOURCE = "time"


def validate_runtime_names(names, *, allow_empty=False):
    """Validate recipient identities and preserve their configured order."""
    if not isinstance(names, list) or not names and not allow_empty:
        raise ValueError("runtime_names must be a list" if allow_empty else "runtime_names must be a nonempty list")
    targets = [validate_agent_name(name) for name in names]
    if len({name.casefold() for name in targets}) != len(targets):
        raise ValueError("runtime_names must be unique")
    return targets


class ManageConnectorRequest(BaseModel):
    """Deliver explicit observations or control one configured collection Job."""

    model_config = ConfigDict(extra="forbid")
    action: Literal["list", "update", "send", "enable", "pause"]
    runtime_names: list[AgentName] | None = None
    events: list[Event] | None = Field(default=None, min_length=1)
    connector_job: AgentName | None = None

    @model_validator(mode="after")
    def validate_action(self):
        """Reject fields that do not belong to the selected operation."""
        supplied = self.model_fields_set - {"action"}
        if self.action == "list":
            if supplied:
                raise ValueError("list accepts only action")
        elif self.action == "update":
            if supplied != {"connector_job", "runtime_names"} or not self.connector_job:
                raise ValueError("update requires only connector_job and runtime_names")
            validate_runtime_names(self.runtime_names, allow_empty=True)
        elif self.action == "send":
            if supplied != {"runtime_names", "events"} or not self.events:
                raise ValueError("send requires only runtime_names and events")
            validate_runtime_names(self.runtime_names)
            if any(event.source == TIME_EVENT_SOURCE for event in self.events):
                raise ValueError("time is a reserved framework source")
        elif supplied != {"connector_job"} or not self.connector_job:
            raise ValueError("enable/pause require only connector_job")
        return self


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
    schedule: str | None = None
    min_trigger_interval_seconds: float = Field(default=0, ge=0)
    notify_window: NotificationWindow | None = None

    @field_validator("min_trigger_interval_seconds", mode="before")
    @classmethod
    def numeric_interval(cls, value):
        """Do not interpret booleans or numeric strings as delays."""
        if type(value) not in (int, float):
            raise ValueError("min_trigger_interval_seconds must be a number")
        return value

    @field_validator("schedule")
    @classmethod
    def validate_schedule(cls, expression):
        """Validate the optional schedule independently of observation processing."""
        if expression is not None:
            from croniter import croniter

            if not expression or not croniter.is_valid(expression):
                raise ValueError("schedule must be a valid expression")
        return expression


class ProactiveRuntimeDefaults(AgentRuntimeBinding):
    """Defaults shared by preset and dynamically created background Runtimes."""

    skills: list[SkillName] = Field(default_factory=lambda: ["proactive-record"])
    max_batch_events: int = Field(default=20, ge=1, strict=True)
    max_local_search_calls: int = Field(default=10, ge=1, strict=True)
    recent_history_hours: int = Field(default=24, ge=1, strict=True)


class ProactiveRuntimeConfig(ProactiveRuntimeDefaults, ProactiveRuntimeSettings):
    """A named task with internal batching limits; requirements live in TASK.md."""

    agent_name: AgentName
    agent_type: Literal["event", "time"]

    def wiring(self):
        """Separate immutable identity and session bindings from editable settings."""
        return self.model_dump(
            include=ProactiveRuntimeConfig.model_fields.keys() - ProactiveRuntimeSettings.model_fields.keys(),
        )

    @field_validator("agent_name")
    @classmethod
    def safe_name(cls, value: str) -> str:
        """Task names identify workspace files; Remy owns the main conversation."""
        if value.casefold() == CHAT_AGENT_NAME.casefold():
            raise ValueError(f"{CHAT_AGENT_NAME} is a reserved agent name")
        return value

    @model_validator(mode="after")
    def validate_trigger(self):
        """The explicit type owns scheduling and event admission policy."""
        if self.agent_type == "time" and self.schedule is None:
            raise ValueError("Time-triggered tasks require a schedule")
        if self.agent_type == "event" and self.schedule is not None:
            raise ValueError("Event-triggered tasks cannot have a schedule")
        return self


class CreateProactiveRuntimeConfig(ProactiveRuntimeConfig):
    """Accept initial task text only when creating an Agent."""

    task_description: str | None = Field(default=None, min_length=1, max_length=100000)

    @field_validator("task_description")
    @classmethod
    def nonblank_task_description(cls, value: str) -> str:
        """Reject blank task descriptions and preserve the original Markdown."""
        if value is not None and not value.strip():
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


class ManageProactiveRuntimeRequest(ProactiveRuntimeSettings):
    """Named task operations expose only the fields relevant to each action."""

    model_config = ConfigDict(extra="forbid")

    action: Literal["list", "create", "status", "update", "delete", "query"]
    agent_name: AgentName | None = None
    agent_type: Literal["event", "time"] | None = None
    skills: list[SkillName] | None = None
    job_tools: list[str] | None = None
    task_description: str | None = Field(default=None, min_length=1, max_length=100000)
    question: str | None = Field(default=None, min_length=1, max_length=100000)

    _valid_names = field_validator("skills", "job_tools", mode="before")(validate_binding_names)

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
        settings = {"enabled", "schedule", "min_trigger_interval_seconds", "notify_window"}
        required, allowed = {
            "list": (set(), set()),
            "create": (
                {"agent_name", "agent_type"},
                settings | {"agent_name", "agent_type", "task_description", "skills", "job_tools"},
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
