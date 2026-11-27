"""Shared agent identities, native sessions, notifications and inbox contracts."""

from typing import Annotated, Any, Literal
import re

from pydantic import (
    AfterValidator,
    AwareDatetime,
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

from ..constants import WINDOWS_RESERVED_DEVICE_NAMES
from .agent_skill import SkillName
from .event import Event


def validate_agent_name(value: str) -> str:
    """Keep application identities portable as workspace directory names."""
    valid_length = isinstance(value, str) and 1 <= len(value) <= 80
    if (
        not valid_length
        or len(value.encode("utf-8")) > 192
        or not re.fullmatch(r"[\w-]+", value)
        or value.upper() in WINDOWS_RESERVED_DEVICE_NAMES
    ):
        raise ValueError("agent_name must contain only letters, numbers, underscores or hyphens")
    return value


AgentName = Annotated[str, Field(min_length=1, max_length=80, pattern=r"^[\w-]+$"), AfterValidator(validate_agent_name)]
AgentType = Literal["chat", "event", "time"]
ChatMessageId = Annotated[str, Field(pattern=r"^[a-zA-Z0-9_-]{1,128}$")]
InboxPriority = Literal["P0", "P1", "P2"]


def validate_binding_names(values):
    """Keep skill and Job references distinct and free of filesystem paths."""
    if not isinstance(values, (list, tuple)) or any(
        not isinstance(name, str) or re.fullmatch(r"[a-zA-Z0-9_][a-zA-Z0-9_-]{0,79}", name) is None for name in values
    ):
        raise ValueError("skills and job_tools must contain valid names")
    if len(set(values)) != len(values):
        raise ValueError("skills and job_tools must be distinct")
    return values


class AgentRuntimeBinding(BaseModel):
    """Trusted component configuration for one Runtime's model sessions."""

    model_config = ConfigDict(extra="forbid")

    skills: list[SkillName] = Field(default_factory=list)
    job_tools: list[str] = Field(default_factory=list)
    injected_job_kwargs: dict[str, Any] = Field(default_factory=dict)
    effort: str | None = None
    sandbox: Literal["read-only", "workspace-write", "danger-full-access"] | None = "workspace-write"
    approval_mode: str = "deny_all"
    prompt_dict: dict[str, str] = Field(default_factory=dict)

    _valid_names = field_validator("skills", "job_tools", mode="before")(validate_binding_names)


class NotificationConfig(BaseModel):
    """Receiver-owned wakeup delays in seconds."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    p0_delay_seconds: float = Field(default=0, ge=0)
    p1_idle_seconds: float = Field(default=10, ge=0)
    p2_idle_seconds: float = Field(default=1800, ge=0)

    @field_validator("p0_delay_seconds", "p1_idle_seconds", "p2_idle_seconds", mode="before")
    @classmethod
    def numeric_seconds(cls, value):
        """Reject booleans and strings rather than treating them as durations."""
        if type(value) not in (int, float):
            raise ValueError("inbox delays must be numbers")
        return value


class AgentSession(BaseModel):
    """One durable native session reference; temporary branches never select the current session."""

    model_config = ConfigDict(extra="forbid")

    session_id: str = Field(min_length=1, max_length=128)
    created_at: AwareDatetime
    temporary: bool = False
    signature: str | None = None


class ChatAgentConfig(BaseModel):
    """Startup settings for Remy's personal memory."""

    model_config = ConfigDict(extra="forbid")

    initial_memory: str = Field(default="", max_length=100000)


class AgentReminder(BaseModel):
    """A daily record reminder and its delivery and Chat processing state."""

    model_config = ConfigDict(extra="forbid")

    priority: InboxPriority
    message: str = Field(min_length=1, max_length=100000)
    id: str = Field(description="Workspace-relative path to the source daily record.")
    delivered_at: AwareDatetime | None = None
    read_at: AwareDatetime | None = None
    chat_message_ids: list[ChatMessageId] = Field(default_factory=list)

    @field_validator("message")
    @classmethod
    def nonblank_message(cls, value: str) -> str:
        """Require user-facing reminder text."""
        if not value.strip():
            raise ValueError("reminder message must not be blank")
        return value

    @field_validator("id")
    @classmethod
    def relative_path(cls, value: str) -> str:
        """Apply the same workspace-relative contract as event inputs."""
        Event.validate_paths([value])
        return value


class ProcessChatNotificationsRequest(BaseModel):
    """A framework-selected batch eligible for automatic processing."""

    model_config = ConfigDict(extra="forbid")

    priority: InboxPriority
    msg_ids: list[str] = Field(min_length=1, max_length=100)

    @field_validator("msg_ids")
    @classmethod
    def relative_paths(cls, value: list[str]) -> list[str]:
        """Batch identifiers are internal references, not user-selected source paths."""
        if len(set(value)) != len(value) or any(not item or len(item) > 128 for item in value):
            raise ValueError("msg_ids must contain distinct valid notification ids")
        return value


class SubmitChatReplyRequest(BaseModel):
    """Explicitly send text or commit a silent decision for the active fork."""

    model_config = ConfigDict(extra="forbid")

    action: Literal["send", "skip"]
    content: str | None = Field(default=None, max_length=100000)

    @model_validator(mode="after")
    def validate_action(self):
        """Reject fields that do not belong to the selected operation."""
        if self.action == "send":
            if self.content is None or not self.content.strip():
                raise ValueError("send requires nonblank content")
            self.content = self.content.strip()
        elif "content" in self.model_fields_set:
            raise ValueError("skip does not accept content")
        return self
