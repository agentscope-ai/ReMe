"""Shared agent identities, native sessions, notifications and inbox contracts."""

from typing import Annotated, Literal
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

from .event import Event


def validate_agent_id(value: str) -> str:
    """Keep application identities portable as workspace directory names."""
    reserved = {"CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(1, 10)), *(f"LPT{i}" for i in range(1, 10))}
    valid_length = isinstance(value, str) and 1 <= len(value) <= 80
    if (
        not valid_length
        or len(value.encode("utf-8")) > 192
        or not re.fullmatch(r"[\w-]+", value)
        or value.upper() in reserved
    ):
        raise ValueError("agent_id must contain only letters, numbers, underscores or hyphens")
    return value


AgentId = Annotated[str, Field(min_length=1, max_length=80, pattern=r"^[\w-]+$"), AfterValidator(validate_agent_id)]
InboxMessageId = Annotated[str, Field(pattern=r"^[a-zA-Z0-9_-]{1,128}$")]
InboxPriority = Literal["P0", "P1", "P2"]


class InboxConfig(BaseModel):
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


class CreateChatAgentConfig(BaseModel):
    """Initial personal memory text, separate from the retained Chat identity."""

    model_config = ConfigDict(extra="forbid")

    description: str = Field(default="", max_length=100000)


class ChatAgentConfig(BaseModel):
    """The fixed foreground identity has no task trigger or temporary lifetime."""

    model_config = ConfigDict(extra="forbid")

    name: Literal["chat"] = "chat"
    type: Literal["chat"] = "chat"
    temporary: Literal[False] = False
    created_at: AwareDatetime


class AgentNotification(BaseModel):
    """Notification content shared by agent output and durable inbox messages."""

    model_config = ConfigDict(extra="forbid")

    priority: InboxPriority
    message: str = Field(min_length=1, max_length=100000)
    paths: list[str] = Field(min_length=1, description="Reminder Markdown first, followed by related workspace files.")

    @field_validator("message")
    @classmethod
    def nonblank_message(cls, value: str) -> str:
        """An intentional empty decision is an empty array, not whitespace content."""
        if not value.strip():
            raise ValueError("reminder message must not be blank")
        return value

    @field_validator("paths")
    @classmethod
    def relative_paths(cls, value: list[str]) -> list[str]:
        """Apply the same workspace-relative contract as event inputs."""
        return Event.validate_paths(value)


class ReminderRef(BaseModel):
    """Locate one message without copying its contents or delivery state."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    file: str
    id: InboxMessageId

    @field_validator("file")
    @classmethod
    def relative_file(cls, value: str) -> str:
        """References always stay workspace-relative."""
        return Event.validate_paths([value])[0]


class AgentReminder(AgentNotification):
    """The single durable message with independent Agent and user acknowledgements."""

    id: InboxMessageId
    created_at: AwareDatetime
    delivered_at: AwareDatetime | None = None
    read_at: AwareDatetime | None = None
    user_read_at: AwareDatetime | None = None


class InboxWakeRequest(BaseModel):
    """A framework-selected batch eligible for automatic processing."""

    model_config = ConfigDict(extra="forbid")

    priority: InboxPriority
    msg_ids: list[InboxMessageId] = Field(min_length=1, max_length=100)


class ReadInboxRequest(BaseModel):
    """Read the active fork's batch at its assigned priority."""

    model_config = ConfigDict(extra="forbid")

    priority: InboxPriority


class SendChatMessageRequest(BaseModel):
    """Submit user-facing text, or an empty decision to stay quiet."""

    model_config = ConfigDict(extra="forbid")

    content: str = Field(max_length=100000)


class AgentMemoryRequest(BaseModel):
    """A write must refer to the exact complete memory previously read."""

    model_config = ConfigDict(extra="forbid")

    action: Literal["read", "write"]
    agent_id: AgentId | None = None
    content: str | None = Field(default=None, max_length=100000)
    expected_revision: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def arguments(self):
        """Reject ambiguous read and unconditional replacement requests."""
        if self.action == "write" and (self.content is None or self.expected_revision is None):
            raise ValueError("write requires content and expected_revision from a complete read")
        if self.action == "read" and self.model_fields_set - {"action", "agent_id"}:
            raise ValueError("read accepts only action and agent_id")
        return self
