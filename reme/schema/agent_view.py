"""Read-only Agent workspaces and current conversation subscription contracts."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, StrictBool

from .agent_runtime import AgentName, AgentType
from .agent_session import SessionSnapshot


class AgentView(BaseModel):
    """A visible identity; paths refer to original workspace files, never copies."""

    model_config = ConfigDict(extra="forbid")

    agent_name: AgentName
    agent_type: AgentType
    enabled: bool = True
    running: bool = False
    status: Literal["active", "idle", "not_loaded", "error"] = "idle"
    session_id: str | None = None
    memory_path: str | None = Field(default=None, exclude_if=lambda value: value is None)
    task_path: str | None = Field(default=None, exclude_if=lambda value: value is None)
    directory: str
    pending_events: int = 0
    last_started_at: str | None = None
    last_completed_at: str | None = None
    next_trigger_at: str | None = None
    error: str | None = None


class AgentRequest(BaseModel):
    """Resolve a conversation by application identity rather than a native thread ID."""

    model_config = ConfigDict(extra="forbid")

    agent_name: AgentName = "Remy"


class AgentChatRequest(AgentRequest):
    """A user message continues Chat or queries a fork of the latest task session."""

    query: str = Field(min_length=1, max_length=100000)


class AgentSessionView(BaseModel):
    """Chat retains full history; event/cron views contain only the latest native turn.

    The snapshot is absent before the Agent has a native session. This view never trims persisted history.
    """

    agent: AgentView
    session_id: str | None = None
    snapshot: SessionSnapshot | None = None


class AgentPageRequest(AgentRequest):
    """Bound resource queries independently of the current conversation."""

    offset: int = Field(default=0, ge=0, strict=True)
    limit: int = Field(default=20, ge=1, le=100, strict=True)


class AgentInboxRequest(AgentPageRequest):
    """Observe inputs without claiming or acknowledging them."""

    state: Literal["all", "pending", "processing", "processed"] = "all"


class AgentOutboxRequest(AgentPageRequest):
    """Page reminders produced by one task Agent."""

    agent_name: AgentName
    state: Literal["all", "pending", "processed"] = "all"


class AgentReminderRequest(AgentRequest):
    """Read a scoped reminder and optionally acknowledge explicit user reading."""

    message_id: str = Field(min_length=1, max_length=128)
    mark_read: StrictBool = False
