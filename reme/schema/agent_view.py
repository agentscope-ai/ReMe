"""Read-only Agent workspaces and current conversation subscription contracts."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, StrictBool, model_validator

from .agent_runtime import AgentId
from .agent_session import SessionSnapshot


class AgentView(BaseModel):
    """A visible identity; paths refer to original workspace files, never copies."""

    model_config = ConfigDict(extra="forbid")

    agent_id: AgentId
    name: str
    kind: Literal["chat", "cron", "event"]
    enabled: bool = True
    running: bool = False
    status: Literal["active", "idle", "not_loaded", "error"] = "idle"
    session_id: str | None = None
    memory_path: str
    directory: str
    pending_events: int = 0
    last_started_at: str | None = None
    last_completed_at: str | None = None
    next_trigger_at: str | None = None
    error: str | None = None


class AgentRequest(BaseModel):
    """Resolve a conversation by application identity rather than a native thread ID."""

    model_config = ConfigDict(extra="forbid")

    agent_id: AgentId = "chat"


class AgentChatRequest(AgentRequest):
    """A user message continues the selected Agent's current formal session."""

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


class AgentEventListRequest(AgentPageRequest):
    """Observe inputs without claiming or acknowledging them."""

    state: Literal["all", "pending", "processing", "processed"] = "all"


class AgentMailboxRequest(AgentPageRequest):
    """Chat sees received mail; task Agents see their own committed output."""

    action: Literal["list", "read"] = "list"
    state: Literal["all", "pending", "processed"] = "all"
    message_id: str | None = Field(default=None, min_length=1, max_length=128)
    mark_read: StrictBool = False

    @model_validator(mode="after")
    def action_arguments(self):
        """User reading never acknowledges Agent processing."""
        if self.action == "read":
            if self.message_id is None:
                raise ValueError("read requires message_id")
            if self.model_fields_set & {"state", "offset", "limit"}:
                raise ValueError("read does not accept listing filters")
        elif self.model_fields_set & {"message_id", "mark_read"}:
            raise ValueError("list does not accept read arguments")
        return self
