"""Read-only Agent workspaces and current conversation subscription contracts."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, StrictBool, model_validator

from ..constants import CHAT_AGENT_NAME
from .agent_runtime import AgentName, AgentType
from .agent_activity import ActivityPhase
from .agent_message import AgentMessage, ClientModel
from .proactive_agent import TaskSettings


class AgentView(TaskSettings):
    """A visible identity; paths refer to original workspace files, never copies."""

    model_config = ConfigDict(extra="forbid")

    agent_name: AgentName
    agent_type: AgentType
    status: Literal["active", "idle", "not_loaded", "error"] = "idle"
    session_id: str | None = None
    memory_path: str | None = None
    task_path: str | None = None
    directory: str
    pending_events: int = 0
    in_flight_events: int = 0
    last_started_at: str | None = None
    last_completed_at: str | None = None
    next_trigger_at: str | None = None
    error: str | None = None


class AgentRequest(BaseModel):
    """Resolve a conversation by application identity rather than a native thread ID."""

    model_config = ConfigDict(extra="forbid")

    agent_name: AgentName = CHAT_AGENT_NAME


class AgentChatRequest(AgentRequest):
    """A user message continues Chat or queries a fork of the latest task session."""

    query: str = Field(min_length=1, max_length=100000)


class AgentTaskRequest(BaseModel):
    """Read or save a task's original TASK.md without accepting arbitrary paths."""

    model_config = ConfigDict(extra="forbid")

    action: Literal["read", "save"]
    agent_name: AgentName
    content: str | None = Field(default=None, min_length=1, max_length=100000)
    expected_mtime: str | None = Field(default=None, min_length=1, max_length=80)

    @model_validator(mode="after")
    def validate_action(self):
        """Require an editor revision for every write and reject irrelevant read inputs."""
        if self.action == "save":
            if self.content is None or not self.content.strip() or not self.expected_mtime:
                raise ValueError("save requires nonblank content and expected_mtime")
        elif self.model_fields_set & {"content", "expected_mtime"}:
            raise ValueError("read does not accept content or expected_mtime")
        return self


class ClientActivity(ClientModel):
    """An activity is identified by its Agent map key, not by redundant task IDs."""

    session_id: str
    turn_id: str
    phase: ActivityPhase
    tool: str | None = None
    started_at: str | None = None


class AgentSessionView(ClientModel):
    """An authoritative Agent state and its ordered display messages."""

    agent: AgentView
    messages: list[AgentMessage] = Field(default_factory=list)
    activities: dict[str, ClientActivity] = Field(default_factory=dict)


class AgentSnapshot(AgentSessionView):
    """Replace state atomically on connection, session switch or history replacement."""

    type: Literal["snapshot"] = "snapshot"
    agents: list[AgentView] | None = None


class AgentUpdate(AgentSessionView):
    """Replace state and upsert complete messages without replaying token deltas."""

    type: Literal["update"] = "update"
    agents: list[AgentView] | None = None


class AgentHeartbeat(ClientModel):
    """Transport liveness carries no application state."""

    type: Literal["heartbeat"] = "heartbeat"


class ClientInfo(ClientModel):
    """Client capabilities independent of component wiring and model credentials."""

    workspace_dir: str
    timezone: str | None
    resource_dir: str
    daily_dir: str
    digest_dir: str
    tag_key: str
    jobs: list[str]
    asr: bool
    tts: bool
    speech_revision: str


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
