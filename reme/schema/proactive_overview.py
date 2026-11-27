"""Read-only, versioned proactive overview subscription contract."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from .agent_inbox import NotificationRef
from .agent_message import ClientModel
from .agent_outbox import NotificationDelivery
from .agent_view import AgentView
from .event import Event
from .proactive_agent import validate_runtime_names
from .proactive_runtime import AgentName
from .proactive_trace import SourceRef
from .proactive_connector import ConnectorView
from ..constants import CHAT_AGENT_NAME


class WatchProactiveOverviewRequest(BaseModel):
    """Omitting the filter includes every background Runtime and shared Remy."""

    model_config = ConfigDict(extra="forbid")
    agent_names: list[AgentName] | None = Field(default=None, min_length=1)

    @field_validator("agent_names")
    @classmethod
    def validate_names(cls, names):
        """Reject duplicate identities and Chat in the background-only filter."""
        if names is not None:
            names = validate_runtime_names(names)
            if any(name.casefold() == CHAT_AGENT_NAME.casefold() for name in names):
                raise ValueError("agent_names selects background Runtimes; Remy is included automatically")
        return names


class OverviewRun(ClientModel):
    """Return observed execution facts; leave unobserved details unset."""

    run_id: str
    event_ids: list[str] = Field(default_factory=list)
    source_refs: list[SourceRef] = Field(default_factory=list)
    phase: str
    tool: str | None = None
    started_at: str | None = None
    completed_at: str | None = None
    session_id: str | None = None
    turn_id: str | None = None
    record_paths: list[str] = Field(default_factory=list)
    notification_requested: bool | None = None
    error: str | None = None


class OverviewAgent(AgentView, ClientModel):
    """Reuse AgentView without changing its existing queue count semantics."""

    active_run: OverviewRun | None = None
    last_run: OverviewRun | None = None
    deliveries: list[NotificationDelivery] = Field(default_factory=list)
    pending_delivery_count: int = 0
    delivered_count: int = 0


class OverviewInboxItem(ClientModel):
    """Queue identity is the pair of agent_name and id."""

    id: str
    agent_name: str
    state: Literal["pending", "processing", "processed"]
    received_at: str | None = None
    consume_ts: str | None = None
    event: Event | None = None
    notification: NotificationRef | None = None
    source_refs: list[SourceRef] = Field(default_factory=list)
    event_ids: list[str] = Field(default_factory=list)
    run_ids: list[str] = Field(default_factory=list)
    record_paths: list[str] = Field(default_factory=list)
    delivery: NotificationDelivery | None = None
    chat_message_ids: list[str] = Field(default_factory=list)


class OverviewQueueCounts(ClientModel):
    """Waiting and processing counts come from the same locked queue view."""

    waiting_count: int = 0
    processing_count: int = 0
    processed_count: int = 0


class OverviewInbox(OverviewQueueCounts):
    """Aggregate counts with a bounded recent-admission window."""

    queues: dict[str, OverviewQueueCounts] = Field(default_factory=dict)
    items: list[OverviewInboxItem] = Field(default_factory=list)


class OverviewReminder(ClientModel):
    """A native user-facing message, never a prepared or aborted intention."""

    id: str
    session_id: str
    summary: str
    delivered_at: str
    status: Literal["sent", "acknowledged"]
    notification_ids: list[str]
    source_refs: list[SourceRef] = Field(default_factory=list)
    event_ids: list[str] = Field(default_factory=list)
    run_ids: list[str] = Field(default_factory=list)
    record_paths: list[str] = Field(default_factory=list)


class OverviewFile(ClientModel):
    """Original workspace references; file content uses existing file jobs."""

    id: str
    agent_name: str | None = None
    role: Literal["resource", "task", "record", "memory"]
    path: str
    mtime: str | None = None
    run_id: str | None = None


class ProactiveOverview(ClientModel):
    """All first-screen facts, plus independently unavailable resources."""

    sources: list[ConnectorView] = Field(default_factory=list)
    agents: list[OverviewAgent] = Field(default_factory=list)
    event_inbox: OverviewInbox = Field(default_factory=OverviewInbox)
    remy_inbox: OverviewInbox = Field(default_factory=OverviewInbox)
    reminders: list[OverviewReminder] = Field(default_factory=list)
    workspace: list[OverviewFile] = Field(default_factory=list)
    errors: dict[str, str] = Field(default_factory=dict)


class OverviewTransition(ClientModel):
    """Stable action identities support at-most-once animation at the client."""

    id: str
    type: Literal[
        "event_admitted",
        "input_claimed",
        "input_released",
        "input_processed",
        "record_committed",
        "notification_prepared",
        "notification_delivered",
        "reminder_sent",
        "run_completed",
        "run_failed",
        "workspace_read",
    ]
    occurred_at: str
    source_id: str | None = None
    agent_name: str | None = None
    target_agent: str | None = None
    event_ids: list[str] = Field(default_factory=list)
    run_id: str | None = None
    run_ids: list[str] = Field(default_factory=list)
    delivery_id: str | None = None
    message_id: str | None = None
    paths: list[str] = Field(default_factory=list)


class OverviewChange(ClientModel):
    """Null removes an entity; non-null replaces that entity atomically."""

    collection: Literal["sources", "agents", "event_inbox", "remy_inbox", "reminders", "workspace", "errors"]
    id: str
    value: ConnectorView | OverviewAgent | OverviewInbox | OverviewReminder | OverviewFile | str | None


class OverviewSnapshot(ClientModel):
    """Replace all state at the supplied watermark without animating history."""

    type: Literal["snapshot"] = "snapshot"
    epoch: str
    revision: int
    as_of: str
    overview: ProactiveOverview


class OverviewUpdate(ClientModel):
    """Apply only to the matching epoch and previous revision."""

    type: Literal["update"] = "update"
    epoch: str
    previous_revision: int
    revision: int
    as_of: str
    changes: list[OverviewChange]
    transitions: list[OverviewTransition]


class OverviewHeartbeat(ClientModel):
    """Transport liveness carries no business state."""

    type: Literal["heartbeat"] = "heartbeat"
