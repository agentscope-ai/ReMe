"""Shared management, event routing and Remy services for Proactive Runtimes."""

from __future__ import annotations

from abc import abstractmethod
from asyncio import CancelledError, gather
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from datetime import datetime
from typing import TYPE_CHECKING

from ..base import BaseComponent
from ..agent_inbox.base_agent_inbox import InvalidInboxBatchError
from ...enumeration import ComponentEnum

if TYPE_CHECKING:
    import asyncio

    from ...utils.files.workspace_files import WorkspaceFiles
    from ...utils.proactive_record import RoundRecord
    from ._internal.notifications import ChatNotifications
    from ._internal.outbox import ReminderOutbox
    from ._internal.registry import RuntimeRegistry
    from ._internal.sessions import AgentSessions
    from ._internal.live_turn import LiveTurns
    from ..agent_wrapper import BaseAgentWrapper
    from ..agent_inbox import BaseAgentInbox
    from ...schema.proactive_agent import ProactiveRuntimeDefinition
    from ...schema.agent_activity import AgentActivity
    from ...schema.event import Event


class TaskNotFoundError(ValueError):
    """A named task is missing or deleted."""


class ScheduledTaskDeliveryError(ValueError):
    """A scheduled task cannot receive external observations."""


class BaseProactiveManager(BaseComponent):
    """Expose owned Agent resources and operations independently of the local backend."""

    component_type = ComponentEnum.PROACTIVE_MANAGER
    agent_wrapper: BaseAgentWrapper
    files: WorkspaceFiles
    registry: RuntimeRegistry
    sessions: AgentSessions
    live_turns: LiveTurns
    inbox: BaseAgentInbox
    notifications: ChatNotifications
    outbox: ReminderOutbox
    chat_lock: asyncio.Lock
    chat_tasks: set[asyncio.Task]
    workers: dict[str, asyncio.Task]
    activities: dict[str, AgentActivity]
    changed: asyncio.Event
    closed: bool
    retry_seconds: float

    @abstractmethod
    def now(self) -> datetime:
        """Read time in the application timezone."""

    @abstractmethod
    async def definition(self, agent_name: str) -> ProactiveRuntimeDefinition:
        """Read the latest task definition."""

    @abstractmethod
    async def create(self, config: dict) -> dict:
        """Create or restore a durable task identity."""

    @abstractmethod
    async def update(self, agent_name: str, config: dict) -> dict:
        """Update execution settings independently of the task description."""

    @abstractmethod
    async def delete(self, agent_name: str) -> dict:
        """Stop owned execution while preserving task history."""

    @abstractmethod
    async def append_events(self, agent_name: str, events: list[Event]) -> dict:
        """Persist external observations for an event-driven task."""

    async def deliver_events(self, task_names: list[str], events: list[Event]) -> dict:
        """Deliver independently and distinguish rejected targets from retryable failures."""
        results = await gather(*(self.append_events(name, events) for name in task_names), return_exceptions=True)
        deliveries, rejected, errors = {}, {}, {}
        for name, result in zip(task_names, results, strict=True):
            if isinstance(result, (TaskNotFoundError, ScheduledTaskDeliveryError, InvalidInboxBatchError)):
                rejected[name] = str(result)
            elif isinstance(result, CancelledError):
                errors[name] = "Delivery interrupted"
            elif isinstance(result, Exception):
                errors[name] = str(result)
            elif isinstance(result, BaseException):
                raise result
            else:
                deliveries[name] = result["event_ids"]
        return {"deliveries": deliveries, "rejected": rejected, "errors": errors}

    @abstractmethod
    async def status(self, agent_name: str) -> dict:
        """Return execution settings, status, and the task document path."""

    @asynccontextmanager
    @abstractmethod
    async def chat_turn(self) -> AsyncIterator[None]:
        """Own one serial Chat turn and its cancellation."""
        yield

    @asynccontextmanager
    @abstractmethod
    async def query_session(self, agent_name: str, options: dict) -> AsyncIterator[str]:
        """Own a fork of the latest task session through query completion and cleanup."""
        yield

    @abstractmethod
    async def list_views(self) -> dict:
        """Read all foreground and task identities with current session state."""

    @abstractmethod
    async def view_agent(self, agent_name: str, *, status: dict | None = None) -> dict:
        """Read one Agent's public settings and state without loading its transcript."""

    @abstractmethod
    async def read_agent(self, agent_name: str) -> dict:
        """Read one current session without creating it."""

    @abstractmethod
    async def watch_agent(self, agent_name: str) -> AsyncIterator[dict]:
        """Observe one current conversation and global Agent state."""
        yield {}

    @abstractmethod
    async def list_inputs(self, agent_name: str, **kwargs) -> dict:
        """Page retained inputs without acquiring a consumption lease."""

    @abstractmethod
    async def list_chat_sessions(self, offset=0, limit=10) -> dict:
        """Inspect the Chat Agent's retained session references."""

    @abstractmethod
    async def run_round(
        self,
        agent_name: str,
        execute: Callable,
        *,
        record: RoundRecord | None = None,
    ) -> dict:
        """Execute one serial input batch and commit its optional round record."""

    @abstractmethod
    async def run(self, stop: asyncio.Event, run_job: Callable) -> None:
        """Drive the Runtime and Remy workers until the background Job stops."""
