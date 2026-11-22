"""Public lifecycle and coordination contract for file-native Agent runtimes."""

from __future__ import annotations

from abc import abstractmethod
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from datetime import datetime
from typing import TYPE_CHECKING

from ..base import BaseComponent
from ...enumeration import ComponentEnum

if TYPE_CHECKING:
    import asyncio

    from ...utils.files.workspace_files import WorkspaceFiles
    from ...utils.proactive_record import RoundRecord
    from ._internal.notifications import ChatNotifications
    from ._internal.outbox import AgentOutbox
    from ._internal.registry import AgentRegistry
    from ._internal.sessions import AgentSessions
    from ._internal.live_turn import LiveTurns
    from ..agent_wrapper import BaseAgentWrapper
    from ..agent_inbox import BaseAgentInbox
    from ...schema.proactive_agent import ProactiveAgent
    from ...schema.agent_activity import AgentActivity


class BaseAgentRuntime(BaseComponent):
    """Expose owned Agent resources and operations independently of the local backend."""

    component_type = ComponentEnum.AGENT_RUNTIME
    agent_wrapper: BaseAgentWrapper
    files: WorkspaceFiles
    agents: AgentRegistry
    sessions: AgentSessions
    live_turns: LiveTurns
    inbox: BaseAgentInbox
    notifications: ChatNotifications
    outbox: AgentOutbox
    chat_lock: asyncio.Lock
    chat_tasks: set[asyncio.Task]
    workers: dict[str, asyncio.Task]
    agent_tasks: dict[asyncio.Task, str]
    activities: dict[str, AgentActivity]
    changed: asyncio.Event
    closed: bool
    retry_seconds: float

    @abstractmethod
    def now(self) -> datetime:
        """Read time in the application timezone."""

    @abstractmethod
    async def prepare(self, agent: ProactiveAgent) -> None:
        """Prepare an Agent's owned execution resources."""

    @abstractmethod
    async def definition(self, agent_name: str) -> ProactiveAgent:
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
    async def append(self, agent_name: str, events) -> dict:
        """Persist external observations for an event Agent; cron inputs belong to the scheduler."""

    @abstractmethod
    async def status(self, agent_name: str) -> dict:
        """Return execution settings, status, and the original memory path."""

    @abstractmethod
    async def list_agents(self) -> dict:
        """List task settings, current status, and memory paths without memory bodies."""

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
    async def view_agent(self, agent_name: str) -> dict:
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
    async def run(
        self,
        agent_name: str,
        execute: Callable,
        *,
        record: RoundRecord | None = None,
    ) -> dict:
        """Execute one serial input batch and commit its optional round record."""

    @abstractmethod
    async def tick(self, run_job: Callable) -> None:
        """Schedule task and Chat work."""

    @abstractmethod
    async def stop_workers(self) -> None:
        """Stop the owned scheduler workers and notification task."""

    @abstractmethod
    async def wait(self, stop: asyncio.Event, seconds=1) -> None:
        """Wait for input, configuration changes or shutdown."""
