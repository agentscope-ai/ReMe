"""Public lifecycle and coordination contract for file-native Agent runtimes."""

from __future__ import annotations

from abc import abstractmethod
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from datetime import datetime
from typing import TYPE_CHECKING

from ..base_component import BaseComponent
from ...enumeration import ComponentEnum

if TYPE_CHECKING:
    import asyncio

    from ._internal.files import AgentFiles
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
    files: AgentFiles
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
    interactive: dict[asyncio.Task, str]
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
    async def definition(self, agent_id: str) -> ProactiveAgent:
        """Read the latest task definition."""

    @abstractmethod
    async def create(self, config: dict) -> dict:
        """Create or restore a durable task identity."""

    @abstractmethod
    async def update(self, agent_id: str, config: dict) -> dict:
        """Update execution settings independently of task memory."""

    @abstractmethod
    async def delete(self, agent_id: str) -> dict:
        """Stop owned execution while preserving task history."""

    @abstractmethod
    async def append(self, agent_id: str, events) -> dict:
        """Persist new task inputs before scheduling work."""

    @abstractmethod
    async def status(self, agent_id: str) -> dict:
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
    async def query_turn(self, agent_id: str) -> AsyncIterator[str]:
        """Own a query from checkpoint selection through temporary session cleanup."""
        yield

    @asynccontextmanager
    @abstractmethod
    async def agent_turn(self, agent_id: str) -> AsyncIterator[None]:
        """Serialize a user turn with the selected Agent's background work."""
        yield

    @abstractmethod
    async def list_views(self) -> dict:
        """Read all foreground and task identities with current session state."""

    @abstractmethod
    async def read_agent(self, agent_id: str) -> dict:
        """Read one current session without creating it."""

    @abstractmethod
    async def watch_agent(self, agent_id: str) -> AsyncIterator[dict]:
        """Observe one current conversation and global Agent state."""
        yield {}

    @abstractmethod
    async def list_inputs(self, agent_id: str, **kwargs) -> dict:
        """Page retained inputs without acquiring a consumption lease."""

    @abstractmethod
    async def list_chat_sessions(self, offset=0, limit=10) -> dict:
        """Inspect the Chat Agent's retained session references."""

    @abstractmethod
    async def run(self, agent_id: str, execute: Callable) -> dict:
        """Execute and commit one serial input batch."""

    @abstractmethod
    async def tick(self, run_job: Callable) -> None:
        """Schedule task and Chat work and collect idle temporary Agents."""

    @abstractmethod
    async def wait(self, stop: asyncio.Event, seconds=1) -> None:
        """Wait for input, configuration changes or shutdown."""
