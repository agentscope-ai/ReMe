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
    from ._internal.inbox import AgentInbox
    from ._internal.outbox import AgentOutbox
    from ._internal.registry import AgentRegistry
    from ._internal.sessions import AgentSessions
    from ..agent_wrapper import BaseAgentWrapper
    from ..event_stream import BaseEventStream
    from ...schema.proactive_agent import ProactiveAgent


class BaseAgentRuntime(BaseComponent):
    """Expose owned Agent resources and operations independently of the local backend."""

    component_type = ComponentEnum.AGENT_RUNTIME
    agent_wrapper: BaseAgentWrapper
    event_stream: BaseEventStream
    files: AgentFiles
    agents: AgentRegistry
    sessions: AgentSessions
    inbox: AgentInbox
    outbox: AgentOutbox
    chat_lock: asyncio.Lock
    chat_tasks: set[asyncio.Task]
    workers: dict[str, asyncio.Task]
    queries: dict[asyncio.Task, str]
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
    async def snapshot(self, agent_id: str) -> dict:
        """Read retained execution facts without invoking a model."""

    @abstractmethod
    async def status(self, agent_id: str) -> dict:
        """Return execution settings, status, and a memory preview with its path."""

    @abstractmethod
    async def list_agents(self) -> dict:
        """List task settings, current status, and memory paths without memory bodies."""

    @asynccontextmanager
    @abstractmethod
    async def main_turn(self) -> AsyncIterator[None]:
        """Own one serial Chat turn and its cancellation."""
        yield

    @asynccontextmanager
    @abstractmethod
    async def query_turn(self, agent_id: str) -> AsyncIterator[str]:
        """Own a query from checkpoint selection through temporary session cleanup."""
        yield

    @abstractmethod
    async def list_chat_sessions(self, offset=0, limit=10) -> dict:
        """Inspect main's retained session references."""

    @abstractmethod
    async def run(self, agent_id: str, execute: Callable) -> dict:
        """Execute and commit one serial input batch."""

    @abstractmethod
    async def tick(self, run_job: Callable, job_name: str) -> None:
        """Schedule eligible work and collect idle temporary Agents."""

    @abstractmethod
    async def wait(self, stop: asyncio.Event, seconds=1) -> None:
        """Wait for input, configuration changes or shutdown."""
