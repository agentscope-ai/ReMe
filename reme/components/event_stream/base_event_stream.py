"""Public contract for durable, single-consumer event streams."""

from __future__ import annotations

from abc import abstractmethod

from ..base_component import BaseComponent
from ...enumeration import ComponentEnum
from ...schema.event import Event, EventConsumption


class BaseEventStream(BaseComponent):
    """Keep input consumption and asynchronous-result retention backend-independent."""

    component_type = ComponentEnum.EVENT_STREAM

    @abstractmethod
    async def create(self, name: str) -> None:
        """Create a stream; duplicate identities fail explicitly."""

    @abstractmethod
    async def mount(self, name: str, directory: str) -> None:
        """Create or restore an identity at a workspace-relative storage location."""

    @abstractmethod
    async def list(self) -> list[dict]:
        """Read stream summaries without consuming inputs."""

    @abstractmethod
    async def snapshot(self, name: str) -> dict:
        """Read the latest retained input, execution receipt and pending counts."""

    @abstractmethod
    async def append(self, name: str, events: Event | dict | list[Event | dict]) -> list[str]:
        """Persist inputs before publishing them to a consumer."""

    @abstractmethod
    async def claim(self, name: str, limit: int | None = None) -> dict | None:
        """Lease one input batch without confirming execution."""

    @abstractmethod
    async def ack(self, name: str, token: str, *, metadata: dict | None = None, retain: bool = False) -> str:
        """Commit execution and optionally protect its inputs until result handoff."""

    @abstractmethod
    async def nack(self, name: str, token: str) -> None:
        """Release a claim without consuming its inputs."""

    @abstractmethod
    async def receipts(self) -> dict[str, list[EventConsumption]]:
        """Read retained execution receipts grouped by stream identity."""

    @abstractmethod
    async def retain(self, name: str, event_ids: list[str], *, release: bool = False) -> None:
        """Protect inputs from history trimming or release completed result handoffs."""

    @abstractmethod
    async def forget(self, name: str) -> None:
        """Unmount an idle identity without deleting source files."""

    @abstractmethod
    async def wait(self, names: str | list[str] | None = None, timeout: float | None = None) -> bool:
        """Wait for claimable inputs or shutdown."""
