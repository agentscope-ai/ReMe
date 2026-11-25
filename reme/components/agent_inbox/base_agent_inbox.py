"""Public contract for durable, single-consumer Agent inboxes."""

from __future__ import annotations

from abc import abstractmethod

from pydantic import BaseModel

from ..base import BaseComponent
from ...enumeration import ComponentEnum
from ...schema.agent_inbox import InboxReceipt


class InvalidInboxBatchError(ValueError):
    """The submitted items cannot be stored unchanged, even when capacity is available."""


class BaseAgentInbox(BaseComponent):
    """Keep input receipt and asynchronous-result retention backend-independent."""

    component_type = ComponentEnum.AGENT_INBOX
    max_read_items: int
    max_pending_items: int
    max_pending_bytes: int
    claim_max_items: int
    claim_max_bytes: int

    @abstractmethod
    async def create(self, name: str) -> None:
        """Create an inbox; duplicate identities fail explicitly."""

    @abstractmethod
    async def mount(self, name: str, directory: str) -> None:
        """Create or restore an identity at a workspace-relative storage location."""

    @abstractmethod
    async def list(self) -> list[dict]:
        """Read inbox summaries without consuming inputs."""

    @abstractmethod
    async def snapshot(self, name: str) -> dict:
        """Read the retained execution receipt and pending counts."""

    @abstractmethod
    async def pending_ids(self, name: str) -> set[str]:
        """Read the identifiers currently admitted to the pending queue."""

    @abstractmethod
    async def read(self, name: str, *, state: str = "all", offset: int = 0, limit: int = 20) -> dict:
        """Page retained inputs and their real processing state without consuming them."""

    @abstractmethod
    async def append(
        self,
        name: str,
        items: BaseModel | dict | list[BaseModel | dict],
        *,
        wait: bool = True,
    ) -> list[str]:
        """Persist inputs before publishing them to a consumer."""

    @abstractmethod
    async def claim(
        self,
        name: str,
        limit: int | None = None,
        *,
        item_ids: list[str] | None = None,
    ) -> dict | None:
        """Lease the next FIFO batch or retry the selected inputs."""

    @abstractmethod
    async def ack(self, name: str, token: str, *, metadata: dict | None = None, retain: bool = False) -> str:
        """Commit execution and optionally protect its inputs until result handoff."""

    @abstractmethod
    async def nack(self, name: str, token: str) -> None:
        """Release a claim without consuming its inputs."""

    @abstractmethod
    async def receipts(self) -> dict[str, list[InboxReceipt]]:
        """Read retained execution receipts grouped by inbox identity."""

    @abstractmethod
    async def retain(self, name: str, item_ids: list[str], *, release: bool = False) -> None:
        """Protect inputs from history trimming or release completed result handoffs."""

    @abstractmethod
    async def forget(self, name: str) -> None:
        """Unmount an idle identity without deleting source files."""

    @abstractmethod
    async def wait(self, names: str | list[str] | None = None, timeout: float | None = None) -> bool:
        """Wait for claimable inputs or shutdown."""
