"""Profile backend abstractions."""

from abc import ABC, abstractmethod

from ....core.schema import MemoryNode


class BaseProfileBackend(ABC):
    """Abstract interface for profile storage backends."""

    def __init__(self, memory_target: str, max_capacity: int = 50):
        self.memory_target = memory_target
        self.max_capacity = max_capacity

    @abstractmethod
    async def get_all(self) -> list[MemoryNode]:
        """Return all profile rows for the current user."""

    @abstractmethod
    async def get_by(self, *, profile_id: str | None = None, profile_key: str | None = None) -> MemoryNode | None:
        """Return one profile row by id or key."""

    @abstractmethod
    async def delete(self, profile_id: str | list[str]) -> bool | int:
        """Delete one or more profile rows."""

    @abstractmethod
    async def delete_all(self) -> int:
        """Delete all profile rows for the current user."""

    @abstractmethod
    async def add(self, message_time: str, profile_key: str, profile_value: str, ref_memory_id: str = "") -> MemoryNode:
        """Add a single profile row."""

    @abstractmethod
    async def add_batch(self, profiles: list[dict], ref_memory_id: str = "") -> list[MemoryNode]:
        """Add multiple profile rows."""

    @abstractmethod
    async def update(
        self,
        profile_id: str,
        message_time: str,
        profile_key: str,
        profile_value: str,
    ) -> MemoryNode | None:
        """Update one profile row."""

    @abstractmethod
    async def search(self, query: str | list[str], limit: int = 5) -> list[MemoryNode]:
        """Search profile rows relevant to the query."""
