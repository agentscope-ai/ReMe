"""Vector-backed profile storage."""

from .profile_backend import BaseProfileBackend
from .profile_vector_handler import ProfileVectorHandler
from ....core import ServiceContext
from ....core.schema import MemoryNode


class VectorProfileBackend(BaseProfileBackend):
    """Persist user profiles in a dedicated vector store."""

    def __init__(
        self,
        memory_target: str,
        service_context: ServiceContext,
        vector_store_name: str = "profile",
        max_capacity: int = 50,
    ):
        super().__init__(memory_target=memory_target, max_capacity=max_capacity)
        self.handler = ProfileVectorHandler(
            memory_target=memory_target,
            service_context=service_context,
            vector_store_name=vector_store_name,
            max_capacity=max_capacity,
        )

    async def get_all(self) -> list[MemoryNode]:
        return await self.handler.get_all()

    async def get_by(self, *, profile_id: str | None = None, profile_key: str | None = None) -> MemoryNode | None:
        return await self.handler.get_by(profile_id=profile_id, profile_key=profile_key)

    async def delete(self, profile_id: str | list[str]) -> bool | int:
        return await self.handler.delete(profile_id)

    async def delete_all(self) -> int:
        return await self.handler.delete_all()

    async def add(self, message_time: str, profile_key: str, profile_value: str, ref_memory_id: str = "") -> MemoryNode:
        return await self.handler.add(message_time, profile_key, profile_value, ref_memory_id)

    async def add_batch(self, profiles: list[dict], ref_memory_id: str = "") -> list[MemoryNode]:
        return await self.handler.add_batch(profiles, ref_memory_id)

    async def update(
        self,
        profile_id: str,
        message_time: str,
        profile_key: str,
        profile_value: str,
    ) -> MemoryNode | None:
        return await self.handler.update(profile_id, message_time, profile_key, profile_value)

    async def search(self, query: str | list[str], limit: int = 5) -> list[MemoryNode]:
        return await self.handler.search(query, limit)
