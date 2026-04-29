"""Vector-backed handler for bounded user profiles."""

import hashlib

from loguru import logger

from ....core import ServiceContext
from ....core.enumeration import MemoryType
from ....core.schema import MemoryNode
from ....core.vector_store import BaseVectorStore


class ProfileVectorHandler:
    """Manage profile rows stored in a dedicated vector collection."""

    PROFILE_KIND = "profile"

    def __init__(
        self,
        memory_target: str,
        service_context: ServiceContext,
        vector_store_name: str = "profile",
        max_capacity: int = 50,
    ):
        self.memory_target = memory_target
        self.service_context = service_context
        self.vector_store_name = vector_store_name
        self.max_capacity = max_capacity
        self.vector_store: BaseVectorStore = service_context.vector_stores[vector_store_name]

    @staticmethod
    def build_retrieval_text(profile_key: str, profile_value: str) -> str:
        """Build the text that will be embedded for semantic profile retrieval."""
        return f"{profile_key}: {profile_value}".strip(": ")

    def build_profile_id(self, profile_key: str) -> str:
        """Build a stable id from user and key."""
        hash_obj = hashlib.sha256(f"{self.memory_target}\n{profile_key}".encode("utf-8"))
        return hash_obj.hexdigest()[:16]

    def _base_filters(self) -> dict:
        return {
            "memory_type": MemoryType.IDENTITY.value,
            "memory_target": self.memory_target,
            "profile_kind": self.PROFILE_KIND,
        }

    def _build_profile_node(self, profile: dict, ref_memory_id: str = "") -> MemoryNode:
        profile_key = profile.get("profile_key", "").strip()
        profile_value = profile.get("profile_value", "").strip()
        message_time = profile.get("message_time", "")
        ref_id = profile.get("ref_memory_id", ref_memory_id)
        metadata = dict(profile.get("metadata", {}))
        metadata.update(
            {
                "profile_key": profile_key,
                "profile_kind": self.PROFILE_KIND,
                "profile_backend": "vector",
            },
        )
        return MemoryNode(
            memory_id=self.build_profile_id(profile_key),
            memory_type=MemoryType.IDENTITY,
            memory_target=self.memory_target,
            when_to_use=self.build_retrieval_text(profile_key, profile_value),
            content=profile_value,
            message_time=message_time,
            ref_memory_id=ref_id,
            metadata=metadata,
        )

    async def get_all(self) -> list[MemoryNode]:
        vector_nodes = await self.vector_store.list(
            filters=self._base_filters(),
            sort_key="message_time",
            reverse=False,
        )
        return [MemoryNode.from_vector_node(node) for node in vector_nodes]

    async def get_by(self, *, profile_id: str | None = None, profile_key: str | None = None) -> MemoryNode | None:
        if not profile_id and not profile_key:
            raise ValueError("Must provide either profile_id or profile_key")

        if profile_id:
            try:
                vector_node = await self.vector_store.get(profile_id)
            except KeyError:
                logger.warning(f"Profile {profile_id} not found in vector store")
                return None
            if vector_node is None:
                logger.warning(f"Profile {profile_id} not found in vector store")
                return None
            memory_node = MemoryNode.from_vector_node(vector_node)
            if memory_node.memory_target != self.memory_target:
                return None
            if memory_node.memory_type is not MemoryType.IDENTITY:
                return None
            if memory_node.metadata.get("profile_kind") != self.PROFILE_KIND:
                return None
            return memory_node

        profile_key = profile_key or ""
        vector_nodes = await self.vector_store.list(filters={**self._base_filters(), "profile_key": profile_key}, limit=1)
        if not vector_nodes:
            return None
        return MemoryNode.from_vector_node(vector_nodes[0])

    async def delete(self, profile_id: str | list[str]) -> bool | int:
        if isinstance(profile_id, list):
            profile_ids = list(dict.fromkeys(pid for pid in profile_id if pid))
            if not profile_ids:
                return 0
            existing_nodes = []
            for pid in profile_ids:
                node = await self.get_by(profile_id=pid)
                if node is not None:
                    existing_nodes.append(node)
            if not existing_nodes:
                return 0
            await self.vector_store.delete([node.memory_id for node in existing_nodes])
            return len(existing_nodes)

        existing_node = await self.get_by(profile_id=profile_id)
        if existing_node is None:
            return False
        await self.vector_store.delete(existing_node.memory_id)
        return True

    async def delete_all(self) -> int:
        nodes = await self.get_all()
        if not nodes:
            return 0
        await self.vector_store.delete([node.memory_id for node in nodes])
        return len(nodes)

    async def add_batch(self, profiles: list[dict], ref_memory_id: str = "") -> list[MemoryNode]:
        if not profiles:
            return []

        deduped_profiles: dict[str, dict] = {}
        for profile in profiles:
            profile_key = profile.get("profile_key", "").strip()
            if not profile_key:
                continue
            deduped_profiles[profile_key] = profile

        new_nodes = [self._build_profile_node(profile, ref_memory_id=ref_memory_id) for profile in deduped_profiles.values()]
        if not new_nodes:
            return []

        await self.vector_store.delete([node.memory_id for node in new_nodes])
        await self.vector_store.insert([node.to_vector_node() for node in new_nodes])
        await self.enforce_capacity()
        return new_nodes

    async def add(self, message_time: str, profile_key: str, profile_value: str, ref_memory_id: str = "") -> MemoryNode:
        nodes = await self.add_batch(
            [
                {
                    "message_time": message_time,
                    "profile_key": profile_key,
                    "profile_value": profile_value,
                },
            ],
            ref_memory_id=ref_memory_id,
        )
        return nodes[0]

    async def update(
        self,
        profile_id: str,
        message_time: str,
        profile_key: str,
        profile_value: str,
    ) -> MemoryNode | None:
        existing_node = await self.get_by(profile_id=profile_id)
        if existing_node is None:
            return None

        new_node = self._build_profile_node(
            {
                "message_time": message_time,
                "profile_key": profile_key,
                "profile_value": profile_value,
                "ref_memory_id": existing_node.ref_memory_id,
                "metadata": existing_node.metadata,
            },
        )

        if existing_node.memory_id != new_node.memory_id:
            await self.vector_store.delete(existing_node.memory_id)
        else:
            await self.vector_store.delete(new_node.memory_id)

        await self.vector_store.insert(new_node.to_vector_node())
        await self.enforce_capacity()
        return new_node

    async def search(self, query: str | list[str], limit: int = 5) -> list[MemoryNode]:
        queries = [query] if isinstance(query, str) else query
        seen_nodes: dict[str, MemoryNode] = {}
        for item in queries:
            if not item or not item.strip():
                continue
            vector_nodes = await self.vector_store.search(item, limit=limit, filters=self._base_filters())
            for vector_node in vector_nodes:
                memory_node = MemoryNode.from_vector_node(vector_node)
                seen_nodes[memory_node.memory_id] = memory_node
        nodes = list(seen_nodes.values())
        nodes.sort(key=lambda node: (node.score, node.message_time), reverse=True)
        return nodes[:limit]

    async def enforce_capacity(self):
        nodes = await self.get_all()
        overflow = len(nodes) - self.max_capacity
        if overflow <= 0:
            return

        to_delete = [node.memory_id for node in nodes[:overflow]]
        await self.vector_store.delete(to_delete)
