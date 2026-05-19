"""Profile handler facade for filesystem and vector backends."""

# pylint: disable=missing-function-docstring

import asyncio
from pathlib import Path

from loguru import logger

from .file_profile_backend import FileProfileBackend
from .profile_backend import BaseProfileBackend
from .vector_profile_backend import VectorProfileBackend
from ....core import ServiceContext
from ....core.schema import MemoryNode


class ProfileHandler:
    """User profile facade with pluggable storage backends."""

    def __init__(
        self,
        memory_target: str,
        profile_path: str | Path | None = None,
        service_context: ServiceContext | None = None,
        profile_backend: str = "filesystem",
        profile_store_name: str = "profile",
        max_capacity: int = 50,
    ):
        self.memory_target = memory_target
        self.profile_backend = profile_backend
        self.profile_store_name = profile_store_name
        self.max_capacity = max_capacity
        self.cache_key = self.memory_target.replace(" ", "_").lower()
        self.backend = self._build_backend(
            profile_path=profile_path,
            service_context=service_context,
        )

    def _build_backend(
        self,
        profile_path: str | Path | None,
        service_context: ServiceContext | None,
    ) -> BaseProfileBackend:
        if self.profile_backend == "filesystem":
            if profile_path is None:
                raise ValueError("profile_path is required for filesystem profile backend")
            return FileProfileBackend(
                profile_path=profile_path,
                memory_target=self.memory_target,
                max_capacity=self.max_capacity,
            )

        if self.profile_backend == "vector":
            if service_context is None:
                raise ValueError("service_context is required for vector profile backend")
            return VectorProfileBackend(
                memory_target=self.memory_target,
                service_context=service_context,
                vector_store_name=self.profile_store_name,
                max_capacity=self.max_capacity,
            )

        raise ValueError(f"Unsupported profile backend: {self.profile_backend}")

    @staticmethod
    def _run_sync(coro):
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(coro)
        raise RuntimeError(
            "Synchronous profile access is not available in an active event loop. Use async methods instead.",
        )

    async def adelete(self, profile_id: str | list[str]) -> bool | int:
        return await self.backend.delete(profile_id)

    async def adelete_all(self) -> int:
        return await self.backend.delete_all()

    async def aadd(
        self,
        message_time: str,
        profile_key: str,
        profile_value: str,
        ref_memory_id: str = "",
    ) -> MemoryNode:
        return await self.backend.add(message_time, profile_key, profile_value, ref_memory_id)

    async def aadd_batch(self, profiles: list[dict], ref_memory_id: str = "") -> list[MemoryNode]:
        return await self.backend.add_batch(profiles, ref_memory_id)

    async def aupdate(
        self,
        profile_id: str,
        message_time: str,
        profile_key: str,
        profile_value: str,
    ) -> MemoryNode | None:
        return await self.backend.update(profile_id, message_time, profile_key, profile_value)

    async def aget_by(self, *, profile_id: str | None = None, profile_key: str | None = None) -> MemoryNode | None:
        return await self.backend.get_by(profile_id=profile_id, profile_key=profile_key)

    async def aget_by_id(self, profile_id: str) -> MemoryNode | None:
        return await self.aget_by(profile_id=profile_id)

    async def aget_by_key(self, profile_key: str) -> MemoryNode | None:
        return await self.aget_by(profile_key=profile_key)

    async def aget_all(self) -> list[MemoryNode]:
        return await self.backend.get_all()

    async def asearch(self, query: str | list[str], limit: int = 5) -> list[MemoryNode]:
        return await self.backend.search(query=query, limit=limit)

    @staticmethod
    def format_node(node: MemoryNode, add_profile_id: bool = False, add_history_id: bool = False) -> str:
        """Render a profile ``MemoryNode`` as a single-line string for tools/logs."""
        parts = []
        profile_key = str(node.metadata.get("profile_key", node.when_to_use))

        if add_profile_id:
            parts.append(f"profile_id={node.memory_id}")

        if node.message_time:
            parts.append(f"[{node.message_time}]")

        parts.append(f"{profile_key}: {node.content}")

        if add_history_id and node.ref_memory_id:
            parts.append(f"history_id={node.ref_memory_id}")

        return " ".join(parts)

    async def aread_all(self, add_profile_id: bool = False, add_history_id: bool = False) -> str:
        nodes = await self.aget_all()
        formatted_profiles = [self.format_node(node, add_profile_id, add_history_id) for node in nodes]
        logger.info(f"Read {len(formatted_profiles)} profiles from {self.cache_key}")
        return "\n".join(formatted_profiles).strip()

    async def aretrieve(
        self,
        query: str | list[str],
        limit: int = 5,
        add_profile_id: bool = True,
        add_history_id: bool = False,
    ) -> tuple[list[MemoryNode], str]:
        nodes = await self.asearch(query=query, limit=limit)
        formatted_profiles = [self.format_node(node, add_profile_id, add_history_id) for node in nodes]
        return nodes, "\n".join(formatted_profiles).strip()

    def delete(self, profile_id: str | list[str]) -> bool | int:
        if isinstance(self.backend, FileProfileBackend):
            return self.backend.delete_sync(profile_id)
        return self._run_sync(self.adelete(profile_id))

    def delete_all(self) -> int:
        if isinstance(self.backend, FileProfileBackend):
            return self.backend.delete_all_sync()
        return self._run_sync(self.adelete_all())

    def add(self, message_time: str, profile_key: str, profile_value: str, ref_memory_id: str = "") -> MemoryNode:
        if isinstance(self.backend, FileProfileBackend):
            return self.backend.add_sync(message_time, profile_key, profile_value, ref_memory_id)
        return self._run_sync(self.aadd(message_time, profile_key, profile_value, ref_memory_id))

    def add_batch(self, profiles: list[dict], ref_memory_id: str = "") -> list[MemoryNode]:
        if isinstance(self.backend, FileProfileBackend):
            return self.backend.add_batch_sync(profiles, ref_memory_id)
        return self._run_sync(self.aadd_batch(profiles, ref_memory_id))

    def update(self, profile_id: str, message_time: str, profile_key: str, profile_value: str) -> MemoryNode | None:
        if isinstance(self.backend, FileProfileBackend):
            return self.backend.update_sync(profile_id, message_time, profile_key, profile_value)
        return self._run_sync(self.aupdate(profile_id, message_time, profile_key, profile_value))

    def get_by(self, *, profile_id: str | None = None, profile_key: str | None = None) -> MemoryNode | None:
        if isinstance(self.backend, FileProfileBackend):
            return self.backend.get_by_sync(profile_id=profile_id, profile_key=profile_key)
        return self._run_sync(self.aget_by(profile_id=profile_id, profile_key=profile_key))

    def get_by_id(self, profile_id: str) -> MemoryNode | None:
        return self._run_sync(self.aget_by_id(profile_id))

    def get_by_key(self, profile_key: str) -> MemoryNode | None:
        return self._run_sync(self.aget_by_key(profile_key))

    def get_all(self) -> list[MemoryNode]:
        if isinstance(self.backend, FileProfileBackend):
            return self.backend.get_all_sync()
        return self._run_sync(self.aget_all())

    def read_all(self, add_profile_id: bool = False, add_history_id: bool = False) -> str:
        return self._run_sync(self.aread_all(add_profile_id, add_history_id))
