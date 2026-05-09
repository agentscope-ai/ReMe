"""Filesystem-backed profile storage."""

from pathlib import Path

from loguru import logger

from .profile_backend import BaseProfileBackend
from ....core.enumeration import MemoryType
from ....core.schema import MemoryNode
from ....core.utils import CacheHandler, deduplicate_memories


class FileProfileBackend(BaseProfileBackend):
    """Persist user profiles in local JSONL cache files."""

    def __init__(self, profile_path: str | Path, memory_target: str, max_capacity: int = 50):
        super().__init__(memory_target=memory_target, max_capacity=max_capacity)
        self.cache_key: str = self.memory_target.replace(" ", "_").lower()
        self.cache_handler: CacheHandler = CacheHandler(profile_path)

    def _load_nodes(self) -> list[MemoryNode]:
        cached_data = self.cache_handler.load(self.cache_key, auto_clean=False)
        if not cached_data:
            return []
        return [MemoryNode(**data) for data in cached_data]

    def _save_nodes(self, nodes: list[MemoryNode], apply_limits: bool = True):
        if apply_limits:
            nodes = deduplicate_memories(nodes)

            if len(nodes) > self.max_capacity:
                sorted_nodes = sorted(nodes, key=lambda n: n.message_time)
                removed_count = len(sorted_nodes) - self.max_capacity
                nodes = sorted_nodes[removed_count:]
                logger.info(
                    f"Capacity limit reached: removed {removed_count} oldest profiles "
                    f"(kept {len(nodes)}/{self.max_capacity})",
                )

        nodes_data = [node.model_dump(exclude_none=True) for node in nodes]
        self.cache_handler.save(self.cache_key, nodes_data)
        logger.info(f"Saved {len(nodes)} profiles to {self.cache_key}")

    def get_all_sync(self) -> list[MemoryNode]:
        """Load all profile nodes from cache, ordered by ``message_time``."""
        nodes = self._load_nodes()
        nodes.sort(key=lambda n: n.message_time)
        return nodes

    def get_by_sync(self, *, profile_id: str | None = None, profile_key: str | None = None) -> MemoryNode | None:
        """Return the first node matching ``profile_id`` or ``profile_key``."""
        if not profile_id and not profile_key:
            raise ValueError("Must provide either profile_id or profile_key")

        for node in self._load_nodes():
            if profile_id and node.memory_id == profile_id:
                return node
            if profile_key and node.when_to_use == profile_key:
                return node
        return None

    def delete_sync(self, profile_id: str | list[str]) -> bool | int:
        """Remove one id, many ids, or none; returns bool, count, or 0/false if nothing removed."""
        nodes = self._load_nodes()
        original_count = len(nodes)

        if isinstance(profile_id, list):
            profile_ids_set = set(profile_id)
            nodes = [n for n in nodes if n.memory_id not in profile_ids_set]
            deleted_count = original_count - len(nodes)
            if deleted_count == 0:
                logger.warning(f"No profiles found to delete from {len(profile_id)} IDs")
                return 0

            self._save_nodes(nodes, apply_limits=False)
            logger.info(f"Batch deleted {deleted_count} profiles")
            return deleted_count

        nodes = [n for n in nodes if n.memory_id != profile_id]
        if len(nodes) == original_count:
            logger.warning(f"Profile {profile_id} not found")
            return False

        self._save_nodes(nodes, apply_limits=False)
        logger.info(f"Deleted profile {profile_id}")
        return True

    def delete_all_sync(self) -> int:
        """Clear every cached profile for this target; returns how many were stored."""
        nodes = self._load_nodes()
        count = len(nodes)
        self._save_nodes([], apply_limits=False)
        logger.info(f"Deleted all {count} profiles")
        return count

    def add_sync(self, message_time: str, profile_key: str, profile_value: str, ref_memory_id: str = "") -> MemoryNode:
        """Append a profile row, replacing any existing row with the same key."""
        nodes = self._load_nodes()

        new_node = MemoryNode(
            memory_type=MemoryType.PERSONAL,
            memory_target=self.memory_target,
            when_to_use=profile_key,
            content=profile_value,
            message_time=message_time,
            ref_memory_id=ref_memory_id,
        )

        original_count = len(nodes)
        nodes = [n for n in nodes if n.when_to_use != profile_key]
        if len(nodes) < original_count:
            logger.info(f"Removed {original_count - len(nodes)} duplicate profile(s) with key: {profile_key}")

        nodes.append(new_node)
        self._save_nodes(nodes)
        logger.info(f"Added profile: {profile_key}={profile_value}")
        return new_node

    def add_batch_sync(self, profiles: list[dict], ref_memory_id: str = "") -> list[MemoryNode]:
        """Insert many profiles in one write, deduping by key against existing rows."""
        if not profiles:
            return []

        nodes = self._load_nodes()
        new_nodes = [
            MemoryNode(
                memory_type=MemoryType.PERSONAL,
                memory_target=self.memory_target,
                when_to_use=p.get("profile_key", ""),
                content=p.get("profile_value", ""),
                message_time=p.get("message_time", ""),
                ref_memory_id=ref_memory_id,
            )
            for p in profiles
        ]

        new_keys = {n.when_to_use for n in new_nodes}
        original_count = len(nodes)
        nodes = [n for n in nodes if n.when_to_use not in new_keys]
        if len(nodes) < original_count:
            logger.info(f"Removed {original_count - len(nodes)} duplicate profile(s) with matching keys")

        nodes.extend(new_nodes)
        self._save_nodes(nodes)
        logger.info(f"Batch added {len(new_nodes)} profiles")
        return new_nodes

    def update_sync(
        self,
        profile_id: str,
        message_time: str,
        profile_key: str,
        profile_value: str,
    ) -> MemoryNode | None:
        """Update fields for ``profile_id``; return ``None`` if that id is missing."""
        nodes = self._load_nodes()
        target_node = None
        for node in nodes:
            if node.memory_id == profile_id:
                node.when_to_use = profile_key
                node.content = profile_value
                node.message_time = message_time
                target_node = node
                break

        if target_node is None:
            logger.warning(f"Profile {profile_id} not found")
            return None

        self._save_nodes(nodes, apply_limits=False)
        logger.info(f"Updated profile {profile_id}: {profile_key}={profile_value}")
        return target_node

    def search_sync(self, query: str | list[str], limit: int = 5) -> list[MemoryNode]:
        """Simple substring/token match over key and content, best matches first."""
        queries = [query] if isinstance(query, str) else query
        query_terms = [q.strip().lower() for q in queries if q and q.strip()]
        if not query_terms:
            return []

        scored_nodes = []
        for node in self.get_all_sync():
            profile_key = str(node.metadata.get("profile_key", node.when_to_use)).lower()
            haystack = f"{profile_key}: {node.content}".lower()
            score = 0
            for term in query_terms:
                if term in haystack:
                    score += len(term) + 10
                else:
                    token_hits = sum(1 for token in term.split() if token and token in haystack)
                    score += token_hits

            if score > 0:
                node.score = float(score)
                scored_nodes.append(node)

        scored_nodes.sort(key=lambda n: (n.score, n.message_time), reverse=True)
        return scored_nodes[:limit]

    async def get_all(self) -> list[MemoryNode]:
        return self.get_all_sync()

    async def get_by(self, *, profile_id: str | None = None, profile_key: str | None = None) -> MemoryNode | None:
        return self.get_by_sync(profile_id=profile_id, profile_key=profile_key)

    async def delete(self, profile_id: str | list[str]) -> bool | int:
        return self.delete_sync(profile_id)

    async def delete_all(self) -> int:
        return self.delete_all_sync()

    async def add(
        self,
        message_time: str,
        profile_key: str,
        profile_value: str,
        ref_memory_id: str = "",
    ) -> MemoryNode:
        return self.add_sync(message_time, profile_key, profile_value, ref_memory_id)

    async def add_batch(self, profiles: list[dict], ref_memory_id: str = "") -> list[MemoryNode]:
        return self.add_batch_sync(profiles, ref_memory_id)

    async def update(
        self,
        profile_id: str,
        message_time: str,
        profile_key: str,
        profile_value: str,
    ) -> MemoryNode | None:
        return self.update_sync(profile_id, message_time, profile_key, profile_value)

    async def search(self, query: str | list[str], limit: int = 5) -> list[MemoryNode]:
        return self.search_sync(query, limit)
