"""Profile Handler for managing user profiles in local memory"""

from pathlib import Path

from loguru import logger

from ....core.enumeration import MemoryType
from ....core.schema import MemoryNode
from ....core.utils import CacheHandler, deduplicate_memories


class ProfileHandler:
    """User profile CRUD handler"""

    def __init__(self, profile_path: str | Path, memory_target: str, max_capacity: int = 50):
        """init"""
        self.memory_target: str = memory_target
        self.cache_key: str = self.memory_target.replace(" ", "_").lower()
        self.cache_handler: CacheHandler = CacheHandler(profile_path)
        self.max_capacity: int = max_capacity

    def _load_nodes(self) -> list[MemoryNode]:
        """Load profile nodes"""
        cached_data = self.cache_handler.load(self.cache_key, auto_clean=False)
        if not cached_data:
            return []
        return [MemoryNode(**data) for data in cached_data]

    def _save_nodes(self, nodes: list[MemoryNode], apply_limits: bool = True):
        """Save nodes with optional deduplication and capacity enforcement"""
        if apply_limits:
            nodes = deduplicate_memories(nodes)

            # Enforce capacity limit by removing the oldest profiles
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

    def delete(self, profile_id: str | list[str]) -> bool | int:
        """Delete profile by ID(s), returns True/False for single ID or count for batch delete"""
        nodes = self._load_nodes()
        original_count = len(nodes)

        # Batch delete mode
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

        # Single delete mode
        nodes = [n for n in nodes if n.memory_id != profile_id]

        if len(nodes) == original_count:
            logger.warning(f"Profile {profile_id} not found")
            return False

        self._save_nodes(nodes, apply_limits=False)
        logger.info(f"Deleted profile {profile_id}")
        return True

    def delete_all(self) -> int:
        """Delete all profiles, returns count deleted"""
        nodes = self._load_nodes()
        count = len(nodes)
        self._save_nodes([], apply_limits=False)
        logger.info(f"Deleted all {count} profiles")
        return count

    def add(self, message_time: str, profile_key: str, profile_value: str, ref_memory_id: str = "") -> MemoryNode:
        """Add new profile, returns created MemoryNode"""
        nodes = self._load_nodes()

        new_node = MemoryNode(
            memory_type=MemoryType.PERSONAL,
            memory_target=self.memory_target,
            when_to_use=profile_key,
            content=profile_value,
            message_time=message_time,
            ref_memory_id=ref_memory_id,
        )

        # Remove existing nodes with the same when_to_use (profile_key)
        original_count = len(nodes)
        nodes = [n for n in nodes if n.when_to_use != profile_key]
        if len(nodes) < original_count:
            logger.info(f"Removed {original_count - len(nodes)} duplicate profile(s) with key: {profile_key}")

        nodes.append(new_node)
        self._save_nodes(nodes)
        logger.info(f"Added profile: {profile_key}={profile_value}")
        return new_node

    def add_batch(self, profiles: list[dict], ref_memory_id: str = "") -> list[MemoryNode]:
        """Add multiple profiles in batch, returns list of created MemoryNodes"""
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

        # Remove existing nodes with the same when_to_use (profile_key)
        new_keys = {n.when_to_use for n in new_nodes}
        original_count = len(nodes)
        nodes = [n for n in nodes if n.when_to_use not in new_keys]
        if len(nodes) < original_count:
            logger.info(f"Removed {original_count - len(nodes)} duplicate profile(s) with matching keys")

        nodes.extend(new_nodes)
        self._save_nodes(nodes)
        logger.info(f"Batch added {len(new_nodes)} profiles")
        return new_nodes

    def update(self, profile_id: str, message_time: str, profile_key: str, profile_value: str) -> MemoryNode | None:
        """Update profile by ID, returns updated node or None if not found"""
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

    def get_by(self, *, profile_id: str | None = None, profile_key: str | None = None) -> MemoryNode | None:
        """Get profile by ID or key"""
        if not profile_id and not profile_key:
            raise ValueError("Must provide either profile_id or profile_key")

        nodes = self._load_nodes()
        for node in nodes:
            if profile_id and node.memory_id == profile_id:
                return node
            if profile_key and node.when_to_use == profile_key:
                return node
        return None

    def get_by_id(self, profile_id: str) -> MemoryNode | None:
        """Get profile by ID (convenience method)"""
        return self.get_by(profile_id=profile_id)

    def get_by_key(self, profile_key: str) -> MemoryNode | None:
        """Get profile by key (convenience method)"""
        return self.get_by(profile_key=profile_key)

    def get_all(self) -> list[MemoryNode]:
        """Get all profiles, sorted by message_time"""
        nodes = self._load_nodes()
        nodes.sort(key=lambda n: n.message_time)
        return nodes

    @staticmethod
    def _format_node(node: MemoryNode, add_profile_id: bool = False, add_history_id: bool = False) -> str:
        """Format a single node to string"""
        parts = []

        if add_profile_id:
            parts.append(f"profile_id={node.memory_id}")

        if node.message_time:
            parts.append(f"[{node.message_time}]")

        parts.append(f"{node.when_to_use}: {node.content}")

        if add_history_id:
            parts.append(f"history_id={node.ref_memory_id}")

        return " ".join(parts)

    def read_all(self, add_profile_id: bool = False, add_history_id: bool = False) -> str:
        """Read all profiles and return formatted string"""
        nodes = self.get_all()
        formatted_profiles = [self._format_node(node, add_profile_id, add_history_id) for node in nodes]
        logger.info(f"Read {len(formatted_profiles)} profiles from {self.cache_key}")
        return "\n".join(formatted_profiles).strip()
