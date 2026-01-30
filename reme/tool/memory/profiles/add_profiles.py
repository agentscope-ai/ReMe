"""Add user profile tool"""

from collections import defaultdict
from loguru import logger

from .profile_handler import ProfileHandler
from ..base_memory_tool import BaseMemoryTool
from ....core.schema import ToolCall


class AddProfiles(BaseMemoryTool):
    """Add user profile entries"""

    def __init__(self, enable_memory_target: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.enable_memory_target = enable_memory_target

    def _build_profile_parameters(self) -> dict:
        """Build profile parameters schema"""
        properties = {
            "message_time": {"type": "string", "description": "Message time, e.g. '2020-01-01 00:00:00'"},
            "profile_key": {"type": "string", "description": "Profile key, e.g. 'name'"},
            "profile_value": {"type": "string", "description": "Profile value, e.g. 'John Smith'"},
        }
        required = ["message_time", "profile_key", "profile_value"]

        if self.enable_memory_target:
            properties["memory_target"] = {"type": "string", "description": "Memory target"}
            required.append("memory_target")

        return {"type": "object", "properties": properties, "required": required}

    def _build_tool_call(self) -> ToolCall:
        """Build single profile tool call"""
        params = self._build_profile_parameters()
        return ToolCall(**{
            "description": "Add a user profile entry.",
            "parameters": {"type": "object", "properties": params["properties"], "required": params["required"]},
        })

    def _build_multiple_tool_call(self) -> ToolCall:
        """Build multiple profiles tool call"""
        return ToolCall(**{
            "description": "Batch add user profiles.",
            "parameters": {
                "type": "object",
                "properties": {
                    "profiles": {
                        "type": "array",
                        "description": "Profiles to add",
                        "items": self._build_profile_parameters(),
                    }
                },
                "required": ["profiles"],
            },
        })

    async def execute(self):
        # Normalize to list format
        if self.enable_multiple:
            profiles = self.context.get("profiles", [])
        else:
            profiles = [self.context] if "profile_key" in self.context else []

        if not profiles:
            return "No profiles to add."

        # Group by target
        profiles_by_target = defaultdict(list)
        for profile in profiles:
            target = profile.get("memory_target", self.memory_target) if self.enable_memory_target else self.memory_target
            profiles_by_target[target].append(profile)

        # Add profiles
        all_nodes = []
        for target, target_profiles in profiles_by_target.items():
            handler = ProfileHandler(profile_path=self.profile_path, memory_target=target)
            nodes = handler.add_batch(profiles=target_profiles, ref_memory_id=self.history_id)
            all_nodes.extend(nodes)

        self.memory_nodes.extend(all_nodes)
        
        message = f"Added {len(profiles)} profile(s)."
        logger.info(message)
        return message
