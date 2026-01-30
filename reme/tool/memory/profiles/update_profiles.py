"""Update user profile tool"""

from collections import defaultdict

from loguru import logger

from .profile_handler import ProfileHandler
from ..base_memory_tool import BaseMemoryTool
from ....core.schema import ToolCall


class UpdateProfiles(BaseMemoryTool):
    """Update existing user profile entries"""

    def __init__(self, name="update_profiles", enable_memory_target: bool = False, **kwargs):
        super().__init__(name=name, **kwargs)
        self.enable_memory_target = enable_memory_target

    def _build_profile_parameters(self) -> dict:
        """Build profile parameters schema"""
        properties = {
            "profile_id": {"type": "string", "description": "Profile ID to update"},
            "message_time": {"type": "string", "description": "Message time, e.g. '2020-01-01 00:00:00'"},
            "profile_key": {"type": "string", "description": "Profile key, e.g. 'name'"},
            "profile_value": {"type": "string", "description": "Profile value, e.g. 'John Smith'"},
        }
        required = ["profile_id", "message_time", "profile_key", "profile_value"]

        if self.enable_memory_target:
            properties["memory_target"] = {"type": "string", "description": "Memory target"}
            required.append("memory_target")

        return {"type": "object", "properties": properties, "required": required}

    def _build_tool_call(self) -> ToolCall:
        """Build single profile tool call"""
        return ToolCall(**{
            "description": "Update a user profile entry.",
            "parameters": self._build_profile_parameters(),
        })

    def _build_multiple_tool_call(self) -> ToolCall:
        """Build multiple profiles tool call"""
        return ToolCall(**{
            "description": "batch update user profiles.",
            "parameters": {
                "type": "object",
                "properties": {
                    "profiles": {
                        "type": "array",
                        "description": "Profiles to update",
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
            profiles = [self.context] if "profile_id" in self.context else []

        if not profiles:
            return "No profiles to update."

        # Group profiles by target
        profiles_by_target = defaultdict(list)
        for profile in profiles:
            target = profile.get("memory_target",
                                 self.memory_target) if self.enable_memory_target else self.memory_target
            profiles_by_target[target].append(profile)

        # Update profiles using the update method
        updated_count = 0
        all_nodes = []
        for target, target_profiles in profiles_by_target.items():
            handler = ProfileHandler(profile_path=self.profile_path, memory_target=target)
            for profile in target_profiles:
                updated_node = handler.update(
                    profile_id=profile.get("profile_id"),
                    message_time=profile.get("message_time"),
                    profile_key=profile.get("profile_key"),
                    profile_value=profile.get("profile_value")
                )
                if updated_node:
                    updated_count += 1
                    all_nodes.append(updated_node)

        self.memory_nodes.extend(all_nodes)

        message = f"Updated {updated_count} profile(s)."
        logger.info(message)
        return message
