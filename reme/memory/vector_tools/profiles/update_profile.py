"""Update user profile tool"""

from loguru import logger

from .profile_handler import ProfileHandler
from ..base_memory_tool import BaseMemoryTool
from ....core.schema import ToolCall


class UpdateProfile(BaseMemoryTool):
    """Tool to update user profile by adding or removing profile entries"""

    def __init__(self, enable_memory_target: bool = False, **kwargs):
        kwargs["enable_multiple"] = True
        super().__init__(**kwargs)
        self.enable_memory_target: bool = enable_memory_target

    def _build_multiple_tool_call(self) -> ToolCall:
        """Build and return the multiple tool call schema"""
        profile_properties = {
            "message_time": {
                "type": "string",
                "description": "Message time, e.g. '2020-01-01 00:00:00'",
            },
            "profile_key": {
                "type": "string",
                "description": "Profile key or category, e.g. 'name'",
            },
            "profile_value": {
                "type": "string",
                "description": "Profile value or content, e.g. 'John Smith'",
            },
        }
        profile_required = ["message_time", "profile_key", "profile_value"]

        if self.enable_memory_target:
            profile_properties["memory_target"] = {
                "type": "string",
                "description": "memory_target",
            }
            profile_required.append("memory_target")

        return ToolCall(
            **{
                "description": "update user profile by removing and adding profile entries.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "profile_ids_to_delete": {
                            "type": "array",
                            "description": "List of profile IDs to delete",
                            "items": {
                                "type": "string",
                            },
                        },
                        "profiles_to_add": {
                            "type": "array",
                            "description": "List of profiles to add",
                            "items": {
                                "type": "object",
                                "properties": profile_properties,
                                "required": profile_required,
                            },
                        },
                    },
                    "required": ["profile_ids_to_delete", "profiles_to_add"],
                },
            },
        )

    async def execute(self):
        # Get parameters
        profile_ids_to_delete = self.context.get("profile_ids_to_delete", [])
        profile_ids_to_delete = sorted({pid for pid in profile_ids_to_delete if pid})
        profiles_to_add = self.context.get("profiles_to_add", [])

        if not profile_ids_to_delete and not profiles_to_add:
            return "No profiles to remove or add, operation completed."

        removed_count = 0
        added_count = 0

        # Delete profiles (using self.memory_target)
        if profile_ids_to_delete:
            profile_handler = ProfileHandler(profile_path=self.profile_path, memory_target=self.memory_target)
            removed_count = profile_handler.delete(profile_ids_to_delete)

        # Add new profiles
        if profiles_to_add:
            if self.enable_memory_target:
                # Group profiles by memory_target
                from collections import defaultdict

                profiles_by_target = defaultdict(list)
                for profile in profiles_to_add:
                    target = profile.get("memory_target", self.memory_target)
                    profiles_by_target[target].append(profile)

                # Add profiles for each target
                for target, target_profiles in profiles_by_target.items():
                    profile_handler = ProfileHandler(profile_path=self.profile_path, memory_target=target)
                    new_nodes = profile_handler.add_batch(profiles=target_profiles, ref_memory_id=self.history_id)
                    self.memory_nodes.extend(new_nodes)
                    added_count += len(new_nodes)
            else:
                # Use self.memory_target for all profiles
                profile_handler = ProfileHandler(profile_path=self.profile_path, memory_target=self.memory_target)
                new_nodes = profile_handler.add_batch(profiles=profiles_to_add, ref_memory_id=self.history_id)
                self.memory_nodes.extend(new_nodes)
                added_count = len(new_nodes)

        # Build output message
        operations = []
        if removed_count > 0:
            operations.append(f"removed {removed_count} old profiles.")
        if added_count > 0:
            operations.append(f"added {added_count} new profiles.")
        operations.append("Operation completed.")
        logger.info("\n".join(operations))
        return "\n".join(operations)
