"""Update user profile tool"""

from loguru import logger

from .profile_handler import ProfileHandler
from ..base_memory_tool import BaseMemoryTool
from ....core.schema import ToolCall


class UpdateProfilesV1(BaseMemoryTool):
    """Tool to update user profile by adding or removing profile entries"""

    def __init__(self, name="update_profiles", enable_memory_target: bool = False, **kwargs):
        kwargs["enable_multiple"] = True
        super().__init__(name=name, **kwargs)
        self.enable_memory_target: bool = enable_memory_target

    def _build_profile_parameters(self, include_profile_id: bool = False) -> dict:
        """Build the profile parameters schema based on enabled features."""
        properties = {}
        required = []

        if include_profile_id:
            properties["profile_id"] = {
                "type": "string",
                "description": "ID of the profile to update",
            }
            required.append("profile_id")

        properties.update(
            {
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
            },
        )
        required.extend(["message_time", "profile_key", "profile_value"])

        if self.enable_memory_target:
            properties["memory_target"] = {
                "type": "string",
                "description": "memory_target",
            }
            required.append("memory_target")

        return {
            "type": "object",
            "properties": properties,
            "required": required,
        }

    def _build_multiple_tool_call(self) -> ToolCall:
        """Build and return the multiple tool call schema"""
        return ToolCall(
            **{
                "description": "Update existing profiles and add new profiles.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "profiles_to_update": {
                            "type": "array",
                            "description": "List of profiles to update",
                            "items": self._build_profile_parameters(include_profile_id=True),
                        },
                        "profiles_to_add": {
                            "type": "array",
                            "description": "List of profiles to add",
                            "items": self._build_profile_parameters(include_profile_id=False),
                        },
                    },
                    "required": ["profiles_to_update", "profiles_to_add"],
                },
            },
        )

    async def execute(self):
        # Get parameters
        profiles_to_update = self.context.get("profiles_to_update", [])
        profiles_to_add = self.context.get("profiles_to_add", [])

        if not profiles_to_update and not profiles_to_add:
            return "No profiles to update or add, operation completed."

        # Step 1: Collect and delete all old profiles that need to be updated
        if profiles_to_update:
            # Group deletion IDs by memory_target if enabled
            if self.enable_memory_target:
                delete_by_target = {}
                for profile in profiles_to_update:
                    target = profile.get("memory_target", self.memory_target)
                    profile_id = profile.get("profile_id")
                    if profile_id:
                        if target not in delete_by_target:
                            delete_by_target[target] = []
                        delete_by_target[target].append(profile_id)
            else:
                delete_by_target = {
                    self.memory_target: [
                        profile.get("profile_id") for profile in profiles_to_update if profile.get("profile_id")
                    ],
                }

            # Delete old profiles for each target
            for target, profile_ids in delete_by_target.items():
                if profile_ids:
                    profile_ids = sorted(set(profile_ids))  # Remove duplicates and sort
                    profile_handler = ProfileHandler(profile_path=self.profile_path, memory_target=target)
                    profile_handler.delete(profile_ids)

        # Step 2: Prepare all profiles to add (both updated and new)
        all_profiles_to_add = []

        # Add profiles from updates
        if profiles_to_update:
            for profile in profiles_to_update:
                target = (
                    profile.get(
                        "memory_target",
                        self.memory_target,
                    )
                    if self.enable_memory_target
                    else self.memory_target
                )
                all_profiles_to_add.append((target, profile))

        # Add new profiles
        if profiles_to_add:
            for profile in profiles_to_add:
                target = (
                    profile.get(
                        "memory_target",
                        self.memory_target,
                    )
                    if self.enable_memory_target
                    else self.memory_target
                )
                all_profiles_to_add.append((target, profile))

        # Step 3: Group all profiles by target and add them in batch
        from collections import defaultdict

        profiles_by_target = defaultdict(list)
        for target, profile in all_profiles_to_add:
            profiles_by_target[target].append(profile)

        # Process each target and add profiles
        all_memory_nodes = []
        updated_count = len(profiles_to_update)
        added_count = len(profiles_to_add)

        for target, target_profiles in profiles_by_target.items():
            profile_handler = ProfileHandler(profile_path=self.profile_path, memory_target=target)
            new_nodes = profile_handler.add_batch(profiles=target_profiles, ref_memory_id=self.history_id)
            all_memory_nodes.extend(new_nodes)

        # Extend memory_nodes for tracking
        self.memory_nodes.extend(all_memory_nodes)

        # Build output message
        operations = []
        if updated_count > 0:
            operations.append(f"updated {updated_count} profiles.")
        if added_count > 0:
            operations.append(f"added {added_count} new profiles.")
        operations.append("Operation completed.")
        logger.info("\n".join(operations))
        return "\n".join(operations)
