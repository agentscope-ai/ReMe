"""Add user profile tool"""

from loguru import logger

from .profile_handler import ProfileHandler
from ..base_memory_tool import BaseMemoryTool
from ....core.schema import ToolCall


class AddProfile(BaseMemoryTool):
    """Tool to add a single profile entry"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _build_tool_call(self) -> ToolCall:
        """Build and return the tool call schema"""
        return ToolCall(
            **{
                "description": "Add a new profile entry for the user.",
                "parameters": {
                    "type": "object",
                    "properties": {
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
                    "required": ["message_time", "profile_key", "profile_value"],
                },
            },
        )

    async def execute(self):
        profile_handler = ProfileHandler(profile_path=self.profile_path, memory_target=self.memory_target)

        # Get parameters
        message_time = self.context.get("message_time", "")
        profile_key = self.context.get("profile_key", "")
        profile_value = self.context.get("profile_value", "")

        if not profile_key or not profile_value:
            return "Missing required parameters (profile_key or profile_value), operation cancelled."

        # Build profile dict
        profile = {
            "message_time": message_time,
            "profile_key": profile_key,
            "profile_value": profile_value,
        }

        # Add profile using ProfileHandler
        new_nodes = profile_handler.add_batch(profiles=[profile], ref_memory_id=self.history_id)
        self.memory_nodes.extend(new_nodes)

        if new_nodes:
            output = f"Successfully added profile: [{profile_key}] = {profile_value}"
            logger.info(output)
            return output
        else:
            output = "Failed to add profile."
            logger.warning(output)
            return output
