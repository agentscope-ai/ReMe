"""Delete user profile tool"""

from loguru import logger

from .profile_handler import ProfileHandler
from ..base_memory_tool import BaseMemoryTool
from ....core.schema import ToolCall


class DeleteProfile(BaseMemoryTool):
    """Tool to delete a single profile entry by ID"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _build_tool_call(self) -> ToolCall:
        """Build and return the tool call schema"""
        return ToolCall(
            **{
                "description": "Delete a profile entry by profile ID.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "profile_id": {
                            "type": "string",
                            "description": "The unique ID of the profile to delete.",
                        },
                    },
                    "required": ["profile_id"],
                },
            },
        )

    async def execute(self):
        profile_handler = ProfileHandler(profile_path=self.profile_path, memory_target=self.memory_target)

        # Get profile_id parameter
        profile_id = self.context.get("profile_id", "")

        if not profile_id:
            return "No profile_id provided, operation cancelled."

        # Delete profile using ProfileHandler
        success = profile_handler.delete(profile_id)

        if success:
            output = f"Successfully deleted profile with ID: {profile_id}"
            logger.info(output)
            return output
        else:
            output = f"Profile with ID '{profile_id}' not found."
            logger.warning(output)
            return output
