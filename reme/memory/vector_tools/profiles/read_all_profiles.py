"""Read user profile tool"""

from loguru import logger

from .profile_handler import ProfileHandler
from ..base_memory_tool import BaseMemoryTool
from ....core.schema import ToolCall


class ReadAllProfiles(BaseMemoryTool):
    """Tool to read all user profiles"""

    def __init__(self, enable_memory_target: bool = False, **kwargs):
        kwargs["enable_multiple"] = False
        super().__init__(**kwargs)
        self.enable_memory_target: bool = enable_memory_target

    def _build_tool_call(self) -> ToolCall:
        """Build and return the tool call schema"""
        properties = {}
        required = []

        if self.enable_memory_target:
            properties["memory_target"] = {
                "type": "string",
                "description": "memory_target",
            }
            required.append("memory_target")

        return ToolCall(
            **{
                "description": "Read all user profiles.",
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                },
            },
        )

    async def execute(self):
        if self.enable_memory_target:
            target = self.context.get("memory_target")
        else:
            target = self.memory_target

        profile_handler = ProfileHandler(profile_path=self.profile_path, memory_target=target)
        profiles_str = profile_handler.read_all(add_profile_id=True)
        if not profiles_str:
            output = "No profiles found."
            logger.info(output)
            return output

        logger.info("Successfully read profiles")
        return profiles_str
