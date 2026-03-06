"""Add draft profile and read all profiles from local storage"""

from loguru import logger

from .profile_handler import ProfileHandler
from ..base_memory_tool import BaseMemoryTool
from ....core.schema import ToolCall


class AddDraftAndReadAllProfiles(BaseMemoryTool):
    """Tool to add draft profile and read all profiles"""

    def __init__(self, enable_memory_target: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.enable_memory_target: bool = enable_memory_target

    def _build_query_parameters(self) -> dict:
        """Build the query parameters schema"""
        properties = {
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
        required = ["message_time", "profile_key", "profile_value"]

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

    def _build_tool_call(self) -> ToolCall:
        return ToolCall(
            **{
                "description": "Add draft profile and read all profiles from local storage.",
                "parameters": self._build_query_parameters(),
            },
        )

    def _build_multiple_tool_call(self) -> ToolCall:
        return ToolCall(
            **{
                "description": "Add draft profile and read all profiles from local storage.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "draft_items": {
                            "type": "array",
                            "description": "List of draft profile items.",
                            "items": self._build_query_parameters(),
                        },
                    },
                    "required": ["draft_items"],
                },
            },
        )

    async def execute(self):
        if self.enable_multiple:
            draft_items = self.context.get("draft_items", [])
        else:
            draft_items = [self.context]

        # Collect all profiles from all targets
        all_profiles = []
        targets_processed = set()

        for item in draft_items:
            if self.enable_memory_target:
                target = item["memory_target"]
            else:
                target = self.memory_target

            # Skip if already processed this target
            if target in targets_processed:
                continue
            targets_processed.add(target)

            profile_handler = ProfileHandler(profile_path=self.profile_path, memory_target=target)

            profiles_str = profile_handler.read_all(add_profile_id=True)
            if profiles_str:
                all_profiles.append(f"## Profiles for {target}:\n{profiles_str}")

        if not all_profiles:
            output = "No profiles found."
            logger.info(output)
            return output

        output = "\n\n".join(all_profiles)
        logger.info(f"Successfully read profiles for {len(targets_processed)} target(s)")
        return output
