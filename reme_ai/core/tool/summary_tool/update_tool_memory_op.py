"""Update tool memory operation for saving tool-specific memories.

This module provides the UpdateToolMemoryOp class for updating tool memories
using file-based storage with CacheHandler.
"""

from ..base_memory_tool_op import BaseMemoryToolOp
from ... import C


@C.register_op()
class UpdateToolMemoryOp(BaseMemoryToolOp):
    """Operation for updating tool memories using file-based storage."""

    def __init__(self, **kwargs):
        kwargs["enable_multiple"] = False
        super().__init__(**kwargs)

    def build_input_schema(self) -> dict:
        """Build input schema for tool memory update.

        Returns:
            dict: Input schema for updating a tool memory.
        """
        return {
            "tool_name": {
                "type": "string",
                "description": self.get_prompt("tool_name"),
                "required": True,
            },
            "tool_memory": {
                "type": "string",
                "description": self.get_prompt("tool_memory"),
                "required": True,
            },
        }

    def _save_tool_memory(self, tool_name: str, tool_memory: str, workspace_id: str) -> bool:
        """Save tool memory to file.

        Args:
            tool_name: The name of the tool.
            tool_memory: The memory content for the tool.
            workspace_id: The workspace ID.

        Returns:
            bool: Whether the save was successful.
        """
        # Get metadata handler for the workspace tool directory
        metadata_handler = self.get_metadata_handler(f"{workspace_id}/tool")
        return self.save_metadata_value(metadata_handler, tool_name, tool_memory)

    def _load_tool_memory(self, tool_name: str, workspace_id: str) -> str:
        """Load tool memory from file.

        Args:
            tool_name: The name of the tool.
            workspace_id: The workspace ID.

        Returns:
            str: The tool memory content, or empty string if not found.
        """
        metadata_handler = self.get_metadata_handler(f"{workspace_id}/tool")
        result = self.load_metadata_value(metadata_handler, tool_name)
        return result if result is not None else ""

    async def async_execute(self):
        """Execute the update tool memory operation.

        Updates tool memory to file storage.
        """
        workspace_id: str = self.workspace_id

        tool_name = self.input_dict.get("tool_name", "")
        tool_memory = self.input_dict.get("tool_memory", "")

        if not tool_name or not tool_memory:
            self.set_output("No valid tool memory provided for update.")
            return

        self._save_tool_memory(tool_name, tool_memory, workspace_id)
        self.set_output(f"Successfully updated tool memory for '{tool_name}' in workspace={workspace_id}.")

