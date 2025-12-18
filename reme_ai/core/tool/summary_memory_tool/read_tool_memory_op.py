"""Read tool memory operation for retrieving tool-specific memories.

This module provides the ReadToolMemoryOp class for reading tool memories
using file-based storage with CacheHandler.
"""

from ..base_memory_tool_op import BaseMemoryToolOp
from ... import C


@C.register_op()
class ReadToolMemoryOp(BaseMemoryToolOp):
    """Operation for reading tool memories using file-based storage."""

    def __init__(self, **kwargs):
        kwargs["enable_multiple"] = False
        super().__init__(**kwargs)

    def build_input_schema(self) -> dict:
        """Build input schema for tool memory read.

        Returns:
            dict: Input schema for reading a tool memory.
        """
        return {
            "tool_name": {
                "type": "string",
                "description": self.get_prompt("tool_name"),
                "required": True,
            },
        }

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
        """Execute the read tool memory operation.

        Reads tool memory from file storage.
        """
        workspace_id: str = self.workspace_id

        tool_name = self.input_dict.get("tool_name", "")

        if not tool_name:
            self.set_output("No tool name provided.")
            return

        tool_memory = self._load_tool_memory(tool_name, workspace_id)
        if tool_memory:
            self.set_output(f"Tool memory for '{tool_name}':\n{tool_memory}")
        else:
            self.set_output(f"No memory found for tool '{tool_name}' in workspace={workspace_id}.")

