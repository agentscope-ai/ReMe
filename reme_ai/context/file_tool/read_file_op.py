"""Read file operation module.

This module provides a tool operation for reading file contents.
It supports reading entire files or specific line ranges for large files.
"""

from flowllm.core.context import C
from flowllm.core.schema import ToolCall
from flowllm.extensions.file_tool import ReadFileOp as FlowReadFileOp


@C.register_op()
class ReadFileOp(FlowReadFileOp):
    """Read file operation.

    This operation reads and returns the content of a specified file.
    For text files, it can read specific line ranges using offset and limit.
    """

    file_path = __file__

    def build_tool_call(self) -> ToolCall:
        """Build and return the tool call schema for this operator."""
        tool_params = {
            "name": "ReadFile",
            "description": self.get_prompt("tool_desc"),
            "input_schema": {
                "absolute_path": {
                    "type": "string",
                    "description": self.get_prompt("absolute_path"),
                    "required": True,
                },
                "offset": {
                    "type": "number",
                    "description": self.get_prompt("offset"),
                    "required": False,
                },
                "limit": {
                    "type": "number",
                    "description": self.get_prompt("limit"),
                    "required": False,
                },
            },
        }

        return ToolCall(**tool_params)
