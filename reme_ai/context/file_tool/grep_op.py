"""Grep text search operation module.

This module provides a tool operation for searching text patterns in files.
It enables efficient content-based search using regular expressions, with support
for glob pattern filtering and result limiting.
"""

from flowllm.core.context import C
from flowllm.core.schema import ToolCall
from flowllm.extensions.file_tool import GrepOp as FlowGrepOp


@C.register_op()
class GrepOp(FlowGrepOp):
    """Grep text search operation.

    This operation searches for text patterns in files using regular expressions.
    Supports glob pattern filtering and result limiting.
    """

    file_path = __file__

    def build_tool_call(self) -> ToolCall:
        """Build and return the tool call schema for this operator."""
        tool_params = {
            "name": "Grep",
            "description": self.get_prompt("tool_desc"),
            "input_schema": {
                "pattern": {
                    "type": "string",
                    "description": self.get_prompt("pattern"),
                    "required": True,
                },
                "path": {
                    "type": "string",
                    "description": self.get_prompt("path"),
                    "required": False,
                },
                "glob": {
                    "type": "string",
                    "description": self.get_prompt("glob"),
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
