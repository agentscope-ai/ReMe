"""Code execution tool for running Python code dynamically.

This module provides an operation that can execute Python code strings
and return the output or error messages.
"""

from ..op import BaseTool
from ..schema import ToolCall
from ..utils import exec_code, async_exec_code


class ExecuteCode(BaseTool):
    """Operation for executing Python code dynamically.

    This operation takes Python code as input, executes it in a safe context,
    and returns the output or any error messages that occur during execution.
    """

    def _build_tool_call(self) -> ToolCall:
        return ToolCall(
            **{
                "description": self.get_prompt("tool"),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "code": {
                            "type": "string",
                            "description": "code",
                        },
                    },
                    "required": ["code"],
                },
            },
        )

    async def execute(self):
        return await async_exec_code(self.context.code)

    def execute_sync(self):
        return exec_code(self.context.code)
