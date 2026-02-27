"""Shell command execution tool.

This module provides an operation that can execute shell commands
asynchronously and return the output, error, and exit code.
"""

from ..op import BaseTool
from ..schema import ToolCall
from ..utils import run_shell_command


class ExecuteShell(BaseTool):
    """Operation for executing shell commands asynchronously.

    This operation takes a shell command as input, executes it asynchronously,
    and returns the stdout, stderr, and exit code in a formatted result.
    """

    def _build_tool_call(self) -> ToolCall:
        return ToolCall(
            **{
                "description": self.get_prompt("tool"),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "command": {
                            "type": "string",
                            "description": "command",
                        },
                    },
                    "required": ["command"],
                },
            },
        )

    async def execute(self):
        command: str = self.context.command
        stdout, stderr, return_code = await run_shell_command(command)
        result_parts = [
            f"Command: {command}",
            f"Output: {stdout if stdout else '(empty)'}",
            f"Error: {stderr if stderr else '(none)'}",
            f"Exit Code: {return_code if return_code is not None else '(none)'}",
        ]

        return "\n".join(result_parts)
