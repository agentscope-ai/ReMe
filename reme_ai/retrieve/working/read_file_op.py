"""Read file operation module.

This module provides a tool operation for reading file contents.
It supports reading entire files or specific line ranges for large files.
"""

from pathlib import Path
from typing import Optional

from flowllm.core.context import C
from flowllm.core.op import BaseAsyncToolOp
from flowllm.core.schema import ToolCall
from reme_ai.utils.op_utils import run_shell_command


@C.register_op()
class ReadFileOp(BaseAsyncToolOp):
    """Read file operation.

    This operation reads and returns the content of a specified file.
    For text files, it can read specific line ranges using offset and limit.
    """

    file_path = __file__

    def __init__(self, **kwargs):
        kwargs.setdefault("raise_exception", False)
        super().__init__(**kwargs)

    def build_tool_call(self) -> ToolCall:
        """Build and return the tool call schema for this operator."""
        tool_params = {
            "name": "ReadFile",
            "description": self.get_prompt("tool_desc"),
            "input_schema": {
                "file_path": {
                    "type": "string",
                    "description": self.get_prompt("file_path"),
                    "required": True,
                },
                "offset": {
                    "type": "number",
                    "description": self.get_prompt("offset"),
                    "required": True,
                },
                "limit": {
                    "type": "number",
                    "description": self.get_prompt("limit"),
                    "required": True,
                },
            },
        }

        return ToolCall(**tool_params)

    async def async_execute(self):
        """Execute the read file operation."""
        file_path: str = self.context.get("file_path", "").strip()
        offset: Optional[int] = int(self.context.get("offset"))
        limit: Optional[int] = int(self.context.get("limit"))

        # Validate and resolve file path
        assert file_path, "The 'file_path' parameter cannot be empty."
        file_path_obj = Path(file_path).expanduser().resolve()
        assert file_path_obj.exists(), f"File not found: {file_path_obj}"
        assert file_path_obj.is_file(), f"Path is not a file: {file_path_obj}"

        # Set default values and validate
        offset = offset or 0
        limit = limit or 1000000
        assert offset >= 0, "Offset must be a non-negative number"
        assert limit > 0, "Limit must be a positive number"

        # Use sed for efficient line reading (1-indexed)
        start_line = offset + 1
        end_line = offset + limit

        cmd = ["sed", "-n", f"{start_line},{end_line}p", str(file_path_obj)]
        stdout, stderr, returncode = await run_shell_command(cmd, timeout=30)

        assert returncode == 0, f"sed command failed: {stderr}"
        content = stdout.rstrip("\n")
        self.set_output(content)

    async def async_default_execute(self, e: Exception = None, **_kwargs):
        """Fill outputs with a default failure message when execution fails."""
        file_path: str = self.context.get("file_path", "").strip()
        error_msg = f'Failed to read file "{file_path}"'
        if e:
            error_msg += f": {str(e)}"
        self.set_output(error_msg)
