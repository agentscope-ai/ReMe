"""Write file operation module.

This module provides a tool operation for writing content to files.
It supports creating new files or overwriting existing files, and automatically
creates parent directories if they don't exist.
"""

from pathlib import Path

from flowllm.core.context import C
from flowllm.core.op import BaseAsyncToolOp
from flowllm.core.schema import ToolCall


@C.register_op()
class WriteFileOp(BaseAsyncToolOp):
    """Write file operation.

    This operation writes content to a specified file. If the file doesn't exist,
    it will be created. If parent directories don't exist, they will be created automatically.
    """

    file_path = __file__

    def __init__(self, **kwargs):
        kwargs.setdefault("raise_exception", False)
        super().__init__(**kwargs)

    def build_tool_call(self) -> ToolCall:
        """Build and return the tool call schema for this operator."""
        tool_params = {
            "name": "WriteFile",
            "description": self.get_prompt("tool_desc"),
            "input_schema": {
                "file_path": {
                    "type": "string",
                    "description": self.get_prompt("file_path"),
                    "required": True,
                },
                "content": {
                    "type": "string",
                    "description": self.get_prompt("content"),
                    "required": True,
                },
            },
        }

        return ToolCall(**tool_params)

    async def async_execute(self):
        """Execute the write file operation."""
        file_path: str = self.context.get("file_path", "").strip()
        content: str = self.context.get("content", "")

        # Validate file_path
        if not file_path:
            raise ValueError("The 'file_path' parameter cannot be empty.")

        # Resolve file path
        file_path_obj = Path(file_path).expanduser().resolve()

        # Check if path is a directory
        if file_path_obj.exists() and file_path_obj.is_dir():
            raise ValueError(f"Path is a directory, not a file: {file_path_obj}")

        # Create parent directories if they don't exist
        file_path_obj.parent.mkdir(parents=True, exist_ok=True)

        # Check if file exists
        file_exists = file_path_obj.exists() and file_path_obj.is_file()

        # Write content to file
        file_path_obj.write_text(content, encoding="utf-8")

        # Format success message
        if file_exists:
            result = f"Successfully overwrote file: {file_path_obj}"
        else:
            result = f"Successfully created and wrote to new file: {file_path_obj}"

        self.set_output(result)

    async def async_default_execute(self, e: Exception = None, **_kwargs):
        """Fill outputs with a default failure message when execution fails."""
        file_path: str = self.context.get("file_path", "").strip()
        error_msg = f'Failed to write file "{file_path}"'
        if e:
            error_msg += f": {str(e)}"
        self.set_output(error_msg)
