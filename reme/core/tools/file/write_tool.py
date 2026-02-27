"""Write tool for creating and overwriting files.

This module provides a tool for writing content to files with:
- Automatic parent directory creation
- File overwriting (creates if doesn't exist, overwrites if exists)
- Path resolution (relative to working directory)
"""

import os

from .base_file_tool import BaseFileTool
from ...schema import ToolCall


class WriteTool(BaseFileTool):
    """Tool for writing content to files.

    Features:
    - Creates file if it doesn't exist, overwrites if it does
    - Automatically creates parent directories
    - Supports both relative and absolute paths
    """

    def __init__(self, cwd: str | None = None):
        """Initialize write tool.

        Args:
            cwd: Working directory (defaults to current directory)
        """
        super().__init__()
        self.cwd = cwd or os.getcwd()

    def _build_tool_call(self) -> ToolCall:
        return ToolCall(
            **{
                "description": (
                    "Write content to a file. Creates the file if it doesn't exist, "
                    "overwrites if it does. Automatically creates parent directories."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Path to the file to write (relative or absolute)",
                        },
                        "content": {
                            "type": "string",
                            "description": "Content to write to the file",
                        },
                    },
                    "required": ["path", "content"],
                },
            },
        )

    async def execute(self) -> str:
        """Execute the write operation."""
        path: str = self.context.path
        content: str = self.context.content

        # Resolve path to absolute
        if not os.path.isabs(path):
            absolute_path = os.path.join(self.cwd, path)
        else:
            absolute_path = path

        absolute_path = os.path.normpath(absolute_path)

        # Create parent directories if needed
        parent_dir = os.path.dirname(absolute_path)
        if parent_dir:
            os.makedirs(parent_dir, exist_ok=True)

        # Write the file
        with open(absolute_path, "w", encoding="utf-8") as f:
            f.write(content)

        # Return success message
        content_bytes = len(content.encode("utf-8"))
        return f"Successfully wrote {content_bytes} bytes to {path}"
