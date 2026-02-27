"""Read file tool with smart truncation and image support.

Features:
- Reads text files with offset/limit support
- Detects and handles image files (jpg, png, gif, webp)
- Smart truncation to prevent memory issues
"""

import os
from pathlib import Path

from .base_file_tool import BaseFileTool
from .truncate import DEFAULT_MAX_BYTES, DEFAULT_MAX_LINES, format_size, truncate_head
from ...schema import ToolCall

# Supported image extensions
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".webp"}


def is_image_file(path: str) -> bool:
    """Check if file is a supported image type.

    Args:
        path: File path to check

    Returns:
        True if file is a supported image
    """
    return Path(path).suffix.lower() in IMAGE_EXTENSIONS


class ReadTool(BaseFileTool):
    """Read file contents with smart truncation.

    Features:
    - Supports text files and images (jpg, png, gif, webp)
    - Smart truncation for large files
    - Offset/limit for reading specific portions
    """

    def __init__(self, cwd: str | None = None):
        """Initialize read tool.

        Args:
            cwd: Working directory (defaults to current directory)
        """
        super().__init__()
        self.cwd = cwd or os.getcwd()

    def _build_tool_call(self) -> ToolCall:
        max_kb = DEFAULT_MAX_BYTES // 1024
        return ToolCall(
            **{
                "description": (
                    f"Read the contents of a file. Supports text files and images "
                    f"(jpg, png, gif, webp). Images are sent as attachments. For text files, "
                    f"output is truncated to {DEFAULT_MAX_LINES} lines or {max_kb}KB "
                    f"(whichever is hit first). Use offset/limit for large files. "
                    f"When you need the full file, continue with offset until complete."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Path to the file to read (relative or absolute)",
                        },
                        "offset": {
                            "type": "number",
                            "description": "Line number to start reading from (1-indexed)",
                        },
                        "limit": {
                            "type": "number",
                            "description": "Maximum number of lines to read",
                        },
                    },
                    "required": ["path"],
                },
            },
        )

    async def execute(self) -> str:
        """Execute the read operation."""
        path: str = self.context.path
        offset: int | None = self.context.get("offset", None)
        limit: int | None = self.context.get("limit", None)

        # Resolve path
        if not os.path.isabs(path):
            absolute_path = os.path.join(self.cwd, path)
        else:
            absolute_path = path
        absolute_path = os.path.normpath(absolute_path)

        # Check file exists and is readable
        if not os.path.exists(absolute_path):
            raise ValueError(f"File not found: {path}")

        if not os.path.isfile(absolute_path):
            raise ValueError(f"Not a file: {path}")

        if not os.access(absolute_path, os.R_OK):
            raise ValueError(f"File not readable: {path}")

        # Check if image
        if is_image_file(absolute_path):
            return await self._read_image(absolute_path, path)
        else:
            return await self._read_text(absolute_path, path, offset, limit)

    @staticmethod
    async def _read_image(absolute_path: str, display_path: str) -> str:
        """Read and return image file information.

        Args:
            absolute_path: Absolute path to image
            display_path: Path to display to user

        Returns:
            Image information text
        """
        # Get file size
        file_size = os.path.getsize(absolute_path)
        file_ext = Path(absolute_path).suffix.lower()

        # For Python tools, we typically can't return image data directly to LLM
        # So we return a descriptive message
        return (
            f"Read image file [{file_ext}]\n"
            f"Path: {display_path}\n"
            f"Size: {format_size(file_size)}\n"
            f"Note: Image content cannot be displayed in text format. "
            f"Use bash tool or other methods to process the image."
        )

    @staticmethod
    async def _read_text(
        absolute_path: str,
        _display_path: str,
        offset: int | None,
        limit: int | None,
    ) -> str:
        """Read text file with smart truncation.

        Args:
            absolute_path: Absolute path to file
            _display_path: Path to display to user
            offset: Starting line (1-indexed)
            limit: Maximum lines to read

        Returns:
            File contents with truncation notices
        """
        # Read file
        try:
            with open(absolute_path, "r", encoding="utf-8") as f:
                content = f.read()
        except UnicodeDecodeError:
            # Try with error handling for binary files
            with open(absolute_path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()

        all_lines = content.split("\n")
        total_file_lines = len(all_lines)

        # Validate and apply offset (1-indexed to 0-indexed)
        if offset is not None:
            if offset < 1:
                raise ValueError(f"offset must be >= 1, got {offset}")
            start_line = offset - 1
        else:
            start_line = 0

        start_line_display = start_line + 1

        # Check offset bounds
        if start_line >= total_file_lines:
            raise IndexError(
                f"Offset {offset} is beyond end of file ({total_file_lines} lines total)",
            )

        # Validate and apply limit
        if limit is not None:
            if limit <= 0:
                raise ValueError(f"limit must be positive, got {limit}")
            end_line = min(start_line + limit, total_file_lines)
        else:
            end_line = total_file_lines

        # Extract selected lines
        selected_content = "\n".join(all_lines[start_line:end_line])

        # Apply truncation
        truncation = truncate_head(selected_content)

        # Build output with truncation notices
        if truncation.truncated:
            # Truncation occurred
            end_line_display = start_line_display + truncation.output_lines - 1
            next_offset = end_line_display + 1

            output_text = truncation.content

            if truncation.truncated_by == "lines":
                output_text += (
                    f"\n\n[Showing lines {start_line_display}-{end_line_display} "
                    f"of {total_file_lines}. Use offset={next_offset} to continue.]"
                )
            else:
                max_kb = DEFAULT_MAX_BYTES // 1024
                output_text += (
                    f"\n\n[Showing lines {start_line_display}-{end_line_display} "
                    f"of {total_file_lines} ({max_kb}KB limit). "
                    f"Use offset={next_offset} to continue.]"
                )
        elif end_line < total_file_lines:
            # User limit reached but no truncation
            remaining = total_file_lines - end_line
            next_offset = end_line + 1

            output_text = truncation.content
            output_text += f"\n\n[{remaining} more lines in file. " f"Use offset={next_offset} to continue.]"
        else:
            # No truncation or limit
            output_text = truncation.content

        return output_text
