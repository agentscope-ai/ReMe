"""Directory listing tool with truncation support."""

import os
from pathlib import Path

from .base_file_tool import BaseFileTool
from .truncate import DEFAULT_MAX_BYTES, truncate_head
from ...schema import ToolCall

DEFAULT_LIMIT = 500


class LsTool(BaseFileTool):
    """List directory contents with smart truncation.

    Features:
    - Returns entries sorted alphabetically (case-insensitive)
    - Directory indicators ('/' suffix)
    - Includes dotfiles
    - Entry count limiting
    - Byte truncation
    """

    def __init__(self, cwd: str | None = None):
        """Initialize ls tool.

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
                    f"List directory contents. Returns entries sorted alphabetically, "
                    f"with '/' suffix for directories. Includes dotfiles. Output is truncated "
                    f"to {DEFAULT_LIMIT} entries or {max_kb}KB (whichever is hit first)."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Directory to list (default: current directory)",
                        },
                        "limit": {
                            "type": "number",
                            "description": f"Maximum number of entries to return (default: {DEFAULT_LIMIT})",
                        },
                    },
                    "required": [],
                },
            },
        )

    async def execute(self) -> str:
        """List directory contents with production-grade features."""
        path: str | None = self.context.get("path", None)
        limit: int | None = self.context.get("limit", None)

        # Resolve directory path
        dir_path = Path(self.cwd) / (path or ".")
        dir_path = dir_path.resolve()
        effective_limit = limit if limit is not None else DEFAULT_LIMIT

        # Check if path exists
        if not dir_path.exists():
            raise FileNotFoundError(f"Path not found: {dir_path}")

        # Check if path is a directory
        if not dir_path.is_dir():
            raise NotADirectoryError(f"Not a directory: {dir_path}")

        # Read directory entries
        entries = list(dir_path.iterdir())

        # Sort alphabetically (case-insensitive)
        entries.sort(key=lambda e: e.name.lower())

        # Format entries with directory indicators
        results: list[str] = []
        entry_limit_reached = False

        for entry in entries:
            if len(results) >= effective_limit:
                entry_limit_reached = True
                break

            try:
                # Add '/' suffix for directories
                suffix = "/" if entry.is_dir() else ""
                results.append(entry.name + suffix)
            except Exception:
                # Skip entries we can't stat
                continue

        # Handle empty directory
        if len(results) == 0:
            return "(empty directory)"

        # Apply byte truncation
        raw_output = "\n".join(results)
        truncation_result = truncate_head(raw_output, max_lines=float("inf"))

        output_text = truncation_result.content

        # Build notices
        notices: list[str] = []

        if entry_limit_reached:
            notices.append(
                f"{effective_limit} entries limit reached. Use limit={effective_limit * 2} for more",
            )

        if truncation_result.truncated:
            max_kb = DEFAULT_MAX_BYTES // 1024
            notices.append(f"{max_kb}KB limit reached")

        if notices:
            output_text += f"\n\n[{'. '.join(notices)}]"

        return output_text
