"""Grep tool for searching file contents using ripgrep.

This module provides a tool for searching file contents with:
- Pattern matching (regex or literal string)
- Smart output truncation (prevents memory issues)
- Context lines support
- Respects .gitignore
"""

import asyncio
import json
import os
import shutil
from pathlib import Path

from .base_file_tool import BaseFileTool
from .truncate import (
    DEFAULT_MAX_BYTES,
    GREP_MAX_LINE_LENGTH,
    format_size,
    truncate_head,
    truncate_line,
)
from ...schema import ToolCall

# Default limits
DEFAULT_LIMIT = 100  # Maximum number of matches


class GrepTool(BaseFileTool):
    """Tool for searching file contents using ripgrep.

    Features:
    - Pattern matching with regex or literal string
    - Context lines support
    - Smart output truncation
    - Respects .gitignore
    """

    def __init__(self, cwd: str | None = None):
        """Initialize grep tool.

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
                    f"Search file contents for a pattern. Returns matching lines with "
                    f"file paths and line numbers. Respects .gitignore. Output is "
                    f"truncated to {DEFAULT_LIMIT} matches or {max_kb}KB (whichever is "
                    f"hit first). Long lines are truncated to {GREP_MAX_LINE_LENGTH} chars."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "pattern": {
                            "type": "string",
                            "description": "Search pattern (regex or literal string)",
                        },
                        "path": {
                            "type": "string",
                            "description": "Directory or file to search (default: current directory)",
                        },
                        "glob": {
                            "type": "string",
                            "description": "Filter files by glob pattern, e.g. '*.ts' or '**/*.spec.ts'",
                        },
                        "ignoreCase": {
                            "type": "boolean",
                            "description": "Case-insensitive search (default: false)",
                        },
                        "literal": {
                            "type": "boolean",
                            "description": "Treat pattern as literal string instead of regex (default: false)",
                        },
                        "contextLines": {
                            "type": "number",
                            "description": "Number of lines to show before and after each match (default: 0)",
                        },
                        "limit": {
                            "type": "number",
                            "description": f"Maximum number of matches to return (default: {DEFAULT_LIMIT})",
                        },
                    },
                    "required": ["pattern"],
                },
            },
        )

    async def execute(self) -> str:
        """Execute the grep search."""
        pattern: str = self.context.pattern
        search_path: str = self.context.get("path", ".")
        glob: str | None = self.context.get("glob", None)
        ignore_case: bool = self.context.get("ignoreCase", False)
        literal: bool = self.context.get("literal", False)
        context_lines: int = self.context.get("contextLines", 0)
        limit: int = self.context.get("limit", DEFAULT_LIMIT)

        # Check if ripgrep is available
        rg_path = shutil.which("rg")
        if not rg_path:
            raise RuntimeError(
                "ripgrep (rg) is not available. Please install it:\n"
                "  macOS: brew install ripgrep\n"
                "  Ubuntu: apt-get install ripgrep\n"
                "  Other: https://github.com/BurntSushi/ripgrep",
            )

        # Resolve search path
        if not os.path.isabs(search_path):
            search_path = os.path.join(self.cwd, search_path)

        # Check if path exists
        if not Path(search_path).exists():
            raise FileNotFoundError(f"Path not found: {search_path}")

        is_directory = Path(search_path).is_dir()
        effective_limit = max(1, limit)

        # Build ripgrep arguments
        args = [
            rg_path,
            "--json",
            "--line-number",
            "--color=never",
            "--hidden",
        ]

        if ignore_case:
            args.append("--ignore-case")

        if literal:
            args.append("--fixed-strings")

        if glob:
            args.extend(["--glob", glob])

        args.extend([pattern, search_path])

        # Execute ripgrep
        process = await asyncio.create_subprocess_exec(
            *args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=self.cwd,
        )

        stdout, stderr = await process.communicate()

        # Parse JSON output
        matches = []
        match_count = 0
        lines_truncated = False

        for line in stdout.decode("utf-8", errors="ignore").splitlines():
            if not line.strip() or match_count >= effective_limit:
                break

            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue

            if event.get("type") == "match":
                match_count += 1
                data = event.get("data", {})
                file_path = data.get("path", {}).get("text", "")
                line_number = data.get("line_number", 0)

                if file_path and line_number:
                    matches.append({"file_path": file_path, "line_number": line_number})

                if match_count >= effective_limit:
                    break

        # Check for errors
        if process.returncode not in (0, 1) and match_count == 0:
            error_msg = stderr.decode("utf-8", errors="ignore").strip()
            if error_msg:
                raise RuntimeError(error_msg)
            raise RuntimeError(f"ripgrep exited with code {process.returncode}")

        # No matches found
        if match_count == 0:
            return "No matches found"

        # Format matches with context
        output_lines = []
        file_cache = {}

        for match in matches:
            file_path = match["file_path"]
            line_number = match["line_number"]

            # Read file if not cached
            if file_path not in file_cache:
                try:
                    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                        file_cache[file_path] = f.read().replace("\r\n", "\n").replace("\r", "\n").split("\n")
                except Exception:
                    file_cache[file_path] = []

            lines = file_cache[file_path]

            # Format relative path
            if is_directory:
                relative_path = os.path.relpath(file_path, search_path)
                if not relative_path.startswith(".."):
                    display_path = relative_path.replace("\\", "/")
                else:
                    display_path = os.path.basename(file_path)
            else:
                display_path = os.path.basename(file_path)

            # Generate context block
            if not lines:
                output_lines.append(f"{display_path}:{line_number}: (unable to read file)")
                continue

            context_value = max(0, context_lines)
            start = max(1, line_number - context_value) if context_value > 0 else line_number
            end = min(len(lines), line_number + context_value) if context_value > 0 else line_number

            for current in range(start, end + 1):
                if current < 1 or current > len(lines):
                    continue

                line_text = lines[current - 1]
                is_match_line = current == line_number

                # Truncate long lines
                truncated_text, was_truncated = truncate_line(line_text)
                if was_truncated:
                    lines_truncated = True

                if is_match_line:
                    output_lines.append(f"{display_path}:{current}: {truncated_text}")
                else:
                    output_lines.append(f"{display_path}-{current}- {truncated_text}")

        # Apply byte truncation
        raw_output = "\n".join(output_lines)
        truncation = truncate_head(raw_output, max_lines=999999999)

        output = truncation.content
        notices = []

        # Add notices
        if match_count >= effective_limit:
            notices.append(
                f"{effective_limit} matches limit reached. "
                f"Use limit={effective_limit * 2} for more, or refine pattern",
            )

        if truncation.truncated:
            notices.append(f"{format_size(DEFAULT_MAX_BYTES)} limit reached")

        if lines_truncated:
            notices.append(
                f"Some lines truncated to {GREP_MAX_LINE_LENGTH} chars. " f"Use read tool to see full lines",
            )

        if notices:
            output += f"\n\n[{'. '.join(notices)}]"

        return output
