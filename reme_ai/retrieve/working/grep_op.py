"""Grep text search operation module.

This module provides a tool operation for searching text patterns in files.
It enables efficient content-based search using regular expressions, with support
for glob pattern filtering and result limiting.
"""

import re
from pathlib import Path

from flowllm.core.context import C
from flowllm.core.op import BaseAsyncToolOp
from flowllm.core.schema import ToolCall
from loguru import logger


@C.register_op()
class GrepOp(BaseAsyncToolOp):
    """Grep text search operation.

    This operation searches for text patterns in files using regular expressions.
    Supports glob pattern filtering and result limiting.
    """

    file_path = __file__

    def __init__(self, **kwargs):
        kwargs.setdefault("raise_exception", False)
        super().__init__(**kwargs)

    def build_tool_call(self) -> ToolCall:
        """Build and return the tool call schema for this operator."""
        tool_params = {
            "name": "Grep",
            "description": self.get_prompt("tool_desc"),
            "input_schema": {
                "file_path": {
                    "type": "string",
                    "description": self.get_prompt("file_path"),
                    "required": True,
                },
                "pattern": {
                    "type": "string",
                    "description": self.get_prompt("pattern"),
                    "required": True,
                },
                "limit": {
                    "type": "number",
                    "description": self.get_prompt("limit"),
                    "required": False,
                },
            },
        }

        return ToolCall(**tool_params)

    async def async_execute(self):
        """Execute the grep search operation."""
        pattern: str = self.context.get("pattern", "").strip()
        file_path: str | None | Path = self.context.get("file_path", "")
        limit: int = int(self.context.get("limit", 50))

        assert pattern, "The 'pattern' parameter cannot be empty."
        assert file_path, "The 'file_path' parameter is required."
        target_file = Path(file_path).expanduser().resolve()
        assert target_file.exists(), f"File does not exist: {target_file}"
        assert target_file.is_file(), f"Path is not a file: {target_file}"

        logger.info(f"Searching for pattern '{pattern}' in {target_file}")

        regex = re.compile(re.escape(pattern), re.IGNORECASE)
        results = []

        with target_file.open("r", encoding="utf-8", errors="ignore") as f:
            for line_num, line in enumerate(f, 1):
                if regex.search(line):
                    results.append(f"{target_file}:{line_num}:{line.rstrip()}")
                    if len(results) >= limit:
                        break

        if not results:
            search_location = f'in file_path "{file_path}"' if file_path else "in the workspace directory"
            result_msg = f'No matches found for pattern "{pattern}" {search_location}.'
        else:
            result_msg = "\n".join(results)

        self.set_output(result_msg)

    async def async_default_execute(self, e: Exception = None, **_kwargs):
        """Fill outputs with a default failure message when execution fails."""
        pattern: str = self.context.get("pattern", "").strip()
        error_msg = f'Failed to search for pattern "{pattern}"'
        if e:
            error_msg += f": {str(e)}"
        self.set_output(error_msg)
