"""File editing tool with exact text replacement."""

import os
from pathlib import Path

from .base_file_tool import BaseFileTool
from .edit_diff import (
    detect_line_ending,
    fuzzy_find_text,
    generate_diff_string,
    normalize_for_fuzzy_match,
    normalize_to_lf,
    restore_line_endings,
    strip_bom,
)
from ...schema import ToolCall


class EditTool(BaseFileTool):
    """Edit a file by replacing exact text."""

    def __init__(self, cwd: str | None = None):
        """Initialize edit tool.

        Args:
            cwd: Working directory (defaults to current directory)
        """
        super().__init__()
        self.cwd = cwd or os.getcwd()

    def _build_tool_call(self) -> ToolCall:
        return ToolCall(
            **{
                "description": (
                    "Edit a file by replacing exact text. The oldText must match exactly "
                    "(including whitespace). Use this for precise, surgical edits."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Path to the file to edit (relative or absolute)",
                        },
                        "oldText": {
                            "type": "string",
                            "description": "Exact text to find and replace (must match exactly)",
                        },
                        "newText": {
                            "type": "string",
                            "description": "New text to replace the old text with",
                        },
                    },
                    "required": ["path", "oldText", "newText"],
                },
            },
        )

    async def execute(self) -> str:
        """Execute the edit operation."""
        path: str = self.context.path
        old_text: str = self.context.oldText
        new_text: str = self.context.newText

        # Resolve path
        if not os.path.isabs(path):
            absolute_path = os.path.join(self.cwd, path)
        else:
            absolute_path = path

        # Check file exists and is writable
        path_obj = Path(absolute_path)
        if not path_obj.exists():
            raise FileNotFoundError(f"File not found: {path}")

        if not os.access(absolute_path, os.R_OK | os.W_OK):
            raise PermissionError(f"File not readable/writable: {path}")

        # Read file
        with open(absolute_path, "r", encoding="utf-8") as f:
            raw_content = f.read()

        # Strip BOM (LLM won't include invisible BOM in oldText)
        bom, content = strip_bom(raw_content)

        original_ending = detect_line_ending(content)
        normalized_content = normalize_to_lf(content)
        normalized_old_text = normalize_to_lf(old_text)
        normalized_new_text = normalize_to_lf(new_text)

        # Find old text using fuzzy matching
        match_result = fuzzy_find_text(normalized_content, normalized_old_text)

        if not match_result.found:
            raise ValueError(
                f"Could not find the exact text in {path}. The old text must match "
                f"exactly including all whitespace and newlines.",
            )

        # Count occurrences for uniqueness check
        fuzzy_content = normalize_for_fuzzy_match(normalized_content)
        fuzzy_old_text = normalize_for_fuzzy_match(normalized_old_text)
        occurrences = fuzzy_content.count(fuzzy_old_text)

        if occurrences > 1:
            raise ValueError(
                f"Found {occurrences} occurrences of the text in {path}. "
                f"The text must be unique. Please provide more context to make it unique.",
            )

        # Perform replacement
        base_content = match_result.content_for_replacement
        new_content = (
            base_content[: match_result.index]
            + normalized_new_text
            + base_content[match_result.index + match_result.match_length :]
        )

        # Verify replacement changed something
        if base_content == new_content:
            raise ValueError(
                f"No changes made to {path}. The replacement produced identical content. "
                f"This might indicate an issue with special characters or the text not "
                f"exist as expected.",
            )

        # Write file
        final_content = bom + restore_line_endings(new_content, original_ending)
        with open(absolute_path, "w", encoding="utf-8") as f:
            f.write(final_content)

        # Generate diff
        diff_result = generate_diff_string(base_content, new_content)

        return f"Successfully replaced text in {path}.\n\n{diff_result.diff}"
