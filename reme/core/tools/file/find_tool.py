"""File search tool using glob patterns with gitignore support."""

import os
from pathlib import Path

from .base_file_tool import BaseFileTool
from .truncate import FIND_MAX_BYTES, FIND_MAX_LINES, format_size, truncate_head
from ...schema import ToolCall


class FindTool(BaseFileTool):
    """Search for files by glob pattern, respecting .gitignore."""

    def __init__(self, cwd: str | None = None):
        """Initialize find tool.

        Args:
            cwd: Working directory (defaults to current directory)
        """
        super().__init__()
        self.cwd = cwd or os.getcwd()

    def _build_tool_call(self) -> ToolCall:
        max_kb = FIND_MAX_BYTES // 1024
        return ToolCall(
            **{
                "description": (
                    f"Search for files by glob pattern. Returns matching file paths relative "
                    f"to the search directory. Respects .gitignore. Output is truncated to "
                    f"1000 results or {max_kb}KB (whichever is hit first)."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "pattern": {
                            "type": "string",
                            "description": "Glob pattern to match files, "
                            "e.g. '*.ts', '**/*.json', or 'src/**/*.spec.ts'",
                        },
                        "path": {
                            "type": "string",
                            "description": "Directory to search in (default: current directory)",
                        },
                        "limit": {
                            "type": "number",
                            "description": "Maximum number of results (default: 1000)",
                        },
                    },
                    "required": ["pattern"],
                },
            },
        )

    def _load_gitignore_patterns(self, search_path: Path) -> list[str]:
        """Load gitignore patterns from directory and subdirectories."""
        patterns = ["**/node_modules/**", "**/.git/**"]

        # Load root .gitignore
        gitignore_path = search_path / ".gitignore"
        if gitignore_path.exists():
            try:
                with open(gitignore_path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith("#"):
                            patterns.append(line)
            except Exception:
                pass  # Ignore errors

        # Load nested .gitignore files
        try:
            for gitignore in search_path.rglob(".gitignore"):
                if gitignore == gitignore_path:
                    continue
                try:
                    with open(gitignore, "r", encoding="utf-8") as f:
                        for line in f:
                            line = line.strip()
                            if line and not line.startswith("#"):
                                patterns.append(line)
                except Exception:
                    pass  # Ignore errors
        except Exception:
            pass  # Ignore glob errors

        return patterns

    def _should_ignore(self, path: Path, ignore_patterns: list[str]) -> bool:
        """Check if path matches any ignore pattern."""
        path_str = str(path)

        for pattern in ignore_patterns:
            # Simple pattern matching (not full gitignore spec)
            if "**" in pattern:
                # Recursive match
                clean_pattern = pattern.replace("**/", "").replace("/**", "")
                if clean_pattern in path_str:
                    return True
            elif "*" in pattern:
                # Wildcard match
                from fnmatch import fnmatch

                if fnmatch(path.name, pattern):
                    return True
            elif pattern in path_str:
                return True

        return False

    async def execute(self) -> str:
        """Execute file search."""
        pattern: str = self.context.pattern
        search_dir: str = self.context.get("path", ".")
        limit: int = self.context.get("limit", 1000)

        # Resolve search path
        if not os.path.isabs(search_dir):
            search_path = Path(self.cwd) / search_dir
        else:
            search_path = Path(search_dir)

        # Check if directory exists
        if not search_path.exists():
            raise FileNotFoundError(f"Path not found: {search_dir}")

        if not search_path.is_dir():
            raise NotADirectoryError(f"Path is not a directory: {search_dir}")

        # Load gitignore patterns
        ignore_patterns = self._load_gitignore_patterns(search_path)

        # Search for files
        results = []
        for file_path in search_path.glob(pattern):
            if len(results) >= limit:
                break

            # Skip if matches ignore patterns
            if self._should_ignore(file_path, ignore_patterns):
                continue

            # Get relative path
            try:
                rel_path = file_path.relative_to(search_path)
                # Add trailing slash for directories
                if file_path.is_dir():
                    results.append(f"{rel_path}/")
                else:
                    results.append(str(rel_path))
            except ValueError:
                # If relative_to fails, use the path as-is
                results.append(str(file_path))

        # Handle no results
        if not results:
            return "No files found matching pattern"

        # Sort results for consistency
        results.sort()

        # Apply limit and truncation
        result_limit_reached = len(results) >= limit
        raw_output = "\n".join(results)
        truncation = truncate_head(raw_output, max_lines=FIND_MAX_LINES, max_bytes=FIND_MAX_BYTES)

        output = truncation.content
        notices = []

        if result_limit_reached:
            notices.append(
                f"{limit} results limit reached. Use limit={limit * 2} for more, or refine pattern",
            )

        if truncation.truncated:
            notices.append(f"{format_size(FIND_MAX_BYTES)} limit reached")

        if notices:
            output += f"\n\n[{'. '.join(notices)}]"

        return output
