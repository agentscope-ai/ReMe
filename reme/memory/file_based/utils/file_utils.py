# -*- coding: utf-8 -*-
"""Shared utilities for file and shell tools."""

# Default truncation limit
DEFAULT_MAX_BYTES = 50 * 1024  # 50KB

# Marker prepended to every truncation notice.
# Split on this to recover the original (un-truncated) portion:
#   original = output.split(TRUNCATION_NOTICE_MARKER)[0]
TRUNCATION_NOTICE_MARKER = "\x00[TRUNCATION_NOTICE]"


def truncate_text_output(
    text: str,
    start_line: int,
    total_lines: int,
    max_bytes: int = DEFAULT_MAX_BYTES,
    file_path: str | None = None,
) -> str:
    """Truncate file output by bytes with line integrity.

    If text is under byte limit, return as-is.
    If over limit, truncate at the last complete line that fits,
    allowing the next read to start from a fresh line.

    Args:
        text: The output text to truncate.
        start_line: The starting line number (1-based).
        total_lines: Total lines in the original file.
        max_bytes: Maximum size in bytes.
        file_path: Optional file path to include in the truncation notice.

    Returns:
        Truncated text with notice if truncated.
    """
    if not text:
        return text

    try:
        text_bytes = text.encode("utf-8")

        # No truncation needed
        if len(text_bytes) <= max_bytes:
            return text

        # Truncate by bytes
        truncated = text_bytes[:max_bytes]
        result = truncated.decode("utf-8", errors="ignore")

        newline_count = result.count("\n")
        ends_with_newline = result.endswith("\n")

        if ends_with_newline:
            end_line = start_line + newline_count - 1
            next_line = end_line + 1
            extra = ""
        elif newline_count == 0:
            # The single line itself exceeds max_bytes; partially shown.
            end_line = start_line
            next_line = start_line + 1
            extra = (
                f" Line {end_line} exceeds the {max_bytes // 1024}KB limit"
                f" and is partially shown."
            )
        else:
            # Ends mid-line: last line is partially shown.
            end_line = start_line + newline_count
            next_line = end_line
            extra = f" Line {end_line} is truncated."

        continuation = (
            "" if next_line > total_lines else f" Use start_line={next_line} to continue."
        )
        file_info = f" {file_path}" if file_path else ""
        notice = (
            TRUNCATION_NOTICE_MARKER
            + f"\n\n[Output truncated:{file_info} showing lines "
            f"{start_line}-{end_line} of {total_lines} "
            f"({max_bytes // 1024}KB limit).{extra}{continuation}]"
        )

        return result + notice
    except Exception:
        return text


def read_file_safe(file_path: str) -> str:
    """Read file with Unicode error handling.

    Args:
        file_path: Path to the file.

    Returns:
        File content as string.
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    except UnicodeDecodeError:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()