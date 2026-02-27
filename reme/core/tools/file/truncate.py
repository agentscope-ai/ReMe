"""fs utils"""

from typing import Literal

from ...schema import TruncationResult

# Default limits for output truncation
DEFAULT_MAX_LINES = 1000  # Maximum lines to keep for tail truncation
DEFAULT_MAX_BYTES = 30 * 1024  # Maximum bytes to keep (30KB)

# Find tool limits
FIND_MAX_LINES = 2000  # Maximum lines for find output
FIND_MAX_BYTES = 50 * 1024  # 50KB for find output

# Grep tool limits
GREP_MAX_LINE_LENGTH = 500  # Maximum line length for grep output


def format_size(num_bytes: int) -> str:
    """Format byte size in human-readable format.

    Args:
        num_bytes: Number of bytes

    Returns:
        Formatted string (e.g., "1.5KB", "2.3MB")
    """
    if num_bytes < 1024:
        return f"{num_bytes}B"
    elif num_bytes < 1024 * 1024:
        return f"{num_bytes / 1024:.1f}KB"
    else:
        return f"{num_bytes / (1024 * 1024):.1f}MB"


def truncate_line(text: str, max_length: int = GREP_MAX_LINE_LENGTH) -> tuple[str, bool]:
    """Truncate a single line if it exceeds max length.

    Args:
        text: Line text
        max_length: Maximum line length

    Returns:
        Tuple of (truncated_text, was_truncated)
    """
    if len(text) <= max_length:
        return text, False
    return text[:max_length] + "...", True


def truncate_tail(
    text: str,
    max_lines: int = DEFAULT_MAX_LINES,
    max_bytes: int = DEFAULT_MAX_BYTES,
) -> TruncationResult:
    """Truncate text to keep only the tail (last portion).

    Keeps the last N lines or M bytes, whichever is hit first.
    This is useful for command outputs where the end is most relevant.

    Args:
        text: The text to truncate
        max_lines: Maximum number of lines to keep
        max_bytes: Maximum bytes to keep

    Returns:
        TruncationResult with truncated content and metadata
    """
    if not text:
        return TruncationResult(
            content="",
            truncated=False,
            total_lines=0,
            output_lines=0,
            total_bytes=0,
            output_bytes=0,
        )

    total_bytes = len(text.encode("utf-8"))
    lines = text.split("\n")
    total_lines = len(lines)

    # Check if we need to truncate
    if total_lines <= max_lines and total_bytes <= max_bytes:
        return TruncationResult(
            content=text,
            truncated=False,
            total_lines=total_lines,
            output_lines=total_lines,
            total_bytes=total_bytes,
            output_bytes=total_bytes,
        )

    # Keep last N lines
    kept_lines = lines[-max_lines:] if total_lines > max_lines else lines
    truncated_by: Literal["lines", "bytes"] = "lines" if total_lines > max_lines else "bytes"

    # Check byte limit on kept lines
    kept_text = "\n".join(kept_lines)
    kept_bytes = len(kept_text.encode("utf-8"))

    # If still over byte limit, truncate further
    last_line_partial = False
    if kept_bytes > max_bytes:
        truncated_by = "bytes"
        # Keep truncating from the start until under byte limit
        while kept_lines and len("\n".join(kept_lines).encode("utf-8")) > max_bytes:
            kept_lines.pop(0)

        # If still over (single line > max_bytes), truncate the line itself
        if kept_lines and len("\n".join(kept_lines).encode("utf-8")) > max_bytes:
            last_line = kept_lines[-1]
            # Binary search to find how much of last line fits
            encoded = last_line.encode("utf-8")
            if len(encoded) > max_bytes:
                last_line_partial = True
                # Take last max_bytes of the line
                kept_lines[-1] = encoded[-max_bytes:].decode("utf-8", errors="ignore")

        kept_text = "\n".join(kept_lines)
        kept_bytes = len(kept_text.encode("utf-8"))

    return TruncationResult(
        content=kept_text,
        truncated=True,
        total_lines=total_lines,
        output_lines=len(kept_lines),
        total_bytes=total_bytes,
        output_bytes=kept_bytes,
        truncated_by=truncated_by,
        last_line_partial=last_line_partial,
    )


def truncate_head(
    text: str,
    max_lines: int = FIND_MAX_LINES,
    max_bytes: int = FIND_MAX_BYTES,
) -> TruncationResult:
    """Truncate text to keep only the head (first portion).

    Keeps the first N lines or M bytes, whichever is hit first.
    Suitable for file reads where you want to see the beginning.

    Args:
        text: The text to truncate
        max_lines: Maximum number of lines to keep
        max_bytes: Maximum bytes to keep

    Returns:
        TruncationResult with truncated content and metadata
    """
    if not text:
        return TruncationResult(
            content="",
            truncated=False,
            total_lines=0,
            output_lines=0,
            total_bytes=0,
            output_bytes=0,
        )

    total_bytes = len(text.encode("utf-8"))
    lines = text.split("\n")
    total_lines = len(lines)

    # Check if no truncation needed
    if total_lines <= max_lines and total_bytes <= max_bytes:
        return TruncationResult(
            content=text,
            truncated=False,
            total_lines=total_lines,
            output_lines=total_lines,
            total_bytes=total_bytes,
            output_bytes=total_bytes,
        )

    # Collect complete lines that fit
    kept_lines = []
    kept_bytes = 0
    truncated_by: Literal["lines", "bytes"] = "lines"

    for i, line in enumerate(lines):
        if i >= max_lines:
            truncated_by = "lines"
            break

        # Calculate bytes for this line (+1 for newline except first line)
        line_bytes = len(line.encode("utf-8")) + (1 if i > 0 else 0)

        if kept_bytes + line_bytes > max_bytes:
            truncated_by = "bytes"
            break

        kept_lines.append(line)
        kept_bytes += line_bytes

    kept_text = "\n".join(kept_lines)
    final_bytes = len(kept_text.encode("utf-8"))

    return TruncationResult(
        content=kept_text,
        truncated=True,
        total_lines=total_lines,
        output_lines=len(kept_lines),
        total_bytes=total_bytes,
        output_bytes=final_bytes,
        truncated_by=truncated_by,
    )
