"""Shared filesystem helpers for CRUD steps (safe read, truncation)."""

import aiofiles
import aiofiles.os

from ...constants import DEFAULT_MAX_BYTES, MAX_FILE_READ_BYTES, TRUNCATION_NOTICE_MARKER
from ...utils import get_logger

logger = get_logger()


async def read_file_safe(file_path, max_bytes: int = MAX_FILE_READ_BYTES) -> str:
    """Read file with utf-8-sig (BOM-tolerant), fallback to errors='ignore'."""
    stat = await aiofiles.os.stat(str(file_path))
    read_size = min(stat.st_size, max_bytes)
    try:
        async with aiofiles.open(str(file_path), "r", encoding="utf-8-sig") as f:
            return await f.read(read_size)
    except UnicodeDecodeError:
        async with aiofiles.open(
            str(file_path),
            "r",
            encoding="utf-8-sig",
            errors="ignore",
        ) as f:
            return await f.read(read_size)


def truncate_text_output(
    text: str,
    *,
    start_line: int = 1,
    total_lines: int = 0,
    max_bytes: int = DEFAULT_MAX_BYTES,
    file_path: str | None = None,
    encoding: str = "utf-8",
) -> str:
    """Truncate text by bytes preserving line integrity; append a continuation notice.

    See qwenpaw `tools/utils.py` for the same semantics. Returns text unchanged when
    it fits within max_bytes, when max_bytes <= 0, or when the last line itself
    exceeds max_bytes (unhandled edge case).
    """
    if not text or max_bytes <= 0:
        return text

    try:
        text_bytes = text.encode(encoding)
        if len(text_bytes) <= max_bytes:
            return text

        truncated = text_bytes[:max_bytes]
        result = truncated.decode(encoding, errors="ignore")
        newline_count = result.count("\n")
        next_line = start_line + max(1, newline_count)

        if next_line <= total_lines:
            read_from = next_line
        elif start_line < total_lines:
            read_from = total_lines
        else:
            return result

        notice = (
            TRUNCATION_NOTICE_MARKER + f"\nThe output above was truncated."
            f"\nThe full content is saved to the file and contains {total_lines} lines in total."
            f"\nThis excerpt starts at line {start_line} and covers the next {max_bytes} bytes."
            f"\nIf the current content is not enough, call `read` with file={file_path or ''} "
            f"start_line={read_from} to read more."
        )
        return result + notice
    except Exception:
        logger.warning("truncate_text_output failed, returning original text", exc_info=True)
        return text
