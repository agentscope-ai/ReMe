"""Shared filesystem helpers for CRUD steps (path gating, safe read, truncation)."""

from pathlib import Path
from typing import Iterable

import aiofiles
import aiofiles.os

from ...constants import DEFAULT_MAX_BYTES, MAX_FILE_READ_BYTES, TRUNCATION_NOTICE_MARKER
from ...utils import get_logger

logger = get_logger()

NON_MD_WARNING = (
    "non-markdown file detected; CRUD operations are recommended on markdown files. "
    "Operating in compatibility mode may carry risks of errors."
)

# Modern text formats — assume UTF-8 by convention.
_STANDARD_TEXT_EXTS = {
    ".md",
    ".py",
    ".js",
    ".ts",
    ".json",
    ".yaml",
    ".yml",
    ".html",
    ".css",
    ".xml",
    ".log",
    ".conf",
    ".ini",
    ".txt",
    ".sh",
}
# Legacy formats that may use ANSI/GBK on Chinese Windows systems.
_NON_STANDARD_EXTS = {".csv", ".bat", ".cmd", ".reg"}


def resolve_path(working_path: Path, raw: str) -> tuple[Path | None, str | None]:
    """Resolve a `path=` argument against ``working_path``.

    Rules:
        - Relative paths are joined under ``working_path``.
        - Absolute paths are accepted and returned as-is; a warning is logged
          recommending relative paths, but the read still proceeds.
    Returns ``(abs_path, None)`` on success, or ``(None, error_message)`` on failure
    (currently only when ``raw`` is empty/blank).
    Filetype-specific gating (e.g. markdown-only / suffix auto-append) is
    layered on top by callers — see ``reme4/steps/crud/_file_io.py::gate_md``.
    """
    if not raw or not str(raw).strip():
        return None, "`path` is required"
    s = str(raw).strip()
    p = Path(s)
    if p.is_absolute():
        logger.info("absolute path detected, recommending relative paths")
        return p, None
    return working_path / p, None


def gate_md(target: Path) -> tuple[Path, bool]:
    """Markdown gate with compatibility fallback.

    Returns ``(path, is_md)``:
        - No suffix → auto-append `.md`, ``is_md=True``.
        - `.md` suffix → ``is_md=True``.
        - Any other suffix → ``is_md=False`` (caller handles degraded mode).
    """
    if target.suffix == "":
        return target.with_suffix(".md"), True
    if target.suffix.lower() != ".md":
        return target, False
    return target, True


def _try_decode(data: bytes, encodings: Iterable[str]) -> tuple[str, str] | None:
    """Return ``(text, encoding)`` for the first encoding that decodes ``data`` cleanly."""
    for enc in encodings:
        try:
            return data.decode(enc), enc
        except (UnicodeDecodeError, LookupError):
            continue
    return None


def decode_known_file(data: bytes, file_extension: str) -> tuple[str, str]:
    """Decode file bytes using the extension as a hint. Returns ``(text, encoding)``.

    Strategy:
        1. BOM-based detection.
        2. Extension-driven defaults:
        3. Last resort → UTF-8 with ``errors='replace'`` so the function never raises.
    """
    if data.startswith(b"\xef\xbb\xbf"):
        return data.decode("utf-8-sig"), "utf-8-sig"
    if data.startswith((b"\xff\xfe", b"\xfe\xff")):
        try:
            return data.decode("utf-16"), "utf-16"
        except UnicodeDecodeError:
            pass

    ext = (file_extension or "").lower()

    if ext in _STANDARD_TEXT_EXTS:
        try:
            return data.decode("utf-8-sig"), "utf-8"
        except UnicodeDecodeError:
            pass  # fall through

    if ext in _NON_STANDARD_EXTS:
        result = _try_decode(data, ("utf-8-sig", "gbk"))
        if result is not None:
            text, enc = result
            return text, "utf-8" if enc == "utf-8-sig" else enc

    # Unknown extension or earlier strategies failed.

    return data.decode("utf-8", errors="replace"), "utf-8"


async def read_file_safe(file_path, max_bytes: int = MAX_FILE_READ_BYTES) -> str:
    """Read file in byte mode and decode to string using extension-aware strategy."""
    stat = await aiofiles.os.stat(str(file_path))
    read_size = min(stat.st_size, max_bytes)
    async with aiofiles.open(str(file_path), "rb") as f:
        data = await f.read(read_size)
    text, _ = decode_known_file(data, Path(file_path).suffix)
    return text


async def detect_file_encoding(file_path, sniff_bytes: int = 8192) -> str:
    """Detect the encoding of an existing file so writes can preserve it.

    Reads up to ``sniff_bytes`` from the head of the file (enough for BOM
    detection and statistical analysis). Falls back to ``utf-8`` if the file
    is unreadable.
    """
    try:
        async with aiofiles.open(str(file_path), "rb") as f:
            data = await f.read(sniff_bytes)
    except Exception:  # pylint: disable=broad-except
        return "utf-8"
    _, enc = decode_known_file(data, Path(file_path).suffix)
    return enc


async def write_file_safe(file_path: Path, content: str | bytes, encoding: str = "utf-8") -> None:
    """Write ``content`` to ``file_path`` in binary mode; creates parent dirs.

    ``str`` input is encoded with ``encoding`` (default UTF-8); callers wanting
    to preserve a file's original encoding should pass the result of
    :func:`detect_file_encoding`. If the requested ``encoding`` can't represent
    some characters, falls back to UTF-8 to avoid data loss.

    ``bytes`` input is written verbatim — callers managing their own encoding
    can pass raw bytes directly.
    """
    file_path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(content, str):
        try:
            payload = content.encode(encoding)
        except (UnicodeEncodeError, LookupError):
            logger.warning(
                "write_file_safe: %r cannot encode all chars, falling back to utf-8",
                encoding,
            )
            payload = content.encode("utf-8")
    else:
        payload = content
    async with aiofiles.open(str(file_path), "wb") as f:
        await f.write(payload)


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
