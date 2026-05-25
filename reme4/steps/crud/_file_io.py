"""Shared filesystem helpers for CRUD steps (path gating, safe read, truncation, OCC writes)."""

import asyncio
import contextlib
import os
import random
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Awaitable, Callable, Iterable

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


# ---------------------------------------------------------------------------
# Optimistic Concurrency Control (OCC) primitives.
#
# All mutating CRUD steps (write / edit / append) share the same flow:
#     read old bytes + stat → compute new bytes → recheck stat → atomic replace
# Concurrent modifications are detected by comparing (mtime_ns, size) between
# the pre-read snapshot and a re-stat just before replace. On mismatch the
# whole loop retries with exponential backoff + jitter.
#
# Atomic replace is done via tmp file (same directory) + fsync + os.replace,
# so partial writes cannot be observed by readers.
# ---------------------------------------------------------------------------

OCC_DEFAULT_MAX_RETRIES = 10
OCC_DEFAULT_BASE_DELAY = 0.05


@dataclass(frozen=True)
class VersionStamp:
    """Cheap version token for OCC; equal stamps imply no observed mutation."""

    mtime_ns: int
    size: int

    @classmethod
    async def of(cls, path: Path) -> "VersionStamp | None":
        """Return a stamp for ``path``, or ``None`` if the file does not exist."""
        try:
            st = await aiofiles.os.stat(str(path))
        except FileNotFoundError:
            return None
        return cls(mtime_ns=st.st_mtime_ns, size=st.st_size)


class ConflictError(Exception):
    """Raised when OCC retries are exhausted due to persistent concurrent writes."""

    def __init__(
        self,
        path: Path,
        attempts: int,
        *,
        v_read: VersionStamp | None = None,
        v_now: VersionStamp | None = None,
    ):
        super().__init__(
            f"write conflict on {path} after {attempts} attempts (concurrent modification detected)",
        )
        self.path = path
        self.attempts = attempts
        self.v_read = v_read
        self.v_now = v_now


async def read_bytes_safe(file_path: Path, max_bytes: int = MAX_FILE_READ_BYTES) -> bytes:
    """Read raw bytes from ``file_path`` (capped by ``max_bytes``)."""
    stat = await aiofiles.os.stat(str(file_path))
    read_size = min(stat.st_size, max_bytes)
    async with aiofiles.open(str(file_path), "rb") as f:
        return await f.read(read_size)


def _sync_atomic_replace(target: Path, payload: bytes) -> None:
    """Blocking implementation of atomic_replace; off-loaded via ``asyncio.to_thread``."""
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=str(target.parent), prefix=target.name + ".tmp.")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(payload)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, str(target))
    except Exception:
        with contextlib.suppress(OSError):
            os.unlink(tmp_path)
        raise


async def atomic_replace(target: Path, payload: bytes) -> None:
    """Atomically replace ``target`` with ``payload``: tmp file + fsync + os.replace.

    Tmp file lives in the same directory as ``target`` so ``os.replace`` is
    atomic on the same filesystem. Partial writes (process crash, OOM) leave
    the original file untouched; the tmp file is cleaned up on error.
    """
    await asyncio.to_thread(_sync_atomic_replace, target, payload)


async def occ_write(
    target: Path,
    compute: Callable[[bytes | None, VersionStamp | None], Awaitable[bytes | None]],
    *,
    max_retries: int = OCC_DEFAULT_MAX_RETRIES,
    base_delay: float = OCC_DEFAULT_BASE_DELAY,
    occ_on_create: bool = True,
) -> tuple[int, VersionStamp | None, int]:
    """Run ``compute`` under OCC and atomically replace ``target`` with its output.

    Loop per attempt:
        1. ``v_read`` = stat(target)  (None if absent)
        2. ``old_bytes`` = read(target) if ``v_read`` else None
        3. ``new_bytes`` = await compute(old_bytes, v_read)
           - ``new_bytes is None`` → no-op, return (0, v_read, attempt)
        4. If ``v_read is None`` and ``occ_on_create is False``:
           - skip recheck (greenfield create), atomic_replace, return
        5. ``v_now`` = stat(target)
        6. If ``v_now != v_read``: backoff(2^attempt) + retry
        7. atomic_replace(target, new_bytes), return (n, stat(target), attempt)

    Raises:
        ConflictError: if all ``max_retries`` attempts hit a version mismatch.
        Any exception raised by ``compute`` propagates unmodified (no retry) —
            callers use this to signal hard errors (e.g. "anchor not found").

    The ``occ_on_create=False`` mode is for write/append, where caller intent
    is "create or overwrite" — concurrent creates between the absent-stat and
    the replace are accepted (last writer wins for the create race only).
    For edit, pass ``occ_on_create=True`` (the default) and let ``compute``
    reject ``old_bytes is None``.
    """
    last_v_read: VersionStamp | None = None
    last_v_now: VersionStamp | None = None
    for attempt in range(1, max_retries + 1):
        v_read = await VersionStamp.of(target)
        old_bytes = await read_bytes_safe(target) if v_read is not None else None

        new_bytes = await compute(old_bytes, v_read)
        if new_bytes is None:
            return 0, v_read, attempt

        if v_read is None and not occ_on_create:
            await atomic_replace(target, new_bytes)
            return len(new_bytes), await VersionStamp.of(target), attempt

        v_now = await VersionStamp.of(target)
        last_v_read, last_v_now = v_read, v_now
        if v_now != v_read:
            delay = base_delay * (2 ** (attempt - 1)) + random.uniform(0, base_delay)
            logger.info(
                f"occ conflict on {target} attempt={attempt} v_read={v_read} v_now={v_now}; "
                f"retrying in {delay:.3f}s",
            )
            await asyncio.sleep(delay)
            continue

        await atomic_replace(target, new_bytes)
        return len(new_bytes), await VersionStamp.of(target), attempt

    raise ConflictError(target, max_retries, v_read=last_v_read, v_now=last_v_now)
