"""Bounded SSE decoding, independent of the event payload schema."""

from dataclasses import dataclass
import re


@dataclass(frozen=True)
class SSEFrame:
    """One terminated block; IDs are stream cursors, not CloudEvent identities."""

    data: str | None
    cursor: str | None
    retry_ms: int | None


async def _lines(response, limit):
    """Split byte chunks in one pass, preserving UTF-8 and CRLF across chunk boundaries."""
    pending = bytearray()
    skip_lf = False
    async for chunk in response.aiter_bytes():
        if not chunk:
            continue
        if skip_lf:
            chunk = chunk.removeprefix(b"\n")
            skip_lf = False
        start = 0
        for match in re.finditer(rb"\r\n?|\n", chunk):
            if len(pending) + match.start() - start > limit:
                raise ValueError("SSE line exceeds max_event_bytes")
            pending.extend(chunk[start : match.start()])
            yield bytes(pending)
            pending.clear()
            start = match.end()
            skip_lf = start == len(chunk) and match.group() == b"\r"
        if len(pending) + len(chunk) - start > limit:
            raise ValueError("SSE line exceeds max_event_bytes")
        pending.extend(chunk[start:])


async def read_sse(response, max_event_bytes):
    """Decode bounded multiline data and cursor-only blocks; comments retain no event state."""
    data, size = [], 0
    cursor, retry = None, None
    changed, first = False, True
    async for raw in _lines(response, max_event_bytes):
        line = raw.decode("utf-8")
        if first:
            line, first = line.removeprefix("\ufeff"), False
        if not line:
            if data or changed or retry is not None:
                yield SSEFrame("\n".join(data) if data else None, cursor, retry)
            data, size, changed, retry = [], 0, False, None
            continue
        if line.startswith(":"):
            continue
        field, _, value = line.partition(":")
        value = value.removeprefix(" ")
        if field == "data":
            size += len(raw) + 1
            if size > max_event_bytes:
                raise ValueError("SSE event exceeds max_event_bytes")
            data.append(value)
        elif field == "id" and "\0" not in value:
            cursor, changed = value, True
        elif field == "retry" and value.isascii() and value.isdecimal() and len(value) <= 9:
            retry = int(value)
    # EOF does not dispatch an unterminated block or advance its cursor.
