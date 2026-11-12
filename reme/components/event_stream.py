"""Named single-consumer event streams backed by paired, append-only JSONL segments.

``create/list/append/claim/ack/nack/wait`` are the complete component API. A claim
keeps its events in memory until acknowledgement is durable. Only fully consumed
segments are eligible for retention cleanup; saturation waits rather than drops.
IDs are idempotent within retained history. Restart invalidates outstanding claim
tokens and restores unacknowledged events. Resource files are never removed here.

Storage is ``<workspace>/<directory>/<stream>/<number>.{events,consumed}.jsonl``.
``max_segments`` counts pairs per stream. Pending budgets are component-wide;
lowering them below an existing backlog fails startup without dropping events.
An I/O error requires closing and restarting the component before further use.
"""

from __future__ import annotations

import asyncio
from collections import deque
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime, timezone
from hashlib import sha256
from itertools import islice
import json
import math
import os
from pathlib import Path, PureWindowsPath
import re
import sys
from uuid import uuid4

from .base_component import BaseComponent
from .component_registry import R
from ..enumeration import ComponentEnum
from ..schema.event import Event, EventConsumption
from ..utils.async_utils import complete_in_thread, finish_cleanup

_NAME = re.compile(r"[A-Za-z0-9_-]{1,128}\Z")
_SEGMENT = re.compile(r"([0-9]{8,})\.(?:events|consumed)\.jsonl\Z")
_RESERVED = {"CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(1, 10)), *(f"LPT{i}" for i in range(1, 10))}


def _name(value: str) -> str:
    if not isinstance(value, str) or not _NAME.fullmatch(value) or value.upper() in _RESERVED:
        raise ValueError("stream names must be portable names containing letters, numbers, '_' or '-'")
    return value


def _positive(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")


def _segment_number(path: Path) -> int:
    match = _SEGMENT.fullmatch(path.name)
    if match is None or int(match[1]) < 1 or match[1] != f"{int(match[1]):08d}":
        raise ValueError(f"invalid segment name: {path.name}")
    return int(match[1])


def _sync(fd: int) -> None:
    # macOS fsync alone does not request flushing the device's write cache.
    if sys.platform == "darwin":
        import fcntl

        fcntl.fcntl(fd, fcntl.F_FULLFSYNC)
    else:
        os.fsync(fd)


def _sync_directory(path: Path) -> None:
    if os.name != "nt":
        fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)


@dataclass(slots=True)
class _Segment:
    number: int
    size: int = 0
    ids: set[str] = field(default_factory=set)
    pending: int = 0


@dataclass(slots=True)
class _Pending:
    event: dict
    size: int
    segment: _Segment


@dataclass(slots=True)
class _Stream:
    segments: list[_Segment] = field(default_factory=list)
    pending: deque[_Pending] = field(default_factory=deque)
    seen: dict[str, bytes] = field(default_factory=dict)
    pending_bytes: int = 0
    claim: tuple[str, int] | None = None


@R.register("local")
class LocalEventStream(BaseComponent):
    """Own one workspace-contained directory and serialize durable writes off the event loop.

    Byte budgets count UTF-8 JSONL bytes, not Python heap size. Limits cover all
    streams together, including claimed events. Each append batch fits one segment;
    each claim stops at a segment boundary. No background tasks or query index are
    needed. Use ``append(name, events)`` to share one sync across a batch.
    """

    component_type = ComponentEnum.EVENT_STREAM

    def __init__(
        self,
        directory: str = "event_stream",
        segment_max_events: int = 1000,
        segment_max_bytes: int = 8 * 1024 * 1024,
        max_segments: int = 10,
        max_pending_events: int = 10000,
        max_pending_bytes: int = 16 * 1024 * 1024,
        max_event_bytes: int = 32 * 1024,
        claim_max_events: int = 20,
        claim_max_bytes: int = 64 * 1024,
        append_timeout_seconds: float = 5,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        for key, value in {
            "segment_max_events": segment_max_events,
            "segment_max_bytes": segment_max_bytes,
            "max_segments": max_segments,
            "max_pending_events": max_pending_events,
            "max_pending_bytes": max_pending_bytes,
            "max_event_bytes": max_event_bytes,
            "claim_max_events": claim_max_events,
            "claim_max_bytes": claim_max_bytes,
        }.items():
            _positive(key, value)
            setattr(self, key, value)
        if max_event_bytes > min(segment_max_bytes, max_pending_bytes, claim_max_bytes):
            raise ValueError("max_event_bytes must fit the segment, pending and claim byte budgets")
        if (
            isinstance(append_timeout_seconds, bool)
            or not isinstance(append_timeout_seconds, (int, float))
            or not math.isfinite(append_timeout_seconds)
            or append_timeout_seconds <= 0
        ):
            raise ValueError("append_timeout_seconds must be positive and finite")
        path = Path(directory)
        if any(
            (
                not directory,
                directory.startswith("~"),
                "\\" in directory,
                PureWindowsPath(directory).anchor,
                path.is_absolute(),
                ".." in path.parts,
                path == Path("."),
            ),
        ):
            raise ValueError("directory must stay inside the workspace")
        self.directory = directory
        self.append_timeout_seconds = append_timeout_seconds
        self._workspace = self.workspace_path.resolve()
        self._root = self._workspace / path
        self._streams: dict[str, _Stream] = {}
        self._pending_count = 0
        self._pending_bytes = 0
        self._changed = asyncio.Condition()
        self._owner_fd: int | None = None
        self._closed = True
        self._failed = False

    def _path(self, name: str = "", segment: _Segment | None = None, *, consumed: bool = False) -> Path:
        path = self._root / name
        if segment is not None:
            suffix = "consumed" if consumed else "events"
            path /= f"{segment.number:08d}.{suffix}.jsonl"
        if not path.is_relative_to(self._workspace) or path.resolve() != path:
            raise ValueError("event stream paths must stay inside the workspace without symlinks")
        return path

    async def _start(self) -> None:
        await complete_in_thread(self._load)
        self._closed = False

    def _load(self) -> None:
        self._streams.clear()
        self._pending_count = self._pending_bytes = 0
        self._failed = False
        root = self._path()
        root.mkdir(parents=True, exist_ok=True)
        self._path()
        lock = root / ".lock"
        if lock.is_symlink():
            raise ValueError("event stream lock must not be a symlink")
        fd = os.open(lock, os.O_CREAT | os.O_RDWR, 0o600)
        try:
            if os.name == "nt":
                import msvcrt

                os.write(fd, b"\0")
                os.lseek(fd, 0, os.SEEK_SET)
                msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            os.close(fd)
            raise RuntimeError("event stream directory already has a writer") from exc
        self._owner_fd = fd
        _sync_directory(root.parent)
        for directory in sorted(root.iterdir()):
            if directory.name == ".lock":
                continue
            name = _name(directory.name)
            if not self._path(name).is_dir():
                raise ValueError(f"unexpected event stream entry: {directory.name}")
            stream = self._load_stream(name)
            self._streams[name] = stream
            if len(stream.segments) > self.max_segments:
                raise ValueError("persisted segments exceed max_segments; increase the limit before restarting")

    def _records(self, path: Path):
        """Recover only an unterminated tail; a corrupt complete record fails closed."""
        if not path.exists():
            return
        with path.open("rb+") as file:
            while raw := file.readline():
                if not raw.endswith(b"\n"):
                    file.truncate(file.tell() - len(raw))
                    _sync(file.fileno())
                    break
                yield raw

    def _load_stream(self, name: str) -> _Stream:
        stream = _Stream()
        directory = self._path(name)
        # An orphan receipt file is a fully consumed segment interrupted during cleanup.
        for path in directory.glob("*.consumed.jsonl"):
            segment = _Segment(_segment_number(path))
            receipt = self._path(name, segment, consumed=True)
            if not self._path(name, segment).exists():
                receipt.unlink()
        pending_started = False
        for number in sorted(_segment_number(path) for path in directory.glob("*.events.jsonl")):
            segment = _Segment(number)
            consumed = set()
            for raw in self._records(self._path(name, segment, consumed=True)):
                consumed.update(EventConsumption.model_validate_json(raw).event_ids)
            for raw in self._records(self._path(name, segment)):
                model = Event.model_validate_json(raw)
                if model.model_fields_set != Event.model_fields.keys():
                    raise ValueError("persisted events must contain all six event fields")
                event = model.model_dump(mode="json")
                event_id = event["id"]
                if event_id in stream.seen:
                    raise ValueError(f"duplicate persisted event id: {event_id}")
                segment.ids.add(event_id)
                segment.size += len(raw)
                stream.seen[event_id] = sha256(self._encode(event)).digest()
                if event_id in consumed:
                    if pending_started:
                        raise ValueError("consumption receipts must confirm a contiguous event prefix")
                    continue
                if len(raw) > self.claim_max_bytes:
                    raise ValueError("persisted event exceeds claim_max_bytes; increase it before restarting")
                pending_started = True
                segment.pending += 1
                self._add_pending(stream, _Pending(event, len(raw), segment))
                if self._pending_count > self.max_pending_events or self._pending_bytes > self.max_pending_bytes:
                    raise ValueError("persisted backlog exceeds pending limits; increase them before restarting")
            if consumed - segment.ids:
                raise ValueError("consumption receipt references unknown events")
            stream.segments.append(segment)
        return stream

    async def _close(self) -> None:
        async with self._changed:
            self._closed = True
            self._changed.notify_all()
            if self._owner_fd is not None:
                os.close(self._owner_fd)
                self._owner_fd = None
            self._streams.clear()
            self._pending_count = self._pending_bytes = 0

    def _check_open(self) -> None:
        if self._closed or self._failed:
            raise RuntimeError("event stream is closed or failed; close and restart before use")

    def _stream(self, name: str) -> _Stream:
        self._check_open()
        return self._streams[_name(name)]

    async def create(self, name: str) -> None:
        """Create an empty named stream. Existing names raise FileExistsError."""
        _name(name)
        async with self._changed:
            self._check_open()
            if name in self._streams:
                raise FileExistsError(name)
            await finish_cleanup(self._create(name))

    async def _create(self, name: str) -> None:
        def persist():
            self._path(name).mkdir()
            _sync_directory(self._root)

        try:
            await asyncio.to_thread(persist)
        except FileExistsError:
            raise
        except OSError:
            self._failed = True
            self._changed.notify_all()
            raise
        self._streams[name] = _Stream()
        self._changed.notify_all()

    async def list(self) -> list[dict]:
        """Return stream summaries without reading or exposing event history."""
        async with self._changed:
            self._check_open()
            return [
                {
                    "name": name,
                    "pending_count": len(stream.pending),
                    "pending_bytes": stream.pending_bytes,
                    "in_flight_count": stream.claim[1] if stream.claim else 0,
                    "segments": len(stream.segments),
                }
                for name, stream in sorted(self._streams.items())
            ]

    @staticmethod
    def _encode(event: dict) -> bytes:
        return (
            json.dumps(event, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
        ).encode()

    def _prepare(self, events: Event | dict | list[Event | dict]) -> list[tuple[dict, bytes, bytes]]:
        if isinstance(events, (Event, dict)):
            events = [events]
        if not isinstance(events, list):
            raise TypeError("events must be an Event, a dict, or a list of events")
        prepared = {}
        for value in events:
            event = Event.model_validate(value).model_dump(mode="json")
            raw = self._encode(event)
            if len(raw) > self.max_event_bytes:
                raise ValueError("event exceeds max_event_bytes; store large payloads in resource files")
            fingerprint = sha256(raw).digest()
            previous = prepared.get(event["id"])
            if previous is not None and previous[2] != fingerprint:
                raise ValueError(f"conflicting event id: {event['id']}")
            prepared[event["id"]] = (event, raw, fingerprint)
        return list(prepared.values())

    async def append(self, name: str, events: Event | dict | list[Event | dict]) -> list[str]:
        """Durably append one segment-sized batch; wait for capacity or raise TimeoutError."""
        prepared = self._prepare(events)
        async with asyncio.timeout(self.append_timeout_seconds) as deadline:
            async with self._changed:
                while True:
                    stream = self._stream(name)
                    fresh = []
                    for item in prepared:
                        previous = stream.seen.get(item[0]["id"])
                        if previous is None:
                            fresh.append(item)
                        elif previous != item[2]:
                            raise ValueError(f"conflicting event id: {item[0]['id']}")
                    count, size = len(fresh), sum(len(item[1]) for item in fresh)
                    if count > min(self.segment_max_events, self.max_pending_events) or size > min(
                        self.segment_max_bytes,
                        self.max_pending_bytes,
                    ):
                        raise ValueError("append batch must fit one segment and the pending budgets")
                    if not fresh:
                        return [item[0]["id"] for item in prepared]
                    segment = stream.segments[-1] if stream.segments else None
                    rotate = segment is None or (
                        len(segment.ids) + count > self.segment_max_events
                        or segment.size + size > self.segment_max_bytes
                    )
                    if (
                        self._pending_count + count <= self.max_pending_events
                        and self._pending_bytes + size <= self.max_pending_bytes
                        and (not rotate or len(stream.segments) < self.max_segments or not stream.segments[0].pending)
                    ):
                        break
                    await self._changed.wait()
                deadline.reschedule(None)
                await finish_cleanup(self._append(name, stream, fresh, rotate))
                return [item[0]["id"] for item in prepared]

    async def _append(self, name: str, stream: _Stream, prepared: list, rotate: bool) -> None:
        segment = _Segment(stream.segments[-1].number + 1 if stream.segments else 1) if rotate else stream.segments[-1]
        try:
            if rotate and len(stream.segments) >= self.max_segments:
                oldest = stream.segments[0]
                await asyncio.to_thread(self._remove_segment, name, oldest)
                stream.segments.pop(0)
                for event_id in oldest.ids:
                    del stream.seen[event_id]
            await asyncio.to_thread(self._write, self._path(name, segment), b"".join(item[1] for item in prepared))
        except OSError:
            self._failed = True
            self._changed.notify_all()
            raise
        if rotate:
            stream.segments.append(segment)
        for event, raw, fingerprint in prepared:
            segment.ids.add(event["id"])
            segment.size += len(raw)
            segment.pending += 1
            stream.seen[event["id"]] = fingerprint
            self._add_pending(stream, _Pending(event, len(raw), segment))
        self._changed.notify_all()

    def _add_pending(self, stream: _Stream, pending: _Pending) -> None:
        stream.pending.append(pending)
        stream.pending_bytes += pending.size
        self._pending_count += 1
        self._pending_bytes += pending.size

    @staticmethod
    def _write(path: Path, raw: bytes) -> None:
        created = not path.exists()
        with path.open("ab", buffering=0) as file:
            size = file.tell()
            try:
                remaining = memoryview(raw)
                while remaining:
                    written = file.write(remaining)
                    if not written:
                        raise OSError("incomplete event log write")
                    remaining = remaining[written:]
                _sync(file.fileno())
            except OSError:
                # In-process failures roll back; a crash can leave an unterminated tail.
                file.truncate(size)
                _sync(file.fileno())
                raise
        if created:
            _sync_directory(path.parent)

    def _remove_segment(self, name: str, segment: _Segment) -> None:
        self._path(name, segment).unlink()
        _sync_directory(self._path(name))
        self._path(name, segment, consumed=True).unlink(missing_ok=True)
        _sync_directory(self._path(name))

    async def claim(self, name: str, limit: int | None = None) -> dict | None:
        """Claim one FIFO batch. Its token is valid until ack, nack or restart."""
        if limit is not None:
            _positive("limit", limit)
        async with self._changed:
            stream = self._stream(name)
            if stream.claim is not None or not stream.pending:
                return None
            events, size = [], 0
            for pending in stream.pending:
                if (
                    len(events) >= min(limit or self.claim_max_events, self.claim_max_events)
                    or size + pending.size > self.claim_max_bytes
                    or pending.segment is not stream.pending[0].segment
                ):
                    break
                events.append(deepcopy(pending.event))
                size += pending.size
            token = uuid4().hex
            stream.claim = (token, len(events))
            return {"token": token, "events": events}

    @staticmethod
    def _claimed(stream: _Stream, token: str) -> int:
        if stream.claim is None or stream.claim[0] != token:
            raise ValueError("invalid or expired claim token")
        return stream.claim[1]

    async def ack(self, name: str, token: str) -> str:
        """Persist a receipt before releasing any claimed event from memory."""
        async with self._changed:
            stream = self._stream(name)
            count = self._claimed(stream, token)
            return await finish_cleanup(self._ack(name, stream, count))

    async def _ack(self, name: str, stream: _Stream, count: int) -> str:
        pending = list(islice(stream.pending, count))
        receipt = EventConsumption(
            event_ids=[item.event["id"] for item in pending],
            consume_ts=datetime.now(timezone.utc),
        )
        try:
            await asyncio.to_thread(
                self._write,
                self._path(name, pending[0].segment, consumed=True),
                (receipt.model_dump_json() + "\n").encode(),
            )
        except OSError:
            self._failed = True
            self._changed.notify_all()
            raise
        for item in pending:
            stream.pending.popleft()
            item.segment.pending -= 1
            stream.pending_bytes -= item.size
            self._pending_count -= 1
            self._pending_bytes -= item.size
        stream.claim = None
        self._changed.notify_all()
        return receipt.consume_ts.isoformat()

    async def nack(self, name: str, token: str) -> None:
        """Release a claim without confirming or removing its events."""
        async with self._changed:
            stream = self._stream(name)
            self._claimed(stream, token)
            stream.claim = None
            self._changed.notify_all()

    async def wait(self, names: str | list[str] | None = None, timeout: float | None = None) -> bool:
        """Wait for claimable events in selected streams, or all streams; False on timeout."""
        if isinstance(names, str):
            names = [names]
        if names is not None:
            if not isinstance(names, list) or not names:
                raise ValueError("names must be a stream name or a nonempty list of names")
            for name in names:
                _name(name)
        if timeout is not None and (
            isinstance(timeout, bool)
            or not isinstance(timeout, (int, float))
            or not math.isfinite(timeout)
            or timeout < 0
        ):
            raise ValueError("timeout must be nonnegative and finite")
        try:
            async with asyncio.timeout(timeout):
                async with self._changed:
                    while True:
                        self._check_open()
                        streams = (
                            [self._stream(name) for name in names] if names is not None else self._streams.values()
                        )
                        if any(stream.pending and stream.claim is None for stream in streams):
                            return True
                        await self._changed.wait()
        except TimeoutError:
            return False
