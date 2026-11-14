"""Durable FIFO inputs in events_unread.jsonl; bounded history in events_read.jsonl.

Claims do not consume events. Acknowledgement atomically publishes read history
before removing unread inputs; recovery reconciles an interrupted move by ID.
History retains at most ``max_read_events`` events, trimming whole oldest batches.
Protected result handoffs block acknowledgement rather than lose their receipts.
IDs remain idempotent within unread inputs and retained history. An I/O failure
requires restarting the component. Resource files are never removed here.
"""

from __future__ import annotations

import asyncio
from collections import deque
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime, timezone
from hashlib import sha256
from itertools import chain, islice
import json
import math
import os
from pathlib import Path, PureWindowsPath
import re
from uuid import uuid4

from .base_event_stream import BaseEventStream
from ..component_registry import R
from ...schema.event import Event, EventConsumption
from ...utils.async_utils import complete_in_thread, finish_cleanup
from ...utils.file_sync import sync_directory, sync_file

_NAME = re.compile(r"[A-Za-z0-9_-]{1,128}\Z")
_RESERVED = {"CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(1, 10)), *(f"LPT{i}" for i in range(1, 10))}


def _name(value: str) -> str:
    if not isinstance(value, str) or not _NAME.fullmatch(value) or value.upper() in _RESERVED:
        raise ValueError("stream names must be portable names containing letters, numbers, '_' or '-'")
    return value


def _positive(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")


@dataclass(slots=True)
class _Consumed:
    events: list[dict]
    receipt: EventConsumption

    def record(self) -> dict:
        """Serialize a consumed batch without duplicating its event IDs."""
        return {"events": self.events, **self.receipt.model_dump(mode="json", exclude={"event_ids"})}


@dataclass(slots=True)
class _Pending:
    event: dict
    size: int


@dataclass(slots=True)
class _Stream:
    history: list[_Consumed] = field(default_factory=list)
    pending: deque[_Pending] = field(default_factory=deque)
    seen: dict[str, bytes] = field(default_factory=dict)
    pending_bytes: int = 0
    claim: tuple[str, int] | None = None
    last_event: dict | None = None
    last_consumption: EventConsumption | None = None
    retained: set[str] = field(default_factory=set)


@R.register("local")
class LocalEventStream(BaseEventStream):
    """Serialize workspace-contained writes and bound pending inputs across all streams."""

    def __init__(
        self,
        directory: str = "event_stream",
        max_read_events: int = 1000,
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
            "max_read_events": max_read_events,
            "max_pending_events": max_pending_events,
            "max_pending_bytes": max_pending_bytes,
            "max_event_bytes": max_event_bytes,
            "claim_max_events": claim_max_events,
            "claim_max_bytes": claim_max_bytes,
        }.items():
            _positive(key, value)
            setattr(self, key, value)
        if max_event_bytes > min(max_pending_bytes, claim_max_bytes):
            raise ValueError("max_event_bytes must fit the pending and claim byte budgets")
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

    def _path(self, name: str = "", filename: str | None = None) -> Path:
        path = self._stream_path(name) if name else self._root
        if filename is not None:
            path /= filename
        if not path.is_relative_to(self._workspace) or path.resolve() != path:
            raise ValueError("event stream paths must stay inside the workspace without symlinks")
        return path

    def _stream_path(self, name: str) -> Path:
        return self._root / name

    def _names(self):
        return (self._normalize_name(path.name) for path in sorted(self._root.iterdir()) if path.name != ".lock")

    @staticmethod
    def _normalize_name(value: str) -> str:
        return _name(value)

    def _fingerprint(self, event: dict) -> bytes:
        return sha256(self._encode(event)).digest()

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
        sync_directory(root.parent)
        for name in self._names():
            if not self._path(name).is_dir():
                raise ValueError(f"unexpected event stream entry: {name}")
            stream = self._load_stream(name)
            self._streams[name] = stream

    def _records(self, path: Path):
        """Recover only an unterminated tail; a corrupt complete record fails closed."""
        if not path.exists():
            return
        with path.open("rb+") as file:
            while raw := file.readline():
                if not raw.endswith(b"\n"):
                    file.truncate(file.tell() - len(raw))
                    sync_file(file.fileno())
                    break
                yield raw

    def _load_stream(self, name: str) -> _Stream:
        stream = _Stream()
        directory = self._path(name)
        if list(directory.glob("*.events.jsonl")) or list(directory.glob("*.consumed.jsonl")):
            raise ValueError("numbered event logs require migration to events_unread.jsonl and events_read.jsonl")
        consumed = set()
        for raw in self._records(self._path(name, "events_read.jsonl")):
            record = json.loads(raw)
            events = [self._event(self._encode(value)) for value in record.pop("events")]
            receipt = EventConsumption(event_ids=[event["id"] for event in events], **record)
            for event in events:
                if event["id"] in stream.seen:
                    raise ValueError(f"duplicate persisted event id: {event['id']}")
                stream.seen[event["id"]] = self._fingerprint(event)
                consumed.add(event["id"])
            stream.history.append(_Consumed(events, receipt))
            stream.last_event = events[-1]
            stream.last_consumption = receipt
        unread = self._path(name, "events_unread.jsonl")
        recovered = False
        unread_ids = set()
        for raw in self._records(unread):
            event = self._event(raw)
            event_id = event["id"]
            if event_id in unread_ids:
                raise ValueError(f"duplicate persisted event id: {event_id}")
            unread_ids.add(event_id)
            fingerprint = self._fingerprint(event)
            if event_id in consumed:
                if stream.seen[event_id] != fingerprint:
                    raise ValueError(f"conflicting persisted event id: {event_id}")
                recovered = True
                continue
            if len(raw) > self.claim_max_bytes:
                raise ValueError("persisted event exceeds claim_max_bytes; increase it before restarting")
            stream.seen[event_id] = fingerprint
            stream.last_event = event
            self._add_pending(stream, _Pending(event, len(raw)))
            if self._pending_count > self.max_pending_events or self._pending_bytes > self.max_pending_bytes:
                raise ValueError("persisted backlog exceeds pending limits; increase them before restarting")
        if sum(len(batch.events) for batch in stream.history) > self.max_read_events:
            raise ValueError("persisted read history exceeds max_read_events; increase it before restarting")
        if recovered:
            self._replace(unread, b"".join(self._encode(item.event) for item in stream.pending))
        return stream

    @staticmethod
    def _event(raw: bytes) -> dict:
        model = Event.model_validate_json(raw)
        if model.model_fields_set != Event.model_fields.keys():
            raise ValueError("persisted events must contain all six event fields")
        return model.model_dump(mode="json")

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
        return self._streams[self._normalize_name(name)]

    async def create(self, name: str) -> None:
        """Create an empty named stream. Existing names raise FileExistsError."""
        self._normalize_name(name)
        async with self._changed:
            self._check_open()
            if name in self._streams:
                raise FileExistsError(name)
            await finish_cleanup(self._create(name))

    async def mount(self, name: str, directory: str) -> None:
        """Mount the storage location supported by this backend's layout."""
        self._normalize_name(name)
        Event.validate_paths([directory])
        if self._workspace / directory != self._path(name):
            raise ValueError("directory does not match the event-stream storage layout")
        async with self._changed:
            await finish_cleanup(self._mount(name))

    async def _mount(self, name: str) -> None:
        self._check_open()
        if name in self._streams:
            return
        if not self._path(name).exists():
            await self._create(name)
            return
        try:
            stream = await asyncio.to_thread(self._load_stream, name)
            self._streams[name] = stream
        except (OSError, ValueError):
            self._failed = True
            self._changed.notify_all()
            raise

    async def receipts(self) -> dict[str, list[EventConsumption]]:
        """Read retained commits without exposing mutable backend state."""
        async with self._changed:
            self._check_open()
            return deepcopy(
                {name: [batch.receipt for batch in stream.history] for name, stream in self._streams.items()},
            )

    async def retain(self, name: str, event_ids: list[str], *, release: bool = False) -> None:
        """Update only known handoffs, preserving concurrent execution commits."""
        async with self._changed:
            retained = self._stream(name).retained
            if release:
                retained.difference_update(event_ids)
            else:
                retained.update(event_ids)
            self._changed.notify_all()

    async def forget(self, name: str) -> None:
        """Unmount idle storage without silently dropping unconsumed input."""
        async with self._changed:
            stream = self._stream(name)
            if stream.pending or stream.claim is not None or stream.retained:
                raise ValueError("cannot unmount a stream with pending inputs or result handoffs")
            del self._streams[name]
            self._changed.notify_all()

    async def _create(self, name: str) -> None:
        def persist():
            self._path(name).mkdir(parents=True)
            sync_directory(self._root)

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

    async def snapshot(self, name: str) -> dict:
        """Read the latest retained input and execution receipt without consuming anything."""
        async with self._changed:
            stream = self._stream(name)
            return deepcopy(
                {
                    "event": stream.last_event,
                    "consumption": stream.last_consumption,
                    "pending_count": len(stream.pending),
                    "in_flight_count": stream.claim[1] if stream.claim else 0,
                },
            )

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
                    "read_count": sum(len(batch.events) for batch in stream.history),
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
            fingerprint = self._fingerprint(event)
            previous = prepared.get(event["id"])
            if previous is not None and previous[2] != fingerprint:
                raise ValueError(f"conflicting event id: {event['id']}")
            prepared[event["id"]] = (event, raw, fingerprint)
        return list(prepared.values())

    async def append(self, name: str, events: Event | dict | list[Event | dict]) -> list[str]:
        """Durably append inputs; wait for pending capacity or raise TimeoutError."""
        prepared = self._prepare(events)
        async with self._changed:
            async with asyncio.timeout(self.append_timeout_seconds):
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
                    if count > self.max_pending_events or size > self.max_pending_bytes:
                        raise ValueError("append batch must fit the pending budgets")
                    if not fresh:
                        return [item[0]["id"] for item in prepared]
                    if (
                        self._pending_count + count <= self.max_pending_events
                        and self._pending_bytes + size <= self.max_pending_bytes
                    ):
                        break
                    await self._changed.wait()
            await finish_cleanup(self._append(name, stream, fresh))
            return [item[0]["id"] for item in prepared]

    async def _append(self, name: str, stream: _Stream, prepared: list) -> None:
        try:
            await asyncio.to_thread(
                self._write,
                self._path(name, "events_unread.jsonl"),
                b"".join(item[1] for item in prepared),
            )
        except OSError:
            self._failed = True
            self._changed.notify_all()
            raise
        for event, raw, fingerprint in prepared:
            stream.seen[event["id"]] = fingerprint
            stream.last_event = event
            self._add_pending(stream, _Pending(event, len(raw)))
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
                sync_file(file.fileno())
            except OSError:
                # In-process failures roll back; a crash can leave an unterminated tail.
                file.truncate(size)
                sync_file(file.fileno())
                raise
        if created:
            sync_directory(path.parent)

    def _replace(self, path: Path, raw: bytes) -> None:
        """Publish a complete file; an interrupted temporary file is safe to discard."""
        temporary = path.with_suffix(".tmp")
        if temporary.is_symlink():
            raise ValueError("event stream paths must stay inside the workspace without symlinks")
        try:
            temporary.unlink(missing_ok=True)
            self._write(temporary, raw)
            os.replace(temporary, path)
            sync_directory(path.parent)
        finally:
            temporary.unlink(missing_ok=True)

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
                    len(events) >= min(limit or self.claim_max_events, self.claim_max_events, self.max_read_events)
                    or size + pending.size > self.claim_max_bytes
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

    async def ack(self, name: str, token: str, *, metadata: dict | None = None, retain: bool = False) -> str:
        """Persist a receipt before releasing any claimed event from memory."""
        async with self._changed:
            stream = self._stream(name)
            count = self._claimed(stream, token)
            return await finish_cleanup(self._ack(name, stream, count, metadata, retain))

    async def _ack(self, name: str, stream: _Stream, count: int, metadata: dict | None, retain: bool) -> str:
        pending = list(islice(stream.pending, count))
        receipt = EventConsumption(
            event_ids=[item.event["id"] for item in pending],
            consume_ts=datetime.now(timezone.utc),
            metadata=metadata or {},
        )
        history = [*stream.history, _Consumed([item.event for item in pending], receipt)]
        total = sum(len(batch.events) for batch in history)
        while total > self.max_read_events:
            oldest = history[0]
            if not stream.retained.isdisjoint(oldest.receipt.event_ids):
                raise TimeoutError("read history is full of pending result handoffs")
            total -= len(oldest.events)
            history.pop(0)

        def persist():
            # Read history is the commit point. Recovery removes these IDs from unread.
            self._replace(
                self._path(name, "events_read.jsonl"),
                b"".join(self._encode(batch.record()) for batch in history),
            )
            self._replace(
                self._path(name, "events_unread.jsonl"),
                b"".join(self._encode(item.event) for item in islice(stream.pending, count, None)),
            )

        try:
            await asyncio.to_thread(persist)
        except (OSError, ValueError):
            self._failed = True
            self._changed.notify_all()
            raise
        if retain:
            stream.retained.update(receipt.event_ids)
        for item in pending:
            stream.pending.popleft()
            stream.pending_bytes -= item.size
            self._pending_count -= 1
            self._pending_bytes -= item.size
        stream.claim = None
        stream.history = history
        stream.seen = {
            event["id"]: self._fingerprint(event)
            for event in chain(
                (event for batch in history for event in batch.events),
                (item.event for item in stream.pending),
            )
        }
        stream.last_consumption = receipt
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
