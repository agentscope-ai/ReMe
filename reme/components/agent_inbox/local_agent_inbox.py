"""Durable FIFO inputs in inbox_unread.jsonl; bounded history in inbox_read.jsonl.

Claims do not consume items. Acknowledgement atomically publishes read history
before removing unread inputs; recovery reconciles an interrupted move by ID.
History retains at most ``max_read_items`` items, trimming whole oldest batches.
Protected result handoffs block acknowledgement rather than lose their receipts.
IDs remain idempotent within unread inputs and retained history. An I/O failure
requires restarting the component. Resource files are never removed here.
"""

from __future__ import annotations

import asyncio
from collections import deque
from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime, timezone
from hashlib import sha256
from itertools import chain
import json
import math
import os
from pathlib import Path, PureWindowsPath
import re
from uuid import uuid4

from pydantic import BaseModel

from ...constants import WINDOWS_RESERVED_DEVICE_NAMES
from .base_agent_inbox import BaseAgentInbox, InvalidInboxBatchError
from ...core.registry import R
from ...schema.agent_inbox import InboxItem, InboxReceipt
from ...schema.event import Event
from ...utils.async_utils import complete_in_thread, finish_cleanup
from ...utils.files.file_sync import sync_directory, sync_file

_NAME = re.compile(r"[A-Za-z0-9_-]{1,128}\Z")


def _name(value: str) -> str:
    if not isinstance(value, str) or not _NAME.fullmatch(value) or value.upper() in WINDOWS_RESERVED_DEVICE_NAMES:
        raise ValueError("inbox names must be portable names containing letters, numbers, '_' or '-'")
    return value


def _positive(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")


@dataclass(slots=True)
class _Consumed:
    items: list[dict]
    receipt: InboxReceipt

    def record(self) -> dict:
        """Serialize a consumed batch without duplicating its item IDs."""
        return {"items": self.items, **self.receipt.model_dump(mode="json", exclude={"item_ids"})}


@dataclass(slots=True)
class _Pending:
    item: dict
    size: int


@dataclass(slots=True)
class _Inbox:
    history: list[_Consumed] = field(default_factory=list)
    pending: deque[_Pending] = field(default_factory=deque)
    seen: dict[str, bytes] = field(default_factory=dict)
    pending_bytes: int = 0
    claim: tuple[str, tuple[str, ...]] | None = None
    last_receipt: InboxReceipt | None = None
    retained: set[str] = field(default_factory=set)


@R.register("local")
class LocalAgentInbox(BaseAgentInbox):
    """Serialize workspace-contained writes and bound pending inputs across all inboxes."""

    def __init__(
        self,
        directory: str = "agent_inbox",
        max_read_items: int = 1000,
        max_pending_items: int = 10000,
        max_pending_bytes: int = 16 * 1024 * 1024,
        max_item_bytes: int = 32 * 1024,
        claim_max_items: int = 20,
        claim_max_bytes: int = 64 * 1024,
        append_timeout_seconds: float = 5,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        for key, value in {
            "max_read_items": max_read_items,
            "max_pending_items": max_pending_items,
            "max_pending_bytes": max_pending_bytes,
            "max_item_bytes": max_item_bytes,
            "claim_max_items": claim_max_items,
            "claim_max_bytes": claim_max_bytes,
        }.items():
            _positive(key, value)
            setattr(self, key, value)
        if max_item_bytes > min(max_pending_bytes, claim_max_bytes):
            raise ValueError("max_item_bytes must fit the pending and claim byte budgets")
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
        self._inboxes: dict[str, _Inbox] = {}
        self._pending_count = 0
        self._pending_bytes = 0
        self._changed = asyncio.Condition()
        self._owner_fd: int | None = None
        self._closed = True
        self._failed = False

    def _path(self, name: str = "", filename: str | None = None) -> Path:
        path = self._inbox_path(name) if name else self._root
        if filename is not None:
            path /= filename
        if not path.is_relative_to(self._workspace) or path.resolve() != path:
            raise ValueError("Agent inbox paths must stay inside the workspace without symlinks")
        return path

    def _inbox_path(self, name: str) -> Path:
        return self._root / name

    def _names(self):
        return (self._normalize_name(path.name) for path in sorted(self._root.iterdir()) if path.name != ".lock")

    @staticmethod
    def _normalize_name(value: str) -> str:
        return _name(value)

    def _fingerprint(self, item: dict) -> bytes:
        return sha256(self._encode(item)).digest()

    async def _start(self) -> None:
        await complete_in_thread(self._load)
        self._closed = False

    def _load(self) -> None:
        self._inboxes.clear()
        self._pending_count = self._pending_bytes = 0
        self._failed = False
        root = self._path()
        root.mkdir(parents=True, exist_ok=True)
        self._path()
        lock = root / ".lock"
        if lock.is_symlink():
            raise ValueError("Agent inbox lock must not be a symlink")
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
            raise RuntimeError("Agent inbox directory already has a writer") from exc
        self._owner_fd = fd
        sync_directory(root.parent)
        for name in self._names():
            if not self._path(name).is_dir():
                raise ValueError(f"unexpected Agent inbox entry: {name}")
            inbox = self._load_inbox(name)
            self._inboxes[name] = inbox

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

    def _load_inbox(self, name: str) -> _Inbox:
        inbox = _Inbox()
        consumed = set()
        for raw in self._records(self._path(name, "inbox_read.jsonl")):
            record = json.loads(raw)
            items = [self._item(self._encode(value)) for value in record.pop("items")]
            receipt = InboxReceipt(item_ids=[item["id"] for item in items], **record)
            for item in items:
                if item["id"] in inbox.seen:
                    raise ValueError(f"duplicate persisted item id: {item['id']}")
                inbox.seen[item["id"]] = self._fingerprint(item)
                consumed.add(item["id"])
            inbox.history.append(_Consumed(items, receipt))
            inbox.last_receipt = receipt
        unread = self._path(name, "inbox_unread.jsonl")
        recovered = False
        unread_ids = set()
        for raw in self._records(unread):
            item = self._item(raw)
            item_id = item["id"]
            if item_id in unread_ids:
                raise ValueError(f"duplicate persisted item id: {item_id}")
            unread_ids.add(item_id)
            fingerprint = self._fingerprint(item)
            if item_id in consumed:
                if inbox.seen[item_id] != fingerprint:
                    raise ValueError(f"conflicting persisted item id: {item_id}")
                recovered = True
                continue
            if len(raw) > self.claim_max_bytes:
                raise ValueError("persisted item exceeds claim_max_bytes; increase it before restarting")
            inbox.seen[item_id] = fingerprint
            self._add_pending(inbox, _Pending(item, len(raw)))
            count, size = self._pending_usage(inbox)
            if count > self.max_pending_items or size > self.max_pending_bytes:
                raise ValueError("persisted backlog exceeds pending limits; increase them before restarting")
        if sum(len(batch.items) for batch in inbox.history) > self.max_read_items:
            raise ValueError("persisted read history exceeds max_read_items; increase it before restarting")
        if recovered:
            self._replace(unread, b"".join(self._encode(item.item) for item in inbox.pending))
        return inbox

    @staticmethod
    def _validate(value) -> dict:
        if isinstance(value, BaseModel):
            value = value.model_dump(mode="json")
        return InboxItem.model_validate(value).model_dump(mode="json")

    def _item(self, raw: bytes) -> dict:
        return self._validate(json.loads(raw))

    @staticmethod
    def _sort_key(item):
        return item["id"]

    async def _close(self) -> None:
        async with self._changed:
            self._closed = True
            self._changed.notify_all()
            if self._owner_fd is not None:
                os.close(self._owner_fd)
                self._owner_fd = None
            self._inboxes.clear()
            self._pending_count = self._pending_bytes = 0

    def _check_open(self) -> None:
        if self._closed or self._failed:
            raise RuntimeError("Agent inbox is closed or failed; close and restart before use")

    def _inbox(self, name: str) -> _Inbox:
        self._check_open()
        return self._inboxes[self._normalize_name(name)]

    async def create(self, name: str) -> None:
        """Create an empty named inbox. Existing names raise FileExistsError."""
        self._normalize_name(name)
        async with self._changed:
            self._check_open()
            if name in self._inboxes:
                raise FileExistsError(name)
            await finish_cleanup(self._create(name))

    async def mount(self, name: str, directory: str) -> None:
        """Mount the storage location supported by this backend's layout."""
        self._normalize_name(name)
        Event.validate_paths([directory])
        if self._workspace / directory != self._path(name):
            raise ValueError("directory does not match the item-inbox storage layout")
        self._check_open()
        if name in self._inboxes:
            return
        async with self._changed:
            await finish_cleanup(self._mount(name))

    async def _mount(self, name: str) -> None:
        self._check_open()
        if name in self._inboxes:
            return
        if not self._path(name).exists():
            await self._create(name)
            return
        try:
            inbox = await asyncio.to_thread(self._load_inbox, name)
            self._inboxes[name] = inbox
        except (OSError, ValueError):
            self._failed = True
            self._changed.notify_all()
            raise

    async def receipts(self) -> dict[str, list[InboxReceipt]]:
        """Read retained commits without exposing mutable backend state."""
        async with self._changed:
            self._check_open()
            return deepcopy(
                {name: [batch.receipt for batch in inbox.history] for name, inbox in self._inboxes.items()},
            )

    async def retain(self, name: str, item_ids: list[str], *, release: bool = False) -> None:
        """Update only known handoffs, preserving concurrent execution commits."""
        async with self._changed:
            retained = self._inbox(name).retained
            if release:
                retained.difference_update(item_ids)
            else:
                retained.update(item_ids)
            self._changed.notify_all()

    async def forget(self, name: str) -> None:
        """Unmount idle storage without silently dropping unconsumed input."""
        async with self._changed:
            inbox = self._inbox(name)
            if inbox.pending or inbox.claim is not None or inbox.retained:
                raise ValueError("cannot unmount an inbox with pending inputs or result handoffs")
            del self._inboxes[name]
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
        self._inboxes[name] = _Inbox()
        self._changed.notify_all()

    async def snapshot(self, name: str) -> dict:
        """Read the retained execution receipt and pending counts without consuming anything."""
        async with self._changed:
            inbox = self._inbox(name)
            return deepcopy(
                {
                    "receipt": inbox.last_receipt,
                    "pending_count": len(inbox.pending),
                    "in_flight_count": len(inbox.claim[1]) if inbox.claim else 0,
                },
            )

    async def list(self) -> list[dict]:
        """Return inbox summaries without reading or exposing item history."""
        async with self._changed:
            self._check_open()
            return [
                {
                    "name": name,
                    "pending_count": len(inbox.pending),
                    "pending_bytes": inbox.pending_bytes,
                    "in_flight_count": len(inbox.claim[1]) if inbox.claim else 0,
                    "read_count": sum(len(batch.items) for batch in inbox.history),
                }
                for name, inbox in sorted(self._inboxes.items())
            ]

    async def read(self, name: str, *, state: str = "all", offset: int = 0, limit: int = 20) -> dict:
        """Snapshot a bounded page under the same lock as claims and acknowledgements."""
        if state not in {"all", "pending", "processing", "processed"}:
            raise ValueError("invalid item state")
        if type(offset) is not int or offset < 0 or type(limit) is not int or not 1 <= limit <= 100:
            raise ValueError("offset must be nonnegative and limit must be between 1 and 100")
        async with self._changed:
            inbox = self._inbox(name)
            claimed = set(inbox.claim[1]) if inbox.claim else set()
            items = [
                {
                    "item": pending.item,
                    "state": "processing" if pending.item["id"] in claimed else "pending",
                    "consume_ts": None,
                }
                for pending in inbox.pending
            ]
            items.extend(
                {"item": item, "state": "processed", "consume_ts": batch.receipt.consume_ts.isoformat()}
                for batch in inbox.history
                for item in batch.items
            )
            items = [item for item in items if state in {"all", item["state"]}]
            items.sort(
                key=lambda item: self._sort_key(item["item"]),
                reverse=True,
            )
            return deepcopy(
                {"items": items[offset : offset + limit], "total": len(items), "offset": offset, "limit": limit},
            )

    @staticmethod
    def _encode(item: dict) -> bytes:
        return (
            json.dumps(item, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
        ).encode()

    def _prepare(self, items: BaseModel | dict | list[BaseModel | dict]) -> list[tuple[dict, bytes, bytes]]:
        if isinstance(items, (BaseModel, dict)):
            items = [items]
        if not isinstance(items, list):
            raise TypeError("items must be a model, a dict, or a list of items")
        prepared = {}
        for value in items:
            item = self._validate(value)
            raw = self._encode(item)
            if len(raw) > self.max_item_bytes:
                raise InvalidInboxBatchError("item exceeds max_item_bytes; store large payloads in resource files")
            fingerprint = self._fingerprint(item)
            previous = prepared.get(item["id"])
            if previous is not None and previous[2] != fingerprint:
                raise InvalidInboxBatchError(f"conflicting item id: {item['id']}")
            prepared[item["id"]] = (item, raw, fingerprint)
        return list(prepared.values())

    async def append(
        self,
        name: str,
        items: BaseModel | dict | list[BaseModel | dict],
        *,
        wait: bool = True,
    ) -> list[str]:
        """Durably append inputs; wait for pending capacity or raise TimeoutError."""
        prepared = self._prepare(items)
        async with asyncio.timeout(self.append_timeout_seconds) as deadline:
            async with self._changed:
                while True:
                    inbox = self._inbox(name)
                    fresh = []
                    for item in prepared:
                        previous = inbox.seen.get(item[0]["id"])
                        if previous is None:
                            fresh.append(item)
                        elif previous != item[2]:
                            raise InvalidInboxBatchError(f"conflicting item id: {item[0]['id']}")
                    count, size = len(fresh), sum(len(item[1]) for item in fresh)
                    if count > self.max_pending_items or size > self.max_pending_bytes:
                        raise InvalidInboxBatchError("append batch must fit the pending budgets")
                    if not fresh:
                        return [item[0]["id"] for item in prepared]
                    pending_count, pending_size = self._pending_usage(inbox)
                    if (
                        pending_count + count <= self.max_pending_items
                        and pending_size + size <= self.max_pending_bytes
                    ):
                        break
                    if not wait:
                        raise TimeoutError("Agent inbox pending capacity is full")
                    await self._changed.wait()
                deadline.reschedule(None)
                await finish_cleanup(self._append(name, inbox, fresh))
                return [item[0]["id"] for item in prepared]

    async def _append(self, name: str, inbox: _Inbox, prepared: list) -> None:
        try:
            await asyncio.to_thread(
                self._write,
                self._path(name, "inbox_unread.jsonl"),
                b"".join(item[1] for item in prepared),
            )
        except OSError:
            self._failed = True
            self._changed.notify_all()
            raise
        for item, raw, fingerprint in prepared:
            inbox.seen[item["id"]] = fingerprint
            self._add_pending(inbox, _Pending(item, len(raw)))
        self._changed.notify_all()

    def _pending_usage(self, _inbox: _Inbox) -> tuple[int, int]:
        """Use the shared capacity budget for local inboxes."""
        return self._pending_count, self._pending_bytes

    def _add_pending(self, inbox: _Inbox, pending: _Pending) -> None:
        inbox.pending.append(pending)
        inbox.pending_bytes += pending.size
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
                        raise OSError("incomplete item log write")
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
            raise ValueError("Agent inbox paths must stay inside the workspace without symlinks")
        try:
            temporary.unlink(missing_ok=True)
            self._write(temporary, raw)
            os.replace(temporary, path)
            sync_directory(path.parent)
        finally:
            temporary.unlink(missing_ok=True)

    def _persist(self, name, history, pending):
        """Publish the read commit before unread inputs."""
        for filename, records in (
            ("inbox_read.jsonl", (batch.record() for batch in history)),
            ("inbox_unread.jsonl", (item.item for item in pending)),
        ):
            path = self._path(name, filename)
            raw = b"".join(self._encode(value) for value in records)
            self._replace(path, raw)

    @staticmethod
    def _claim_candidates(inbox: _Inbox, item_ids: list[str] | None):
        if item_ids is None:
            return iter(inbox.pending)
        selected = set(item_ids)
        if not selected <= {pending.item["id"] for pending in inbox.pending}:
            raise ValueError("unknown pending item ID")
        return (pending for pending in inbox.pending if pending.item["id"] in selected)

    async def peek(self, name: str, *, item_ids: list[str] | None = None) -> dict | None:
        """Inspect the first selected item without creating a claim."""
        async with self._changed:
            inbox = self._inbox(name)
            if inbox.claim is not None or not inbox.pending:
                return None
            pending = next(self._claim_candidates(inbox, item_ids), None)
            return deepcopy(pending.item) if pending is not None else None

    async def claim(
        self,
        name: str,
        limit: int | None = None,
        *,
        item_ids: list[str] | None = None,
        batch_key: Callable[[dict], str | None] | None = None,
    ) -> dict | None:
        """Claim selected IDs or the next FIFO batch; release ownership on ack, nack or restart."""
        if limit is not None:
            _positive("limit", limit)
        async with self._changed:
            inbox = self._inbox(name)
            if inbox.claim is not None or not inbox.pending:
                return None
            items, size = [], 0
            first_key = None
            for pending in self._claim_candidates(inbox, item_ids):
                if (
                    len(items) >= min(limit or self.claim_max_items, self.claim_max_items, self.max_read_items)
                    or size + pending.size > self.claim_max_bytes
                ):
                    break
                if batch_key is not None:
                    key = batch_key(pending.item)
                    if items and key != first_key:
                        break
                    first_key = key
                items.append(deepcopy(pending.item))
                size += pending.size
            if not items:
                return None
            token = uuid4().hex
            inbox.claim = (token, tuple(item["id"] for item in items))
            return {"token": token, "items": items}

    @staticmethod
    def _claimed(inbox: _Inbox, token: str) -> tuple[str, ...]:
        if inbox.claim is None or inbox.claim[0] != token:
            raise ValueError("invalid or expired claim token")
        return inbox.claim[1]

    async def ack(self, name: str, token: str, *, metadata: dict | None = None, retain: bool = False) -> str:
        """Persist a receipt before releasing any claimed item from memory."""
        async with self._changed:
            inbox = self._inbox(name)
            identifiers = self._claimed(inbox, token)
            return await finish_cleanup(self._ack(name, inbox, identifiers, metadata, retain))

    async def _ack(
        self,
        name: str,
        inbox: _Inbox,
        identifiers: tuple[str, ...],
        metadata: dict | None,
        retain: bool,
    ) -> str:
        selected = set(identifiers)
        pending = [item for item in inbox.pending if item.item["id"] in selected]
        remaining = deque(item for item in inbox.pending if item.item["id"] not in selected)
        receipt = InboxReceipt(
            item_ids=[item.item["id"] for item in pending],
            consume_ts=datetime.now(timezone.utc),
            metadata=metadata or {},
        )
        history = [*inbox.history, _Consumed([item.item for item in pending], receipt)]
        total = sum(len(batch.items) for batch in history)
        while total > self.max_read_items:
            oldest = history[0]
            if not inbox.retained.isdisjoint(oldest.receipt.item_ids):
                raise TimeoutError("read history is full of pending result handoffs")
            total -= len(oldest.items)
            history.pop(0)

        try:
            await asyncio.to_thread(self._persist, name, history, remaining)
        except (OSError, ValueError):
            self._failed = True
            self._changed.notify_all()
            raise
        if retain:
            inbox.retained.update(receipt.item_ids)
        for item in pending:
            inbox.pending_bytes -= item.size
            self._pending_count -= 1
            self._pending_bytes -= item.size
        inbox.pending = remaining
        inbox.claim = None
        inbox.history = history
        inbox.seen = {
            item["id"]: self._fingerprint(item)
            for item in chain(
                (item for batch in history for item in batch.items),
                (item.item for item in inbox.pending),
            )
        }
        inbox.last_receipt = receipt
        self._changed.notify_all()
        return receipt.consume_ts.isoformat()

    async def nack(self, name: str, token: str) -> None:
        """Release a claim without confirming or removing its items."""
        async with self._changed:
            inbox = self._inbox(name)
            self._claimed(inbox, token)
            inbox.claim = None
            self._changed.notify_all()

    async def wait(self, names: str | list[str] | None = None, timeout: float | None = None) -> bool:
        """Wait for claimable items in selected inboxes, or all inboxes; False on timeout."""
        if isinstance(names, str):
            names = [names]
        if names is not None:
            if not isinstance(names, list) or not names:
                raise ValueError("names must be an inbox name or a nonempty list of names")
            for name in names:
                self._normalize_name(name)
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
                        inboxes = [self._inbox(name) for name in names] if names is not None else self._inboxes.values()
                        if any(inbox.pending and inbox.claim is None for inbox in inboxes):
                            return True
                        await self._changed.wait()
        except TimeoutError:
            return False

    async def pending_ids(self, name: str) -> set[str]:
        """Read scheduling membership without copying or paging event bodies."""
        async with self._changed:
            return {entry.item["id"] for entry in self._inbox(name).pending}
