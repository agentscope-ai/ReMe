"""One durable single-consumer queue, owned by one Agent Runtime."""

import asyncio
from collections import OrderedDict
from copy import deepcopy
from datetime import datetime, timezone
import json
import math
import os
from uuid import uuid4

from ...base import BaseComponent
from ....enumeration import ComponentEnum
from ....schema.agent_inbox import InboxItem, InboxReceipt
from ....utils.async_utils import finish_cleanup
from ....utils.files.file_sync import sync_directory, sync_file
from ....utils.files.workspace_files import WorkspaceFiles


class InvalidInboxBatchError(ValueError):
    """The target cannot accept this batch even when its queue is empty."""


class BaseInbox(BaseComponent):
    """Share durable storage without owning Runtime policy or a worker."""

    component_type = ComponentEnum.AGENT_INBOX
    keep_processed = False

    def __init__(
        self,
        directory,
        max_read_items=1000,
        max_pending_items=10000,
        max_pending_bytes=16 * 1024 * 1024,
        max_item_bytes=32 * 1024,
        claim_max_items=20,
        claim_max_bytes=64 * 1024,
        append_timeout_seconds=5,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.files = WorkspaceFiles(self.workspace_path)
        self.directory = self.files.path(directory)
        if self.directory == self.workspace_path:
            raise ValueError("Inbox directory must be a workspace subdirectory")
        limits = {
            "max_read_items": max_read_items,
            "max_pending_items": max_pending_items,
            "max_pending_bytes": max_pending_bytes,
            "max_item_bytes": max_item_bytes,
            "claim_max_items": claim_max_items,
            "claim_max_bytes": claim_max_bytes,
        }
        for key, value in limits.items():
            if type(value) is not int or value < 1:
                raise ValueError(f"{key} must be a positive integer")
            setattr(self, key, value)
        if max_item_bytes > min(max_pending_bytes, claim_max_bytes):
            raise ValueError("max_item_bytes must fit pending and claim budgets")
        if (
            type(append_timeout_seconds) not in (int, float)
            or not math.isfinite(append_timeout_seconds)
            or append_timeout_seconds <= 0
        ):
            raise ValueError("append_timeout_seconds must be positive and finite")
        self.append_timeout_seconds = append_timeout_seconds
        self.changed = asyncio.Condition()
        self._pending = OrderedDict()
        self._history = []
        self._seen = {}
        self._retained = set()
        self._claim = None
        self._owner_fd = None
        self.closed = True
        self.failed = False

    def _path(self, filename):
        return self.files.checked(self.directory / filename)

    @staticmethod
    def _encode(value):
        return (
            json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False, separators=(",", ":")) + "\n"
        ).encode()

    def validate_item(self, value):
        """Validate and normalize one durable queue payload."""
        if hasattr(value, "model_dump"):
            value = value.model_dump(mode="json")
        return InboxItem.model_validate(value).model_dump(mode="json")

    @staticmethod
    def sort_key(_item):
        """Keep producer order unless a typed Inbox supplies an ordering."""
        return 0

    def _check_open(self):
        if self.closed or self.failed:
            raise RuntimeError("Inbox is closed or requires recovery")

    async def _start(self):
        await finish_cleanup(asyncio.to_thread(self._load))
        self.closed = self.failed = False

    def _lines(self, path):
        if not path.exists():
            return
        with path.open("rb+") as file:
            while raw := file.readline():
                if not raw.endswith(b"\n"):
                    file.truncate(file.tell() - len(raw))
                    sync_file(file.fileno())
                    break
                yield json.loads(raw)

    def _load(self):
        self.files.checked(self.directory).mkdir(parents=True, exist_ok=True)
        lock = self._path(".lock")
        if lock.is_symlink():
            raise ValueError("Inbox lock must not be a symlink")
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
            self._pending.clear()
            self._history.clear()
            self._seen.clear()
            self._retained.clear()
            self._claim = None
            for row in self._lines(self._path("inbox_read.jsonl")):
                items = [self.validate_item(item) for item in row.pop("items")]
                receipt = InboxReceipt(item_ids=[item["id"] for item in items], **row)
                for item in items:
                    if item["id"] in self._seen:
                        raise ValueError("Duplicate committed Inbox item")
                    self._seen[item["id"]] = self._encode(item)
                self._history.append((items, receipt))
            for value in self._lines(self._path("inbox_unread.jsonl")):
                item = self.validate_item(value)
                raw = self._encode(item)
                previous = self._seen.get(item["id"])
                if previous is not None:
                    if previous != raw:
                        raise ValueError("Conflicting persisted Inbox item")
                    continue
                if len(raw) > self.max_item_bytes or item["id"] in self._pending:
                    raise ValueError("Invalid persisted Inbox item")
                self._seen[item["id"]] = raw
                self._pending[item["id"]] = item
            if len(self._pending) > self.max_pending_items or self._pending_size() > self.max_pending_bytes:
                raise ValueError("Persisted Inbox exceeds pending limits")
            self._replace(self._path("inbox_unread.jsonl"), b"".join(self._encode(v) for v in self._pending.values()))
        except BaseException:
            os.close(fd)
            raise
        self._owner_fd = fd

    async def _close(self):
        async with self.changed:
            self.closed = True
            self._claim = None
            self.changed.notify_all()
            if self._owner_fd is not None:
                os.close(self._owner_fd)
                self._owner_fd = None

    def _pending_size(self):
        return sum(len(self._encode(item)) for item in self._pending.values())

    @staticmethod
    def _replace(path, raw):
        temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
        try:
            with temporary.open("xb") as file:
                file.write(raw)
                file.flush()
                sync_file(file.fileno())
            os.replace(temporary, path)
            sync_directory(path.parent)
        finally:
            temporary.unlink(missing_ok=True)

    @staticmethod
    def _write(path, raw):
        with path.open("ab") as file:
            file.write(raw)
            file.flush()
            sync_file(file.fileno())
        sync_directory(path.parent)

    async def _persist(self, function, *args):
        try:
            await finish_cleanup(asyncio.to_thread(function, *args))
        except (OSError, ValueError):
            self.failed = True
            self.changed.notify_all()
            raise

    async def append(self, items, *, wait=False):
        """Persist an idempotent batch, optionally waiting for pending capacity."""
        values = items if isinstance(items, list) else [items]
        prepared = {}
        for value in values:
            item = self.validate_item(value)
            raw = self._encode(item)
            if len(raw) > self.max_item_bytes:
                raise InvalidInboxBatchError("item exceeds max_item_bytes")
            previous = prepared.get(item["id"])
            if previous is not None and self._encode(previous) != raw:
                raise InvalidInboxBatchError(f"conflicting item id: {item['id']}")
            prepared[item["id"]] = item
        async with asyncio.timeout(self.append_timeout_seconds) as deadline:
            async with self.changed:
                while True:
                    self._check_open()
                    fresh = {}
                    for key, value in prepared.items():
                        raw = self._encode(value)
                        if key in self._seen and self._seen[key] != raw:
                            raise InvalidInboxBatchError(f"conflicting item id: {key}")
                        if key not in self._seen:
                            fresh[key] = value
                    size = sum(len(self._encode(v)) for v in fresh.values())
                    if len(fresh) > self.max_pending_items or size > self.max_pending_bytes:
                        raise InvalidInboxBatchError("batch exceeds pending limits")
                    if (
                        len(self._pending) + len(fresh) <= self.max_pending_items
                        and self._pending_size() + size <= self.max_pending_bytes
                    ):
                        break
                    if not wait:
                        raise TimeoutError("Inbox pending capacity is full")
                    await self.changed.wait()
                if fresh:
                    deadline.reschedule(None)

                    async def commit_append():
                        await self._persist(
                            self._write,
                            self._path("inbox_unread.jsonl"),
                            b"".join(self._encode(v) for v in fresh.values()),
                        )
                        self._pending.update(fresh)
                        self._seen.update({k: self._encode(v) for k, v in fresh.items()})
                        self.changed.notify_all()

                    await finish_cleanup(commit_append())
                return list(prepared)

    async def claim(self, limit=None, *, item_ids=None):
        """Exclusively select a bounded batch without consuming its inputs."""
        if limit is not None and (type(limit) is not int or limit < 1):
            raise ValueError("limit must be positive")
        async with self.changed:
            self._check_open()
            if self._claim is not None:
                return None
            selected = None if item_ids is None else set(item_ids)
            items, size = [], 0
            for item in sorted(self._pending.values(), key=self.sort_key):
                if selected is not None and item["id"] not in selected:
                    continue
                length = len(self._encode(item))
                if (
                    len(items) >= min(limit or self.claim_max_items, self.claim_max_items)
                    or size + length > self.claim_max_bytes
                ):
                    break
                items.append(item)
                size += length
            if not items:
                return None
            token = uuid4().hex
            self._claim = (token, tuple(item["id"] for item in items))
            return {"token": token, "items": deepcopy(items)}

    async def ack(self, token, *, metadata=None, retain=False):
        """Commit the processing receipt before removing unread inputs."""
        async with self.changed:
            self._check_open()
            if self._claim is None or self._claim[0] != token:
                raise ValueError("Inbox claim expired")
            identifiers = self._claim[1]
            items = [self._pending[key] for key in identifiers]
            receipt = InboxReceipt(
                item_ids=list(identifiers),
                consume_ts=datetime.now(timezone.utc),
                metadata=metadata or {},
            )
            history = [*self._history, (items, receipt)]
            protected = self._retained | (set(identifiers) if retain else set())
            if not self.keep_processed:
                total = sum(len(batch) for batch, _ in history)
                while total > self.max_read_items:
                    old, _ = history[0]
                    if protected.intersection(item["id"] for item in old):
                        raise TimeoutError("Inbox history contains pending handoffs")
                    total -= len(history.pop(0)[0])
            remaining = OrderedDict((key, value) for key, value in self._pending.items() if key not in identifiers)

            def commit():
                raw = b"".join(
                    self._encode({"items": batch, **value.model_dump(mode="json", exclude={"item_ids"})})
                    for batch, value in history
                )
                self._replace(self._path("inbox_read.jsonl"), raw)
                self._replace(self._path("inbox_unread.jsonl"), b"".join(self._encode(v) for v in remaining.values()))

            async def commit_ack():
                await self._persist(commit)
                self._history, self._pending, self._claim = history, remaining, None
                self._retained = protected
                self._seen = {item["id"]: self._encode(item) for batch, _ in history for item in batch}
                self._seen.update({key: self._encode(value) for key, value in remaining.items()})
                self.changed.notify_all()
                return receipt.consume_ts.isoformat()

            return await finish_cleanup(commit_ack())

    async def nack(self, token):
        """Release a live claim while preserving its unread inputs."""
        async with self.changed:
            if self._claim is not None and self._claim[0] == token:
                self._claim = None
                self.changed.notify_all()

    async def snapshot(self):
        """Read queue counts and the latest durable receipt without claiming."""
        async with self.changed:
            self._check_open()
            return deepcopy(
                {
                    "receipt": self._history[-1][1] if self._history else None,
                    "pending_count": len(self._pending),
                    "in_flight_count": len(self._claim[1]) if self._claim else 0,
                },
            )

    async def read(self, *, state="all", offset=0, limit=20):
        """Return an isolated read-only projection."""
        if state not in {"all", "pending", "processing", "processed"}:
            raise ValueError("Invalid Inbox state")
        if type(offset) is not int or offset < 0 or type(limit) is not int or not 1 <= limit <= 100:
            raise ValueError("Invalid Inbox pagination")
        async with self.changed:
            self._check_open()
            claimed = set(self._claim[1]) if self._claim else set()
            items = [
                {"item": item, "state": "processed", "consume_ts": receipt.consume_ts.isoformat()}
                for batch, receipt in self._history
                for item in batch
            ]
            items.extend(
                {"item": item, "state": "processing" if item["id"] in claimed else "pending", "consume_ts": None}
                for item in self._pending.values()
            )
            items = [v for v in items if state in {"all", v["state"]}]
            items.sort(key=lambda row: self.sort_key(row["item"]), reverse=True)
            return deepcopy(
                {"items": items[offset : offset + limit], "total": len(items), "offset": offset, "limit": limit},
            )

    async def pending_ids(self):
        """Return unread identities, including the active claim."""
        async with self.changed:
            self._check_open()
            return set(self._pending)

    async def receipts(self):
        """Return durable processing facts without changing queue state."""
        async with self.changed:
            self._check_open()
            return deepcopy([receipt for _, receipt in self._history])

    async def has_receipt(self, item_ids):
        """Check commitment, including a receipt persisted before an acknowledgement error."""
        identifiers = set(item_ids)
        async with self.changed:
            if not self.failed:
                self._check_open()
                return any(identifiers.intersection(receipt.item_ids) for _, receipt in self._history)

            def persisted():
                return any(
                    identifiers.intersection(item["id"] for item in row["items"])
                    for row in self.files.iter_jsonl(self._path("inbox_read.jsonl"))
                )

            return await asyncio.to_thread(persisted)

    async def retain(self, item_ids, *, release=False):
        """Protect unresolved handoff receipts from history compaction."""
        async with self.changed:
            if release:
                self._retained.difference_update(item_ids)
            else:
                self._retained.update(item_ids)

    async def wait(self, timeout=None):
        """Wait for readable input, shutdown or a bounded timeout."""
        async with self.changed:
            if self.closed or self._pending and self._claim is None:
                return not self.closed
            try:
                async with asyncio.timeout(timeout):
                    await self.changed.wait()
            except TimeoutError:
                return False
            return not self.closed and bool(self._pending) and self._claim is None
