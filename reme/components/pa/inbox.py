"""A durable Markdown inbox with non-destructive consumption and request receipts."""

# pylint: disable=protected-access

import asyncio
from hashlib import sha256
import json
import os
from stat import S_ISREG
from uuid import uuid4

from ...constants import DEFAULT_MAX_FILE_BYTES
from ...utils.async_utils import complete_in_thread


class ProactiveInbox:
    """Retain user edits; acknowledge a content prefix instead of clearing the source."""

    def __init__(self, runtime):
        self.runtime = runtime
        self.lock = asyncio.Lock()

    @property
    def cursor_path(self):
        """Locate the consumed-prefix checkpoint beside the source inbox."""
        return self.runtime.files.metadata_path("proactive_inbox_cursor.json")

    @property
    def receipts_path(self):
        """Locate the append-only message receipts."""
        return self.runtime.files.metadata_path("proactive_inbox_receipts.jsonl")

    def initialize(self):
        """Seed missing files and recover accepted requests without clearing input."""
        path = self.runtime.proactive_inbox_path
        legacy = self.runtime.files.metadata_path("proactive_inbox.txt")
        if not path.exists() and legacy.exists() and self.runtime.files.read(legacy).strip():
            raise ValueError("Legacy proactive inbox requires offline migration before startup")
        self.runtime.files.write_text(path, "", create_only=True)
        self.runtime.files.write_text(self.receipts_path, "", create_only=True)
        # Requests accepted before a crash must not silently disappear.
        for record in self.receipts().values():
            if record["status"] == "staging":
                self._publish(record)

    def receipts(self):
        """Read the latest persisted state for each request identity."""
        records = {}
        if self.receipts_path.exists():
            descriptor = os.open(self.receipts_path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
            with os.fdopen(descriptor, "rb") as handle:
                if not S_ISREG(os.fstat(handle.fileno()).st_mode):
                    raise ValueError("Proactive requests must be a regular file")
                while line := handle.readline(DEFAULT_MAX_FILE_BYTES + 1):
                    if len(line) > DEFAULT_MAX_FILE_BYTES:
                        raise ValueError("Proactive request exceeds the record read limit")
                    if line.strip():
                        record = json.loads(line)
                        records[record["id"]] = record
        return records

    def _record(self, record):
        content = json.dumps(record, ensure_ascii=False) + "\n"
        if len(content.encode("utf-8")) > DEFAULT_MAX_FILE_BYTES:
            raise ValueError("Proactive request exceeds the record read limit")
        self.runtime.files.append_text(self.receipts_path, content)

    def version(self):
        """Fingerprint current feedback for stale-decision checks."""
        return sha256(self.runtime.files.read(self.runtime.proactive_inbox_path).encode("utf-8")).hexdigest()

    @staticmethod
    def entry(record):
        """Render the receipt-owned Markdown section shared by consumers and views."""
        return f"\n## {record['id']}\n\n{record['created_at']} · Main Agent\n\n{record['message']}\n"

    @staticmethod
    def _display_text(text, receipts):
        """Hide generated request metadata while preserving the current message bodies."""
        result = text
        for identifier, record in receipts.items():
            timestamp = record["created_at"]
            prefix = f"\n## {identifier}\n\n{timestamp} · Main Agent\n\n"
            result = result.replace(prefix, f"\n{timestamp} ", 1)
        return result.strip("\n") if result != text else text

    def _publish(self, record):
        path = self.runtime.proactive_inbox_path
        text = self.runtime.files.read(path)
        heading = f"\n## {record['id']}\n"
        if heading not in text:
            entry = self.entry(record)
            if len((text + entry).encode("utf-8")) > DEFAULT_MAX_FILE_BYTES:
                raise ValueError("Proactive inbox exceeds the file read limit")
            self.runtime.files.append_text(path, entry)
            if heading not in self.runtime.files.read(path):
                raise OSError("Proactive inbox changed during publication; the staged request is retained")
        record = {**record, "status": "queued"}
        self._record(record)
        return record

    async def write(self, message: str):
        """Queue Main's message for the proactive agent to interpret on its next wakeup."""
        if not isinstance(message, str) or not message.strip():
            raise ValueError("message must be a non-empty string")

        def publish():
            record = {
                "id": uuid4().hex,
                "message": message.strip(),
                "created_at": self.runtime.now().isoformat(),
                "status": "staging",
            }
            text = self.runtime.files.read(self.runtime.proactive_inbox_path)
            if len((text + self.entry(record)).encode("utf-8")) > DEFAULT_MAX_FILE_BYTES:
                raise ValueError("Proactive inbox exceeds the file read limit")
            self._record(record)
            return self._publish(record)

        async with self.lock:
            record = await complete_in_thread(publish)
        await self.runtime.event_stream.announce_messages()
        return {
            **record,
            "path": self.runtime.to_workspace_relative(self.runtime.proactive_inbox_path),
            "receipt_path": self.runtime.to_workspace_relative(self.receipts_path),
        }

    def consumed_bytes(self, raw: bytes) -> int:
        """Validate the same consumed prefix for agent reads and desktop views."""
        if not self.cursor_path.exists():
            return 0
        cursor = json.loads(self.runtime.files.read(self.cursor_path))
        if not isinstance(cursor, dict):
            raise ValueError("Invalid proactive inbox cursor")
        offset = cursor.get("bytes", 0)
        if type(offset) is not int or offset < 0:
            raise ValueError("Invalid proactive inbox cursor")
        if offset <= len(raw) and sha256(raw[:offset]).hexdigest() == cursor.get("sha256"):
            return offset
        return 0

    async def read(self, *, round_id, max_bytes) -> str:
        """Return unconsumed text and advance its durable cursor."""
        if type(max_bytes) is not int or max_bytes < 1:
            raise ValueError("max_bytes must be a positive integer")
        current = self.runtime.current_round
        if current is None or current.id != round_id:
            raise ValueError("Inbox consumption requires a live proactive round")
        async with self.lock:
            raw = (await asyncio.to_thread(self.runtime.files.read, self.runtime.proactive_inbox_path)).encode("utf-8")
            if len(raw) > max_bytes:
                raise ValueError("Proactive inbox exceeds the read limit; consumption was not advanced")
            offset = await asyncio.to_thread(self.consumed_bytes, raw)
            text = raw[offset:].decode("utf-8")
            receipts = await asyncio.to_thread(self.receipts)
            result = self._display_text(text, receipts)
            for identifier, record in receipts.items():
                if f"\n## {identifier}\n" in text and record["status"] == "queued":
                    record = {
                        **record,
                        "status": "consumed",
                        "consumed_at": self.runtime.now().isoformat(),
                        "round_id": round_id,
                    }
                    await complete_in_thread(self._record, record)
                    receipts[identifier] = record
            await complete_in_thread(
                self.runtime.files.write_json,
                self.cursor_path,
                {"bytes": len(raw), "sha256": sha256(raw).hexdigest()},
            )
            current.feedback_version = sha256(raw).hexdigest()
            return result
