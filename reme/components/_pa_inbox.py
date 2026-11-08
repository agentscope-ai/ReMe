"""A durable Markdown inbox with non-destructive consumption and request receipts."""

# pylint: disable=protected-access

import asyncio
from hashlib import sha256
import json
import os
from stat import S_ISREG
from uuid import uuid4

from ..constants import DEFAULT_MAX_FILE_BYTES
from ..utils.async_utils import complete_in_thread


class ProactiveInbox:
    """Retain user edits; acknowledge a content prefix instead of clearing the source."""

    def __init__(self, runtime):
        self.runtime = runtime
        self.lock = asyncio.Lock()

    @property
    def cursor_path(self):
        """Locate the consumed-prefix checkpoint beside the source inbox."""
        return self.runtime._metadata_file("proactive_inbox_cursor.json")

    @property
    def receipts_path(self):
        """Locate the append-only message receipts."""
        return self.runtime._metadata_file("proactive_inbox_receipts.jsonl")

    def initialize(self):
        """Seed missing files and recover accepted requests without clearing input."""
        path = self.runtime.proactive_inbox_path
        legacy = self.runtime._metadata_file("proactive_inbox.txt")
        if not path.exists() and legacy.exists() and self.runtime.tasks.read(legacy).strip():
            raise ValueError("Legacy proactive inbox requires offline migration before startup")
        self.runtime._write_text(path, "", create_only=True)
        self.runtime._write_text(self.receipts_path, "", create_only=True)
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
        self.runtime._append_text(self.receipts_path, content)

    def version(self):
        """Fingerprint current feedback for stale-decision checks."""
        return sha256(self.runtime.tasks.read(self.runtime.proactive_inbox_path).encode("utf-8")).hexdigest()

    def pending(self):
        """Detect unread feedback or a settings phase that failed after consuming it."""
        content = self.runtime.tasks.read(self.runtime.proactive_inbox_path)
        if not content:
            return False
        raw = content.encode("utf-8")
        cursor = json.loads(self.runtime.tasks.read(self.cursor_path)) if self.cursor_path.exists() else {}
        if cursor.get("bytes") != len(raw) or cursor.get("sha256") != sha256(raw).hexdigest():
            return True
        pending = False
        with self.runtime._work_context_lock:
            if self.runtime.proactive_context_path.exists():
                for line in self.runtime.tasks.read(self.runtime.proactive_context_path).splitlines():
                    record = json.loads(line)
                    if record["kind"] == "feedback" and record["value"] in content:
                        pending = True
                    elif record["kind"] == "query_config":
                        pending = False
        return pending

    @staticmethod
    def _entry(record):
        return f"\n## {record['id']}\n\n{record['created_at']} · Main Agent\n\n{record['message']}\n"

    def _publish(self, record):
        path = self.runtime.proactive_inbox_path
        text = self.runtime.tasks.read(path)
        heading = f"\n## {record['id']}\n"
        if heading not in text:
            entry = self._entry(record)
            if len((text + entry).encode("utf-8")) > DEFAULT_MAX_FILE_BYTES:
                raise ValueError("Proactive inbox exceeds the file read limit")
            self.runtime._append_text(path, entry)
            if heading not in self.runtime.tasks.read(path):
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
                "created_at": self.runtime._now().isoformat(),
                "status": "staging",
            }
            text = self.runtime.tasks.read(self.runtime.proactive_inbox_path)
            if len((text + self._entry(record)).encode("utf-8")) > DEFAULT_MAX_FILE_BYTES:
                raise ValueError("Proactive inbox exceeds the file read limit")
            self._record(record)
            return self._publish(record)

        async with self.lock:
            record = await complete_in_thread(publish)
        await self.runtime.announce_messages()
        return {
            **record,
            "path": self.runtime.to_workspace_relative(self.runtime.proactive_inbox_path),
            "receipt_path": self.runtime.to_workspace_relative(self.receipts_path),
        }

    async def read(self, *, round_id, max_bytes):
        """Checkpoint unconsumed input before advancing its durable cursor."""
        if type(max_bytes) is not int or max_bytes < 1:
            raise ValueError("max_bytes must be a positive integer")
        current = self.runtime.current_round
        if current is None or current.id != round_id:
            raise ValueError("Inbox consumption requires a live proactive round")
        async with self.lock:
            raw = (await asyncio.to_thread(self.runtime.tasks.read, self.runtime.proactive_inbox_path)).encode("utf-8")
            if len(raw) > max_bytes:
                raise ValueError("Proactive inbox exceeds the read limit; consumption was not advanced")
            cursor = {}
            if self.cursor_path.exists():
                cursor = json.loads(await asyncio.to_thread(self.runtime.tasks.read, self.cursor_path))
            offset = cursor.get("bytes", 0)
            if type(offset) is not int or offset < 0:
                raise ValueError("Invalid proactive inbox cursor")
            prefix_matches = offset <= len(raw) and sha256(raw[:offset]).hexdigest() == cursor.get("sha256")
            text = raw[offset if prefix_matches else 0 :].decode("utf-8")
            if text:
                await self.runtime.checkpoint_proactive(round_id, "feedback", text)
            receipts = await asyncio.to_thread(self.receipts)
            for identifier, record in receipts.items():
                if f"\n## {identifier}\n" in text and record["status"] == "queued":
                    record = {
                        **record,
                        "status": "consumed",
                        "consumed_at": self.runtime._now().isoformat(),
                        "round_id": round_id,
                    }
                    await complete_in_thread(self._record, record)
                    receipts[identifier] = record
            await complete_in_thread(
                self.runtime._write_json,
                self.cursor_path,
                {"bytes": len(raw), "sha256": sha256(raw).hexdigest()},
            )
            current.feedback_version = sha256(raw).hexdigest()
            return {
                "input_path": self.runtime.to_workspace_relative(self.runtime.proactive_context_path) if text else None,
                "round_id": round_id,
                "record_kind": "feedback",
                "receipt_path": self.runtime.to_workspace_relative(self.receipts_path),
                "source_path": self.runtime.to_workspace_relative(self.runtime.proactive_inbox_path),
            }
