"""Claude Code transcript projection for the shared incremental dialogue index."""

from datetime import datetime
import json
from uuid import UUID

from .base_dialogue_index import BaseDialogueIndex
from .....constants import DEFAULT_MAX_FILE_BYTES


class ClaudeCodeDialogueIndex(BaseDialogueIndex):
    """Aggregate native API blocks while retaining their real message and turn identities."""

    def paths(self):
        """Only watch top-level sessions, never nested subagent transcripts."""
        for path in sorted(self.root.glob("*/*.jsonl")):
            self.check_stopped()
            try:
                UUID(path.stem)
            except ValueError:
                continue
            if path.resolve() == path and path.is_file():
                yield path

    @staticmethod
    def _visible(record):
        raw = record.get("message") or {}
        if not isinstance(raw, dict):
            raise ValueError("Claude Code message must be an object")
        return (
            not any(record.get(key) for key in ("isSidechain", "isMeta", "isCompactSummary", "isSynthetic", "teamName"))
            and raw.get("model") != "<synthetic>"
        )

    @staticmethod
    def _content(raw):
        content = raw.get("content", [])
        if isinstance(content, str):
            return [{"type": "text", "text": content}]
        if not isinstance(content, list) or any(not isinstance(part, dict) for part in content):
            raise ValueError("Claude Code content must be text or content blocks")
        return content

    @classmethod
    def _prompt(cls, record):
        if record.get("type") != "user" or not cls._visible(record):
            return False
        origin = record.get("origin") or {}
        if not isinstance(origin, dict) or origin.get("kind") not in (None, "human"):
            return False
        parts = cls._content(record.get("message") or {})
        text = "\n".join(part["text"] for part in parts if part.get("type") == "text")
        return not text.lstrip().startswith(("<local-command-", "<command-name>", "<command-message>")) and any(
            part.get("type") in ("text", "image") for part in parts
        )

    def _meta(self, handle, offset=0):
        meta = {"user": True, "turn_id": None, "pending": None}
        # Recover an active pre-baseline turn without importing any of its messages.
        end = offset
        while end:
            self.check_stopped()
            start = self._complete_end(handle, end - 1)
            handle.seek(start)
            record = json.loads(handle.read(end - start))
            if self._prompt(record):
                meta["turn_id"] = record["uuid"]
                break
            end = start
        return meta

    @staticmethod
    def _row(record, meta, identifier, text, attachments=0):
        stamp = datetime.fromisoformat(record["timestamp"])
        if stamp.tzinfo is None:
            raise ValueError("Claude Code message timestamps must include a timezone")
        return {
            "timestamp": stamp.isoformat(),
            "thread_id": record["sessionId"],
            "session_id": record["sessionId"],
            "turn_id": meta["turn_id"],
            "message_id": identifier,
            "cwd": record["cwd"],
            "role": record["type"],
            "content": text,
            "image_attachments": attachments,
        }

    @staticmethod
    def _flush(meta, started_at):
        pending, meta["pending"] = meta["pending"], None
        if pending is None:
            return []
        row = pending["row"]
        row["content"] = "\n".join(pending["blocks"].values())
        row["phase"] = "final_answer" if pending["stop_reason"] == "end_turn" else "commentary"
        if not row["content"] or datetime.fromisoformat(row["timestamp"]) < started_at:
            return []
        return [(row, [])]

    def parse(self, record, meta, started_at):
        role = record.get("type")
        if role not in ("user", "assistant"):
            return (
                self._flush(meta, started_at) if role == "system" and record.get("subtype") == "turn_duration" else []
            )
        if not self._visible(record):
            return []
        raw = record.get("message") or {}
        parts = self._content(raw)
        if role == "user":
            if any(part.get("type") == "tool_result" for part in parts):
                return self._flush(meta, started_at)
            if not self._prompt(record):
                return []
            rows = self._flush(meta, started_at)
            meta["turn_id"] = record["uuid"]
            text = "\n".join(part["text"] for part in parts if part.get("type") == "text")
            attachments = [{"image_url": self._image_url(part)} for part in parts if part.get("type") == "image"]
            row = self._row(record, meta, record["uuid"], text, len(attachments))
            if datetime.fromisoformat(row["timestamp"]) >= started_at and (text or attachments):
                rows.append((row, attachments))
            return rows
        identifier = raw.get("id") or record["uuid"]
        pending = meta["pending"]
        rows = self._flush(meta, started_at) if pending and pending["row"]["message_id"] != identifier else []
        if meta["pending"] is None:
            meta["pending"] = {
                "row": self._row(record, meta, identifier, ""),
                "blocks": {},
                "stop_reason": None,
            }
        pending = meta["pending"]
        text = "\n".join(part["text"] for part in parts if part.get("type") == "text")
        if text:
            pending["blocks"][str(record.get("apiBlockIndex", record["uuid"]))] = text
        if raw.get("stop_reason"):
            pending["stop_reason"] = raw["stop_reason"]
        if sum(len(value.encode()) for value in pending["blocks"].values()) > DEFAULT_MAX_FILE_BYTES:
            raise ValueError("Claude Code message exceeds the record limit")
        return rows

    def finish(self, meta, started_at):
        pending = meta["pending"]
        return self._flush(meta, started_at) if pending and pending["stop_reason"] else []

    @staticmethod
    def _image_url(part):
        source = part.get("source") or {}
        if isinstance(source, dict) and source.get("type") == "base64":
            return f"data:{source.get('media_type')};base64,{source.get('data', '')}"
        return None
