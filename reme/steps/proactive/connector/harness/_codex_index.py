"""Codex native rollout adapter for the shared dialogue index."""

from datetime import datetime
import json

from ._dialogue_index import BaseDialogueIndex
from .....constants import DEFAULT_MAX_FILE_BYTES


class CodexDialogueIndex(BaseDialogueIndex):
    """Read displayed Codex messages, excluding ReMe and subagent sessions."""

    def paths(self):
        """Discover native rollouts across all date directories."""
        for path in sorted(self.root.glob("*/*/*/rollout-*.jsonl")):
            self.check_stopped()
            if path.resolve() == path and path.is_file():
                yield path

    def parse(self, record, meta, started_at):
        """Codex completed items already have stable identity and phase."""
        message = self._message(record, meta, started_at)
        return [] if message is None else [message]

    @staticmethod
    def _meta(handle, offset=0):
        handle.seek(0)
        line = handle.readline(DEFAULT_MAX_FILE_BYTES + 1)
        if not line.endswith(b"\n"):
            return None
        if len(line) > DEFAULT_MAX_FILE_BYTES:
            raise ValueError("Codex session metadata exceeds the record limit")
        record = json.loads(line)
        if record.get("type") != "session_meta":
            raise ValueError("Codex rollout must begin with session_meta")
        payload = record["payload"]
        source = payload.get("source")
        return {
            "thread_id": payload["id"],
            "session_id": payload["session_id"],
            "cwd": payload["cwd"],
            "user": payload.get("thread_source", "user") == "user"
            and not isinstance(source, dict)
            and payload.get("originator") != "reme",
        }

    @staticmethod
    def _message(record, meta, started_at):
        if record.get("type") != "event_msg":
            return None
        payload = record.get("payload", {})
        if payload.get("type") != "item_completed":
            return None
        item = payload.get("item", {})
        kind = item.get("type")
        is_user = kind == "UserMessage"
        phase = item.get("phase")
        if not is_user and not (kind == "AgentMessage" and phase in {"commentary", "final_answer"}):
            return None
        timestamp = datetime.fromisoformat(record["timestamp"].replace("Z", "+00:00"))
        if timestamp.tzinfo is None:
            raise ValueError("Codex message timestamps must include a timezone")
        if timestamp < started_at:
            return None
        parts = item["content"]
        text_type = "text" if is_user else "Text"
        text = "\n".join(part["text"] for part in parts if part["type"] == text_type)
        attachments = [part for part in parts if part["type"] in {"image", "local_image"}] if is_user else []
        if not text and not attachments:
            return None
        message = {
            "timestamp": timestamp.isoformat(),
            "thread_id": meta["thread_id"],
            "session_id": meta["session_id"],
            "turn_id": payload["turn_id"],
            "message_id": item["id"],
            "cwd": meta["cwd"],
            "role": "user" if is_user else "assistant",
            "content": text,
            "image_attachments": len(attachments),
        }
        if not is_user:
            message["phase"] = phase
        if attachments:
            message["image_paths"] = []
        return message, attachments

    @staticmethod
    def _input_images(record):
        """Find the model-facing image references without collecting the model-facing text."""
        payload = record.get("payload", {})
        if record.get("type") != "response_item" or payload.get("type") != "message" or payload.get("role") != "user":
            return []
        return [part for part in payload.get("content", []) if part.get("type") == "input_image"]

    def _previous_input_images(self, handle, message_offset):
        """Read the preceding complete record without changing the scan position."""
        end = handle.tell()
        try:
            start = self._complete_end(handle, message_offset - 1)
            length = message_offset - start
            if length <= DEFAULT_MAX_FILE_BYTES:
                handle.seek(start)
                return self._input_images(json.loads(handle.read(length)))
        except ValueError:
            pass
        finally:
            handle.seek(end)
        return []

    def _attachment_files(self, handle, message_offset, attachments, key, budget):
        """Recover local-image bytes from Codex's adjacent user input, including after a scan boundary."""
        local = any(part["type"] == "local_image" for part in attachments)
        input_images = self._previous_input_images(handle, message_offset) if local else []
        use_input = len(input_images) == len(attachments)
        resolved = [
            input_images[index] if use_input and part["type"] == "local_image" else part
            for index, part in enumerate(attachments)
        ]
        return super()._attachment_files(handle, message_offset, resolved, key, budget)
