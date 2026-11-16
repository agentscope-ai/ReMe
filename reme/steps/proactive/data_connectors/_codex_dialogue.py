"""Read Codex display messages incrementally; all methods run on a worker thread."""

from datetime import datetime
from hashlib import sha256
import json
import os
from pathlib import Path
from stat import S_ISREG
from typing import Callable

from ....components.agent_runtime._internal.files import AgentFiles
from ....constants import DEFAULT_MAX_FILE_BYTES

_WINDOW = 4096


class CodexDialogueIndex:
    """Keep Codex collection progress and unfinished delivery under its resource source."""

    def __init__(self, files: AgentFiles, directory: Path, home: Path, check_stopped: Callable):
        self.files = files
        self.directory = files.checked(directory)
        self.root = (home.expanduser().absolute() / "sessions").resolve()
        self.check_stopped = check_stopped
        self.progress_path = directory / "progress.jsonl"
        self.pending_path = directory / "pending.json"

    def paths(self):
        """Restrict discovery to native rollout files, including old date directories."""
        for path in sorted(self.root.glob("*/*/*/rollout-*.jsonl")):
            self.check_stopped()
            if path.resolve() == path and path.is_file():
                yield path

    @staticmethod
    def _open(path):
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0))
        handle = os.fdopen(descriptor, "rb")
        if not S_ISREG(os.fstat(handle.fileno()).st_mode):
            handle.close()
            raise ValueError("Codex rollout must be a regular file")
        return handle

    def _complete_end(self, handle, size):
        """Find the last newline without parsing history, including an incomplete startup tail."""
        if size:
            handle.seek(size - 1)
            if handle.read(1) == b"\n":
                return size
        end = size
        while end:
            self.check_stopped()
            start = max(0, end - 65536)
            handle.seek(start)
            block = handle.read(end - start)
            if (position := block.rfind(b"\n")) >= 0:
                return start + position + 1
            if size - start > DEFAULT_MAX_FILE_BYTES:
                raise ValueError("Codex rollout has an oversized incomplete record")
            end = start
        return 0

    @staticmethod
    def _signature(handle, offset):
        handle.seek(max(0, offset - _WINDOW))
        return sha256(handle.read(min(offset, _WINDOW))).hexdigest()

    def _cursor(self, handle, stat, offset, meta=None):
        return {
            "dev": stat.st_dev,
            "ino": stat.st_ino,
            "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
            "offset": offset,
            "signature": self._signature(handle, offset),
            "meta": meta,
        }

    def load(self, now):
        """Only first activation skips history; restarts retain their original cursors."""
        records = self.files.iter_jsonl(self.progress_path, incomplete_ok=True)
        state = next(records, None)
        if state is None:
            if self.progress_path.exists() or self.pending_path.exists():
                raise ValueError("Codex progress journal is missing a complete baseline")
            state = {"version": 1, "root": str(self.root), "started_at": now.isoformat(), "files": {}}
            for path in self.paths():
                try:
                    with self._open(path) as handle:
                        stat = os.fstat(handle.fileno())
                        offset = self._complete_end(handle, stat.st_size)
                        state["files"][path.relative_to(self.root).as_posix()] = self._cursor(handle, stat, offset)
                except FileNotFoundError:
                    continue
            self._validate(state)
            self.files.write_jsonl(self.progress_path, [{"type": "baseline", **state}])
            state["known_ids"] = {}
        else:
            if not isinstance(state, dict) or state.pop("type", None) != "baseline":
                raise ValueError("Codex progress journal must begin with a baseline")
            self._validate(state)
            state["known_ids"] = {}
            for record in records:
                self.check_stopped()
                self._apply(state, record)
        return state

    def _apply(self, state, record):
        if not isinstance(record, dict) or record.get("type") != "commit":
            raise ValueError("Invalid Codex progress record")
        fields = {"id": str, "updates": dict, "next_path": str, "message_ids": list}
        if any(not isinstance(record.get(key), kind) for key, kind in fields.items()):
            raise ValueError("Invalid Codex progress record")
        if any(not isinstance(value, str) for value in record["message_ids"]):
            raise ValueError("Invalid Codex progress record")
        self._validate({**state, "files": record["updates"]})
        state["files"].update(record["updates"])
        state["last_batch"] = record["id"]
        state["next_path"] = record["next_path"]
        state["known_ids"].update(dict.fromkeys(record["message_ids"]))

    def _validate(self, state):
        """Reject corrupt offsets before they can become unbounded reads or skipped data."""
        if state.get("version") != 1 or state.get("root") != str(self.root):
            raise ValueError("Codex checkpoint belongs to a different source directory or version")
        if not isinstance(state.get("files"), dict):
            raise ValueError("Invalid Codex file index")
        if datetime.fromisoformat(state["started_at"]).tzinfo is None:
            raise ValueError("Codex baseline timestamp must include a timezone")
        for cursor in state["files"].values():
            if not isinstance(cursor, dict) or any(
                type(cursor.get(key)) is not int or cursor[key] < 0
                for key in ("dev", "ino", "size", "mtime_ns", "offset")
            ):
                raise ValueError("Invalid Codex file cursor")
            if cursor["offset"] > cursor["size"] or not isinstance(cursor.get("signature"), str):
                raise ValueError("Invalid Codex file cursor boundary")

    def commit(self, state, transaction):
        """Append confirmed progress once before clearing the unfinished batch."""
        unchanged_cursor = transaction["next_path"] == state.get("next_path", "")
        if not transaction["updates"] and not transaction["message_ids"] and unchanged_cursor:
            return
        if state.get("last_batch") != transaction["id"]:
            record = {
                "type": "commit",
                **{key: transaction[key] for key in ("id", "updates", "message_ids", "next_path")},
            }
            updated = {**state, "files": dict(state["files"]), "known_ids": dict(state["known_ids"])}
            self._apply(updated, record)
            self.files.append_jsonl(self.progress_path, record)
            state.update(updated)

    @staticmethod
    def _meta(handle):
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
            "session_id": payload.get("session_id") or payload["id"],
            "cwd": payload.get("cwd", ""),
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
        if kind != "UserMessage" and not (kind == "AgentMessage" and item.get("phase") == "final_answer"):
            return None
        timestamp = datetime.fromisoformat(record["timestamp"].replace("Z", "+00:00"))
        if timestamp.tzinfo is None:
            raise ValueError("Codex message timestamps must include a timezone")
        if timestamp < started_at:
            return None
        parts = item.get("content", [])
        text = "\n".join(part["text"] for part in parts if part.get("type") in {"text", "Text"})
        attachments = sum(part.get("type") in {"image", "local_image", "Image"} for part in parts)
        if not text and not attachments:
            return None
        return {
            "timestamp": timestamp.isoformat(),
            "thread_id": meta["thread_id"],
            "session_id": meta["session_id"],
            "turn_id": payload.get("turn_id"),
            "message_id": item["id"],
            "cwd": meta["cwd"],
            "role": "user" if kind == "UserMessage" else "assistant",
            "content": text,
            "image_attachments": attachments,
        }

    def scan(self, state, max_scan_bytes, max_output_bytes):
        """Bound each pass; tool-only records advance cursors without producing events."""
        updates, messages, message_ids, errors = {}, [], set(), {}
        known = state["known_ids"]
        scanned, output_size = 0, 0
        started_at = datetime.fromisoformat(state["started_at"])
        paths = list(self.paths())
        # Resume at the next file after a bounded pass so a busy old thread cannot starve other projects.
        anchor = state.get("next_path", "")
        paths.sort(key=lambda path: (path.relative_to(self.root).as_posix() < anchor, str(path)))
        next_path = ""
        for path in paths:
            self.check_stopped()
            relative = path.relative_to(self.root).as_posix()
            previous = state["files"].get(relative)
            initial_output_size = output_size
            budget_reached = False
            try:
                stat = path.stat()
                if (
                    previous
                    and all(
                        previous[key] == value
                        for key, value in (
                            ("dev", stat.st_dev),
                            ("ino", stat.st_ino),
                            ("size", stat.st_size),
                            ("mtime_ns", stat.st_mtime_ns),
                        )
                    )
                    and previous["offset"] == stat.st_size
                ):
                    continue
                with self._open(path) as handle:
                    stat = os.fstat(handle.fileno())
                    reset = previous is None or (previous["dev"], previous["ino"]) != (stat.st_dev, stat.st_ino)
                    offset = 0 if previous is None else previous["offset"]
                    if offset > stat.st_size or (previous and self._signature(handle, offset) != previous["signature"]):
                        reset = True
                    if previous and stat.st_size == previous["size"] and stat.st_mtime_ns != previous["mtime_ns"]:
                        reset = True
                    meta = self._meta(handle) if reset or not previous.get("meta") else previous["meta"]
                    if meta is None:
                        continue
                    if not meta["user"]:
                        offset = self._complete_end(handle, stat.st_size)
                        updates[relative] = self._cursor(handle, stat, offset, meta)
                        continue
                    if reset:
                        offset = 0
                    handle.seek(offset)
                    file_messages, file_ids, file_bytes = [], set(), 0
                    while handle.tell() < stat.st_size:
                        self.check_stopped()
                        start = handle.tell()
                        line = handle.readline(min(DEFAULT_MAX_FILE_BYTES + 1, stat.st_size - start))
                        if len(line) > DEFAULT_MAX_FILE_BYTES:
                            raise ValueError("Codex record exceeds the record limit; cursor was retained")
                        if not line.endswith(b"\n"):
                            break
                        if scanned + file_bytes and scanned + file_bytes + len(line) > max_scan_bytes:
                            budget_reached = True
                            break
                        message = self._message(json.loads(line), meta, started_at)
                        if message is not None:
                            key = f"{message['session_id']}:{message['message_id']}"
                            if key not in message_ids and key not in file_ids and key not in known:
                                size = len((json.dumps(message, ensure_ascii=False) + "\n").encode())
                                if output_size + size > max_output_bytes:
                                    if size > max_output_bytes:
                                        raise ValueError("Codex dialogue message exceeds the resource byte limit")
                                    budget_reached = True
                                    break
                                file_messages.append(message)
                                file_ids.add(key)
                                output_size += size
                        offset = handle.tell()
                        file_bytes += len(line)
                    updates[relative] = self._cursor(handle, stat, offset, meta)
                    messages.extend(file_messages)
                    message_ids.update(file_ids)
                    scanned += file_bytes
            except FileNotFoundError:
                continue
            except (OSError, ValueError, KeyError, TypeError) as exc:
                output_size = initial_output_size
                errors[relative] = str(exc)
            if budget_reached or scanned >= max_scan_bytes or output_size >= max_output_bytes:
                next_path = relative + "\x00"
                break
        return {
            "updates": updates,
            "message_ids": sorted(message_ids),
            "messages": messages,
            "next_path": next_path,
            "errors": errors,
            "scanned_bytes": scanned,
        }
