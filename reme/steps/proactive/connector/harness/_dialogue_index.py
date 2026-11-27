"""Shared bounded JSONL scanning, durable cursors and normalized dialogue attachments."""

from abc import ABC, abstractmethod
from base64 import b64decode
from copy import deepcopy
from datetime import datetime
from hashlib import sha256
import json
import os
from pathlib import Path
from stat import S_ISREG
from typing import Callable

from ....file_io._path import IMAGE_MIME_BY_EXT
from .....utils.files.workspace_files import WorkspaceFiles
from .....constants import DEFAULT_MAX_FILE_BYTES

_WINDOW = 4096
_IMAGE_EXTENSIONS = {mime: extension for extension, mime in IMAGE_MIME_BY_EXT.items()}


class _OutputFull(Exception):
    """Retry the current record after publishing the current bounded package."""


class BaseDialogueIndex(ABC):
    """Keep source-owned parser state inside durable, shared file cursors."""

    def __init__(self, files: WorkspaceFiles, directory: Path, root: Path, check_stopped: Callable):
        self.files = files
        self.directory = files.checked(directory)
        self.root = root.expanduser().resolve()
        self.check_stopped = check_stopped
        self.progress_path = directory / "progress.jsonl"
        self.pending_path = directory / "pending.json"

    @abstractmethod
    def paths(self):
        """Yield native top-level transcript paths without following symlinks."""

    @abstractmethod
    def _meta(self, handle, offset=0):
        """Read source context, including the active turn at an initial baseline."""

    @abstractmethod
    def parse(self, record, meta, started_at):
        """Advance parser state and return normalized (message, attachment) pairs."""

    def finish(self, _meta, _started_at):
        """Release source messages whose completion is established at the file tail."""
        return []

    @staticmethod
    def _open(path):
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0))
        handle = os.fdopen(descriptor, "rb")
        if not S_ISREG(os.fstat(handle.fileno()).st_mode):
            handle.close()
            raise ValueError("Harness rollout must be a regular file")
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
                raise ValueError("Harness rollout has an oversized incomplete record")
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
                raise ValueError("Harness progress journal is missing a complete baseline")
            state = {"version": 1, "root": str(self.root), "started_at": now.isoformat(), "files": {}}
            for path in self.paths():
                try:
                    with self._open(path) as handle:
                        stat = os.fstat(handle.fileno())
                        offset = self._complete_end(handle, stat.st_size)
                        state["files"][path.relative_to(self.root).as_posix()] = self._cursor(
                            handle,
                            stat,
                            offset,
                            self._meta(handle, offset),
                        )
                except FileNotFoundError:
                    continue
            self._validate(state)
            self.files.write_jsonl(self.progress_path, [{"type": "baseline", **state}])
            state["known_ids"] = {}
        else:
            if not isinstance(state, dict) or state.pop("type", None) != "baseline":
                raise ValueError("Harness progress journal must begin with a baseline")
            self._validate(state)
            state["known_ids"] = {}
            for record in records:
                self.check_stopped()
                self._apply(state, record)
        return state

    def _apply(self, state, record):
        if not isinstance(record, dict) or record.get("type") != "commit":
            raise ValueError("Invalid Harness progress record")
        fields = {"id": str, "updates": dict, "next_path": str, "message_ids": list}
        if any(not isinstance(record.get(key), kind) for key, kind in fields.items()):
            raise ValueError("Invalid Harness progress record")
        if any(not isinstance(value, str) for value in record["message_ids"]):
            raise ValueError("Invalid Harness progress record")
        self._validate({**state, "files": record["updates"]})
        state["files"].update(record["updates"])
        state["last_batch"] = record["id"]
        state["next_path"] = record["next_path"]
        state["known_ids"].update(dict.fromkeys(record["message_ids"]))

    def _validate(self, state):
        """Reject corrupt offsets before they can become unbounded reads or skipped data."""
        if state.get("version") != 1 or state.get("root") != str(self.root):
            raise ValueError("Harness checkpoint belongs to a different source directory or version")
        if not isinstance(state.get("files"), dict):
            raise ValueError("Invalid Harness file index")
        if datetime.fromisoformat(state["started_at"]).tzinfo is None:
            raise ValueError("Harness baseline timestamp must include a timezone")
        for cursor in state["files"].values():
            if not isinstance(cursor, dict) or any(
                type(cursor.get(key)) is not int or cursor[key] < 0
                for key in ("dev", "ino", "size", "mtime_ns", "offset")
            ):
                raise ValueError("Invalid Harness file cursor")
            if cursor["offset"] > cursor["size"] or not isinstance(cursor.get("signature"), str):
                raise ValueError("Invalid Harness file cursor boundary")

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

    def _attachment_files(self, _handle, _offset, attachments, key, budget):
        """Decode embedded images without reading arbitrary source-provided paths."""
        identity = sha256(key.encode()).hexdigest()
        files = {}
        for index, part in enumerate(attachments):
            self.check_stopped()
            url = part.get("image_url")
            if not isinstance(url, str):
                continue
            header, separator, encoded = url.partition(",")
            extension = _IMAGE_EXTENSIONS.get(header.removeprefix("data:").removesuffix(";base64"))
            if not separator or extension is None or not header.startswith("data:") or not header.endswith(";base64"):
                continue
            name = f"image-{identity}-{index + 1}{extension}"
            overhead = len(name) + 4  # Bounds the JSON quotes and separator for this path.
            try:
                data = b64decode(encoded, validate=True)
            except ValueError:
                continue
            if data and len(data) + overhead <= budget:
                files[name] = data
                budget -= len(data) + overhead
        return files

    def _pack(self, parsed, handle, offset, known, budget, limit):
        messages, images, identifiers, size = [], {}, set(), 0
        for message, attachments in parsed:
            key = f"{message['session_id']}:{message['message_id']}"
            if key in identifiers or any(key in group for group in known):
                continue
            if attachments:
                message["image_paths"] = []
            length = len((json.dumps(message, ensure_ascii=False) + "\n").encode())
            if length > limit:
                raise ValueError("Harness dialogue message exceeds the resource byte limit")
            if attachments:
                payloads = self._attachment_files(handle, offset, attachments, key, limit - length)
                message["image_paths"] = list(payloads)
                length += sum(len(data) + len(name) + 4 for name, data in payloads.items())
                images.update(payloads)
            size += length
            if size > limit:
                raise ValueError("Harness record produces dialogue exceeding the resource byte limit")
            if size > budget:
                raise _OutputFull
            messages.append(message)
            identifiers.add(key)
        return messages, images, identifiers, size

    def scan(self, state, max_scan_bytes, max_output_bytes):
        """Bound each pass; tool-only records advance cursors without producing events."""
        updates, messages, images, message_ids, errors = {}, [], {}, set(), {}
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
                    and not (previous.get("meta") or {}).get("pending")
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
                    meta = self._meta(handle) if reset or not previous.get("meta") else deepcopy(previous["meta"])
                    if meta is None:
                        continue
                    if not meta["user"]:
                        offset = self._complete_end(handle, stat.st_size)
                        updates[relative] = self._cursor(handle, stat, offset, meta)
                        continue
                    if reset:
                        offset = 0
                    handle.seek(offset)
                    file_messages, file_images, file_ids, file_bytes = [], {}, set(), 0
                    while True:
                        self.check_stopped()
                        start = handle.tell()
                        at_end = start == stat.st_size
                        line = handle.readline(min(DEFAULT_MAX_FILE_BYTES + 1, stat.st_size - start))
                        if len(line) > DEFAULT_MAX_FILE_BYTES:
                            raise ValueError("Harness record exceeds the record limit; cursor was retained")
                        if not at_end and not line.endswith(b"\n"):
                            break
                        if not at_end and scanned + file_bytes and scanned + file_bytes + len(line) > max_scan_bytes:
                            budget_reached = True
                            break
                        before = deepcopy(meta)
                        if at_end:
                            if os.fstat(handle.fileno()).st_size != stat.st_size:
                                break
                            parsed = self.finish(meta, started_at)
                        else:
                            record = json.loads(line)
                            if not isinstance(record, dict):
                                raise ValueError("Harness record must be an object")
                            parsed = self.parse(record, meta, started_at)
                        try:
                            rows, payloads, identifiers, size = self._pack(
                                parsed,
                                handle,
                                start,
                                (known, message_ids, file_ids),
                                max_output_bytes - output_size,
                                max_output_bytes,
                            )
                        except _OutputFull:
                            meta = before
                            budget_reached = True
                            break
                        file_messages.extend(rows)
                        file_images.update(payloads)
                        file_ids.update(identifiers)
                        output_size += size
                        offset = handle.tell()
                        file_bytes += len(line)
                        if at_end:
                            break
                    cursor = self._cursor(handle, stat, offset, meta)
                    if cursor != previous:
                        updates[relative] = cursor
                    messages.extend(file_messages)
                    images.update(file_images)
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
            "images": images,
            "next_path": next_path,
            "errors": errors,
            "scanned_bytes": scanned,
        }
