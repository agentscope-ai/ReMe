"""Bounded, workspace-contained persistence for agent configuration and receipts."""

import json
import os
from pathlib import Path, PureWindowsPath
from stat import S_ISREG
from uuid import uuid4

from ....constants import DEFAULT_MAX_FILE_BYTES
from ....utils.file_sync import sync_directory, sync_file


class AgentFiles:
    """Atomically publish complete files; reject aliases and workspace escapes."""

    def __init__(self, workspace: Path):
        self.workspace = workspace.resolve()

    def path(self, relative: str | Path) -> Path:
        """Resolve only workspace-relative paths without following symlinks."""
        raw = str(relative)
        value = Path(raw)
        if not raw or raw.startswith("~") or "\\" in raw or PureWindowsPath(raw).anchor or value.is_absolute():
            raise ValueError("agent runtime paths must be workspace-relative")
        if ".." in value.parts:
            raise ValueError("agent runtime paths must stay inside the workspace")
        return self.checked(self.workspace / value)

    def checked(self, path: Path) -> Path:
        """Validate internal absolute paths again at each filesystem boundary."""
        path = path.absolute()
        if path.resolve() != path or not path.is_relative_to(self.workspace):
            raise ValueError("agent runtime paths must stay inside the workspace without symlinks")
        return path

    def read(self, path: Path) -> str:
        """Read a complete regular UTF-8 file within the application byte limit."""
        path = self.checked(path)
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0))
        with os.fdopen(descriptor, "rb") as handle:
            if not S_ISREG(os.fstat(handle.fileno()).st_mode):
                raise ValueError("agent runtime state must be a regular file")
            data = handle.read(DEFAULT_MAX_FILE_BYTES + 1)
        if len(data) > DEFAULT_MAX_FILE_BYTES:
            raise ValueError("agent runtime file exceeds the byte limit")
        return data.decode("utf-8")

    def read_json(self, path: Path, default):
        """Only an absent file receives a default; corruption remains explicit."""
        try:
            return json.loads(self.read(path))
        except FileNotFoundError:
            return default

    def _mkdir(self, path):
        missing, parent = [], path
        while not parent.exists():
            missing.append(parent)
            parent = parent.parent
        path.mkdir(parents=True, exist_ok=True)
        self.checked(path)
        for directory in reversed(missing):
            sync_directory(directory.parent)

    def write(self, path: Path, content: str, *, create_only: bool = False) -> None:
        """Finish a bounded, atomic replacement before its caller acknowledges work."""
        path = self.checked(path)
        data = content.encode("utf-8")
        if len(data) > DEFAULT_MAX_FILE_BYTES:
            raise ValueError("agent runtime file exceeds the byte limit")
        self._mkdir(path.parent)
        self.checked(path)
        if path.exists():
            if not S_ISREG(path.stat().st_mode):
                raise ValueError("agent runtime state must be a regular file")
            if create_only:
                return
            with path.open("r+"):
                pass
        temporary = path.with_name(f".agent-{uuid4().hex}.tmp")
        try:
            with temporary.open("xb") as handle:
                os.chmod(temporary, 0o600)
                handle.write(data)
                handle.flush()
                sync_file(handle.fileno())
            if create_only:
                try:
                    os.link(temporary, path)
                except FileExistsError:
                    pass
            else:
                os.replace(temporary, path)
            sync_directory(path.parent)
        finally:
            temporary.unlink(missing_ok=True)

    def write_json(self, path: Path, value) -> None:
        """Keep user-owned configuration readable without private database state."""
        self.write(path, json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")

    def write_jsonl(self, path: Path, values) -> None:
        """Replace a bounded reference view only when its contents changed."""
        content = "".join(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n" for value in values)
        try:
            if self.read(path) == content:
                return
        except FileNotFoundError:
            pass
        self.write(path, content)

    def iter_jsonl(self, path: Path, *, incomplete_ok=False):
        """Stream an unbounded session history while bounding each complete record."""
        path = self.checked(path)
        try:
            descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0))
        except FileNotFoundError:
            return
        with os.fdopen(descriptor, "rb") as handle:
            if not S_ISREG(os.fstat(handle.fileno()).st_mode):
                raise ValueError("session history must be a regular file")
            while line := handle.readline(DEFAULT_MAX_FILE_BYTES + 1):
                if len(line) > DEFAULT_MAX_FILE_BYTES:
                    raise ValueError("JSONL contains an oversized record")
                if not line.endswith(b"\n"):
                    if incomplete_ok:
                        return
                    raise ValueError("JSONL contains an incomplete record")
                yield json.loads(line)

    def append_jsonl(self, path: Path, value) -> None:
        """Publish one session reference durably without rewriting retained history."""
        path = self.checked(path)
        data = (json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")
        if len(data) > DEFAULT_MAX_FILE_BYTES:
            raise ValueError("session record exceeds the byte limit")
        self._mkdir(path.parent)
        self.checked(path)
        flags = os.O_RDWR | os.O_APPEND | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
        descriptor = os.open(path, flags, 0o600)
        with os.fdopen(descriptor, "ab+") as handle:
            if not S_ISREG(os.fstat(handle.fileno()).st_mode):
                raise ValueError("session history must be a regular file")
            handle.seek(0, os.SEEK_END)
            size = handle.tell()
            if size:
                handle.seek(-1, os.SEEK_END)
                if handle.read(1) != b"\n":
                    handle.seek(max(0, size - DEFAULT_MAX_FILE_BYTES))
                    tail = handle.read()
                    if b"\n" not in tail and size > len(tail):
                        raise ValueError("JSONL contains an oversized incomplete record")
                    size -= len(tail) - tail.rfind(b"\n") - 1
                    handle.truncate(size)
            try:
                handle.write(data)
                handle.flush()
                sync_file(handle.fileno())
                sync_directory(path.parent)
            except BaseException:
                handle.truncate(size)
                handle.flush()
                sync_file(handle.fileno())
                raise
