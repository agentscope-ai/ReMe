"""Workspace-contained, durable file operations shared by PA domains."""

from __future__ import annotations

import json
import os
from pathlib import Path
import re
from stat import S_ISREG
from typing import TYPE_CHECKING
from uuid import uuid4

import frontmatter

from ...constants import DEFAULT_MAX_FILE_BYTES

if TYPE_CHECKING:
    from .runtime import PaRuntime


class PaFiles:
    """Own bounded reads, atomic replacement and durable append operations."""

    def __init__(self, runtime: PaRuntime):
        self.runtime = runtime

    def metadata_path(self, name: str) -> Path:
        """Resolve a PA metadata path without symlink aliases."""
        path = self.runtime.component_metadata_path / name
        if (
            not path.resolve().is_relative_to(self.runtime.workspace_path.resolve())
            or path.resolve() != path.absolute()
        ):
            raise ValueError("PA metadata must stay inside the workspace without symlinks")
        return path

    def write_text(self, path: Path, content: str, *, create_only: bool = False) -> None:
        """Atomically replace one contained file, preserving read-only failures."""
        path = self.checked(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            if not path.is_file():
                raise ValueError(f"PA state must be a regular file: {path}")
            if create_only:
                return
            # Respect read-only state files instead of replacing them through directory permissions.
            with path.open("r+"):
                pass
        temporary = path.with_name(f".pa-{uuid4().hex}.tmp")
        try:
            with temporary.open("x", encoding="utf-8") as handle:
                os.chmod(temporary, 0o600)
                handle.write(content)
                handle.flush()
                os.fsync(handle.fileno())
            if create_only:
                try:
                    os.link(temporary, path)
                except FileExistsError:
                    pass
            else:
                os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)

    def write_json(self, path: Path, data) -> None:
        """Atomically replace a human-readable JSON document."""
        self.write_text(path, json.dumps(data, ensure_ascii=False, indent=2) + "\n")

    def append_text(self, path: Path, content: str) -> None:
        """Finish a durable append without following a symlink."""
        path = self.checked(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        flags = os.O_WRONLY | os.O_CREAT | os.O_APPEND | getattr(os, "O_NOFOLLOW", 0)
        with os.fdopen(os.open(path, flags, 0o600), "a", encoding="utf-8") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())

    def append_jsonl(self, path: Path, data: dict) -> None:
        """Append one complete JSONL event."""
        self.append_text(path, json.dumps(data, ensure_ascii=False) + "\n")

    @staticmethod
    def reverse_jsonl_lines(handle):
        """Scan JSONL from EOF without loading its entire history."""
        position = handle.seek(0, os.SEEK_END)
        pending = b""
        while position:
            size = min(position, 8192)
            position -= size
            handle.seek(position)
            lines = (handle.read(size) + pending).split(b"\n")
            pending = lines[0]
            for line in reversed(lines[1:]):
                if line.strip():
                    yield line
        if pending.strip():
            yield pending

    def checked(self, path: Path) -> Path:
        """Reject workspace escapes and symlink aliases before any file operation."""
        resolved = path.resolve()
        if resolved != path.absolute() or not resolved.is_relative_to(self.runtime.workspace_path.resolve()):
            raise ValueError("PA files must stay inside the workspace without symlinks")
        return path

    def identity_path(self, root: Path, identifier: str) -> Path:
        """Resolve a task identity beneath its own domain directory."""
        if not isinstance(identifier, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", identifier):
            raise ValueError("Invalid task_id")
        return self.checked(root / identifier)

    def read(self, path: Path) -> str:
        """Read a complete bounded UTF-8 document."""
        target = self.checked(path)
        if not S_ISREG(target.stat().st_mode):
            raise ValueError(f"Task file must be a regular file: {path}")
        with target.open("rb") as handle:
            content = handle.read(DEFAULT_MAX_FILE_BYTES + 1)
        if len(content) > DEFAULT_MAX_FILE_BYTES:
            raise ValueError(f"Task file exceeds the read limit: {path}")
        return content.decode("utf-8")

    @staticmethod
    def markdown(post: frontmatter.Post) -> str:
        """Serialize a complete task document within the same limit used when reading it."""
        content = frontmatter.dumps(post) + "\n"
        if len(content.encode("utf-8")) > DEFAULT_MAX_FILE_BYTES:
            raise ValueError("Task document exceeds the file read limit after metadata")
        return content
