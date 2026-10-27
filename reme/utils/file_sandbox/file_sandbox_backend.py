"""Common contract and path validation for subprocess file sandboxes."""

import asyncio
import os
import sys
from abc import abstractmethod
from collections import deque
from pathlib import Path

from agentscope.tool import ExecResult

from ..local_backend import LocalProcessBackend


def contains(root: str, path: str) -> bool:
    """Whether a path is inside a configured root."""
    return os.path.commonpath((root, path)) == root


class FileSandboxBackend(LocalProcessBackend):
    """Execute subprocesses with explicit host-path read and write access."""

    def __init__(self, *, cwd: str | os.PathLike[str], read_only_paths: list[str], writable_paths: list[str]) -> None:
        self._cwd = self._validate_path(os.fspath(cwd))
        if not os.path.isdir(self._cwd):
            raise ValueError("sandbox working directory must be a directory")
        self._read_only_paths = self._validate_paths(read_only_paths, "read_only_paths")
        self._writable_paths = self._validate_paths(writable_paths, "writable_paths")
        if not any(contains(path, self._cwd) for path in self._read_only_paths + self._writable_paths):
            self._read_only_paths.append(self._cwd)
        # A virtual environment can point outside sys.prefix to its base
        # interpreter and standard library (notably with Homebrew Python).
        for runtime in (sys.prefix, sys.base_prefix, sys.executable):
            path = os.path.realpath(runtime)
            if path in ("/", "/usr", "/usr/local") or path.startswith("/usr/"):
                continue
            if not any(contains(root, path) for root in self._read_only_paths + self._writable_paths):
                self._read_only_paths.append(self._validate_path(path))
        paths = self._read_only_paths + self._writable_paths
        if len(set(paths)) != len(paths):
            raise ValueError("sandbox paths must not contain duplicates")
        self._path_identities = {path: self._identity(path) for path in paths}

    @staticmethod
    def _runtime_paths() -> tuple[set[str], set[str]]:
        """Collect runtime symlinks and traversal paths without exposing parent trees."""
        links = set()
        metadata = set()
        for runtime in (sys.prefix, sys.base_prefix, sys.executable):
            pending = deque(Path(runtime).parts)
            current = "/"
            followed = 0
            while pending:
                part = pending.popleft()
                if part == "/":
                    current = "/"
                    continue
                if part == "..":
                    current = os.path.dirname(current)
                    continue
                path = os.path.join(current, part)
                if os.path.islink(path):
                    followed += 1
                    if followed > 40:
                        raise RuntimeError(f"too many runtime symlinks: {runtime}")
                    links.add(path)
                    pending.extendleft(reversed(Path(os.readlink(path)).parts))
                else:
                    metadata.add(path)
                    current = path
        return links, metadata

    @staticmethod
    def _identity(path: str) -> tuple[int, int]:
        info = os.stat(path, follow_symlinks=False)
        return info.st_dev, info.st_ino

    @staticmethod
    def _validate_path(raw: str) -> str:
        if not isinstance(raw, str) or not raw or not os.path.isabs(raw):
            raise ValueError("sandbox paths must be non-empty absolute paths")
        path = os.path.normpath(raw)
        if path == "/" or not os.path.exists(path) or os.path.realpath(path) != path:
            raise ValueError(f"sandbox path must exist without symlinks and cannot be root: {raw}")
        return path

    @classmethod
    def _validate_paths(cls, raw: list[str], name: str) -> list[str]:
        if not isinstance(raw, list) or any(not isinstance(item, str) for item in raw):
            raise ValueError(f"{name} must be a list of absolute paths")
        return [cls._validate_path(path) for path in raw]

    def _check_paths(self, cwd: str) -> None:
        if not any(contains(path, cwd) for path in self._read_only_paths + self._writable_paths):
            raise ValueError("command working directory must be inside a sandbox path")
        for path, identity in self._path_identities.items():
            if not os.path.exists(path) or os.path.realpath(path) != path or self._identity(path) != identity:
                raise RuntimeError(f"sandbox path changed before execution: {path}")

    async def getcwd(self) -> str:
        """Return the host-identical sandbox working directory."""
        return self._cwd

    @abstractmethod
    async def _start_process(self, command: list[str], cwd: str, *, stdin=asyncio.subprocess.DEVNULL):
        """Start one sandboxed process, never falling back to host execution."""

    async def exec_shell(
        self,
        command: list[str],
        *,
        cwd: str | None = None,
        timeout: float | None = None,
    ) -> ExecResult:
        """Execute an argv list with a timeout inside the platform sandbox."""
        return await self._run_job(command, cwd or self._cwd, data=None, timeout=timeout)

    async def _run_job(self, command: list[str], cwd: str, *, data: bytes | None, timeout: float | None) -> ExecResult:
        """Run with optional stdin; platform backends may own a different job lifecycle."""
        process = await self._start_process(
            command,
            cwd,
            stdin=asyncio.subprocess.PIPE if data is not None else asyncio.subprocess.DEVNULL,
        )
        return await self._communicate(process, data=data, timeout=timeout)

    async def read_file(self, path: str) -> bytes:
        """Read through a sandboxed command so file rules still apply."""
        result = await self.exec_shell(["cat", path])
        if not result.ok():
            raise OSError(result.stderr.decode(errors="replace"))
        return result.stdout

    async def write_file(self, path: str, data: bytes) -> None:
        """Write through a sandboxed command so file rules still apply."""
        if not isinstance(data, bytes):
            raise TypeError("data must be bytes")
        script = 'mkdir -p -- "$(dirname -- "$1")" && cat > "$1"'
        result = await self._run_job(
            ["/bin/sh", "-c", script, "sh", path],
            self._cwd,
            data=data,
            timeout=None,
        )
        if not result.ok():
            raise OSError(result.stderr.decode(errors="replace"))
