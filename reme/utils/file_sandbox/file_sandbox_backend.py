"""Common contract and path validation for subprocess file sandboxes."""

import asyncio
import os
import signal
import sys
from abc import abstractmethod

import psutil
from agentscope.tool import BackendBase, ExecResult

_CLEANUP_TIMEOUT = 2.0


def contains(root: str, path: str) -> bool:
    """Whether a path is inside a configured root."""
    return os.path.commonpath((root, path)) == root


class FileSandboxBackend(BackendBase):
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

    def _release_process(self, process: asyncio.subprocess.Process) -> None:
        """Release platform resources after the child has exited."""

    async def exec_shell(
        self,
        command: list[str],
        *,
        cwd: str | None = None,
        timeout: float | None = None,
    ) -> ExecResult:
        """Execute an argv list with a timeout inside the platform sandbox."""
        process = await self._start_process(command, cwd or self._cwd)
        return await self._communicate(process, timeout=timeout)

    async def _communicate(
        self,
        process: asyncio.subprocess.Process,
        *,
        data: bytes | None = None,
        timeout: float | None = None,
    ) -> ExecResult:
        """Drain output and release process resources on success, timeout, or cancellation."""
        communication = asyncio.create_task(process.communicate(data))
        try:
            stdout, stderr = await asyncio.wait_for(asyncio.shield(communication), timeout=timeout)
        except TimeoutError:
            await self._stop_process(process, communication)
            return ExecResult(exit_code=-1, stdout=b"", stderr=b"timed out")
        except BaseException:
            await self._stop_process(process, communication)
            raise
        finally:
            self._release_process(process)
        return ExecResult(exit_code=process.returncode, stdout=stdout, stderr=stderr)

    @staticmethod
    def _kill_process(process: asyncio.subprocess.Process) -> None:
        """Stop the launcher and known descendants, including detached children."""
        if sys.platform == "darwin":
            try:
                os.killpg(process.pid, signal.SIGSTOP)
            except ProcessLookupError:
                pass
            try:
                descendants = psutil.Process(process.pid).children(recursive=True)
            except psutil.Error:
                descendants = []
            for child in reversed(descendants):
                try:
                    child.kill()
                except psutil.Error:
                    pass
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass

    async def _stop_process(self, process: asyncio.subprocess.Process, communication: asyncio.Task) -> None:
        """Bound pipe draining even if an untracked child kept a pipe open."""
        self._kill_process(process)
        try:
            await asyncio.wait_for(asyncio.shield(communication), timeout=_CLEANUP_TIMEOUT)
        except TimeoutError:
            communication.cancel()
            try:
                await asyncio.wait_for(process.wait(), timeout=_CLEANUP_TIMEOUT)
            except TimeoutError:
                pass

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
        process = await self._start_process(
            ["/bin/sh", "-c", script, "sh", path],
            self._cwd,
            stdin=asyncio.subprocess.PIPE,
        )
        result = await self._communicate(process, data=data)
        if not result.ok():
            raise OSError(result.stderr.decode(errors="replace"))
