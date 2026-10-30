"""macOS Seatbelt file sandbox with host-identical paths."""

import asyncio
import errno
import json
import os
import plistlib
import shutil
import sys
import tempfile
import uuid
import warnings
from pathlib import Path
from types import SimpleNamespace

import psutil
from agentscope.tool import ExecResult

from .file_sandbox_backend import FileSandboxBackend, contains
from ..async_utils import finish_cleanup
from ..local_backend import (
    LocalProcessBackend,
    OutputLimitExceeded,
    capture_output,
    output_limit_result,
)


class LaunchctlError(RuntimeError):
    """Preserve launchctl's error code so missing services can be handled narrowly."""

    def __init__(self, message: str, exit_code: int):
        super().__init__(message)
        self.exit_code = exit_code


class PathSeatbeltBackend(FileSandboxBackend):
    """Run each command under a deny-by-default Seatbelt profile."""

    def __init__(
        self,
        *,
        cwd: str | os.PathLike[str],
        read_only_paths: list[str],
        writable_paths: list[str],
        memory_limit_mib: int = 0,
    ) -> None:
        if isinstance(memory_limit_mib, bool) or not isinstance(memory_limit_mib, int) or memory_limit_mib < 0:
            raise ValueError("memory_limit_mib must be a non-negative integer")
        if memory_limit_mib:
            warnings.warn(
                "PathSeatbeltBackend does not support memory_limit_mib; the limit is ignored on macOS",
                RuntimeWarning,
                stacklevel=2,
            )
        super().__init__(cwd=cwd, read_only_paths=read_only_paths, writable_paths=writable_paths)
        self._process_tempdirs: dict[int, str] = {}

    @staticmethod
    def _path_rule(path: str) -> str:
        """Quote a path as one Seatbelt profile string, never as profile syntax."""
        return f"(subpath {json.dumps(path)})" if os.path.isdir(path) else f"(literal {json.dumps(path)})"

    @classmethod
    def _runtime_link_rules(cls) -> list[str]:
        """Permit runtime symlink traversal without granting their parent trees."""
        links, metadata = cls._runtime_paths()
        return [f"(allow file-read* (literal {json.dumps(path)}))" for path in sorted(links)] + [
            f"(allow file-read-metadata (literal {json.dumps(path)}))" for path in sorted(metadata - links)
        ]

    def _profile(self, tempdir: str) -> str:
        """Allow only configured files plus the system runtime needed to launch tools."""
        system_roots = ("/System", "/usr", "/bin", "/sbin", "/private/etc", "/dev")
        read_paths = [path for path in system_roots if os.path.exists(path)]
        read_paths += self._read_only_paths + self._writable_paths + [tempdir]
        lines = [
            "(version 1)",
            "(deny default)",
            "(allow process*)",
            "(allow sysctl-read)",
            "(allow mach-lookup)",
            # dyld opens the root directory; /bin/sh reads the system shell selector.
            '(allow file-read* (literal "/") (literal "/private/var/select/sh"))',
        ]
        ancestors = set()
        for path in read_paths:
            parent = os.path.dirname(path)
            while parent != "/":
                ancestors.add(parent)
                parent = os.path.dirname(parent)
        lines.extend(f"(allow file-read-metadata (literal {json.dumps(path)}))" for path in sorted(ancestors))
        lines.extend(self._runtime_link_rules())
        lines.extend(f"(allow file-read* {self._path_rule(path)})" for path in read_paths)
        # Shell redirection to the null device is safe and is accepted by
        # AutoBash's read-only policy; do not grant writes to the rest of /dev.
        lines.append('(allow file-write* (literal "/dev/null"))')
        for path in self._writable_paths + [tempdir]:
            # Seatbelt combines allow rules with OR, unlike ordered bwrap mounts.
            # Exclude read-only descendants from each writable parent's grant.
            exclusions = [
                f"(require-not {self._path_rule(read_only)})"
                for read_only in self._read_only_paths
                if contains(path, read_only)
            ]
            rule = self._path_rule(path)
            if exclusions:
                rule = f"(require-all {rule} {' '.join(exclusions)})"
            lines.append(f"(allow file-write* {rule})")
        # A writable ancestor can otherwise be renamed, moving a protected
        # subtree out of the path-based exclusion above. Pin the protected
        # path and every ancestor that falls within a writable root.
        pinned = set()
        for read_only in self._read_only_paths:
            current = read_only
            while current != "/":
                if any(contains(writable, current) for writable in self._writable_paths):
                    pinned.add(current)
                current = os.path.dirname(current)
        for path in sorted(pinned):
            rule = f"(literal {json.dumps(path)})"
            lines.append(f"(deny file-write-unlink {rule})")
            lines.append(f"(deny file-write-create {rule})")
        return "\n".join(lines)

    async def _start_process(self, command: list[str], cwd: str, *, stdin=asyncio.subprocess.DEVNULL):
        """Start Seatbelt directly when the GUI launchd domain is unavailable."""
        tempdir = tempfile.mkdtemp(prefix="reme-seatbelt-", dir=os.path.realpath(tempfile.gettempdir()))
        try:
            process = await self._spawn_process(
                ["/usr/bin/sandbox-exec", "-p", self._profile(tempdir), *command],
                cwd=cwd,
                env={**os.environ, "HOME": self._cwd, "TMPDIR": tempdir},
                stdin=stdin,
            )
        except BaseException:
            shutil.rmtree(tempdir)
            raise
        self._process_tempdirs[process.pid] = tempdir
        return process

    async def _release_process(self, process: asyncio.subprocess.Process) -> None:
        tempdir = self._process_tempdirs.pop(process.pid, None)
        if tempdir is not None:
            shutil.rmtree(tempdir)

    @staticmethod
    async def _has_gui_launchd(domain: str) -> bool:
        result = await LocalProcessBackend().exec_shell(["/bin/launchctl", "print", domain], timeout=5)
        return result.ok()

    @staticmethod
    async def _launchctl(*arguments: str) -> str:
        result = await LocalProcessBackend().exec_shell(["/bin/launchctl", *arguments], timeout=5)
        if not result.ok():
            raise LaunchctlError(
                f"launchctl {arguments[0]} failed: {result.stderr.decode(errors='replace').strip()}",
                result.exit_code,
            )
        return result.stdout.decode(errors="replace").strip()

    async def _bootstrap(self, domain: str, plist: Path, service: str) -> None:
        """Resolve registration before rolling it back, even when its reply is interrupted."""
        registration = asyncio.create_task(self._launchctl("bootstrap", domain, str(plist)))
        try:
            await asyncio.shield(registration)
        except BaseException:

            async def rollback():
                # Killing the client cannot undo a registration accepted by launchd.
                # Wait for the bounded launchctl operation before issuing bootout.
                await asyncio.gather(registration, return_exceptions=True)
                try:
                    await self._launchctl("bootout", service)
                except LaunchctlError as exc:
                    if exc.exit_code != errno.ESRCH:
                        raise

            await finish_cleanup(rollback())
            raise

    @staticmethod
    async def _wait_status(status: Path, pid: int) -> int:
        while True:
            if status.exists():
                return int(status.read_text(encoding="ascii"))
            if not psutil.pid_exists(pid):
                raise RuntimeError("launchd sandbox job exited without reporting a status")
            await asyncio.sleep(0.02)

    @staticmethod
    async def _read_fifo(fd: int, buffer: bytearray, name: str, exceeded: asyncio.Future, finished: asyncio.Event):
        """Drain launchd output through a bounded pipe instead of an unbounded disk file."""
        while True:
            try:
                chunk = os.read(fd, 65536)
            except BlockingIOError:
                if finished.is_set():
                    return
                await asyncio.sleep(0.005)
                continue
            capture_output(buffer, chunk, name, exceeded)
            await asyncio.sleep(0)

    async def _run_job(self, command: list[str], cwd: str, *, data: bytes | None, timeout: float | None) -> ExecResult:
        if sys.platform != "darwin":
            raise RuntimeError("PathSeatbeltBackend requires macOS")
        if shutil.which("sandbox-exec") is None:
            raise RuntimeError("PathSeatbeltBackend requires sandbox-exec on PATH")
        if not command:
            raise ValueError("command must not be empty")
        self._check_paths(cwd)
        domain = f"gui/{os.getuid()}"
        if not await self._has_gui_launchd(domain):
            warnings.warn(
                "No macOS GUI launchd domain; process-group cleanup is unavailable, running directly under Seatbelt",
                RuntimeWarning,
                stacklevel=2,
            )
            return await super()._run_job(command, cwd, data=data, timeout=timeout)
        control = Path(tempfile.mkdtemp(prefix="reme-seatbelt-", dir=os.path.realpath(tempfile.gettempdir())))
        sandbox_tmp = control / "sandbox-tmp"
        sandbox_tmp.mkdir()
        stdout_path = control / "stdout"
        stderr_path = control / "stderr"
        status_path = control / "status"
        stdin_path = control / "stdin"
        label = f"ai.agentscope.reme.sandbox.{uuid.uuid4().hex}"
        service = f"{domain}/{label}"
        registered = False
        pid = None
        stdout, stderr = bytearray(), bytearray()
        exceeded = asyncio.get_running_loop().create_future()
        finished = asyncio.Event()
        output_fds = []
        readers = []

        async def cleanup(completed):
            try:
                if registered:
                    if not completed and pid is not None:
                        self._kill_process(SimpleNamespace(pid=pid))
                    await self._launchctl("bootout", service)
            finally:
                finished.set()
                try:
                    await asyncio.wait_for(asyncio.gather(*readers), timeout=0.25)
                except TimeoutError:
                    pass

        try:
            for path, buffer, name in ((stdout_path, stdout, "stdout"), (stderr_path, stderr, "stderr")):
                os.mkfifo(path, 0o600)
                # Holding both ends open prevents a transient EOF before launchd
                # opens its writer. Kernel pipe capacity bounds pending output.
                fd = os.open(path, os.O_RDWR | os.O_NONBLOCK)
                output_fds.append(fd)
                readers.append(asyncio.create_task(self._read_fifo(fd, buffer, name, exceeded, finished)))
            if data is not None:
                stdin_path.write_bytes(data)
            # The shell reports the command's status in a control file. Its
            # child inherits the launchd job's process group, so launchd can
            # reap same-group background processes when the shell exits.
            script = '"$@"; result=$?; printf "%s\\n" "$result" > "$0.tmp"; mv "$0.tmp" "$0"'
            job = {
                "Label": label,
                "ProgramArguments": [
                    "/bin/sh",
                    "-c",
                    script,
                    str(status_path),
                    "/usr/bin/sandbox-exec",
                    "-p",
                    self._profile(str(sandbox_tmp)),
                    *command,
                ],
                "WorkingDirectory": cwd,
                "EnvironmentVariables": {**os.environ, "HOME": self._cwd, "TMPDIR": str(sandbox_tmp)},
                "StandardInPath": str(stdin_path) if data is not None else "/dev/null",
                "StandardOutPath": str(stdout_path),
                "StandardErrorPath": str(stderr_path),
                "RunAtLoad": False,
                "KeepAlive": False,
                "AbandonProcessGroup": False,
            }
            plist_path = control / "job.plist"
            with os.fdopen(os.open(plist_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "wb") as stream:
                plistlib.dump(job, stream)
            await self._bootstrap(domain, plist_path, service)
            registered = True

            async def run():
                nonlocal pid
                pid = int(await self._launchctl("kickstart", "-p", service))
                return await self._watch_output(self._wait_status(status_path, pid), exceeded)

            try:
                exit_code = await self._execute(run(), timeout=timeout, cleanup=cleanup)
            except TimeoutError:
                return ExecResult(exit_code=-1, stdout=b"", stderr=b"timed out")
            except OutputLimitExceeded as exc:
                return output_limit_result(bytes(stdout), bytes(stderr), str(exc))
            if exceeded.done():
                return output_limit_result(bytes(stdout), bytes(stderr), exceeded.result())
            return ExecResult(
                exit_code=exit_code,
                stdout=bytes(stdout),
                stderr=bytes(stderr),
            )
        finally:

            async def release():
                for reader in readers:
                    reader.cancel()
                await asyncio.gather(*readers, return_exceptions=True)
                for fd in output_fds:
                    os.close(fd)
                shutil.rmtree(control)

            await finish_cleanup(release())
