"""macOS Seatbelt file sandbox with host-identical paths."""

import asyncio
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
            process = await asyncio.create_subprocess_exec(
                "/usr/bin/sandbox-exec",
                "-p",
                self._profile(tempdir),
                *command,
                cwd=cwd,
                env={**os.environ, "HOME": self._cwd, "TMPDIR": tempdir},
                stdin=stdin,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                start_new_session=True,
            )
        except BaseException:
            shutil.rmtree(tempdir)
            raise
        self._process_tempdirs[process.pid] = tempdir
        return process

    def _release_process(self, process: asyncio.subprocess.Process) -> None:
        tempdir = self._process_tempdirs.pop(process.pid, None)
        if tempdir is not None:
            shutil.rmtree(tempdir)

    @staticmethod
    async def _has_gui_launchd(domain: str) -> bool:
        process = await asyncio.create_subprocess_exec(
            "/bin/launchctl",
            "print",
            domain,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )
        return await process.wait() == 0

    @staticmethod
    async def _launchctl(*arguments: str) -> str:
        process = await asyncio.create_subprocess_exec(
            "/bin/launchctl",
            *arguments,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await process.communicate()
        if process.returncode:
            raise RuntimeError(f"launchctl {' '.join(arguments[:1])} failed: {stderr.decode(errors='replace').strip()}")
        return stdout.decode(errors="replace").strip()

    @staticmethod
    async def _wait_status(status: Path, pid: int) -> int:
        while True:
            if status.exists():
                return int(status.read_text(encoding="ascii"))
            if not psutil.pid_exists(pid):
                raise RuntimeError("launchd sandbox job exited without reporting a status")
            await asyncio.sleep(0.02)

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
            process = await self._start_process(
                command,
                cwd,
                stdin=asyncio.subprocess.PIPE if data is not None else asyncio.subprocess.DEVNULL,
            )
            return await self._communicate(process, data=data, timeout=timeout)
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
        timed_out = False
        completed = False
        pid = None
        stdout = b""
        stderr = b""
        try:
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
            await self._launchctl("bootstrap", domain, str(plist_path))
            registered = True
            pid = int(await self._launchctl("kickstart", "-p", service))
            try:
                exit_code = await asyncio.wait_for(self._wait_status(status_path, pid), timeout=timeout)
            except TimeoutError:
                timed_out = True
            else:
                completed = True
        finally:
            try:
                if registered:
                    if not completed and pid is not None:
                        self._kill_process(SimpleNamespace(pid=pid))
                    await self._launchctl("bootout", service)
            finally:
                if not timed_out:
                    stdout = stdout_path.read_bytes() if stdout_path.exists() else b""
                    stderr = stderr_path.read_bytes() if stderr_path.exists() else b""
                shutil.rmtree(control)
        if timed_out:
            return ExecResult(exit_code=-1, stdout=b"", stderr=b"timed out")
        return ExecResult(exit_code=exit_code, stdout=stdout, stderr=stderr)

    async def exec_shell(
        self,
        command: list[str],
        *,
        cwd: str | None = None,
        timeout: float | None = None,
    ) -> ExecResult:
        return await self._run_job(command, cwd or self._cwd, data=None, timeout=timeout)

    async def write_file(self, path: str, data: bytes) -> None:
        if not isinstance(data, bytes):
            raise TypeError("data must be bytes")
        script = 'mkdir -p -- "$(dirname -- "$1")" && cat > "$1"'
        result = await self._run_job(["/bin/sh", "-c", script, "sh", path], self._cwd, data=data, timeout=None)
        if not result.ok():
            raise OSError(result.stderr.decode(errors="replace"))
