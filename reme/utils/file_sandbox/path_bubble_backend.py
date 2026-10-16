"""Bubblewrap execution with host-identical, explicitly mounted paths."""

import asyncio
import os
import shutil
import uuid
from pathlib import Path

import sys

from .file_sandbox_backend import FileSandboxBackend

# Run after exec of a fresh interpreter: Python preexec_fn can deadlock when
# the parent has threads. The requested command starts only after enrollment.
_CGROUP_EXEC = """
import os
import sys
with open(sys.argv[1], 'w', encoding='ascii') as group:
    group.write(str(os.getpid()))
os.execvp(sys.argv[2], sys.argv[2:])
"""


class PathBubbleBackend(FileSandboxBackend):
    """Run commands with selected host paths at the same sandbox paths."""

    def __init__(
        self,
        *,
        cwd: str | os.PathLike[str],
        read_only_paths: list[str],
        writable_paths: list[str],
        memory_limit_mib: int = 4096,
    ) -> None:
        if isinstance(memory_limit_mib, bool) or not isinstance(memory_limit_mib, int) or memory_limit_mib < 0:
            raise ValueError("memory_limit_mib must be a non-negative integer")
        self._memory_limit_mib = memory_limit_mib
        self._process_cgroups: dict[int, Path] = {}
        super().__init__(cwd=cwd, read_only_paths=read_only_paths, writable_paths=writable_paths)

    def _bwrap_argv(self, command: list[str], cwd: str) -> list[str]:
        """Build one isolated execution with host-identical bind targets."""
        if not command:
            raise ValueError("command must not be empty")
        self._check_paths(cwd)
        args = ["bwrap", "--die-with-parent", "--new-session", "--unshare-all", "--proc", "/proc", "--dev", "/dev"]
        for path in ("/usr", "/usr/local"):
            if os.path.isdir(path):
                args.extend(("--ro-bind", path, path))
        for path in ("/bin", "/sbin", "/lib", "/lib64"):
            if os.path.islink(path):
                args.extend(("--symlink", os.readlink(path), path))
            elif os.path.exists(path):
                args.extend(("--ro-bind", path, path))
        args.extend(("--dir", "/etc"))
        for path in ("/etc/passwd", "/etc/group", "/etc/nsswitch.conf", "/etc/hosts", "/etc/resolv.conf", "/etc/ssl"):
            if os.path.exists(path):
                args.extend(("--ro-bind", path, path))
        args.extend(("--tmpfs", "/tmp", "--tmpfs", "/run"))
        mounts = [(path, "--ro-bind") for path in self._read_only_paths]
        mounts += [(path, "--bind") for path in self._writable_paths]
        for path, mode in sorted(mounts, key=lambda item: (item[0].count("/"), item[0])):
            args.extend((mode, path, path))
        binary_paths = [str(Path(sys.prefix) / "bin"), "/usr/local/bin", "/usr/bin", "/bin"]
        args.extend(("--clearenv", "--setenv", "HOME", self._cwd, "--setenv", "TMPDIR", "/tmp"))
        args.extend(("--setenv", "PATH", ":".join(binary_paths), "--chdir", cwd, "--"))
        return args + command

    async def _start_process(self, command: list[str], cwd: str, *, stdin=asyncio.subprocess.DEVNULL):
        """Start bwrap without falling back to host execution."""
        if not sys.platform.startswith("linux"):
            raise RuntimeError("PathBubbleBackend requires Linux")
        if shutil.which("bwrap") is None:
            raise RuntimeError("PathBubbleBackend requires bwrap on PATH")
        argv = self._bwrap_argv(command, cwd)
        group = self._create_cgroup() if self._memory_limit_mib else None
        try:
            if group is not None:
                argv = [sys.executable, "-I", "-S", "-c", _CGROUP_EXEC, str(group / "cgroup.procs"), *argv]
            process = await asyncio.create_subprocess_exec(
                *argv,
                stdin=stdin,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                start_new_session=True,
            )
        except BaseException:
            if group is not None:
                group.rmdir()
            raise
        if group is not None:
            self._process_cgroups[process.pid] = group
        return process

    def _create_cgroup(self) -> Path:
        """Create a separate cgroup v2 for one execution, failing closed if unavailable."""
        try:
            entries = Path("/proc/self/cgroup").read_text(encoding="ascii").splitlines()
            relative = next(line.split("::", 1)[1] for line in entries if line.startswith("0::"))
            root = Path("/sys/fs/cgroup")
            parent = root / relative.lstrip("/")
            if ".." in Path(relative).parts or not parent.resolve().is_relative_to(root):
                raise RuntimeError("invalid cgroup v2 path")
            group = parent / f"reme-{uuid.uuid4().hex}"
            group.mkdir()
            try:
                (group / "memory.max").write_text(str(self._memory_limit_mib * 1024**2), encoding="ascii")
            except BaseException:
                group.rmdir()
                raise
            return group
        except (OSError, StopIteration) as exc:
            raise RuntimeError("PathBubbleBackend requires a writable delegated cgroup v2 for memory limits") from exc

    def _release_process(self, process: asyncio.subprocess.Process) -> None:
        """Release the per-execution group after its process has been reaped."""
        group = self._process_cgroups.pop(process.pid, None)
        if group is None:
            return
        try:
            group.rmdir()
        except OSError as exc:
            try:
                (group / "cgroup.kill").write_text("1", encoding="ascii")
                group.rmdir()
            except OSError as cleanup_error:
                raise RuntimeError(f"could not remove sandbox cgroup {group}: {cleanup_error}") from exc
