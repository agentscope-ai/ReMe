"""Path-preserving Bubblewrap mounts and job integration."""

import asyncio
import errno
import sys
from types import SimpleNamespace
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
import yaml

from agentscope.tool import ExecResult

from reme.utils.file_sandbox.path_bubble_backend import PathBubbleBackend
from reme.components.application_context import ApplicationContext
from reme.steps.common.python_execute import PythonExecuteStep


@pytest.fixture(autouse=True)
def linux_backend_selection(monkeypatch):
    """Keep existing integration assertions on the Linux backend."""
    monkeypatch.setattr("reme.utils.file_sandbox.sys.platform", "linux")


def _mounts(argv):
    return [
        (argv[index], argv[index + 1], argv[index + 2])
        for index in range(len(argv) - 2)
        if argv[index] in ("--bind", "--ro-bind")
    ]


def test_path_bubble_backend_preserves_host_paths_and_mount_order(tmp_path):
    """Nested writable mounts override a read-only parent at the same path."""
    output = tmp_path / "output"
    output.mkdir()
    backend = PathBubbleBackend(cwd=tmp_path, read_only_paths=[str(tmp_path)], writable_paths=[str(output)])

    argv = backend._bwrap_argv(["/bin/sh", "-c", "pwd"], str(tmp_path))  # pylint: disable=protected-access
    mounts = _mounts(argv)

    assert ("--ro-bind", str(tmp_path), str(tmp_path)) in mounts
    assert ("--bind", str(output), str(output)) in mounts
    assert mounts.index(("--ro-bind", str(tmp_path), str(tmp_path))) < mounts.index(
        ("--bind", str(output), str(output)),
    )
    assert argv[-3:] == ["/bin/sh", "-c", "pwd"]
    assert argv[argv.index("--chdir") + 1] == str(tmp_path)
    assert "--share-net" not in argv


def test_path_bubble_backend_adds_workspace_read_only(tmp_path):
    """The working directory is always accessible inside the sandbox."""
    backend = PathBubbleBackend(cwd=tmp_path, read_only_paths=[], writable_paths=[])
    argv = backend._bwrap_argv(["true"], str(tmp_path))  # pylint: disable=protected-access
    assert ("--ro-bind", str(tmp_path), str(tmp_path)) in _mounts(argv)


@pytest.mark.parametrize("relative", [False, True])
def test_runtime_symlink_chains_preserve_interpreter_paths(tmp_path, monkeypatch, relative):
    """External runtime aliases work without exposing siblings or duplicating mounted links."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    environment = tmp_path / "venv"
    runtime = tmp_path / "runtime"
    (environment / "bin").mkdir(parents=True)
    (runtime / "bin").mkdir(parents=True)
    (runtime / "bin" / "python").touch()
    targets = {
        tmp_path / "alias": "venv-alias",
        tmp_path / "venv-alias": "venv",
        tmp_path / "base-alias": "runtime",
        tmp_path / "launcher": "runtime/bin/python",
    }
    for path, target in targets.items():
        path.symlink_to(target if relative else tmp_path / target)
    # This link is already available through the venv's read-only mount, but
    # its target requires another link outside that mount.
    (environment / "bin" / "python").symlink_to("../../launcher")
    interpreter = str(tmp_path / "alias" / "bin" / "python")
    monkeypatch.setattr(sys, "prefix", str(tmp_path / "alias"))
    monkeypatch.setattr(sys, "base_prefix", str(tmp_path / "base-alias"))
    monkeypatch.setattr(sys, "executable", interpreter)
    backend = PathBubbleBackend(cwd=workspace, read_only_paths=[], writable_paths=[])

    argv = backend._bwrap_argv([interpreter, "-c", "print(1)"], str(workspace))  # pylint: disable=protected-access
    links = {argv[index + 2]: argv[index + 1] for index in range(len(argv) - 2) if argv[index] == "--symlink"}

    for path, target in targets.items():
        assert links[str(path)] == (target if relative else str(tmp_path / target))
    assert str(environment / "bin" / "python") not in links
    assert ("--ro-bind", str(environment), str(environment)) in _mounts(argv)
    assert ("--ro-bind", str(runtime), str(runtime)) in _mounts(argv)
    assert all(source != str(tmp_path) for _, source, _ in _mounts(argv))
    assert not any(mode == "--bind" for mode, _, _ in _mounts(argv))
    assert argv[-3:] == [interpreter, "-c", "print(1)"]


@pytest.mark.parametrize("limit", [-1, True, 1.5, "4096"])
def test_path_bubble_backend_rejects_invalid_memory_limit(tmp_path, limit):
    """Memory limits must be non-negative whole MiB."""
    with pytest.raises(ValueError, match="memory_limit_mib"):
        PathBubbleBackend(cwd=tmp_path, read_only_paths=[], writable_paths=[], memory_limit_mib=limit)


def test_one_mib_is_written_as_bytes_to_cgroup(tmp_path, monkeypatch):
    """The public MiB limit is converted only at the Linux cgroup boundary."""
    proc_file = tmp_path / "proc-self-cgroup"
    proc_file.write_text("0::/\n", encoding="ascii")
    cgroup_root = tmp_path / "cgroup"
    cgroup_root.mkdir()
    original_path = Path

    def sandbox_path(value):
        if value == "/proc/self/cgroup":
            return proc_file
        if value == "/sys/fs/cgroup":
            return cgroup_root
        return original_path(value)

    monkeypatch.setattr("reme.utils.file_sandbox.path_bubble_backend.Path", sandbox_path)
    backend = PathBubbleBackend(cwd=tmp_path, read_only_paths=[], writable_paths=[], memory_limit_mib=1)
    group = backend._create_cgroup()  # pylint: disable=protected-access
    assert (group / "memory.max").read_text(encoding="ascii") == str(1024**2)


@pytest.mark.asyncio
async def test_each_execution_gets_separate_memory_cgroup(tmp_path, monkeypatch):
    """Each process receives its own cgroup before bubblewrap is executed."""
    backend = PathBubbleBackend(cwd=tmp_path, read_only_paths=[], writable_paths=[], memory_limit_mib=1024)
    groups = []
    subprocess_options = []
    subprocess_commands = []

    def create_group():
        group = tmp_path / f"group-{len(groups)}"
        group.mkdir()
        groups.append(group)
        return group

    async def create_process(*argv, **kwargs):
        subprocess_commands.append(argv)
        subprocess_options.append(kwargs)
        return SimpleNamespace(pid=100 + len(subprocess_options))

    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr("reme.utils.file_sandbox.path_bubble_backend.shutil.which", lambda _: "/usr/bin/bwrap")
    monkeypatch.setattr(backend, "_create_cgroup", create_group)
    monkeypatch.setattr("reme.utils.file_sandbox.path_bubble_backend.asyncio.create_subprocess_exec", create_process)
    first = await backend._start_process(["true"], str(tmp_path))  # pylint: disable=protected-access
    second = await backend._start_process(["true"], str(tmp_path))  # pylint: disable=protected-access

    assert len(groups) == 2 and groups[0] != groups[1]
    assert all("preexec_fn" not in options for options in subprocess_options)
    for argv, group in zip(subprocess_commands, groups):
        assert argv[:4] == (sys.executable, "-I", "-S", "-c")
        assert argv[5:7] == (str(group / "cgroup.procs"), "bwrap")
    await backend._release_process(first)  # pylint: disable=protected-access
    await backend._release_process(second)  # pylint: disable=protected-access
    assert not any(group.exists() for group in groups)


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", ["delayed_exit", "cancel", "busy", "permission", "kill_error"])
async def test_cgroup_cleanup_waits_without_blocking_and_reports_failures(tmp_path, monkeypatch, outcome):
    """Model kernel EBUSY until members exit, including repeated cancellation during cleanup."""
    backend = PathBubbleBackend(cwd=tmp_path, read_only_paths=[], writable_paths=[])
    group = tmp_path / "cgroup"
    group.mkdir()
    process = SimpleNamespace(pid=123, returncode=0)
    backend._process_cgroups[process.pid] = group  # pylint: disable=protected-access
    entered, empty = asyncio.Event(), asyncio.Event()
    kills = []
    original_rmdir, original_write = Path.rmdir, Path.write_text

    def rmdir(path):
        if path == group:
            if outcome == "permission":
                raise PermissionError(errno.EACCES, "permission denied")
            if not empty.is_set():
                entered.set()
                raise OSError(errno.EBUSY, "members still exiting")
        original_rmdir(path)

    def write(path, data, **kwargs):
        if path == group / "cgroup.kill":
            kills.append(data)
            if outcome == "kill_error":
                raise PermissionError(errno.EACCES, "kill denied")
            return len(data)
        return original_write(path, data, **kwargs)

    monkeypatch.setattr(Path, "rmdir", rmdir)
    monkeypatch.setattr(Path, "write_text", write)
    cleanup_timeout = 1 if outcome in ("delayed_exit", "cancel") else 0.03
    monkeypatch.setattr("reme.utils.file_sandbox.path_bubble_backend._CGROUP_CLEANUP_TIMEOUT", cleanup_timeout)
    monkeypatch.setattr(backend, "_kill_process", lambda _: None)
    monkeypatch.setattr(backend, "_collect_output", AsyncMock(return_value=(b"ok", b"")))
    task = asyncio.create_task(backend._communicate(process))  # pylint: disable=protected-access
    if outcome in ("delayed_exit", "cancel"):
        await asyncio.wait_for(entered.wait(), timeout=1)
        if outcome == "cancel":
            task.cancel()
            await asyncio.sleep(0)
            task.cancel()
        await asyncio.sleep(0.01)
        assert not task.done()
        empty.set()
        if outcome == "cancel":
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(task, timeout=1)
        else:
            assert (await asyncio.wait_for(task, timeout=1)).stdout == b"ok"
        assert not group.exists()
        assert not backend._process_cgroups  # pylint: disable=protected-access
    else:
        with pytest.raises(RuntimeError, match="could not remove sandbox cgroup") as error:
            await asyncio.wait_for(task, timeout=1)
        assert error.value.__cause__.errno == (errno.EBUSY if outcome == "busy" else errno.EACCES)
        assert group.exists()
        assert backend._process_cgroups[process.pid] == group  # pylint: disable=protected-access
    assert kills == ([] if outcome == "permission" else ["1"])


@pytest.mark.asyncio
async def test_zero_memory_limit_skips_cgroup(tmp_path, monkeypatch):
    """A zero limit bypasses cgroup creation and child enrollment."""
    backend = PathBubbleBackend(cwd=tmp_path, read_only_paths=[], writable_paths=[], memory_limit_mib=0)
    options = {}

    async def create_process(*_argv, **kwargs):
        options.update(kwargs)
        return SimpleNamespace(pid=123)

    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr("reme.utils.file_sandbox.path_bubble_backend.shutil.which", lambda _: "/usr/bin/bwrap")
    monkeypatch.setattr("reme.utils.file_sandbox.path_bubble_backend.asyncio.create_subprocess_exec", create_process)
    await backend._start_process(["true"], str(tmp_path))  # pylint: disable=protected-access
    assert "preexec_fn" not in options


@pytest.mark.parametrize("path", ["relative", "/", "/does-not-exist"])
def test_path_bubble_backend_rejects_invalid_mounts(tmp_path, path):
    """Only existing absolute non-root paths can be mounted."""
    with pytest.raises(ValueError):
        PathBubbleBackend(cwd=tmp_path, read_only_paths=[path], writable_paths=[])


def test_path_bubble_backend_rejects_symlink_and_duplicates(tmp_path):
    """Ambiguous mount sources are rejected before a process starts."""
    alias = tmp_path.parent / "alias"
    alias.symlink_to(tmp_path)
    try:
        with pytest.raises(ValueError):
            PathBubbleBackend(cwd=tmp_path, read_only_paths=[str(alias)], writable_paths=[])
        with pytest.raises(ValueError):
            PathBubbleBackend(cwd=tmp_path, read_only_paths=[str(tmp_path)], writable_paths=[str(tmp_path)])
    finally:
        alias.unlink()


def test_path_bubble_backend_rejects_replaced_mount_source(tmp_path):
    """A path swapped after validation cannot become a new mount source."""
    source = tmp_path / "source"
    source.mkdir()
    backend = PathBubbleBackend(cwd=tmp_path, read_only_paths=[str(source)], writable_paths=[])
    source.rename(tmp_path / "old-source")
    source.mkdir()

    with pytest.raises(RuntimeError, match="sandbox path changed"):
        backend._bwrap_argv(["true"], str(tmp_path))  # pylint: disable=protected-access


@pytest.mark.asyncio
@pytest.mark.parametrize("memory_options", [{}, {"memory_limit_mib": 0}])
async def test_python_execute_uses_path_backend_when_paths_supplied(tmp_path, monkeypatch, memory_options):
    """Python execution uses the same backend and timeout units."""
    seen = {}

    async def fake_exec(self, command, *, cwd=None, timeout=None):
        seen.update(command=command, cwd=cwd, timeout=timeout, backend=self)
        return ExecResult(exit_code=0, stdout=b"ok\n", stderr=b"")

    monkeypatch.setattr(PathBubbleBackend, "exec_shell", fake_exec)
    step = PythonExecuteStep(app_context=ApplicationContext(workspace_dir=str(tmp_path)))
    response = await step(code="print('ok')", read_only_paths=[], writable_paths=[str(tmp_path)], **memory_options)

    assert response.success is True
    assert response.answer == "ok\n"
    assert seen["command"] == [sys.executable, "-c", "print('ok')"]
    assert seen["cwd"] == str(tmp_path)
    assert seen["timeout"] == 60
    expected_limit = memory_options.get("memory_limit_mib", 4096)
    assert seen["backend"]._memory_limit_mib == expected_limit  # pylint: disable=protected-access


@pytest.mark.asyncio
@pytest.mark.parametrize("limit", [0, 1, -1])
async def test_python_execute_rejects_memory_limit_without_path_policy(tmp_path, limit):
    """A requested memory policy must never be silently ignored."""
    step = PythonExecuteStep(app_context=ApplicationContext(workspace_dir=str(tmp_path)))
    response = await step(code="print('should not run')", memory_limit_mib=limit)
    assert response.success is False
    assert response.answer == "memory_limit_mib requires read_only_paths or writable_paths to enable the sandbox"


@pytest.mark.asyncio
async def test_step_rejects_invalid_mount_parameters(tmp_path):
    """Malformed path injection cannot silently disable sandboxing."""
    step = PythonExecuteStep(app_context=ApplicationContext(workspace_dir=str(tmp_path)))
    response = await step(code="print('unsafe')", writable_paths="/tmp")
    assert response.success is False
    assert "writable_paths must be a list" in response.answer


@pytest.mark.asyncio
async def test_sandbox_start_failure_does_not_run_on_host(tmp_path, monkeypatch):
    """Python execution reports a sandbox failure instead of running on the host."""

    async def unavailable(*_args, **_kwargs):
        raise RuntimeError("bwrap unavailable")

    monkeypatch.setattr(PathBubbleBackend, "_start_process", unavailable)
    app_context = ApplicationContext(workspace_dir=str(tmp_path))
    python_step = PythonExecuteStep(app_context=app_context)
    python_result = await python_step(code="print('host')", read_only_paths=[])

    assert python_result.success is False
    assert "bwrap unavailable" in python_result.answer


def test_job_hides_injected_path_policy():
    """Path policy reaches the Step through context without becoming a tool argument."""
    config_path = Path(__file__).parents[2] / "reme" / "config" / "benchmark.yaml"
    parameters = yaml.safe_load(config_path.read_text(encoding="utf-8"))["jobs"]["python_execute"]["parameters"]
    assert "read_only_paths" not in parameters["properties"]
    assert "writable_paths" not in parameters["properties"]
    assert "default" not in parameters["properties"]["memory_limit_mib"]
