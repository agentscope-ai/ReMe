"""Shared file sandbox selection and macOS policy construction."""

import asyncio
import plistlib
import sys
from pathlib import Path

import pytest
from agentscope.tool import ExecResult

from reme.components.application_context import ApplicationContext
from reme.steps.common.python_execute import PythonExecuteStep
from reme.utils.file_sandbox import create_file_sandbox_backend
from reme.utils.file_sandbox.path_bubble_backend import PathBubbleBackend
from reme.utils.file_sandbox.path_seatbelt_backend import PathSeatbeltBackend


def test_platform_selection_and_memory_defaults(tmp_path, monkeypatch):
    """Linux retains cgroups while macOS defaults to no unsupported memory cap."""
    monkeypatch.setattr("reme.utils.file_sandbox.sys.platform", "linux")
    linux = create_file_sandbox_backend(cwd=tmp_path, read_only_paths=[], writable_paths=[])
    assert isinstance(linux, PathBubbleBackend)
    assert linux._memory_limit_mib == 4096  # pylint: disable=protected-access

    monkeypatch.setattr("reme.utils.file_sandbox.sys.platform", "darwin")
    mac = create_file_sandbox_backend(cwd=tmp_path, read_only_paths=[], writable_paths=[])
    assert isinstance(mac, PathSeatbeltBackend)
    with pytest.warns(RuntimeWarning, match="limit is ignored on macOS"):
        limited = create_file_sandbox_backend(
            cwd=tmp_path,
            read_only_paths=[],
            writable_paths=[],
            memory_limit_mib=1024,
        )
    assert isinstance(limited, PathSeatbeltBackend)


@pytest.mark.asyncio
async def test_unsupported_platform_requires_backend_only_for_path_policy(tmp_path, monkeypatch):
    """An unsupported platform runs ordinary Python but rejects a requested sandbox."""
    monkeypatch.setattr("reme.utils.file_sandbox.sys.platform", "unsupported")
    step = PythonExecuteStep(app_context=ApplicationContext(workspace_dir=str(tmp_path)))

    plain = await step(code="print('host')")
    assert plain.success is True
    assert plain.answer == "host\n"

    restricted = await step(code="print('blocked')", read_only_paths=[])
    assert restricted.success is False
    assert restricted.answer == "No file sandbox backend for unsupported"


def test_seatbelt_profile_limits_file_access(tmp_path):
    """The profile grants configured paths and a separate per-run temporary directory."""
    output = tmp_path / "output"
    output.mkdir()
    backend = PathSeatbeltBackend(cwd=tmp_path, read_only_paths=[str(tmp_path)], writable_paths=[str(output)])
    profile = backend._profile(str(tmp_path / "temporary"))  # pylint: disable=protected-access
    assert "(deny default)" in profile
    assert f'(allow file-read* (subpath "{tmp_path}"))' in profile
    assert f'(allow file-write* (subpath "{output}"))' in profile
    assert f'(allow file-write* (subpath "{tmp_path}"))' not in profile
    assert '(allow file-write* (literal "/dev/null"))' in profile
    assert '(allow file-write* (subpath "/dev"))' not in profile
    assert "network*" not in profile


def test_sandbox_reads_virtual_environment_base_runtime(tmp_path, monkeypatch):
    """A virtual environment grants its real base interpreter without widening writes."""
    workspace = tmp_path / "workspace"
    venv = tmp_path / "venv"
    runtime = tmp_path / "runtime"
    interpreter = tmp_path / "interpreter" / "python"
    workspace.mkdir()
    (venv / "bin").mkdir(parents=True)
    runtime.mkdir()
    interpreter.parent.mkdir(parents=True)
    interpreter.touch()
    (venv / "bin" / "python").symlink_to(interpreter)
    monkeypatch.setattr(sys, "prefix", str(venv))
    monkeypatch.setattr(sys, "base_prefix", str(runtime))
    monkeypatch.setattr(sys, "executable", str(venv / "bin" / "python"))

    backend = PathSeatbeltBackend(cwd=workspace, read_only_paths=[], writable_paths=[str(workspace)])
    profile = backend._profile(str(workspace / "temporary"))  # pylint: disable=protected-access

    assert f'(allow file-read* (subpath "{runtime}"))' in profile
    assert f'(allow file-read* (literal "{interpreter}"))' in profile
    assert f'(allow file-write* (subpath "{runtime}"))' not in profile
    assert f'(allow file-write* (literal "{interpreter}"))' not in profile


def test_seatbelt_excludes_read_only_child_from_writable_parent(tmp_path):
    """A parent write grant must not override a nested read-only restriction."""
    protected = tmp_path / "protected"
    protected.mkdir()
    output = protected / "output"
    output.mkdir()
    backend = PathSeatbeltBackend(
        cwd=tmp_path,
        read_only_paths=[str(protected)],
        writable_paths=[str(tmp_path), str(output)],
    )
    profile = backend._profile(str(tmp_path / "temporary"))  # pylint: disable=protected-access
    assert (
        f'(allow file-write* (require-all (subpath "{tmp_path}") ' f'(require-not (subpath "{protected}"))))'
    ) in profile
    assert f'(allow file-write* (subpath "{output}"))' in profile


@pytest.mark.asyncio
async def test_sandbox_timeout_does_not_wait_forever_for_inherited_pipe(tmp_path, monkeypatch):
    """A surviving child with an output pipe must not hold the timeout response open."""
    backend = PathBubbleBackend(cwd=tmp_path, read_only_paths=[], writable_paths=[], memory_limit_mib=0)

    class StuckProcess:
        """Model a child that keeps its output pipe open after termination."""

        pid = 12345

        async def communicate(self, _data=None):
            """Wait forever until the caller cancels pipe draining."""
            await asyncio.Event().wait()

        async def wait(self):
            """Model an already reaped launcher."""
            return 0

    async def start(*_args, **_kwargs):
        return StuckProcess()

    monkeypatch.setattr(backend, "_start_process", start)
    monkeypatch.setattr(backend, "_kill_process", lambda _process: None)
    monkeypatch.setattr("reme.utils.file_sandbox.file_sandbox_backend._CLEANUP_TIMEOUT", 0.01)

    result = await asyncio.wait_for(backend.exec_shell(["true"], timeout=0.01), timeout=1)
    assert result.exit_code == -1 and result.stderr == b"timed out"


@pytest.mark.asyncio
async def test_seatbelt_launch_failure_cleans_up_without_host_fallback(tmp_path, monkeypatch):
    """A failed sandbox launcher never starts the requested command directly."""
    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setattr("reme.utils.file_sandbox.path_seatbelt_backend.shutil.which", lambda _: "/usr/bin/sandbox-exec")
    seen = {}

    async def fail(*arguments):
        seen["arguments"] = arguments
        with open(arguments[-1], "rb") as stream:
            seen["job"] = plistlib.load(stream)
        raise RuntimeError("launchd unavailable")

    backend = PathSeatbeltBackend(cwd=tmp_path, read_only_paths=[], writable_paths=[])
    monkeypatch.setattr(backend, "_launchctl", fail)

    async def available(_domain):
        return True

    monkeypatch.setattr(backend, "_has_gui_launchd", available)
    with pytest.raises(RuntimeError, match="launchd unavailable"):
        await backend.exec_shell(["/bin/echo", "unsafe"])
    assert seen["arguments"][0] == "bootstrap"
    assert seen["job"]["AbandonProcessGroup"] is False
    assert seen["job"]["ProgramArguments"][-2:] == ["/bin/echo", "unsafe"]
    assert not Path(seen["arguments"][-1]).exists()


@pytest.mark.asyncio
async def test_python_job_selects_seatbelt_on_macos(tmp_path, monkeypatch):
    """Path injection reaches the macOS backend with no implicit memory cap."""
    monkeypatch.setattr("reme.utils.file_sandbox.sys.platform", "darwin")
    seen = {}

    async def fake_exec(self, command, *, cwd=None, timeout=None):
        seen.update(backend=self, command=command, cwd=cwd, timeout=timeout)
        return ExecResult(exit_code=0, stdout=b"ok\n", stderr=b"")

    monkeypatch.setattr(PathSeatbeltBackend, "exec_shell", fake_exec)
    step = PythonExecuteStep(app_context=ApplicationContext(workspace_dir=str(tmp_path)))
    response = await step(code="print('ok')", read_only_paths=[])
    assert response.success is True
    assert isinstance(seen["backend"], PathSeatbeltBackend)
    assert seen["command"] == [sys.executable, "-c", "print('ok')"]
