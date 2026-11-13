"""Shared file sandbox selection and macOS policy construction."""

import asyncio
import os
import plistlib
import sys
from pathlib import Path

import pytest
from agentscope.tool import ExecResult

from reme.components.application_context import ApplicationContext
from reme.steps.common.python_execute import PythonExecuteStep
from reme.utils.local_backend import LocalProcessBackend, ProcessTimeout
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
async def test_unsupported_platform_never_runs_python_on_host(tmp_path, monkeypatch):
    """Default execution fails closed when the platform has no sandbox."""
    monkeypatch.setattr("reme.utils.file_sandbox.sys.platform", "unsupported")
    step = PythonExecuteStep(app_context=ApplicationContext(workspace_dir=str(tmp_path)))

    marker = tmp_path / "host-execution"
    restricted = await step(code=f"from pathlib import Path; Path({str(marker)!r}).touch()")
    assert restricted.success is False
    assert restricted.answer == "No file sandbox backend for unsupported"
    assert not marker.exists()


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


@pytest.mark.parametrize("redirected", [False, True])
@pytest.mark.parametrize(
    "runtime",
    ["/Library/Developer/CommandLineTools", "/Applications/Xcode.app/Contents/Developer"],
)
def test_seatbelt_developer_runtime_grants_are_read_only(tmp_path, monkeypatch, redirected, runtime):
    """Standard developer tools are readable, but symlinks cannot expose other trees."""
    original_isdir = os.path.isdir
    original_realpath = os.path.realpath
    monkeypatch.setattr(os.path, "isdir", lambda path: path == runtime or original_isdir(path))
    monkeypatch.setattr(
        os.path,
        "realpath",
        lambda path: (str(tmp_path) if redirected else runtime) if path == runtime else original_realpath(path),
    )
    backend = PathSeatbeltBackend(cwd=tmp_path, read_only_paths=[], writable_paths=[])
    profile = backend._profile(str(tmp_path / "temporary"))  # pylint: disable=protected-access

    assert (f'(allow file-read* (subpath "{runtime}"))' in profile) is not redirected
    assert f'(allow file-write* (subpath "{runtime}"))' not in profile
    for path in ("/Library", "/var", "/private/var"):
        assert f'(allow file-read* (subpath "{path}"))' not in profile
        assert f'(allow file-write* (subpath "{path}"))' not in profile
    for path in ("/etc", "/var", "/var/select/developer_dir", "/private/var/db/xcode_select_link"):
        assert f'(allow file-read* (literal "{path}"))' in profile
    assert '(allow file-read-metadata (literal "/private/var/select"))' in profile


@pytest.mark.parametrize("redirected", [False, True])
def test_seatbelt_developer_runtime_grants_are_narrow(tmp_path, monkeypatch, redirected):
    """Xcode's sibling runtime is readable, without trusting symlinked installations."""
    directories = {
        "/Library/Developer/CommandLineTools",
        "/Applications/Xcode.app/Contents/Developer",
        "/Applications/Xcode.app/Contents/Frameworks",
        "/Applications/Xcode.app/Contents/SharedFrameworks",
    }
    runtime = directories | {
        "/Applications/Xcode.app/Contents/Info.plist",
        "/Applications/Xcode.app/Contents/version.plist",
    }
    exists, isdir, realpath = os.path.exists, os.path.isdir, os.path.realpath
    monkeypatch.setattr(os.path, "exists", lambda path: path in runtime or exists(path))
    monkeypatch.setattr(os.path, "isdir", lambda path: path in directories or isdir(path))
    # Model both canonical and redirected installations independently of the
    # host's Xcode.app symlink (common on GitHub's macOS runners).
    monkeypatch.setattr(
        os.path,
        "realpath",
        lambda path: (str(tmp_path / "untrusted") if redirected else path) if path in runtime else realpath(path),
    )
    backend = PathSeatbeltBackend(cwd=tmp_path, read_only_paths=[], writable_paths=[])
    profile = backend._profile(str(tmp_path / "temporary"))  # pylint: disable=protected-access

    for path in runtime:
        rule = f'(subpath "{path}")' if path in directories else f'(literal "{path}")'
        assert (f"(allow file-read* {rule})" in profile) is not redirected
        assert f"(allow file-write* {rule})" not in profile
    for parent in ("/Applications", "/Applications/Xcode.app/Contents", "/Library/Developer", "/private/var/db"):
        assert f'(allow file-read* (subpath "{parent}"))' not in profile
        assert f'(allow file-write* (subpath "{parent}"))' not in profile
    assert '(allow file-read* (literal "/private/var/db/xcode_select_link"))' in profile


def test_seatbelt_environment_drops_host_values(tmp_path, monkeypatch):
    """The environment is an allowlist, including for future secret and hook names."""
    values = {
        "REME_SANDBOX_TEST_SECRET": "secret",
        "PYTHONPATH": str(tmp_path),
        "PYTHONSTARTUP": str(tmp_path / "hook.py"),
        "DYLD_INSERT_LIBRARIES": str(tmp_path / "hook.dylib"),
        "DEVELOPER_DIR": str(tmp_path / "developer"),
        "PATH": str(tmp_path / "untrusted"),
        "GIT_TRACE_FUTURE": str(tmp_path / "trace"),
    }
    for name, value in values.items():
        monkeypatch.setenv(name, value)
    backend = PathSeatbeltBackend(cwd=tmp_path, read_only_paths=[], writable_paths=[])
    environment = backend._environment(str(tmp_path / "temporary"))  # pylint: disable=protected-access
    assert environment == {
        "HOME": str(tmp_path),
        "TMPDIR": str(tmp_path / "temporary"),
        "PATH": f"{sys.prefix}/bin:/usr/local/bin:/usr/bin:/bin",
    }
    assert all(os.environ[name] == value for name, value in values.items())


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "developer,usable",
    [
        ("/Library/Developer/CommandLineTools", True),
        ("/Applications/Xcode.app/Contents/Developer", True),
        ("/Applications/Other.app/Contents/Developer", False),
        ("/tmp/developer", False),
        ("relative/developer", False),
        ("", False),
    ],
)
@pytest.mark.parametrize("redirected", [False, True])
async def test_seatbelt_resolves_only_trusted_selected_git(tmp_path, monkeypatch, developer, usable, redirected):
    """Host selection is sanitized, and only an unredirected runtime can replace Apple's shim."""
    from unittest.mock import AsyncMock

    monkeypatch.setenv("DEVELOPER_DIR", "/untrusted/developer")
    monkeypatch.setenv("DYLD_INSERT_LIBRARIES", "/untrusted/loader")
    backend = PathSeatbeltBackend(cwd=tmp_path, read_only_paths=[], writable_paths=[])
    process = object()
    spawn = AsyncMock(return_value=process)
    communicate = AsyncMock(return_value=ExecResult(exit_code=0, stdout=f"{developer}\n".encode(), stderr=b""))
    monkeypatch.setattr(LocalProcessBackend, "_spawn_process", spawn)
    monkeypatch.setattr(LocalProcessBackend, "_communicate", communicate)
    git = f"{developer}/usr/bin/git"
    realpath = os.path.realpath
    monkeypatch.setattr(os.path, "realpath", lambda path: "/untrusted/git" if redirected and path == git else path)
    monkeypatch.setattr(os.path, "isfile", lambda path: path == git)
    monkeypatch.setattr(os, "access", lambda path, _mode: path == git)
    temporary = str(tmp_path / "temporary")
    command = ["/usr/bin/git", "status", "--short"]

    prepared, environment = await backend._prepare_command(command, temporary)  # pylint: disable=protected-access

    spawn.assert_awaited_once_with(
        ["/usr/bin/xcode-select", "--print-path"],
        cwd="/",
        env={},
        stdin=asyncio.subprocess.DEVNULL,
    )
    communicate.assert_awaited_once_with(process, timeout=5)
    selected = usable and not redirected
    assert prepared == ([git, "status", "--short"] if selected else command)
    expected_bin = f"{developer}/usr/bin:" if selected else ""
    assert environment == {
        "HOME": str(tmp_path),
        "TMPDIR": temporary,
        "PATH": f"{sys.prefix}/bin:/usr/local/bin:{expected_bin}/usr/bin:/bin",
    }
    assert command == ["/usr/bin/git", "status", "--short"]
    # A shell's script and quoting remain intact; it finds Git through PATH.
    shell = ["/bin/sh", "-c", "git rev-parse --show-toplevel"]
    prepared, _ = await backend._prepare_command(shell, temporary)  # pylint: disable=protected-access
    assert prepared == shell
    monkeypatch.setattr(os.path, "realpath", realpath)


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["missing", "timeout", "failed", "not-executable", "not-file"])
async def test_seatbelt_git_selection_failure_preserves_sandbox(tmp_path, monkeypatch, failure):
    """Unavailable tools never turn into arbitrary host execution or broader path grants."""
    from unittest.mock import AsyncMock

    backend = PathSeatbeltBackend(cwd=tmp_path, read_only_paths=[], writable_paths=[])
    spawn = AsyncMock(side_effect=FileNotFoundError() if failure == "missing" else None)
    developer = "/Library/Developer/CommandLineTools"
    result = ExecResult(
        exit_code=0 if failure in {"not-executable", "not-file"} else -1,
        stdout=f"{developer}\n".encode(),
        stderr=b"timed out" if failure == "timeout" else b"",
    )
    monkeypatch.setattr(LocalProcessBackend, "_spawn_process", spawn)
    monkeypatch.setattr(LocalProcessBackend, "_communicate", AsyncMock(return_value=result))
    monkeypatch.setattr(os.path, "isfile", lambda _: failure != "not-file")
    monkeypatch.setattr(os, "access", lambda *_: failure != "not-executable")
    command = ["/usr/bin/git", "status", "--short"]
    temporary = str(tmp_path / "temporary")

    prepared, environment = await backend._prepare_command(command, temporary)  # pylint: disable=protected-access

    assert prepared == command
    assert environment == backend._environment(temporary)  # pylint: disable=protected-access


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


def test_seatbelt_runtime_links_have_only_literal_read_grants(tmp_path, monkeypatch):
    """Follow relative link chains while keeping user-configured symlinks invalid."""
    workspace = tmp_path / "workspace"
    environment = tmp_path / "venv"
    workspace.mkdir()
    (environment / "bin").mkdir(parents=True)
    (environment / "bin" / "python").symlink_to("python3")
    (environment / "bin" / "python3").symlink_to(sys.executable)
    alias = tmp_path / "alias"
    alias.symlink_to("venv", target_is_directory=True)
    monkeypatch.setattr(sys, "prefix", str(alias))
    monkeypatch.setattr(sys, "executable", str(alias / "bin" / "python"))

    backend = PathSeatbeltBackend(cwd=workspace, read_only_paths=[], writable_paths=[])
    profile = backend._profile(str(workspace / "temporary"))  # pylint: disable=protected-access
    for link in (alias, environment / "bin" / "python", environment / "bin" / "python3"):
        assert f'(allow file-read* (literal "{link}"))' in profile
        assert f'(allow file-read* (subpath "{link}"))' not in profile
    assert f'(allow file-read* (subpath "{tmp_path}"))' not in profile
    with pytest.raises(ValueError, match="without symlinks"):
        PathSeatbeltBackend(cwd=workspace, read_only_paths=[str(alias)], writable_paths=[])


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
@pytest.mark.parametrize("data", [b"", b"hello\x00\n\xff"])
@pytest.mark.parametrize("relative", [False, True])
async def test_sandbox_file_io_round_trip(tmp_path, monkeypatch, data, relative):
    """The common file helpers preserve binary stdin, literal paths and write errors."""
    backend = PathBubbleBackend(cwd=tmp_path, read_only_paths=[], writable_paths=[str(tmp_path)])
    # Exercise shared file I/O with real pipes; OS isolation is checked separately.
    monkeypatch.setattr(
        backend,
        "_start_process",
        LocalProcessBackend()._start_process,  # pylint: disable=protected-access
    )
    path = "nested directory/literal ' ; name.bin"
    if not relative:
        path = str(tmp_path / path)
    await backend.write_file(path, data)
    assert await backend.read_file(path) == data
    with pytest.raises(TypeError, match="data must be bytes"):
        await backend.write_file(path, "invalid")
    with pytest.raises(OSError):
        await backend.write_file(str(tmp_path), data)


@pytest.mark.asyncio
@pytest.mark.parametrize("backend_class", [PathBubbleBackend, PathSeatbeltBackend])
@pytest.mark.parametrize("target", ["absolute", "relative", "parent", "symlink", "no_grants", "file_sibling"])
async def test_sandbox_write_rejects_ungranted_paths_before_execution(tmp_path, monkeypatch, backend_class, target):
    """Private temporary mounts must not make ungranted host-path writes look successful."""
    from unittest.mock import AsyncMock

    output = tmp_path / "output"
    output.mkdir()
    outside = tmp_path / "outside.bin"
    outside.write_bytes(b"original")
    (output / "link").symlink_to(outside)
    granted_file = output / "allowed.bin"
    granted_file.touch()
    paths = {
        "absolute": str(outside),
        "relative": "outside.bin",
        "parent": "output/../outside.bin",
        "symlink": str(output / "link"),
        "no_grants": str(output / "new.bin"),
        "file_sibling": str(output / "sibling.bin"),
    }
    writable = [] if target == "no_grants" else [str(granted_file if target == "file_sibling" else output)]
    backend = backend_class(cwd=tmp_path, read_only_paths=[], writable_paths=writable)
    run_job = AsyncMock(return_value=ExecResult(exit_code=0, stdout=b"", stderr=b""))
    monkeypatch.setattr(backend, "_run_job", run_job)

    with pytest.raises(PermissionError, match="outside writable_paths"):
        await backend.write_file(paths[target], b"forbidden")

    run_job.assert_not_awaited()
    assert outside.read_bytes() == b"original"


@pytest.mark.asyncio
async def test_seatbelt_launch_failure_cleans_up_without_host_fallback(tmp_path, monkeypatch):
    """A failed sandbox launcher never starts the requested command directly."""
    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setenv("REME_TEST_SECRET", "must-not-be-inherited")
    monkeypatch.setenv("PYTHONPATH", "/untrusted/python")
    monkeypatch.setattr("reme.utils.file_sandbox.path_seatbelt_backend.shutil.which", lambda _: "/usr/bin/sandbox-exec")
    seen = {}

    async def fail(*arguments):
        if arguments[0] == "bootout":
            import errno
            from reme.utils.file_sandbox.path_seatbelt_backend import LaunchctlError

            raise LaunchctlError("service not found", errno.ESRCH)
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
    assert "REME_TEST_SECRET" not in seen["job"]["EnvironmentVariables"]
    assert "PYTHONPATH" not in seen["job"]["EnvironmentVariables"]
    argv = seen["job"]["ProgramArguments"]
    assert argv[argv.index("/usr/bin/env") + 1] == "-i"
    assert not any("must-not-be-inherited" in argument for argument in argv)
    assert not Path(seen["arguments"][-1]).exists()


@pytest.mark.asyncio
async def test_python_job_selects_seatbelt_on_macos(tmp_path, monkeypatch):
    """Default execution selects macOS isolation with a read-only workspace."""
    monkeypatch.setattr("reme.utils.file_sandbox.sys.platform", "darwin")
    seen = {}

    async def fake_run_job(self, command, cwd, *, data, timeout, raise_errors=False):
        assert data is None
        assert raise_errors is True
        seen.update(backend=self, command=command, cwd=cwd, timeout=timeout)
        return ExecResult(exit_code=0, stdout=b"ok\n", stderr=b"")

    monkeypatch.setattr(PathSeatbeltBackend, "_run_job", fake_run_job)
    step = PythonExecuteStep(app_context=ApplicationContext(workspace_dir=str(tmp_path)))
    response = await step(code="print('ok')")
    assert response.success is True
    assert isinstance(seen["backend"], PathSeatbeltBackend)
    assert seen["command"] == [sys.executable, "-c", "print('ok')"]
    assert seen["cwd"] == str(tmp_path)
    assert seen["timeout"] == 60
    profile = seen["backend"]._profile(str(tmp_path / "temporary"))  # pylint: disable=protected-access
    assert f'(allow file-read* (subpath "{tmp_path}"))' in profile
    assert f'(allow file-write* (subpath "{tmp_path}"))' not in profile


@pytest.mark.asyncio
@pytest.mark.parametrize("platform", ["linux", "darwin"])
@pytest.mark.parametrize("grant", ["default", "workspace", "output"])
async def test_python_workspace_grants(tmp_path, monkeypatch, platform, grant):
    """Backend completion keeps defaults read-only and accepts explicit write grants."""
    monkeypatch.setattr("reme.utils.file_sandbox.sys.platform", platform)
    output = tmp_path / "output"
    output.mkdir()
    writable = [] if grant == "default" else [str(tmp_path if grant == "workspace" else output)]
    backend_class = PathBubbleBackend if platform == "linux" else PathSeatbeltBackend

    async def inspect_exec(backend, _command, **kwargs):
        assert kwargs["raise_errors"] is True
        assert backend._writable_paths == writable  # pylint: disable=protected-access
        read_only = backend._read_only_paths  # pylint: disable=protected-access
        assert (str(tmp_path) in read_only) is (grant != "workspace")
        return ExecResult(exit_code=0, stdout=b"ok\n", stderr=b"")

    monkeypatch.setattr(backend_class, "exec_shell", inspect_exec)
    step = PythonExecuteStep(
        app_context=ApplicationContext(workspace_dir=str(tmp_path)),
        **({"writable_paths": writable} if writable else {}),
    )
    response = await step(code="print('ok')")
    assert response.success is True
    assert response.answer == "ok\n"


@pytest.mark.asyncio
@pytest.mark.parametrize("platform", ["linux", "darwin"])
@pytest.mark.parametrize("name", ["read_only_paths", "writable_paths"])
async def test_python_rejects_request_path_grants(tmp_path, monkeypatch, platform, name):
    """Neither platform accepts request-level path authorization."""
    monkeypatch.setattr("reme.utils.file_sandbox.sys.platform", platform)
    step = PythonExecuteStep(app_context=ApplicationContext(workspace_dir=str(tmp_path)))
    response = await step(code="print('blocked')", **{name: [str(tmp_path)]})
    assert response.success is False
    assert response.answer == f"{name} must be configured on the Step, not supplied by the caller"


@pytest.mark.asyncio
async def test_seatbelt_direct_execution_does_not_inherit_service_environment(tmp_path, monkeypatch):
    """The non-launchd path strips credentials and interpreter injection settings too."""
    monkeypatch.setenv("REME_TEST_SECRET", "must-not-be-inherited")
    monkeypatch.setenv("PYTHONPATH", "/untrusted/python")
    monkeypatch.setenv("DYLD_INSERT_LIBRARIES", "/untrusted/library")
    backend = PathSeatbeltBackend(cwd=tmp_path, read_only_paths=[], writable_paths=[])
    seen = {}

    async def spawn(_command, **kwargs):
        seen.update(kwargs)
        raise RuntimeError("stop before spawning")

    monkeypatch.setattr(backend, "_spawn_process", spawn)
    with pytest.raises(RuntimeError, match="stop before spawning"):
        await backend._start_process(  # pylint: disable=protected-access
            [sys.executable, "-c", "print('ok')"],
            str(tmp_path),
        )
    assert set(seen["env"]) == {"HOME", "TMPDIR", "PATH"}
    assert seen["env"]["HOME"] == str(tmp_path)
    assert not Path(seen["env"]["TMPDIR"]).exists()


@pytest.mark.asyncio
@pytest.mark.parametrize("raise_errors", [False, True])
@pytest.mark.parametrize(
    "mode",
    ["success", "failure", "timeout", "timeout_status", "cancel", "error", "kickstart_error", "write"],
)
async def test_seatbelt_launchd_lifecycle(tmp_path, monkeypatch, mode, raise_errors):
    """Every launchd outcome unregisters the job and removes its control files."""
    from unittest.mock import AsyncMock, Mock

    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setattr("reme.utils.file_sandbox.path_seatbelt_backend.shutil.which", lambda _: "sandbox-exec")
    backend = PathSeatbeltBackend(
        cwd=tmp_path,
        read_only_paths=[],
        writable_paths=[str(tmp_path)] if mode == "write" else [],
    )
    monkeypatch.setattr(backend, "_has_gui_launchd", AsyncMock(return_value=True))
    killed = Mock()
    monkeypatch.setattr(backend, "_kill_process", killed)
    calls = []
    control = None
    started = asyncio.Event()

    async def launchctl(*args):
        nonlocal control
        calls.append(args[0])
        if args[0] == "bootstrap":
            control = Path(args[-1]).parent
            if mode == "write":
                with open(args[-1], "rb") as stream:
                    job = plistlib.load(stream)
                assert Path(job["StandardInPath"]).read_bytes() == b"binary\x00\xff"
        if args[0] == "kickstart" and mode == "kickstart_error":
            raise RuntimeError("kickstart failed")
        if args[0] == "bootout":
            # Output is collected after launchd has reaped the process group.
            (control / "stdout").write_bytes(b"output")
            (control / "stderr").write_bytes(b"diagnostic")
            if mode == "timeout_status":
                (control / "status").write_text("137\n", encoding="ascii")
        return "12345"

    async def wait_status(*_args):
        started.set()
        if mode == "error":
            raise RuntimeError("status failed")
        if mode in ("timeout", "timeout_status", "cancel"):
            await asyncio.Event().wait()
        return 7 if mode == "failure" else 0

    monkeypatch.setattr(backend, "_launchctl", launchctl)
    monkeypatch.setattr(backend, "_wait_status", wait_status)
    operation = (
        backend.write_file(str(tmp_path / "note.bin"), b"binary\x00\xff")
        if mode == "write"
        else backend.exec_shell(
            ["true"],
            timeout=0.01 if mode.startswith("timeout") else None,
            raise_errors=raise_errors,
        )
    )
    task = asyncio.create_task(operation)
    if mode == "cancel":
        await asyncio.wait_for(started.wait(), timeout=1)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, timeout=1)
    elif mode in ("error", "kickstart_error"):
        with pytest.raises(RuntimeError, match="failed"):
            await asyncio.wait_for(task, timeout=1)
    elif mode.startswith("timeout") and raise_errors:
        with pytest.raises(ProcessTimeout) as caught:
            await asyncio.wait_for(task, timeout=1)
        assert caught.value.result == ExecResult(
            exit_code=137 if mode == "timeout_status" else -1,
            stdout=b"output",
            stderr=b"diagnostic",
        )
    else:
        result = await asyncio.wait_for(task, timeout=1)
        if mode == "write":
            assert result is None
        elif mode.startswith("timeout"):
            assert result == ExecResult(exit_code=-1, stdout=b"", stderr=b"timed out")
        else:
            assert result.exit_code == (7 if mode == "failure" else 0)
            assert result.stdout == b"output" and result.stderr == b"diagnostic"
    assert calls == ["bootstrap", "kickstart", "bootout"]
    assert not control.exists()
    assert killed.call_count == int(mode in ("timeout", "timeout_status", "cancel", "error"))


@pytest.mark.asyncio
@pytest.mark.parametrize("backend_class", [PathBubbleBackend, PathSeatbeltBackend])
async def test_python_timeout_preserves_exit_status_and_stderr(tmp_path, monkeypatch, backend_class):
    """Real pipes exercise sandbox timeout propagation without OS isolation requirements."""
    from unittest.mock import AsyncMock

    backend = backend_class(cwd=tmp_path, read_only_paths=[], writable_paths=[])

    async def start(command, cwd, *, stdin):
        return await LocalProcessBackend()._start_process(command, cwd, stdin=stdin)  # pylint: disable=protected-access

    monkeypatch.setattr(backend, "_start_process", start)
    if backend_class is PathSeatbeltBackend:
        monkeypatch.setattr(sys, "platform", "darwin")
        monkeypatch.setattr("reme.utils.file_sandbox.path_seatbelt_backend.shutil.which", lambda _: "sandbox-exec")
        monkeypatch.setattr(backend, "_has_gui_launchd", AsyncMock(return_value=False))
        monkeypatch.setattr("reme.utils.file_sandbox.path_seatbelt_backend.warnings.warn", lambda *a, **kw: None)
    monkeypatch.setattr(
        "reme.steps.common.python_execute.file_sandbox_for_step",
        lambda _step: backend,
    )
    step = PythonExecuteStep(app_context=ApplicationContext(workspace_dir=str(tmp_path)))
    response = await step(
        code="import sys, time; print('diagnostic', file=sys.stderr, flush=True); time.sleep(60)",
        python_timeout=0.3,
    )
    assert response.success is False
    assert response.answer == "Python execution timed out after 0.3s"
    assert response.metadata == {
        "returncode": -9 if os.name == "posix" else 1,
        "stderr": "diagnostic\n",
        "python_timeout": 0.3,
    }
