"""Exercise macOS sandbox startup and file isolation using real subprocesses."""

import asyncio
import os
import sys
import sysconfig
import tempfile
import venv
from pathlib import Path

import psutil

import pytest

from reme.utils.file_sandbox.path_seatbelt_backend import PathSeatbeltBackend

pytestmark = pytest.mark.skipif(sys.platform != "darwin", reason="Seatbelt requires macOS")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "command",
    [
        ["/bin/sh", "-c", "pwd"],
        [sys.executable, "-c", "import os; print(os.getcwd())"],
    ],
)
async def test_seatbelt_starts_shell_and_python(tmp_path, command):
    """System runtime access permits startup and resolution of the working directory."""
    workspace = tmp_path.resolve()
    backend = PathSeatbeltBackend(cwd=workspace, read_only_paths=[], writable_paths=[])

    result = await backend.exec_shell(command, timeout=10)

    assert result.exit_code == 0, result.stderr.decode(errors="replace")
    assert result.stdout.decode().strip() == str(workspace)
    assert result.stderr == b""


@pytest.mark.asyncio
async def test_seatbelt_starts_virtual_environment_python(tmp_path):
    """A symlinked virtual environment interpreter can load its base runtime."""
    workspace = tmp_path.resolve()
    environment = workspace / "venv"
    venv.EnvBuilder(with_pip=False, symlinks=True).create(environment)
    backend = PathSeatbeltBackend(cwd=workspace, read_only_paths=[], writable_paths=[])

    result = await backend.exec_shell(
        [str(environment / "bin" / "python"), "-c", "import os; print(os.getcwd())"],
        timeout=10,
    )

    assert result.exit_code == 0, result.stderr.decode(errors="replace")
    assert result.stdout.decode().strip() == str(workspace)


@pytest.mark.asyncio
async def test_seatbelt_external_virtual_environment_through_symlinks(tmp_path, monkeypatch):
    """An external venv reached through /tmp keeps its packages and file isolation."""
    workspace = tmp_path.resolve()
    with tempfile.TemporaryDirectory(prefix="reme-venv-test-", dir="/private/tmp") as directory:
        root = Path(directory)
        environment = root / "venv"
        venv.EnvBuilder(with_pip=False, symlinks=True).create(environment)
        (root / "alias").symlink_to("venv", target_is_directory=True)
        alias = Path("/tmp") / root.name / "alias"
        interpreter = alias / "bin" / "python"
        packages = Path(sysconfig.get_path("purelib", vars={"base": str(environment), "platbase": str(environment)}))
        (packages / "reme_venv_marker.py").write_text("VALUE = 'venv-package'\n", encoding="utf-8")
        secret = root / "outside.txt"
        secret.write_text("private", encoding="utf-8")
        # Match the runtime paths reported when ReMe is launched through this venv.
        monkeypatch.setattr(sys, "prefix", str(alias))
        monkeypatch.setattr(sys, "executable", str(interpreter))
        backend = PathSeatbeltBackend(cwd=workspace, read_only_paths=[], writable_paths=[])
        code = f"""
import os, sys
from pathlib import Path
import reme_venv_marker
assert os.path.realpath(sys.prefix) == {str(environment)!r}
assert sys.prefix != sys.base_prefix
assert reme_venv_marker.VALUE == 'venv-package'
for path in [{str(secret)!r}, {str(Path('/tmp') / root.name / 'outside.txt')!r}]:
    try:
        Path(path).read_text()
    except PermissionError:
        pass
    else:
        raise AssertionError(f'outside file readable: {{path}}')
print('ready')
"""
        # Exercise real Seatbelt directly, independent of GUI launchd availability.
        process = await backend._start_process(  # pylint: disable=protected-access
            [str(interpreter), "-c", code],
            str(workspace),
        )
        result = await backend._communicate(process, data=None, timeout=10)  # pylint: disable=protected-access
        assert result.exit_code == 0, result.stderr.decode(errors="replace")
        assert result.stdout == b"ready\n"
        assert os.path.realpath(interpreter) != str(interpreter)


@pytest.mark.asyncio
async def test_seatbelt_allows_null_device_redirection(tmp_path):
    """The shell can discard output through /dev/null inside a file sandbox."""
    backend = PathSeatbeltBackend(cwd=tmp_path.resolve(), read_only_paths=[], writable_paths=[])

    result = await backend.exec_shell(["/bin/sh", "-c", "echo hi > /dev/null"], timeout=5)

    assert result.exit_code == 0, result.stderr.decode(errors="replace")
    assert result.stdout == b""


@pytest.mark.asyncio
async def test_seatbelt_write_file_with_launchd_stdin(tmp_path):
    """The launchd job delivers input bytes through its configured stdin file."""
    workspace = tmp_path.resolve()
    backend = PathSeatbeltBackend(cwd=workspace, read_only_paths=[], writable_paths=[str(workspace)])
    target = workspace / "nested" / "note.txt"

    await backend.write_file(str(target), b"hello\n")

    assert target.read_bytes() == b"hello\n"


@pytest.mark.asyncio
async def test_seatbelt_without_gui_launchd_uses_direct_sandbox(tmp_path, monkeypatch):
    """A missing GUI domain warns while Seatbelt still enforces the path policy."""
    root = tmp_path.resolve()
    workspace = root / "workspace"
    workspace.mkdir()
    outside = root / "outside.txt"
    backend = PathSeatbeltBackend(cwd=workspace, read_only_paths=[], writable_paths=[str(workspace)])

    async def missing(_domain):
        return False

    monkeypatch.setattr(backend, "_has_gui_launchd", missing)
    code = (
        "from pathlib import Path;"
        "Path('inside.txt').write_text('allowed');"
        f"Path({str(outside)!r}).write_text('denied')"
    )
    with pytest.warns(RuntimeWarning, match="No macOS GUI launchd domain"):
        result = await backend.exec_shell([sys.executable, "-c", code], timeout=5)
    assert result.exit_code != 0
    assert (workspace / "inside.txt").read_text() == "allowed"
    assert not outside.exists()

    with pytest.warns(RuntimeWarning, match="No macOS GUI launchd domain"):
        await backend.write_file(str(workspace / "input.txt"), b"hello")
    assert (workspace / "input.txt").read_bytes() == b"hello"


@pytest.mark.asyncio
async def test_seatbelt_runtime_permissions_preserve_file_isolation(tmp_path):
    """Startup grants do not expose sibling files or writable access to protected children."""
    root = tmp_path.resolve()
    workspace = root / "workspace"
    protected = workspace / "protected"
    output = protected / "output"
    output.mkdir(parents=True)
    (protected / "note.txt").write_text("readable", encoding="utf-8")
    (root / "outside.txt").write_text("private", encoding="utf-8")
    backend = PathSeatbeltBackend(
        cwd=workspace,
        read_only_paths=[str(protected)],
        writable_paths=[str(workspace), str(output)],
    )
    code = """
from pathlib import Path
assert Path('protected/note.txt').read_text() == 'readable'
Path('allowed.txt').write_text('ok')
Path('protected/output/allowed.txt').write_text('ok')
for path, operation in [
    (Path('protected/denied.txt'), 'write'),
    (Path('../outside.txt'), 'read'),
    (Path('../outside.txt'), 'write'),
]:
    try:
        path.read_text() if operation == 'read' else path.write_text('denied')
    except PermissionError:
        pass
    else:
        raise AssertionError(f'{operation} unexpectedly allowed: {path}')
"""

    result = await backend.exec_shell([sys.executable, "-c", code], timeout=10)

    assert result.exit_code == 0, result.stderr.decode(errors="replace")
    assert (workspace / "allowed.txt").read_text() == "ok"
    assert (output / "allowed.txt").read_text() == "ok"
    assert not (protected / "denied.txt").exists()
    assert (root / "outside.txt").read_text() == "private"


@pytest.mark.asyncio
async def test_seatbelt_cannot_move_read_only_ancestor(tmp_path):
    """Moving a writable ancestor must not change a protected file's effective path."""
    workspace = tmp_path.resolve()
    protected = workspace / "parent" / "protected"
    protected.mkdir(parents=True)
    note = protected / "note.txt"
    note.write_text("original", encoding="utf-8")
    backend = PathSeatbeltBackend(cwd=workspace, read_only_paths=[str(protected)], writable_paths=[str(workspace)])

    result = await backend.exec_shell(
        ["/bin/sh", "-c", "mv parent renamed && echo changed > renamed/protected/note.txt"],
        timeout=5,
    )
    assert result.exit_code != 0
    assert note.read_text(encoding="utf-8") == "original"
    assert not (workspace / "renamed").exists()

    sibling = await backend.exec_shell(["/bin/sh", "-c", "echo allowed > sibling.txt"], timeout=5)
    assert sibling.exit_code == 0, sibling.stderr.decode(errors="replace")
    assert (workspace / "sibling.txt").read_text(encoding="utf-8").strip() == "allowed"


@pytest.mark.asyncio
async def test_seatbelt_reaps_same_group_child_after_parent_exits(tmp_path):
    """launchd clears a background child after its foreground command returns."""
    workspace = tmp_path.resolve()
    backend = PathSeatbeltBackend(cwd=workspace, read_only_paths=[], writable_paths=[str(workspace)])
    marker = workspace / "late.txt"
    child_code = f"import time;from pathlib import Path;time.sleep(1);Path({str(marker)!r}).write_text('late')"
    parent_code = (
        "import subprocess,sys;"
        f"subprocess.Popen([sys.executable,'-c',{child_code!r}],"
        "stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)"
    )

    result = await backend.exec_shell([sys.executable, "-c", parent_code], timeout=5)

    assert result.exit_code == 0, result.stderr.decode(errors="replace")
    await asyncio.sleep(1.2)
    assert not marker.exists()


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel", [False, True], ids=["timeout", "cancel"])
async def test_seatbelt_stops_detached_child(tmp_path, cancel):
    """A child that left the launcher's process group should be stopped when observed."""
    workspace = tmp_path.resolve()
    backend = PathSeatbeltBackend(cwd=workspace, read_only_paths=[], writable_paths=[str(workspace)])
    pidfile = workspace / "child.pid"
    marker = workspace / "late.txt"
    output = workspace / "child.out"
    child_code = (
        f"import os,time;os.setsid();open({str(pidfile)!r},'w').write(str(os.getpid()));"
        f"time.sleep(1.2);open({str(marker)!r},'w').write('late')"
    )
    parent_code = (
        f"import subprocess,sys,time;f=open({str(output)!r},'w');"
        f"subprocess.Popen([sys.executable,'-c',{child_code!r}],stdin=f,stdout=f,stderr=f);"
        "f.close();time.sleep(10)"
    )
    task = asyncio.create_task(backend.exec_shell([sys.executable, "-c", parent_code], timeout=10 if cancel else 0.5))
    try:
        for _ in range(100):
            if pidfile.exists():
                break
            await asyncio.sleep(0.02)
        assert pidfile.exists(), output.read_text() if output.exists() else "child did not start"
        if cancel:
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(task, timeout=4)
        else:
            result = await asyncio.wait_for(task, timeout=4)
            assert result.exit_code == -1 and result.stderr == b"timed out"
        await asyncio.sleep(1.2)
        assert not marker.exists()
    finally:
        if not task.done():
            task.cancel()
        if pidfile.exists():
            try:
                psutil.Process(int(pidfile.read_text(encoding="utf-8"))).kill()
            except psutil.Error:
                pass
