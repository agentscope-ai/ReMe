"""Execution output stays bounded across pipe and launchd backends."""

# pylint: disable=protected-access

import asyncio
import os
import plistlib
import stat
import sys
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from reme.components.application_context import ApplicationContext
from reme.steps.common.python_execute import PythonExecuteStep
from reme.utils.file_sandbox.path_seatbelt_backend import PathSeatbeltBackend
from reme.utils.local_backend import LocalProcessBackend, OUTPUT_LIMIT_BYTES


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", ["stdout", "stderr", "both"])
@pytest.mark.parametrize("extra", [0, 1])
async def test_pipe_output_boundary(tmp_path, stream, extra):
    """Each stream has its own budget; crossing either budget fails execution."""
    backend = LocalProcessBackend()
    size = OUTPUT_LIMIT_BYTES + extra
    code = "import os;" + ";".join(
        f"os.write({fd}, b'x' * {size})" for fd in (1, 2) if stream in ("both", "stdout" if fd == 1 else "stderr")
    )
    result = await backend.exec_shell([sys.executable, "-c", code], cwd=str(tmp_path), timeout=5)
    assert len(result.stdout) <= OUTPUT_LIMIT_BYTES
    assert len(result.stderr) <= OUTPUT_LIMIT_BYTES
    if extra:
        assert not result.ok()
        assert b"Output limit exceeded:" in result.stderr
    else:
        assert result.ok(), result.stderr
        assert result.stdout == (b"x" * size if stream in ("stdout", "both") else b"")
        assert result.stderr == (b"x" * size if stream in ("stderr", "both") else b"")


@pytest.mark.asyncio
@pytest.mark.parametrize("fd", [1, 2])
async def test_continuous_output_stops_process(tmp_path, fd):
    """Overflow terminates a live writer before it can perform a later write."""
    marker = tmp_path / "late.txt"
    code = (
        "import os,time;from pathlib import Path;"
        f"os.write({fd}, b'x' * {OUTPUT_LIMIT_BYTES + 1});"
        f"time.sleep(1);Path({str(marker)!r}).write_text('late')"
    )
    backend = LocalProcessBackend()
    process = await backend._start_process([sys.executable, "-c", code], str(tmp_path))
    result = await asyncio.wait_for(backend._communicate(process), timeout=3)
    assert not result.ok()
    assert b"Output limit exceeded:" in result.stderr
    assert process.returncode is not None
    assert not marker.exists()


@pytest.mark.asyncio
async def test_python_step_reports_output_overflow(tmp_path):
    """Python callers receive bounded partial output and failure metadata."""
    step = PythonExecuteStep(app_context=ApplicationContext(workspace_dir=str(tmp_path)))
    response = await step(code=f"import os;os.write(1, b'x' * {OUTPUT_LIMIT_BYTES + 1})")
    assert response.success is False
    assert len(response.answer) == OUTPUT_LIMIT_BYTES
    assert "Output limit exceeded: stdout" in response.metadata["stderr"]
    assert response.metadata["returncode"] != 0


@pytest.mark.asyncio
@pytest.mark.skipif(os.name != "posix", reason="launchd uses POSIX FIFOs")
@pytest.mark.parametrize("stream", ["stdout", "stderr", "both"])
@pytest.mark.parametrize("extra", [0, 1])
async def test_launchd_output_boundary(tmp_path, monkeypatch, stream, extra):
    """Real FIFO writers obey each stream's budget and are reaped on overflow."""
    backend = PathSeatbeltBackend(cwd=tmp_path, read_only_paths=[], writable_paths=[])
    monkeypatch.setattr("reme.utils.file_sandbox.path_seatbelt_backend.sys.platform", "darwin")
    monkeypatch.setattr("reme.utils.file_sandbox.path_seatbelt_backend.shutil.which", lambda _: "sandbox-exec")
    monkeypatch.setattr(backend, "_has_gui_launchd", AsyncMock(return_value=True))
    size = OUTPUT_LIMIT_BYTES + extra
    code = "import os,time;" + ";".join(
        f"os.write({fd}, b'x' * {size})" for fd in (1, 2) if stream in ("both", "stdout" if fd == 1 else "stderr")
    )
    if extra:
        code += ";time.sleep(60)"
    job = {}
    process = None
    control = None
    calls = []

    async def launchctl(*args):
        nonlocal job, process, control
        calls.append(args[0])
        if args[0] == "bootstrap":
            control = Path(args[-1]).parent
            with open(args[-1], "rb") as source:
                job = plistlib.load(source)
            for key in ("StandardOutPath", "StandardErrorPath"):
                assert stat.S_ISFIFO(os.stat(job[key]).st_mode)
        elif args[0] == "kickstart":
            with open(job["StandardOutPath"], "wb") as stdout, open(job["StandardErrorPath"], "wb") as stderr:
                process = await asyncio.create_subprocess_exec(
                    sys.executable,
                    "-c",
                    code,
                    stdout=stdout,
                    stderr=stderr,
                    start_new_session=True,
                )
            return str(process.pid)
        elif args[0] == "bootout":
            if process.returncode is None:
                process.kill()
            await process.wait()
        return ""

    async def wait_status(*_args):
        return await process.wait()

    monkeypatch.setattr(backend, "_launchctl", launchctl)
    monkeypatch.setattr(backend, "_wait_status", wait_status)
    try:
        result = await asyncio.wait_for(backend.exec_shell(["test-command"], timeout=5), timeout=7)
        assert len(result.stdout) <= OUTPUT_LIMIT_BYTES
        assert len(result.stderr) <= OUTPUT_LIMIT_BYTES
        assert result.ok() is (extra == 0)
        if extra:
            assert b"Output limit exceeded:" in result.stderr
        else:
            assert result.stdout == (b"x" * size if stream in ("stdout", "both") else b"")
            assert result.stderr == (b"x" * size if stream in ("stderr", "both") else b"")
        assert calls == ["bootstrap", "kickstart", "bootout"]
        assert process.returncode is not None
        assert not control.exists()
    finally:
        if process is not None and process.returncode is None:
            process.kill()
            await process.wait()
