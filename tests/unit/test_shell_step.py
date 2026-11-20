"""Tests for asynchronous shell command execution."""

import asyncio
import os
import shlex
import sys
import time
from unittest.mock import AsyncMock

import psutil
import pytest

from reme.core.context import ApplicationContext
from reme.enumeration import ComponentEnum
from reme.steps.common.shell import DEFAULT_TIMEOUT, ShellStep
from reme.core.registry import R
from reme.utils.local_backend import OUTPUT_LIMIT_BYTES


def test_shell_step_is_registered():
    """Importing common steps makes shell_step discoverable."""
    assert R.get(ComponentEnum.STEP, "shell_step") is ShellStep


def test_shell_step_default_timeout_is_one_day():
    """Shell commands may run for one day when no timeout is supplied."""
    assert DEFAULT_TIMEOUT == 86400


@pytest.mark.asyncio
async def test_shell_step_returns_stdout_from_workspace(tmp_path):
    """Successful commands return stdout and run in the configured workspace."""

    step = ShellStep(app_context=ApplicationContext(workspace_dir=str(tmp_path)))
    response = await step(cmd="pwd")

    assert response.success is True
    assert response.answer.strip() == str(tmp_path)
    assert response.metadata["returncode"] == 0
    assert response.metadata["stderr"] == ""


@pytest.mark.asyncio
async def test_shell_step_reports_stderr_on_failure():
    """Failed commands expose stderr and their non-zero exit status."""

    step = ShellStep()
    response = await step(cmd="echo boom >&2; exit 7")

    assert response.success is False
    assert response.answer == "boom\n"
    assert response.metadata["returncode"] == 7
    assert response.metadata["stderr"] == "boom\n"


@pytest.mark.asyncio
async def test_shell_step_times_out():
    """Timeout terminates child processes as well as their parent shell."""

    step = ShellStep()
    started = time.monotonic()
    response = await step(cmd="sleep 3 & wait", shell_timeout=0.01)

    assert response.success is False
    assert response.answer == "Shell command timed out after 0.01s"
    assert response.metadata["shell_timeout"] == 0.01
    assert time.monotonic() - started < 1


@pytest.mark.asyncio
@pytest.mark.skipif(os.name != "posix", reason="POSIX process groups")
async def test_cancelling_shell_step_reaps_its_process_group(tmp_path):
    """Cancelling a caller must not leave its shell command running."""
    marker = tmp_path / "pid"
    script = f"import os,time; open({str(marker)!r}, 'w').write(str(os.getpid())); time.sleep(60)"
    task = asyncio.create_task(ShellStep()(cmd=f"{shlex.quote(sys.executable)} -c {shlex.quote(script)}"))
    pid = None
    try:
        async with asyncio.timeout(5):
            while not marker.exists() or not marker.read_text():
                await asyncio.sleep(0.01)
        pid = int(marker.read_text())
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert not psutil.pid_exists(pid) or psutil.Process(pid).status() == psutil.STATUS_ZOMBIE
    finally:
        if pid is not None and psutil.pid_exists(pid):
            try:
                psutil.Process(pid).kill()
            except psutil.NoSuchProcess:
                pass
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)


@pytest.mark.asyncio
@pytest.mark.skipif(os.name != "posix", reason="POSIX command quoting")
async def test_shell_output_uses_the_shared_byte_limit():
    """Shell jobs enforce the same output budget as other local commands."""
    script = f"import sys; sys.stdout.write('x' * {OUTPUT_LIMIT_BYTES + 1})"
    response = await ShellStep()(cmd=f"{shlex.quote(sys.executable)} -c {shlex.quote(script)}")
    assert not response.success
    assert "Output limit exceeded" in response.metadata["stderr"]
    assert len(response.answer.encode()) <= OUTPUT_LIMIT_BYTES


@pytest.mark.parametrize("value", [True, False, 0, -1, float("nan"), float("inf"), "invalid"])
@pytest.mark.asyncio
async def test_shell_timeout_requires_a_finite_positive_number(value, monkeypatch):
    """Invalid deadlines must be rejected before spawning a command."""
    execute = AsyncMock()
    monkeypatch.setattr("reme.steps.common.shell.LocalProcessBackend.exec_shell", execute)
    response = await ShellStep()(cmd="printf unreachable", shell_timeout=value)
    assert not response.success
    assert response.answer == "shell_timeout must be a positive number"
    execute.assert_not_awaited()


@pytest.mark.asyncio
async def test_shell_step_requires_a_command():
    """Blank commands are rejected without creating a subprocess."""

    response = await ShellStep()(cmd="  ")

    assert response.success is False
    assert response.answer == "cmd is required"


@pytest.mark.asyncio
async def test_shell_step_does_not_accept_legacy_parameter_names():
    """Only cmd and shell_timeout configure shell execution."""

    response = await ShellStep()(command="printf legacy", timeout=1)

    assert response.success is False
    assert response.answer == "cmd is required"

    response = await ShellStep()(cmd="printf current", timeout=1)

    assert response.success is True
    assert response.answer == "current"
    assert response.metadata["shell_timeout"] == DEFAULT_TIMEOUT
