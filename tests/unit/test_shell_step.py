"""Tests for asynchronous shell command execution."""

import time

import pytest

from reme.components.application_context import ApplicationContext
from reme.enumeration import ComponentEnum
from reme.steps.common.shell import DEFAULT_TIMEOUT, ShellStep
from reme.components import R


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
