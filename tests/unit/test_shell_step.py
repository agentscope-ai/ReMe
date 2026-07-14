"""Tests for asynchronous shell command execution."""

import asyncio

from reme.components.application_context import ApplicationContext
from reme.enumeration import ComponentEnum
from reme.steps.common.shell import DEFAULT_TIMEOUT, ShellStep
from reme.components import R


def _run(coro):
    """Run a coroutine on a fresh event loop."""
    asyncio.run(coro)


def test_shell_step_is_registered():
    """Importing common steps makes shell_step discoverable."""
    assert R.get(ComponentEnum.STEP, "shell_step") is ShellStep


def test_shell_step_default_timeout_is_one_day():
    """Shell commands may run for one day when no timeout is supplied."""
    assert DEFAULT_TIMEOUT == 86400


def test_shell_step_returns_stdout_from_workspace(tmp_path):
    """Successful commands return stdout and run in the configured workspace."""

    async def run():
        step = ShellStep(app_context=ApplicationContext(workspace_dir=str(tmp_path)))
        response = await step(command="pwd")

        assert response.success is True
        assert response.answer.strip() == str(tmp_path)
        assert response.metadata["returncode"] == 0
        assert response.metadata["stderr"] == ""

    _run(run())


def test_shell_step_reports_stderr_on_failure():
    """Failed commands expose stderr and their non-zero exit status."""

    async def run():
        step = ShellStep()
        response = await step(command="echo boom >&2; exit 7")

        assert response.success is False
        assert response.answer == "boom\n"
        assert response.metadata["returncode"] == 7
        assert response.metadata["stderr"] == "boom\n"

    _run(run())


def test_shell_step_times_out():
    """Commands exceeding the configured timeout return a failed response."""

    async def run():
        step = ShellStep()
        response = await step(command="sleep 0.1", timeout=0.01)

        assert response.success is False
        assert response.answer == "Shell command timed out after 0.01s"
        assert response.metadata["timeout"] == 0.01

    _run(run())


def test_shell_step_requires_a_command():
    """Blank commands are rejected without creating a subprocess."""

    async def run():
        response = await ShellStep()(command="  ")

        assert response.success is False
        assert response.answer == "command is required"

    _run(run())
