"""Real macOS execution, deliberately separate from the mocked command tests."""

import shlex
import subprocess
import sys

import pytest
from reme_home.auto_bash import AutoBash, AutoBashStep

from reme.components.application_context import ApplicationContext
from reme.steps.common.python_execute import PythonExecuteStep
from reme.utils.file_sandbox.path_seatbelt_backend import PathSeatbeltBackend

pytestmark = pytest.mark.skipif(sys.platform != "darwin", reason="Seatbelt requires macOS")


@pytest.mark.asyncio
async def test_auto_bash_git_preserves_sandbox(tmp_path, monkeypatch):
    """Apple Git works under the default sandbox without granting external access."""
    root = tmp_path.resolve()
    workspace = root / "workspace"
    workspace.mkdir()
    # Use the Apple dispatcher rather than a developer's Homebrew Git.
    subprocess.run(["/usr/bin/git", "init", "-q", str(workspace)], check=True)
    (workspace / "note.txt").write_text("hello", encoding="utf-8")
    outside = root / "outside.txt"
    outside.write_text("private", encoding="utf-8")
    monkeypatch.setenv("REME_SANDBOX_TEST_SECRET", "must-not-inherit")
    context = ApplicationContext(workspace_dir=str(workspace))
    step = AutoBashStep(app_context=context)

    # The default PATH may select Homebrew Git. Explicitly exercise Apple's
    # dispatcher inside the same default backend to cover xcode-select access.
    backend = AutoBash(cwd=workspace)._backend  # pylint: disable=protected-access
    assert isinstance(backend, PathSeatbeltBackend)
    for command, expected in (("status", "?? note.txt"), ("rev-parse", str(workspace))):
        flag = "--short" if command == "status" else "--show-toplevel"
        result = await backend.exec_shell(["/usr/bin/git", command, flag], timeout=10)
        assert result.exit_code == 0, result.stderr.decode(errors="replace")
        assert expected in result.stdout.decode()

    for read_only in (True, False):
        status = await step(command="git status --short", is_read_only=read_only)
        assert status.success, status.answer
        assert "?? note.txt" in status.answer
        toplevel = await step(command="git rev-parse --show-toplevel", is_read_only=read_only)
        assert toplevel.success, toplevel.answer
        assert str(workspace) in toplevel.answer

    denied = await step(command=f"cat {shlex.quote(str(outside))}")
    assert not denied.success
    denied = await step(command=f"echo changed > {shlex.quote(str(outside))}", is_read_only=False)
    assert not denied.success
    assert outside.read_text(encoding="utf-8") == "private"
    denied = await step(command="echo changed > note.txt", is_read_only=False)
    assert not denied.success
    assert (workspace / "note.txt").read_text(encoding="utf-8") == "hello"
    denied = await step(command="true", is_read_only=False, writable_paths=[str(root)])
    assert not denied.success
    assert "must be configured on the Step" in denied.answer

    # Both public executors share the backend affected by the runtime grants.
    code = (
        "import os; from pathlib import Path; "
        "assert 'REME_SANDBOX_TEST_SECRET' not in os.environ; "
        f"Path({str(outside)!r}).read_text()"
    )
    python = PythonExecuteStep(app_context=context)
    denied = await python(code=code)
    assert not denied.success
    assert "PermissionError" in denied.metadata["stderr"]
    clean = await step(
        command=shlex.join([sys.executable, "-c", "import os; print(os.getenv('REME_SANDBOX_TEST_SECRET', 'clean'))"]),
        is_read_only=False,
    )
    assert clean.success, clean.answer
    assert clean.answer.strip() == "clean"
