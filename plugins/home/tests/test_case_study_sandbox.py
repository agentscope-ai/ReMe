"""Home runtime adaptation preserves the core sandbox's path and process contracts."""

# pylint: disable=missing-function-docstring,protected-access

import asyncio
import os
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from reme_home.agent import sandbox
from reme.utils.file_sandbox import PathBubbleBackend, PathSeatbeltBackend


def test_macos_grants_only_selected_developer_runtime(tmp_path, monkeypatch):
    developer = tmp_path / "developer tools"
    (developer / "usr/bin").mkdir(parents=True)
    (developer / "usr/bin/git").touch()
    workspace = tmp_path / "case"
    workspace.mkdir()
    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setattr(
        sandbox.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=0, stdout=str(developer)),
    )
    monkeypatch.setattr(sandbox, "create_file_sandbox_backend", PathSeatbeltBackend)
    roots = [str(workspace)]
    adapted = sandbox.create_case_study_sandbox(
        cwd=workspace,
        read_only_paths=roots,
        writable_paths=[],
        memory_limit_mib=0,
    )
    assert roots == [str(workspace)]
    profile = adapted.backend._profile(str(workspace / "temporary"))
    assert f'(allow file-read* (subpath "{developer}"))' in profile
    assert f'(allow file-write* (subpath "{developer}"))' not in profile
    assert f'(allow file-read* (subpath "{tmp_path}"))' not in profile
    assert adapted.environment["DEVELOPER_DIR"] == str(developer)
    paths = adapted.environment["PATH"].split(os.pathsep)
    assert paths[0] == str(sandbox.Path(sys.prefix) / "bin")
    assert paths.index(str(developer / "usr/bin")) < paths.index("/usr/bin")


@pytest.mark.parametrize("result", [(1, ""), (0, ""), (0, "/nonexistent/reme-toolchain")])
def test_failed_macos_lookup_adds_no_grant(tmp_path, monkeypatch, result):
    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setattr(
        sandbox.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=result[0], stdout=result[1]),
    )

    def factory(**kwargs):
        return SimpleNamespace(policy=kwargs)

    monkeypatch.setattr(sandbox, "create_file_sandbox_backend", factory)
    adapted = sandbox.create_case_study_sandbox(cwd=tmp_path, read_only_paths=[str(tmp_path)], writable_paths=[])
    assert adapted.backend.policy["read_only_paths"] == [str(tmp_path)]
    assert "DEVELOPER_DIR" not in adapted.environment


@pytest.mark.asyncio
async def test_linux_environment_is_inside_bubblewrap(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(sandbox, "create_file_sandbox_backend", PathBubbleBackend)
    adapted = sandbox.create_case_study_sandbox(
        cwd=tmp_path,
        read_only_paths=[str(tmp_path)],
        writable_paths=[],
        memory_limit_mib=0,
    )
    backend = adapted.backend
    backend.exec_shell = AsyncMock(return_value="result")
    command = [sys.executable, "-I", "-B", "-c", "print('literal $HOME; `pwd`')"]
    assert await adapted.exec_shell(command, cwd=str(tmp_path), timeout=7) == "result"
    args, kwargs = backend.exec_shell.call_args
    assert kwargs == {"cwd": str(tmp_path), "timeout": 7}
    assert args[0][-len(command) :] == command
    argv = backend._bwrap_argv(args[0], str(tmp_path))
    assert "--clearenv" in argv and "--unshare-all" in argv
    assert argv.index("--") < argv.index("/usr/bin/env")
    assert "GIT_CONFIG_NOSYSTEM=1" in argv and "GIT_CONFIG_GLOBAL=/dev/null" in argv
    assert adapted.environment["PATH"].split(":")[:2] == list(
        dict.fromkeys([str(sandbox.Path(sys.prefix) / "bin"), str(sandbox.Path(sys.base_prefix) / "bin")]),
    )
    assert backend._writable_paths == []
    assert await adapted.getcwd() == str(tmp_path)
    backend.exec_shell.side_effect = asyncio.CancelledError
    with pytest.raises(asyncio.CancelledError):
        await adapted.exec_shell(command)
    with pytest.raises(ValueError, match="empty"):
        await adapted.exec_shell([])
