"""Shell job and its optional read-only execution guard."""

import asyncio
import inspect
import os
import subprocess
import threading
from importlib.metadata import EntryPoint
from pathlib import Path

import pytest
import psutil
import yaml

from agentscope.message import ToolResultState
from agentscope.permission import PermissionBehavior
from agentscope.tool import ExecResult
from reme_home.auto_bash import AutoBash, AutoBashStep
from reme.utils.local_backend import LocalProcessBackend

from reme.utils.file_sandbox.path_bubble_backend import PathBubbleBackend
from reme.components.application_context import ApplicationContext
from reme.config.config_parser import _load_config
from reme.enumeration import ComponentEnum
from reme.plugin import resolve_plugin_runtime


@pytest.fixture(autouse=True)
def linux_backend_selection(monkeypatch):
    """Keep existing backend integration assertions on the Linux implementation."""
    monkeypatch.setattr("reme.utils.file_sandbox.sys.platform", "linux")


@pytest.fixture(autouse=True)
def isolated_git_config(tmp_path, monkeypatch):
    """Keep Git tests independent of the developer's global and system config."""
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", str(tmp_path / "global.gitconfig"))
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")


@pytest.mark.asyncio
async def test_auto_bash_runs_in_workspace(tmp_path):
    """Approved commands run relative to the configured workspace."""
    (tmp_path / "note.txt").write_text("hello", encoding="utf-8")
    step = AutoBashStep(app_context=ApplicationContext(workspace_dir=str(tmp_path)))

    response = await step(command="cat note.txt")

    assert response.success is True
    assert response.answer == "hello"


@pytest.mark.asyncio
@pytest.mark.skipif(os.name != "posix", reason="POSIX shell pipelines")
@pytest.mark.parametrize("background", [False, True])
@pytest.mark.parametrize("cancel", [False, True])
async def test_local_pipeline_timeout_and_cancellation(tmp_path, monkeypatch, background, cancel):
    """Timeouts and cancellation clean up pipelines even after the shell exits."""
    (tmp_path / "note.txt").write_text("hello\n", encoding="utf-8")
    step = AutoBashStep(app_context=ApplicationContext(workspace_dir=str(tmp_path)))
    processes = []
    descendants = []
    original_start = asyncio.create_subprocess_exec
    original_stop = LocalProcessBackend._stop_process  # pylint: disable=protected-access
    started = asyncio.Event()

    def group_children(pid):
        children = []
        for child in psutil.process_iter():
            try:
                if child.pid != pid and os.getpgid(child.pid) == pid:
                    children.append(child)
            except (ProcessLookupError, PermissionError):
                pass
        return children

    async def start(*args, **kwargs):
        process = await original_start(*args, **kwargs)
        processes.append(process)
        started.set()
        return process

    async def stop(self, process, communication):
        descendants.extend(group_children(process.pid))
        await original_stop(self, process, communication)

    monkeypatch.setattr(asyncio, "create_subprocess_exec", start)
    monkeypatch.setattr(LocalProcessBackend, "_stop_process", stop)
    command = "tail -f note.txt | cat" + (" &" if background else "")
    assert AutoBash._is_read_only(command)  # pylint: disable=protected-access
    task = asyncio.create_task(step(command=command, timeout=60000 if cancel else 100))
    try:
        await asyncio.wait_for(started.wait(), timeout=2)
        if cancel:
            # Wait until pipeline children exist before cancelling the invocation.
            async def pipeline_started():
                while len(group_children(processes[0].pid)) < 2:
                    await asyncio.sleep(0.01)

            await asyncio.wait_for(pipeline_started(), timeout=2)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(task, timeout=0.8)
        else:
            response = await asyncio.wait_for(task, timeout=0.8)
            assert response.success is False
            assert "timed out after 100ms" in response.answer
        assert descendants
        assert processes[0].returncode is not None

        async def descendants_stopped():
            # Pipe EOF can precede exit after SIGKILL, especially for orphaned background children.
            pending = descendants.copy()
            while pending:
                for child in pending.copy():
                    try:
                        if not child.is_running() or child.status() == psutil.STATUS_ZOMBIE:
                            pending.remove(child)
                    except psutil.NoSuchProcess:
                        pending.remove(child)
                if pending:
                    await asyncio.sleep(0.01)

        await asyncio.wait_for(descendants_stopped(), timeout=1)
    finally:
        if not task.done():
            task.cancel()
        for process in processes:
            try:
                os.killpg(process.pid, 9)
            except ProcessLookupError:
                pass
        await asyncio.gather(task, return_exceptions=True)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "command",
    [
        "touch denied.txt",
        "cat note.txt > denied.txt",
        "echo hello > /dev/null-extra",
        "cat $(touch denied.txt)",
        "echo ${X:-$(touch denied.txt)}",
        "cat note.txt; touch denied.txt",
        "cat note.txt\ntouch denied.txt",
        "tee denied.txt",
        "find . -delete",
        "find . -exec touch denied.txt {} \\;",
        "find . -fprint denied.txt",
        "sort -o denied.txt note.txt",
        "sort -T denied.txt note.txt",
        "sort -rTdenied.txt note.txt",
        "sort --compress-program='touch denied.txt' note.txt",
        "diff --output=denied.txt note.txt note.txt",
        "git add note.txt",
        "git branch new-branch",
        "git diff --output=denied.txt",
        "git log --ext-diff",
        "git -c alias.status=add status",
        "GIT_OPTIONAL_LOCKS=1 git status",
        "GIT_PAGER=sh git log",
        "nop",
        "rg hello note.txt",
        "file -C -m magic",
        "file -bC -m magic",
        "file --comp -m magic",
        r"file -\C -m magic",
        "file -z note.txt",
        "file -p note.txt",
        "echo hello >| denied.txt",
        ": <> denied.txt",
        "sort --out=denied.txt note.txt",
        "git diff --out=denied.txt",
        "git diff --ext-di",
        "git show --textcon HEAD",
        "git log --show-signature",
        'git log --format="%G?"',
        "git log --pretty=verify",
        "git log --pre=verify",
        "git show --format=verify",
        "git show --remerge-diff",
        "for PATH in .; do git status; done",
        "git() { echo overridden; }; git status",
        "file -* magic",
    ],
)
async def test_auto_bash_denies_mutation_before_execution(tmp_path, command):
    """Writing and dynamic shell forms fail before a child process starts."""
    (tmp_path / "note.txt").write_text("hello", encoding="utf-8")
    tool = AutoBash(cwd=tmp_path)

    chunks = [chunk async for chunk in tool.call(command)]

    assert len(chunks) == 1
    assert chunks[0].state == ToolResultState.ERROR
    assert "Permission denied" in chunks[0].content[0].text
    assert not (tmp_path / "denied.txt").exists()


@pytest.mark.asyncio
async def test_step_returns_permission_error(tmp_path):
    """The job exposes a denied command as an unsuccessful response."""
    step = AutoBashStep(app_context=ApplicationContext(workspace_dir=str(tmp_path)))

    response = await step(command="touch denied.txt")

    assert response.success is False
    assert "Permission denied" in response.answer
    assert not (tmp_path / "denied.txt").exists()


@pytest.mark.asyncio
async def test_step_can_run_unrestricted_commands(tmp_path):
    """Job input can disable the guard for commands that write files."""
    step = AutoBashStep(app_context=ApplicationContext(workspace_dir=str(tmp_path)))

    response = await step(command="echo hello > written.txt", is_read_only=False)

    assert response.success is True
    assert (tmp_path / "written.txt").read_text(encoding="utf-8") == "hello\n"


@pytest.mark.asyncio
async def test_unsupported_platform_runs_without_paths_but_rejects_path_policy(tmp_path, monkeypatch):
    """AutoBash never drops an explicitly requested sandbox on unsupported systems."""
    monkeypatch.setattr("reme.utils.file_sandbox.sys.platform", "unsupported")
    step = AutoBashStep(app_context=ApplicationContext(workspace_dir=str(tmp_path)))

    plain = await step(command="echo host", is_read_only=False)
    assert plain.success is True
    assert plain.answer.strip() == "host"

    restricted = await step(command="echo blocked", is_read_only=False, read_only_paths=[])
    assert restricted.success is False
    assert restricted.answer == "No file sandbox backend for unsupported"


@pytest.mark.asyncio
@pytest.mark.parametrize("limit", [0, 1, -1])
async def test_step_rejects_memory_limit_without_path_policy(tmp_path, limit):
    """An explicitly supplied limit cannot be ignored by host execution."""
    step = AutoBashStep(app_context=ApplicationContext(workspace_dir=str(tmp_path)))
    response = await step(command="echo should-not-run", is_read_only=False, memory_limit_mib=limit)
    assert response.success is False
    assert response.answer == "memory_limit_mib requires read_only_paths or writable_paths to enable the sandbox"


@pytest.mark.asyncio
async def test_step_configuration_can_disable_read_only(tmp_path):
    """Step configuration applies when the job does not supply the flag."""
    step = AutoBashStep(app_context=ApplicationContext(workspace_dir=str(tmp_path)), is_read_only=False)

    response = await step(command="touch written.txt")

    assert response.success is True
    assert (tmp_path / "written.txt").exists()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("inputs", "message"),
    [
        ({"command": " "}, "command is required"),
        ({"command": 1}, "command is required"),
        ({"command": "echo ok", "timeout": 0}, "timeout must be a positive integer in milliseconds"),
        ({"command": "echo ok", "timeout": True}, "timeout must be a positive integer in milliseconds"),
        ({"command": "echo ok", "timeout": "1"}, "timeout must be a positive integer in milliseconds"),
        ({"command": "touch denied.txt", "is_read_only": "false"}, "is_read_only must be a boolean"),
        ({"command": "", "timeout": 0, "is_read_only": "false"}, "command is required"),
    ],
)
async def test_step_parameter_errors_do_not_execute(tmp_path, monkeypatch, inputs, message):
    """Parameter errors preserve their messages and precedence without executing."""

    def unexpected_backend(_step):
        pytest.fail("invalid parameters reached backend selection")

    monkeypatch.setattr("reme_home.auto_bash.file_sandbox_for_step", unexpected_backend)
    response = await AutoBashStep(app_context=ApplicationContext(workspace_dir=str(tmp_path)))(**inputs)
    assert response.success is False
    assert response.answer == message
    assert not (tmp_path / "denied.txt").exists()


@pytest.mark.asyncio
async def test_permission_decisions_never_ask(tmp_path):
    """Direct AgentScope use makes allow/deny decisions without ASK."""
    tool = AutoBash(cwd=tmp_path)

    allowed = await tool.check_permissions({"command": "ls"}, None)
    denied = await tool.check_permissions({"command": "touch denied.txt"}, None)

    assert allowed.behavior == PermissionBehavior.ALLOW
    assert denied.behavior == PermissionBehavior.DENY


@pytest.mark.asyncio
async def test_read_only_parsing_does_not_block_event_loop(tmp_path, monkeypatch):
    """A slow synchronous parser runs in a worker thread during permission checks."""
    started = threading.Event()
    release = threading.Event()

    def slow_parse(_command):
        started.set()
        release.wait(timeout=2)
        return True

    tool = AutoBash(cwd=tmp_path)
    monkeypatch.setattr(tool, "_is_read_only", slow_parse)
    task = asyncio.create_task(tool.check_read_only({"command": "echo hello"}))
    try:
        assert await asyncio.wait_for(asyncio.to_thread(started.wait), timeout=1)
        assert not task.done()
    finally:
        release.set()
    assert await asyncio.wait_for(task, timeout=1)


@pytest.mark.asyncio
async def test_unrestricted_tool_allows_commands_without_read_only_check(tmp_path, monkeypatch):
    """The unrestricted tool never invokes the read-only parser."""
    tool = AutoBash(cwd=tmp_path, is_read_only=False)
    monkeypatch.setattr(tool, "_is_read_only", lambda _command: pytest.fail("read-only parser was called"))

    decision = await tool.check_permissions({"command": "touch written.txt"}, None)
    chunks = [chunk async for chunk in tool.call("touch written.txt")]

    assert decision.behavior == PermissionBehavior.ALLOW
    assert all(chunk.state != ToolResultState.ERROR for chunk in chunks)
    assert (tmp_path / "written.txt").exists()


@pytest.mark.asyncio
async def test_auto_bash_uses_path_backend_when_paths_supplied(tmp_path, monkeypatch):
    """AutoBash routes an injected path policy through the backend."""
    seen = {}

    async def fake_exec(self, command, *, cwd=None, timeout=None):
        seen.update(command=command, cwd=cwd, timeout=timeout, backend=self)
        return ExecResult(exit_code=0, stdout=b"ok", stderr=b"")

    monkeypatch.setattr(PathBubbleBackend, "exec_shell", fake_exec)
    step = AutoBashStep(app_context=ApplicationContext(workspace_dir=str(tmp_path)))
    response = await step(command="printf ok", is_read_only=False, read_only_paths=[], writable_paths=[str(tmp_path)])

    assert response.success is True
    assert response.answer == "ok"
    assert seen["command"] == ["/bin/sh", "-c", "printf ok"]
    assert seen["cwd"] == str(tmp_path)
    assert seen["timeout"] == 60


@pytest.mark.asyncio
async def test_sandbox_start_failure_does_not_run_on_host(tmp_path, monkeypatch):
    """The plugin reports a sandbox failure instead of executing on the host."""

    async def unavailable(*_args, **_kwargs):
        raise RuntimeError("bwrap unavailable")

    monkeypatch.setattr(PathBubbleBackend, "_start_process", unavailable)
    step = AutoBashStep(app_context=ApplicationContext(workspace_dir=str(tmp_path)))
    result = await step(command="printf host", is_read_only=False, read_only_paths=[])

    assert result.success is False
    assert "bwrap unavailable" in result.answer


def test_home_manifest_declares_step_and_job():
    """The installed manifest declares the job, Step, and tool defaults."""
    manifest = Path(__file__).parents[1] / "src" / "reme_home" / "plugin.yaml"
    config = yaml.safe_load(manifest.read_text(encoding="utf-8"))
    assert config["backends"]["auto_bash_step"] == "reme_home.auto_bash:AutoBashStep"
    job = config["application_defaults"]["jobs"]["auto_bash"]
    assert job["steps"] == [{"backend": "auto_bash_step"}]
    assert inspect.signature(AutoBash.call).parameters["timeout"].default == 60000
    assert AutoBash.input_schema["properties"]["timeout"]["default"] == 60000
    assert job["parameters"]["properties"]["timeout"]["default"] == 60000
    assert job["parameters"]["properties"]["is_read_only"]["default"] is True
    assert job["parameters"]["properties"]["read_only_paths"]["type"] == "array"
    assert job["parameters"]["properties"]["writable_paths"]["type"] == "array"
    assert job["parameters"]["properties"]["memory_limit_mib"]["minimum"] == 0


def test_home_plugin_registers_job_and_step(monkeypatch):
    """Enabling home adds its job and Step to one application runtime."""
    entry = EntryPoint(name="home", value="reme_home", group="reme.plugins")
    monkeypatch.setattr("reme.plugin.find_entry_points", lambda group, name: [entry] if name == "home" else [])
    config = _load_config("default")
    assert "auto_bash" not in config["jobs"]
    config["plugins"] = ["home"]

    runtime = resolve_plugin_runtime(config)

    assert runtime.registry.get(ComponentEnum.STEP, "auto_bash_step") is AutoBashStep
    assert runtime.config["jobs"]["auto_bash"]["steps"] == [{"backend": "auto_bash_step"}]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "command",
    [
        ":",
        "cd . && pwd",
        "echo hello",
        "true",
        "false",
        "grep hello note.txt",
        "find . -name note.txt",
        "diff note.txt note.txt",
        "cmp note.txt note.txt",
        "git status --short",
        "git diff --stat",
        "git rev-parse HEAD",
        "git ls-files",
        "echo hello > /dev/null",
        "cat < note.txt",
        "echo hello >| /dev/null",
        "echo hello 2>&1",
        "find . -name '*.txt'",
        "(cat note.txt) | head -n 1",
    ],
)
async def test_safe_command_forms_are_allowed(tmp_path, command):
    """Common non-writing commands pass the policy regardless of exit code."""
    (tmp_path / "note.txt").write_text("hello\n", encoding="utf-8")
    tool = AutoBash(cwd=tmp_path)

    assert await tool.check_read_only({"command": command})


@pytest.mark.asyncio
async def test_sort_does_not_create_temporary_files(tmp_path):
    """Read-only sorting works in memory and fails closed when a spill is needed."""
    (tmp_path / "small.txt").write_text("b\na\n", encoding="utf-8")
    tool = AutoBash(cwd=tmp_path)

    small = [chunk async for chunk in tool.call("sort small.txt")]
    assert "".join(block.text for chunk in small for block in chunk.content) == "a\nb\n"

    with (tmp_path / "large.txt").open("w", encoding="utf-8") as output:
        for number in range(100000, 0, -1):
            output.write(f"{number:012d} abcdefghijklmnopqrstuvwxyz\n")
    large = [chunk async for chunk in tool.call("sort -S 1K large.txt > /dev/null")]
    assert "sort:" in "".join(block.text for chunk in large for block in chunk.content)


@pytest.mark.asyncio
@pytest.mark.parametrize("trace_variable", ["GIT_TRACE_REFS", "GIT_TRACE", "GIT_TRACE2_EVENT"])
async def test_read_only_git_cannot_write_inherited_traces(tmp_path, monkeypatch, trace_variable):
    """Allowed metadata queries must not write Git traces from the parent environment."""

    def git(*args):
        return subprocess.run(["git", *args], cwd=tmp_path, check=True, capture_output=True)

    git("init", "-q")
    git(
        "-c",
        "user.name=Test",
        "-c",
        "user.email=test@example.org",
        "-c",
        "commit.gpgsign=false",
        "commit",
        "--allow-empty",
        "-qm",
        "test",
    )
    expected = git("rev-parse", "HEAD").stdout.decode().strip()
    marker = tmp_path / "trace output.txt"
    tool = AutoBash(cwd=tmp_path)
    # Set after construction to check that each invocation uses the current environment.
    monkeypatch.setenv(trace_variable, str(marker))

    chunks = [chunk async for chunk in tool.call("git rev-parse HEAD")]

    assert chunks
    assert all(chunk.state != ToolResultState.ERROR for chunk in chunks)
    assert "".join(block.text for chunk in chunks for block in chunk.content).strip() == expected
    assert not marker.exists()

    # Control: the same environment still enables tracing for unrestricted execution.
    unrestricted = AutoBash(cwd=tmp_path, is_read_only=False)
    chunks = [chunk async for chunk in unrestricted.call("git rev-parse HEAD")]
    assert all(chunk.state != ToolResultState.ERROR for chunk in chunks)
    assert marker.stat().st_size > 0


@pytest.mark.asyncio
async def test_read_only_git_clears_future_trace_variables(tmp_path, monkeypatch):
    """Trace isolation uses a prefix rule instead of an enumeration of known names."""
    helper = tmp_path / "git"
    helper.write_text('#!/bin/sh\nprintf "%s" "${GIT_TRACE_FUTURE-unset}"\n', encoding="utf-8")
    helper.chmod(0o700)
    monkeypatch.setenv("PATH", str(tmp_path) + os.pathsep + os.environ["PATH"])
    monkeypatch.setenv("GIT_TRACE_FUTURE", "inherited")
    tool = AutoBash(cwd=tmp_path)

    chunks = [chunk async for chunk in tool.call("git rev-parse HEAD")]

    assert all(chunk.state != ToolResultState.ERROR for chunk in chunks)
    assert "".join(block.text for chunk in chunks for block in chunk.content) == "unset"
    assert os.environ["GIT_TRACE_FUTURE"] == "inherited"


@pytest.mark.asyncio
async def test_git_pretty_config_cannot_launch_gpg(tmp_path):
    """Git format aliases and defaults cannot smuggle signature verification."""
    repo = tmp_path / "repo"
    repo.mkdir()

    def git(*args, input_data=None):
        return subprocess.run(["git", *args], cwd=repo, input=input_data, check=True, capture_output=True)

    git("init", "-q")
    git("config", "user.name", "Test")
    git("config", "user.email", "test@example.org")
    (repo / "note.txt").write_text("hello\n", encoding="utf-8")
    git("add", "note.txt")
    git("commit", "-qm", "test")
    commit = git("cat-file", "commit", "HEAD").stdout
    signed = commit.replace(b"\n\n", b"\ngpgsig -----BEGIN PGP SIGNATURE-----\n \n -----END PGP SIGNATURE-----\n\n", 1)
    signed_hash = git("hash-object", "-t", "commit", "-w", "--stdin", input_data=signed).stdout.strip().decode()
    git("update-ref", "HEAD", signed_hash)

    marker = tmp_path / "gpg-ran"
    helper = tmp_path / "fake-gpg"
    helper.write_text(f'#!/bin/sh\nprintf called > "{marker}"\nexit 1\n', encoding="utf-8")
    helper.chmod(0o700)
    git("config", "pretty.verify", "format:%G?")
    git("config", "pretty.medium", "format:%G?")
    git("config", "pretty.oneline", "format:%G?")
    git("config", "format.pretty", "format:%G?")
    git("config", "gpg.program", str(helper))
    tool = AutoBash(cwd=repo)

    for command in ("git log", "git show HEAD", "git log --oneline"):
        chunks = [chunk async for chunk in tool.call(command)]
        assert chunks
        assert all(chunk.state != ToolResultState.ERROR for chunk in chunks)
        assert not marker.exists()

    for command in ("git log --pretty=verify", "git show --format=verify"):
        chunks = [chunk async for chunk in tool.call(command)]
        assert chunks[-1].state == ToolResultState.ERROR
        assert not marker.exists()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("subcommand", "flag"),
    [(name, "--help") for name in ("status", "diff", "log", "show", "rev-parse", "ls-files")]
    + [("status", flag) for flag in ("--hel", "--he", "--h", "-h")],
)
async def test_git_help_flags_are_denied(tmp_path, monkeypatch, subcommand, flag):
    """All allowlisted Git commands reject help before starting a process."""

    async def unexpected_process(*_args, **_kwargs):
        pytest.fail("Git help must be denied before execution")

    monkeypatch.setattr(asyncio, "create_subprocess_exec", unexpected_process)
    tool = AutoBash(cwd=tmp_path)
    command = f"git {subcommand} {flag}"

    assert not await tool.check_read_only({"command": command})
    chunks = [chunk async for chunk in tool.call(command)]
    assert len(chunks) == 1
    assert chunks[0].state == ToolResultState.ERROR
    assert "Permission denied" in chunks[0].content[0].text


@pytest.mark.asyncio
@pytest.mark.parametrize("subcommand", ["rev-parse", "ls-files"])
async def test_git_help_cannot_run_configured_man_viewer(tmp_path, monkeypatch, subcommand):
    """Help flags must not dispatch repository-configured commands on the host."""
    monkeypatch.delenv("GIT_MAN_VIEWER", raising=False)

    def git(*args):
        return subprocess.run(["git", *args], cwd=tmp_path, check=True, capture_output=True)

    git("init", "-q")
    marker = tmp_path / "denied.txt"
    git("config", "man.viewer", "reme-test")
    git("config", "man.reme-test.cmd", f'echo ran > "{marker}"')
    # Prove that this fixture would execute the helper without the guard.
    git(subcommand, "--help")
    assert marker.exists()
    marker.unlink()
    tool = AutoBash(cwd=tmp_path)
    command = f"git {subcommand} --help"

    chunks = [chunk async for chunk in tool.call(command)]

    assert not marker.exists()
    assert not await tool.check_read_only({"command": command})
    assert len(chunks) == 1
    assert chunks[0].state == ToolResultState.ERROR
    assert "Permission denied" in chunks[0].content[0].text


@pytest.mark.asyncio
@pytest.mark.parametrize("helper", ["fsmonitor", "diff", "textconv", "clean"])
async def test_git_configured_helpers_cannot_write(tmp_path, helper):
    """Repository configuration cannot turn an allowed read into helper execution."""

    def git(*args):
        return subprocess.run(["git", *args], cwd=tmp_path, check=True, capture_output=True)

    git("init", "-q")
    (tmp_path / "note.txt").write_text("before\n", encoding="utf-8")
    (tmp_path / ".gitattributes").write_text("note.txt diff=test filter=test\n", encoding="utf-8")
    git("add", "note.txt", ".gitattributes")
    hook = tmp_path / "helper"
    hook.write_text('#!/bin/sh\ntouch "' + str(tmp_path / "denied.txt") + '"\ncat\n', encoding="utf-8")
    hook.chmod(0o700)
    key = {
        "fsmonitor": "core.fsmonitor",
        "diff": "diff.external",
        "textconv": "diff.test.textconv",
        "clean": "filter.test.clean",
    }[helper]
    git("config", key, str(hook))
    (tmp_path / "note.txt").write_text("after\n", encoding="utf-8")
    command = "git status --short" if helper == "fsmonitor" else "git diff"
    tool = AutoBash(cwd=tmp_path)

    chunks = [chunk async for chunk in tool.call(command)]

    assert not (tmp_path / "denied.txt").exists()
    if helper == "clean":
        assert chunks[-1].state == ToolResultState.ERROR
        assert "external filters" in chunks[-1].content[0].text
    else:
        assert all(chunk.state != ToolResultState.ERROR for chunk in chunks)


@pytest.mark.asyncio
async def test_global_git_lfs_config_allows_metadata_queries(tmp_path, monkeypatch):
    """A global filter alone must not block Git queries that never use it."""
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", str(repo)], check=True, capture_output=True)
    (repo / "note.txt").write_text("hello\n", encoding="utf-8")
    subprocess.run(["git", "add", "note.txt"], cwd=repo, check=True, capture_output=True)

    global_config = tmp_path / "global.gitconfig"
    subprocess.run(
        ["git", "config", "--file", str(global_config), "filter.lfs.process", "/bin/false"],
        check=True,
        capture_output=True,
    )
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", str(global_config))
    tool = AutoBash(cwd=repo)

    for command, expected in (("git rev-parse --git-dir", ".git"), ("git ls-files", "note.txt")):
        chunks = [chunk async for chunk in tool.call(command)]
        assert chunks
        assert all(chunk.state != ToolResultState.ERROR for chunk in chunks)
        assert expected in "".join(block.text for chunk in chunks for block in chunk.content)

    chunks = [chunk async for chunk in tool.call("git diff")]
    assert chunks[-1].state == ToolResultState.ERROR
    assert "external filters" in chunks[-1].content[0].text


@pytest.mark.asyncio
async def test_allow_and_denied_command_options_are_combined(tmp_path, monkeypatch):
    """Allow constrains the first argument while denied still checks later flags."""
    from reme_home import auto_bash

    commands = auto_bash._READ_COMMANDS  # pylint: disable=protected-access
    # Rebuild specs from the temporary policy rather than the production cache.
    monkeypatch.setattr(auto_bash, "_read_specs", auto_bash._read_specs.__wrapped__)  # pylint: disable=protected-access
    monkeypatch.setitem(commands, "echo", {"allow": ["hello"]})
    tool = AutoBash(cwd=tmp_path)

    assert await tool.check_read_only({"command": "echo hello"})
    assert not await tool.check_read_only({"command": "echo goodbye"})

    monkeypatch.setitem(commands, "echo", {"allow": ["hello"], "denied": ["-n"]})
    assert not await tool.check_read_only({"command": "echo hello -n"})
    assert await tool.check_read_only({"command": "find . -name note.txt"})
    assert not await tool.check_read_only({"command": "find . -delete"})
