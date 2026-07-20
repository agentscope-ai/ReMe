"""Unit tests for the Codex agent wrapper and its FastMCP bridge."""

# pylint: disable=missing-class-docstring,missing-function-docstring,protected-access

import asyncio
from dataclasses import dataclass
import json
from pathlib import Path
import stat
import sys
from types import SimpleNamespace

import pytest
from openai_codex.generated.v2_all import TokenUsageBreakdown
from pydantic import BaseModel

from reme.components.agent_wrapper.codex_agent_wrapper import CodexAgentWrapper
from reme.components.agent_wrapper.codex_mcp_server import _prepare_config
from reme.components.job import BackgroundJob
from reme.config import resolve_app_config
from reme.enumeration import ChunkEnum, ComponentEnum
from reme.schema import ApplicationConfig, Response


class _Job:
    def __init__(self, name="search"):
        self.name = name
        self.description = "Search memory"
        self.parameters = {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        }
        self.calls = []

    async def __call__(self, **kwargs):
        self.calls.append(kwargs)
        return Response(answer=f"found:{kwargs['query']}")


def _wrapper(tmp_path, **kwargs):
    job = _Job()
    config = SimpleNamespace(
        workspace_dir=str(tmp_path),
        mem_session_dir="mem_session",
        environment={},
        components={ComponentEnum.AS_LLM: {}},
        model_dump=lambda **_kwargs: {
            "workspace_dir": str(tmp_path),
            "enable_logo": False,
            "log_to_console": False,
            "log_to_file": False,
            "jobs": {},
            "components": {},
        },
    )
    context = SimpleNamespace(app_config=config, jobs={job.name: job})
    return CodexAgentWrapper(app_context=context, **kwargs), job


def test_mcp_config_uses_stdio_bridge_and_selected_jobs(tmp_path):
    wrapper, _job = _wrapper(tmp_path, mcp_config="custom.yaml")

    config = wrapper._mcp_server_config(  # pylint: disable=protected-access
        {"job_tools": ["search", "search"], "tool_context_id": "ctx-1"},
    )

    assert config["command"]
    assert config["enabled_tools"] == ["search"]
    assert config["args"].count("--job") == 1
    assert config["args"][config["args"].index("--job") + 1] == "search"
    assert "reme.components.agent_wrapper.codex_mcp_server" in config["args"]
    assert config["args"][config["args"].index("--config") + 1] == str(tmp_path / "custom.yaml")
    assert config["args"][config["args"].index("--tool-context-id") + 1] == "ctx-1"


def test_thread_config_preserves_other_mcp_servers(tmp_path):
    wrapper, _job = _wrapper(tmp_path)
    config = wrapper._thread_config(  # pylint: disable=protected-access
        {
            "job_tools": ["search"],
            "config": {"mcp_servers": {"docs": {"url": "https://example.test/mcp"}}},
        },
    )

    assert "docs" in config["mcp_servers"]
    assert len(config["mcp_servers"]) == 2
    assert next(name for name in config["mcp_servers"] if name != "docs").startswith("reme_jobs_")


def test_mcp_config_rejects_background_jobs(tmp_path):
    wrapper, _job = _wrapper(tmp_path)
    wrapper.app_context.jobs["watch"] = BackgroundJob(name="watch", app_context=wrapper.app_context)

    with pytest.raises(TypeError, match="non-stream request jobs"):
        wrapper._mcp_server_config({"job_tools": ["watch"]})


def test_prepare_config_reuses_selected_stdio_mcp_service():
    prepared = _prepare_config(
        {
            "service": {"backend": "http"},
            "jobs": {
                "selected": {"backend": "base", "enable_serve": False},
                "helper": {"backend": "base"},
                "watch": {"backend": "background"},
            },
        },
        ["selected"],
        "ctx-1",
    )

    assert prepared["service"] == {
        "backend": "mcp",
        "transport": "stdio",
        "jobs": ["selected"],
        "tool_error_on_failure": True,
        "injected_job_kwargs": {"tool_context_id": "ctx-1"},
    }
    assert set(prepared["jobs"]) == {"selected", "helper"}
    assert prepared["jobs"]["selected"]["enable_serve"] is True


def test_prepare_config_rejects_missing_or_background_selected_jobs():
    with pytest.raises(KeyError, match="missing"):
        _prepare_config({"jobs": {}}, ["missing"])
    with pytest.raises(KeyError, match="watch"):
        _prepare_config({"jobs": {"watch": {"backend": "background"}}}, ["watch"])


@pytest.mark.asyncio
async def test_stdio_bridge_starts_and_lists_selected_job(tmp_path):
    from fastmcp import Client
    from fastmcp.client import StdioTransport

    config_path = tmp_path / "bridge.json"
    config_path.write_text(
        json.dumps(
            {
                "service": {"backend": "mcp"},
                "workspace_dir": str(tmp_path / "workspace"),
                "jobs": {
                    "empty": {
                        "backend": "base",
                        "description": "Return an empty response",
                        "parameters": {"type": "object", "properties": {}},
                        "steps": [],
                    },
                },
            },
        ),
        encoding="utf-8",
    )
    transport = StdioTransport(
        command=sys.executable,
        args=[
            "-m",
            "reme.components.agent_wrapper.codex_mcp_server",
            "--config",
            str(config_path),
            "--workspace",
            str(tmp_path / "workspace"),
            "--job",
            "empty",
        ],
        cwd=str(Path(__file__).resolve().parents[2]),
    )

    async with Client(transport, timeout=10) as client:
        tools = await client.list_tools()

    assert [tool.name for tool in tools] == ["empty"]


@pytest.mark.asyncio
async def test_stdio_bridge_stdout_is_protocol_clean(tmp_path):
    config_path = tmp_path / "bridge.json"
    config_path.write_text(
        json.dumps(
            {
                "service": {"backend": "mcp"},
                "workspace_dir": str(tmp_path / "workspace"),
                "jobs": {
                    "empty": {
                        "backend": "base",
                        "description": "Empty",
                        "parameters": {"type": "object", "properties": {}},
                        "steps": [],
                    },
                },
            },
        ),
        encoding="utf-8",
    )
    proc = await asyncio.create_subprocess_exec(
        sys.executable,
        "-m",
        "reme.components.agent_wrapper.codex_mcp_server",
        "--config",
        str(config_path),
        "--workspace",
        str(tmp_path / "workspace"),
        "--job",
        "empty",
        cwd=str(Path(__file__).resolve().parents[2]),
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    request = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {
            "protocolVersion": "2025-06-18",
            "capabilities": {},
            "clientInfo": {"name": "test", "version": "1"},
        },
    }
    assert proc.stdin is not None and proc.stdout is not None and proc.stderr is not None
    proc.stdin.write((json.dumps(request) + "\n").encode())
    await proc.stdin.drain()
    first_line = await asyncio.wait_for(proc.stdout.readline(), timeout=10)
    message = json.loads(first_line)
    assert message["jsonrpc"] == "2.0"
    assert message["id"] == 1
    proc.terminate()
    await asyncio.wait_for(proc.wait(), timeout=10)
    stdout = first_line + await proc.stdout.read()
    stderr = (await proc.stderr.read()).decode()
    assert b"Loading config" not in stdout
    assert b"INFO" not in stdout
    assert b"WARNING" not in stdout
    assert b"2026-" not in stdout
    assert "Failed to parse JSONRPC message" not in stderr
    assert "Invalid JSON" not in stderr


def test_sdk_responds_to_interactive_approval_server_requests():
    from openai_codex.client import CodexClient

    client = CodexClient()
    assert client._default_approval_handler(
        "item/commandExecution/requestApproval",
        {},
    ) == {  # pylint: disable=protected-access
        "decision": "accept",
    }
    assert client._default_approval_handler(
        "item/fileChange/requestApproval",
        {},
    ) == {  # pylint: disable=protected-access
        "decision": "accept",
    }


def test_event_to_chunks_maps_content_usage_and_completion():
    content_event = SimpleNamespace(
        method="item/agentMessage/delta",
        payload=SimpleNamespace(item_id="item-1", delta="hello"),
    )
    usage = TokenUsageBreakdown(
        cachedInputTokens=1,
        inputTokens=3,
        outputTokens=5,
        reasoningOutputTokens=2,
        totalTokens=8,
    )
    usage_event = SimpleNamespace(
        method="thread/tokenUsage/updated",
        payload=SimpleNamespace(token_usage=SimpleNamespace(last=usage)),
    )
    completed_event = SimpleNamespace(
        method="turn/completed",
        payload=SimpleNamespace(
            turn=SimpleNamespace(id="turn-1", status=SimpleNamespace(value="completed"), duration_ms=10, error=None),
        ),
    )

    content = CodexAgentWrapper._event_to_chunks(content_event, "thread-1")  # pylint: disable=protected-access
    usage_chunks = CodexAgentWrapper._event_to_chunks(usage_event, "thread-1")  # pylint: disable=protected-access
    completed = CodexAgentWrapper._event_to_chunks(completed_event, "thread-1")  # pylint: disable=protected-access

    assert content[0].chunk_type == ChunkEnum.CONTENT
    assert content[0].chunk == "hello"
    assert usage_chunks[0].chunk_type == ChunkEnum.USAGE
    assert usage_chunks[0].input_tokens == 3
    assert usage_chunks[0].output_tokens == 5
    assert completed[0].chunk_type == ChunkEnum.REPLY_END
    assert completed[0].metadata["status"] == "completed"


def test_event_to_chunks_preserves_new_turn_scoped_notifications():
    from openai_codex.types import Notification
    from openai_codex.generated.v2_all import TurnDiffUpdatedNotification

    event = Notification(
        method="turn/diff/updated",
        payload=TurnDiffUpdatedNotification(
            threadId="thread-1",
            turnId="turn-1",
            diff="diff --git a/a b/a",
        ),
    )

    chunk = CodexAgentWrapper._event_to_chunks(event, "thread-1")[0]  # pylint: disable=protected-access

    assert chunk.chunk_type == ChunkEnum.DATA
    assert chunk.chunk == {
        "threadId": "thread-1",
        "turnId": "turn-1",
        "diff": "diff --git a/a b/a",
    }
    assert chunk.metadata == {"codex_method": "turn/diff/updated"}


@dataclass
class _TurnResult:
    final_response: str
    status: str = "completed"


def test_reply_returns_thread_id_and_structured_output(tmp_path, monkeypatch):
    wrapper, _job = _wrapper(tmp_path, auth_mode="oauth")

    class FakeThread:
        id = "thread-1"

        async def run(self, inputs, **kwargs):
            assert inputs == "answer"
            assert kwargs["output_schema"] == {"type": "object"}
            return _TurnResult(final_response=json.dumps({"ok": True}))

    class FakeCodex:
        def __init__(self, _config):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, _exc_type, _exc, _tb):
            return None

        async def close(self):
            return None

        async def account(self):
            return SimpleNamespace(account=SimpleNamespace())

        async def thread_start(self, **_kwargs):
            return FakeThread()

    monkeypatch.setattr("reme.components.agent_wrapper.codex_agent_wrapper.AsyncCodex", FakeCodex)

    result = asyncio.run(wrapper.reply("answer", output_schema={"type": "object"}))

    assert result["session_id"] == "thread-1"
    assert result["structured_output"] == {"ok": True}


@pytest.mark.asyncio
async def test_reply_accepts_latest_sdk_run_input(tmp_path, monkeypatch):
    from openai_codex import LocalImageInput, TextInput

    wrapper, _job = _wrapper(tmp_path)
    inputs = [TextInput("describe this image"), LocalImageInput("image.png")]
    observed = {}

    class FakeThread:
        id = "thread-1"

        async def run(self, run_input, **_kwargs):
            observed["input"] = run_input
            return _TurnResult(final_response="done")

    async def get_codex(_kwargs):
        return SimpleNamespace()

    async def open_thread(_codex, _kwargs):
        return FakeThread()

    monkeypatch.setattr(wrapper, "_get_codex", get_codex)
    monkeypatch.setattr(wrapper, "_open_thread", open_thread)

    result = await wrapper.reply(inputs)

    assert observed["input"] is inputs
    assert result["last_message"] == "done"


def test_codex_skills_add_all_without_deleting_existing_content(tmp_path):
    wrapper, _job = _wrapper(tmp_path)
    for name in ("reme_memory", "qwenpaw_memory"):
        source = tmp_path / "skills" / name
        source.mkdir(parents=True)
        (source / "SKILL.md").write_text(f"# {name}", encoding="utf-8")
    existing = tmp_path / ".agents" / "skills" / "user_skill"
    existing.mkdir(parents=True)
    marker = existing / "marker"
    marker.write_text("keep", encoding="utf-8")

    wrapper._ensure_skills("all")  # pylint: disable=protected-access

    assert marker.read_text(encoding="utf-8") == "keep"
    for name in ("reme_memory", "qwenpaw_memory"):
        target = tmp_path / ".agents" / "skills" / name
        assert target.is_symlink()
        assert target.resolve() == (tmp_path / "skills" / name).resolve()


def test_codex_skills_support_single_name_and_are_idempotent(tmp_path):
    wrapper, _job = _wrapper(tmp_path)
    source = tmp_path / "skills" / "one"
    source.mkdir(parents=True)
    (source / "SKILL.md").write_text("# one", encoding="utf-8")

    wrapper._ensure_skills("one")  # pylint: disable=protected-access
    wrapper._ensure_skills(["one"])  # pylint: disable=protected-access

    assert (tmp_path / ".agents" / "skills" / "one").resolve() == source.resolve()


@pytest.mark.parametrize("kind", ["directory", "external_link"])
def test_codex_skills_preserve_conflicts(tmp_path, kind):
    wrapper, _job = _wrapper(tmp_path)
    source = tmp_path / "skills" / "one"
    source.mkdir(parents=True)
    (source / "SKILL.md").write_text("# one", encoding="utf-8")
    target = tmp_path / ".agents" / "skills" / "one"
    target.parent.mkdir(parents=True)
    if kind == "directory":
        target.mkdir()
        (target / "marker").write_text("keep", encoding="utf-8")
    else:
        external = tmp_path / "external"
        external.mkdir()
        target.symlink_to(external, target_is_directory=True)

    with pytest.raises(FileExistsError, match="Codex skill conflict"):
        wrapper._ensure_skills("one")  # pylint: disable=protected-access
    assert target.exists()


@pytest.mark.parametrize("create_dir", [False, True])
def test_codex_skills_reject_missing_or_invalid_skill(tmp_path, create_dir):
    wrapper, _job = _wrapper(tmp_path)
    if create_dir:
        (tmp_path / "skills" / "missing_manifest").mkdir(parents=True)
        name = "missing_manifest"
    else:
        name = "missing"
    with pytest.raises(FileNotFoundError):
        wrapper._ensure_skills(name)  # pylint: disable=protected-access


def test_codex_skills_do_not_modify_codex_home(tmp_path):
    codex_home = tmp_path / "codex-home"
    marker = codex_home / "skills" / "marker"
    marker.parent.mkdir(parents=True)
    marker.write_text("keep", encoding="utf-8")
    wrapper, _job = _wrapper(tmp_path, codex_home=codex_home)
    source = tmp_path / "skills" / "one"
    source.mkdir(parents=True)
    (source / "SKILL.md").write_text("# one", encoding="utf-8")

    wrapper._ensure_skills("one")  # pylint: disable=protected-access

    assert marker.read_text(encoding="utf-8") == "keep"


def test_effective_mcp_config_snapshot_is_private_and_removed_on_close(tmp_path):
    wrapper, _job = _wrapper(tmp_path)
    config = wrapper._mcp_server_config({"job_tools": ["search"]})  # pylint: disable=protected-access
    snapshot = Path(config["args"][config["args"].index("--config") + 1])
    assert snapshot.exists()
    assert stat.S_IMODE(snapshot.stat().st_mode) == 0o600
    assert json.loads(snapshot.read_text(encoding="utf-8"))["workspace_dir"] == str(tmp_path)

    async def close_started_wrapper():
        await wrapper.start()
        await wrapper.close()

    asyncio.run(close_started_wrapper())
    assert not snapshot.exists()


@pytest.mark.asyncio
async def test_effective_snapshot_exposes_parent_only_custom_job(tmp_path):
    from fastmcp import Client
    from fastmcp.client import StdioTransport

    app_config = ApplicationConfig(
        workspace_dir=str(tmp_path),
        enable_logo=False,
        log_to_console=False,
        log_to_file=False,
        service={"backend": "mcp"},
        jobs={
            "only_custom": {
                "backend": "base",
                "description": "Parent-only inline job",
                "parameters": {"type": "object", "properties": {}},
                "steps": [],
            },
            "referenced_helper": {
                "backend": "base",
                "description": "A normal job that custom jobs may reference.",
                "parameters": {"type": "object", "properties": {}},
                "steps": [],
            },
        },
    )
    job = _Job("only_custom")
    context = SimpleNamespace(app_config=app_config, jobs={"only_custom": job})
    wrapper = CodexAgentWrapper(app_context=context)
    server_config = wrapper._mcp_server_config({"job_tools": ["only_custom"]})  # pylint: disable=protected-access
    transport = StdioTransport(
        command=server_config["command"],
        args=server_config["args"],
        cwd=server_config["cwd"],
    )

    await wrapper.start()
    async with Client(transport, timeout=10) as client:
        tools = await client.list_tools()
        snapshot = Path(server_config["args"][server_config["args"].index("--config") + 1])
        assert snapshot.exists()
        assert set(json.loads(snapshot.read_text(encoding="utf-8"))["jobs"]) == {
            "only_custom",
            "referenced_helper",
        }
    await wrapper.close()

    assert [tool.name for tool in tools] == ["only_custom"]
    assert not snapshot.exists()


@pytest.mark.asyncio
async def test_open_thread_defaults_to_full_access(tmp_path):
    from openai_codex import ApprovalMode, Sandbox

    wrapper, _job = _wrapper(tmp_path)
    observed = {}

    class FakeCodex:
        async def thread_start(self, **kwargs):
            observed.update(kwargs)
            return SimpleNamespace(id="thread-1")

    await wrapper._open_thread(FakeCodex(), {})  # pylint: disable=protected-access

    assert observed["approval_mode"] == ApprovalMode.auto_review
    assert observed["sandbox"] == Sandbox.full_access


@pytest.mark.asyncio
async def test_open_thread_forwards_latest_sdk_options(tmp_path):
    from openai_codex.types import Personality, ThreadSource, ThreadStartSource

    wrapper, _job = _wrapper(tmp_path)
    observed = {}

    class FakeCodex:
        async def thread_start(self, **kwargs):
            observed["start"] = kwargs
            return SimpleNamespace(id="thread-1")

        async def thread_resume(self, _thread_id, **kwargs):
            observed["resume"] = kwargs
            return SimpleNamespace(id="thread-1")

        async def thread_fork(self, _thread_id, **kwargs):
            observed["fork"] = kwargs
            return SimpleNamespace(id="thread-2")

    codex = FakeCodex()
    await wrapper._open_thread(  # pylint: disable=protected-access
        codex,
        {
            "personality": "friendly",
            "service_name": "reme",
            "session_start_source": "startup",
            "thread_source": "user",
        },
    )
    await wrapper._open_thread(  # pylint: disable=protected-access
        codex,
        {"session_id": "thread-1", "personality": "pragmatic"},
    )
    await wrapper._open_thread(  # pylint: disable=protected-access
        codex,
        {"session_id": "thread-1", "fork_session": True, "thread_source": "subagent"},
    )

    assert observed["start"]["personality"] == Personality.friendly
    assert observed["start"]["service_name"] == "reme"
    assert observed["start"]["session_start_source"] == ThreadStartSource.startup
    assert observed["start"]["thread_source"] == ThreadSource.user
    assert observed["resume"]["personality"] == Personality.pragmatic
    assert observed["fork"]["thread_source"] == ThreadSource.subagent


@pytest.mark.asyncio
async def test_resume_reuses_tool_context_and_rejects_context_change(tmp_path):
    wrapper, _job = _wrapper(tmp_path)

    class FakeCodex:
        async def thread_start(self, **_kwargs):
            return SimpleNamespace(id="thread-1")

        async def thread_resume(self, _thread_id, **_kwargs):
            return SimpleNamespace(id="thread-1")

    await wrapper._open_thread(FakeCodex(), {"tool_context_id": "ctx-a"})  # pylint: disable=protected-access
    await wrapper._open_thread(  # pylint: disable=protected-access
        FakeCodex(),
        {"resume": "thread-1", "tool_context_id": "ctx-a"},
    )
    with pytest.raises(ValueError, match="cannot change"):
        await wrapper._open_thread(  # pylint: disable=protected-access
            FakeCodex(),
            {"resume": "thread-1", "tool_context_id": "ctx-b"},
        )


class _StructuredModel(BaseModel):
    ok: bool


@pytest.mark.parametrize("schema", [_StructuredModel(ok=True), str])
def test_output_schema_rejects_instances_and_arbitrary_classes(tmp_path, schema):
    wrapper, _job = _wrapper(tmp_path)
    with pytest.raises(TypeError, match="JSON schema dict or BaseModel class"):
        wrapper._merged_kwargs({"output_schema": schema})  # pylint: disable=protected-access


def test_output_schema_normalizes_model_class_and_preserves_dict(tmp_path):
    wrapper, _job = _wrapper(tmp_path)
    schema = {"type": "object", "properties": {"ok": {"type": "boolean"}}}

    assert (
        wrapper._merged_kwargs({"output_schema": _StructuredModel})["output_schema"]  # pylint: disable=protected-access
        == _StructuredModel.model_json_schema()
    )
    assert (
        wrapper._merged_kwargs({"output_schema": schema})["output_schema"] is schema
    )  # pylint: disable=protected-access


@pytest.mark.asyncio
async def test_reply_normalizes_schema_and_reuses_persistent_client(tmp_path, monkeypatch):
    wrapper, _job = _wrapper(tmp_path, auth_mode="oauth")
    clients = []
    observed_schemas = []
    close_count = 0

    class FakeThread:
        id = "thread-1"

        async def run(self, _inputs, **kwargs):
            observed_schemas.append(kwargs["output_schema"])
            return _TurnResult(final_response=json.dumps({"ok": True}))

    class FakeCodex:
        def __init__(self, _config):
            clients.append(self)

        async def __aenter__(self):
            return self

        async def close(self):
            nonlocal close_count
            close_count += 1

        async def account(self):
            return SimpleNamespace(account=SimpleNamespace())

        async def thread_start(self, **_kwargs):
            return FakeThread()

    monkeypatch.setattr("reme.components.agent_wrapper.codex_agent_wrapper.AsyncCodex", FakeCodex)

    await wrapper.start()
    result = await wrapper.reply("first", output_schema=_StructuredModel)
    await wrapper.close()

    assert result["structured_output"] == {"ok": True}
    assert observed_schemas == [_StructuredModel.model_json_schema()]
    assert len(clients) == 1
    assert close_count == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("schema", [{}, _StructuredModel])
async def test_reply_stream_rejects_output_schema(tmp_path, schema):
    wrapper, _job = _wrapper(tmp_path)

    with pytest.raises(NotImplementedError, match="Structured output is not supported"):
        await anext(wrapper.reply_stream("answer", output_schema=schema))


@pytest.mark.asyncio
async def test_reply_stream_interrupts_turn_when_consumer_closes_early(tmp_path, monkeypatch):
    wrapper, _job = _wrapper(tmp_path)
    stream_closed = False
    interrupt_count = 0

    class FakeTurn:
        id = "turn-1"

        async def stream(self):
            nonlocal stream_closed
            try:
                yield SimpleNamespace(
                    method="turn/started",
                    payload=SimpleNamespace(turn=SimpleNamespace(id=self.id)),
                )
                await asyncio.Event().wait()
            finally:
                stream_closed = True

        async def interrupt(self):
            nonlocal interrupt_count
            interrupt_count += 1

    class FakeThread:
        id = "thread-1"

        async def turn(self, _inputs, **_kwargs):
            return FakeTurn()

    async def get_codex(_kwargs):
        return SimpleNamespace()

    async def open_thread(_codex, _kwargs):
        return FakeThread()

    monkeypatch.setattr(wrapper, "_get_codex", get_codex)
    monkeypatch.setattr(wrapper, "_open_thread", open_thread)

    stream = wrapper.reply_stream("answer")
    first = await anext(stream)
    assert first.chunk_type == ChunkEnum.REPLY_START
    await stream.aclose()

    assert interrupt_count == 1
    assert stream_closed


@pytest.mark.asyncio
async def test_persistent_client_rejects_launch_config_changes(tmp_path, monkeypatch):
    wrapper, _job = _wrapper(tmp_path)

    class FakeCodex:
        def __init__(self, _config):
            pass

        async def login_api_key(self, _api_key):
            return None

        async def close(self):
            return None

    monkeypatch.setattr("reme.components.agent_wrapper.codex_agent_wrapper.AsyncCodex", FakeCodex)

    await wrapper.start()
    first = await wrapper._get_codex({"api_key": "one"})  # pylint: disable=protected-access
    assert await wrapper._get_codex({"api_key": "one"}) is first  # pylint: disable=protected-access
    with pytest.raises(RuntimeError, match="configuration changed"):
        await wrapper._get_codex({"api_key": "two"})  # pylint: disable=protected-access
    await wrapper.close()


def test_oauth_mode_ignores_api_credentials_and_forces_chatgpt(tmp_path, monkeypatch):
    wrapper, _job = _wrapper(tmp_path)
    wrapper.app_context.app_config.environment = {"TOOL_ENV": "preserved"}
    monkeypatch.setenv("CODEX_API_KEY", "ambient-key")
    monkeypatch.setenv("CODEX_BASE_URL", "https://ambient.example.test/v1")

    auth = wrapper._resolve_auth_config(  # pylint: disable=protected-access
        {
            "auth_mode": "oauth",
            "api_key": "explicit-key",
            "base_url": "https://explicit.example.test/v1",
        },
    )
    config = wrapper._build_client_config(  # pylint: disable=protected-access
        {"launch_args_override": ["codex", "app-server", "--listen", "stdio://"]},
        auth,
    )

    assert auth.mode == "oauth"
    assert auth.api_key == ""
    assert auth.base_url == ""
    assert "OPENAI_API_KEY" not in config.env
    assert config.env["TOOL_ENV"] == "preserved"
    assert config.launch_args_override == ("codex", "app-server", "--listen", "stdio://")
    assert "CODEX_HOME" not in wrapper.app_context.app_config.environment
    assert 'forced_login_method="chatgpt"' in config.config_overrides
    assert not any(value.startswith("openai_base_url=") for value in config.config_overrides)


@pytest.mark.asyncio
async def test_api_key_mode_logs_in_app_server_explicitly(tmp_path, monkeypatch):
    wrapper, _job = _wrapper(tmp_path)
    observed = {}

    class FakeCodex:
        def __init__(self, config):
            observed["config"] = config

        async def login_api_key(self, api_key):
            observed["api_key"] = api_key

        async def close(self):
            observed["closed"] = True

    monkeypatch.setattr("reme.components.agent_wrapper.codex_agent_wrapper.AsyncCodex", FakeCodex)

    await wrapper.start()
    await wrapper._get_codex(  # pylint: disable=protected-access
        {
            "auth_mode": "api_key",
            "api_key": "explicit-key",
            "base_url": "https://proxy.example.test/v1",
        },
    )
    await wrapper.close()

    config = observed["config"]
    assert observed["api_key"] == "explicit-key"
    assert "OPENAI_API_KEY" not in config.env
    assert 'openai_base_url="https://proxy.example.test/v1"' in config.config_overrides
    assert 'forced_login_method="api"' in config.config_overrides
    assert observed["closed"] is True


def test_api_key_mode_uses_only_wrapper_config(tmp_path, monkeypatch):
    wrapper, _job = _wrapper(tmp_path)
    wrapper.app_context.app_config.components[ComponentEnum.AS_LLM]["default"] = SimpleNamespace(
        credential={"api_key": "default-key", "base_url": "https://default.example.test/v1"},
    )
    for name in ("CODEX_API_KEY", "OPENAI_API_KEY", "LLM_API_KEY"):
        monkeypatch.setenv(name, "ambient-key")
    for name in ("CODEX_BASE_URL", "OPENAI_BASE_URL", "LLM_BASE_URL"):
        monkeypatch.setenv(name, "https://ambient.example.test/v1")

    auth = wrapper._resolve_auth_config(  # pylint: disable=protected-access
        {
            "auth_mode": "api_key",
            "api_key": "configured-key",
            "credential": {"api_key": "nested-key", "base_url": "https://nested.example.test/v1"},
        },
    )

    assert auth.mode == "api_key"
    assert auth.api_key == "configured-key"
    assert auth.base_url == ""

    with pytest.raises(ValueError, match="requires a non-empty API key"):
        wrapper._resolve_auth_config(  # pylint: disable=protected-access
            {
                "auth_mode": "api_key",
                "credential": {"api_key": "nested-key", "base_url": "https://nested.example.test/v1"},
            },
        )


@pytest.mark.parametrize("review_status", ["approved", "denied"])
def test_event_to_chunks_maps_approval_started_and_completed(review_status):
    action = {"type": "futureApprovalAction", "value": "preserved"}
    review = {"status": review_status, "rationale": "policy"}
    started = SimpleNamespace(
        method="item/autoApprovalReview/started",
        payload=SimpleNamespace(
            action=action,
            review=review,
            review_id="review-1",
            target_item_id="item-1",
            turn_id="turn-1",
        ),
    )
    completed = SimpleNamespace(
        method="item/autoApprovalReview/completed",
        payload=SimpleNamespace(
            action=action,
            review=review,
            review_id="review-1",
            target_item_id="item-1",
            turn_id="turn-1",
            decision_source={"type": "guardian"},
        ),
    )

    started_chunk = CodexAgentWrapper._event_to_chunks(started, "thread-1")[0]  # pylint: disable=protected-access
    completed_chunk = CodexAgentWrapper._event_to_chunks(completed, "thread-1")[0]  # pylint: disable=protected-access

    assert started_chunk.chunk_type == ChunkEnum.APPROVAL
    assert started_chunk.chunk == action
    assert started_chunk.metadata["status"] == "started"
    assert completed_chunk.metadata["status"] == "completed"
    assert completed_chunk.metadata["review"]["status"] == review_status
    assert completed_chunk.metadata["decision_source"] == {"type": "guardian"}


def test_codex_home_expands_user_directory(tmp_path):
    wrapper, _job = _wrapper(tmp_path, codex_home="~/.codex")
    assert wrapper.session_path == Path.home() / ".codex"


def test_named_default_mcp_config_remains_supported(tmp_path):
    wrapper, _job = _wrapper(tmp_path, mcp_config="default")
    assert wrapper._mcp_config_source({}) == "default"  # pylint: disable=protected-access


def test_default_config_provides_codex_oauth_wrapper(monkeypatch):
    monkeypatch.delenv("CODEX_HOME", raising=False)
    config = resolve_app_config(log_config=False)
    oauth = config["components"]["agent_wrapper"]["codex_oauth"]
    codex = config["components"]["agent_wrapper"]["codex"]
    assert oauth["backend"] == "codex"
    assert oauth["auth_mode"] == "oauth"
    assert oauth["codex_home"] == "~/.codex"
    assert oauth["sandbox"] == "full-access"
    assert "api_key" not in oauth
    assert codex["auth_mode"] == "api_key"
    assert codex["sandbox"] == "full-access"
