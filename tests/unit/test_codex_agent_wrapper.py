"""Unit tests for the Codex agent wrapper and its FastMCP bridge."""

# pylint: disable=missing-class-docstring,missing-function-docstring,protected-access

import asyncio
from dataclasses import dataclass
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
from openai_codex.generated.v2_all import TokenUsageBreakdown

from reme.components.agent_wrapper.codex_agent_wrapper import CodexAgentWrapper
from reme.components.agent_wrapper.codex_mcp_server import _load_job_names, _make_tool, build_server
from reme.components.job import BackgroundJob
from reme.config import resolve_app_config
from reme.enumeration import ChunkEnum, ComponentEnum
from reme.schema import Response


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
        components={ComponentEnum.AS_LLM: {}},
    )
    context = SimpleNamespace(app_config=config, jobs={job.name: job})
    return CodexAgentWrapper(app_context=context, **kwargs), job


def test_mcp_config_uses_stdio_bridge_and_selected_jobs(tmp_path):
    wrapper, _job = _wrapper(tmp_path, mcp_config="custom.yaml")

    config = wrapper._mcp_server_config(  # pylint: disable=protected-access
        {"job_tools": ["search"], "tool_context_id": "ctx-1"},
    )

    assert config["command"]
    assert config["enabled_tools"] == ["search"]
    assert "reme.components.agent_wrapper.codex_mcp_server" in config["args"]
    assert config["args"][config["args"].index("--config") + 1] == "custom.yaml"
    assert config["args"][config["args"].index("--tool-context-id") + 1] == "ctx-1"


def test_thread_config_preserves_other_mcp_servers(tmp_path):
    wrapper, _job = _wrapper(tmp_path)
    config = wrapper._thread_config(  # pylint: disable=protected-access
        {
            "job_tools": ["search"],
            "config": {"mcp_servers": {"docs": {"url": "https://example.test/mcp"}}},
        },
    )

    assert set(config["mcp_servers"]) == {"docs", "reme_jobs"}


def test_mcp_config_rejects_background_jobs(tmp_path):
    wrapper, _job = _wrapper(tmp_path)
    wrapper.app_context.jobs["watch"] = BackgroundJob(name="watch", app_context=wrapper.app_context)

    with pytest.raises(TypeError, match="non-stream request jobs"):
        wrapper._mcp_server_config({"job_tools": ["watch"]})


def test_bridge_tool_injects_tool_context_id():
    async def run():
        job = _Job()
        tool = _make_tool(job, "ctx-1")
        result = await tool.run({"query": "alpha"})
        assert job.calls == [{"query": "alpha", "tool_context_id": "ctx-1"}]
        assert "found:alpha" in str(result.content)

    asyncio.run(run())


def test_bridge_rejects_caller_tool_context_id():
    async def run():
        job = _Job()
        tool = _make_tool(job, "ctx-1")
        with pytest.raises(Exception, match="managed by the Codex agent wrapper"):
            await tool.run({"query": "alpha", "tool_context_id": "caller"})

    asyncio.run(run())


def test_build_server_registers_only_selected_jobs():
    app = SimpleNamespace(
        context=SimpleNamespace(jobs={"one": _Job("one"), "two": _Job("two")}),
        start=lambda: None,
        close=lambda: None,
    )

    async def run():
        server = build_server(app, ["two"])
        tools = await server.list_tools(run_middleware=False)
        assert [tool.name for tool in tools] == ["two"]

    asyncio.run(run())


def test_load_job_names_validates_json_array():
    assert _load_job_names('["one", "two"]') == ["one", "two"]
    with pytest.raises(ValueError, match="JSON array"):
        _load_job_names('{"one": true}')


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
            "--jobs",
            '["empty"]',
        ],
        cwd=str(Path(__file__).resolve().parents[2]),
    )

    async with Client(transport, timeout=10) as client:
        tools = await client.list_tools()

    assert [tool.name for tool in tools] == ["empty"]


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


@dataclass
class _TurnResult:
    final_response: str
    status: str = "completed"


def test_reply_returns_thread_id_and_structured_output(tmp_path, monkeypatch):
    wrapper, _job = _wrapper(tmp_path)

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

        async def thread_start(self, **_kwargs):
            return FakeThread()

    monkeypatch.setattr("openai_codex.AsyncCodex", FakeCodex)
    monkeypatch.setattr("reme.components.agent_wrapper.codex_agent_wrapper.load_env", lambda *_args: {})

    result = asyncio.run(wrapper.reply("answer", output_schema={"type": "object"}))

    assert result["session_id"] == "thread-1"
    assert result["structured_output"] == {"ok": True}


def test_individual_codex_skills_fail_explicitly(tmp_path):
    wrapper, _job = _wrapper(tmp_path)
    (tmp_path / "skills" / "one").mkdir(parents=True)
    with pytest.raises(NotImplementedError, match="skills='all'"):
        wrapper._ensure_skills(["one"])  # pylint: disable=protected-access


def test_codex_home_expands_user_directory(tmp_path):
    wrapper, _job = _wrapper(tmp_path, codex_home="~/.codex")
    assert wrapper.session_path == Path.home() / ".codex"


def test_default_config_provides_codex_oauth_wrapper(monkeypatch):
    monkeypatch.delenv("CODEX_HOME", raising=False)
    config = resolve_app_config(log_config=False)
    oauth = config["components"]["agent_wrapper"]["codex_oauth"]
    assert oauth["backend"] == "codex"
    assert oauth["codex_home"] == "~/.codex"
    assert "api_key" not in oauth
