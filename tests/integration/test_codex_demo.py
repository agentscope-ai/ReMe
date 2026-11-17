"""Opt-in live coverage of both demo Steps using Codex API-key authentication."""

# pylint: disable=redefined-outer-name

import asyncio
from collections import Counter
import json
import os
from pathlib import Path

import pytest
import pytest_asyncio

from reme.application import Application
from reme.config import resolve_app_config
from reme.enumeration import ChunkEnum
from reme.steps.common.stream_llm_demo import StreamLLMDemoStep
from reme.utils import global_counter_get, parse_env_file

pytestmark = pytest.mark.skipif(
    os.getenv("REME_CODEX_INTEGRATION") != "1",
    reason="set REME_CODEX_INTEGRATION=1 to run live Codex tests",
)


@pytest_asyncio.fixture
async def demo_app(tmp_path, monkeypatch, request):
    """Load the requested LLM credentials while keeping all runtime state temporary."""
    env_path = Path(__file__).resolve().parents[2] / ".env"
    environment = {**(parse_env_file(env_path) if env_path.is_file() else {}), **os.environ}
    for name in ("LLM_API_KEY", "LLM_BASE_URL"):
        assert environment.get(name), f"{name} is required for the live Codex demo tests"
        monkeypatch.setenv(name, environment[name])
    config = resolve_app_config(
        config="demo",
        log_config=False,
        workspace_dir=str(tmp_path),
        enable_logo=False,
        log_to_console=False,
        log_to_file=False,
        service={"backend": "cli"},
    )
    approval_mode = getattr(request, "param", "deny_all")
    config["components"]["agent_wrapper"]["codex"]["approval_mode"] = approval_mode
    server_config = config["components"]["codex_app_server"]["default"]
    assert server_config["auth_mode"] == "api_key"
    assert server_config["api_key"] == environment["LLM_API_KEY"]
    assert server_config["base_url"] == environment["LLM_BASE_URL"]
    app = Application(**config)
    server = app.context.components["codex_app_server"]["default"]
    async with app:
        yield app
    models = set()
    reasoning = set()
    for path in (server.session_path / "sessions").rglob("*.jsonl"):
        for line in path.read_text(encoding="utf-8").splitlines():
            record = json.loads(line)
            payload = record.get("payload", {})
            if record.get("type") == "event_msg" and payload.get("type") == "thread_settings_applied":
                settings = payload["thread_settings"]
                models.add(settings["model"])
                reasoning.add((settings["model"], settings.get("reasoning_effort")))
    print(f"\nCodex models observed: {sorted(models)}")
    print(f"Codex reasoning observed: {sorted(reasoning)}")
    assert ("qwen3.8-max", "xhigh") in reasoning
    assert {effort for _, effort in reasoning} == {"xhigh"}
    assert models == (
        {"qwen3.8-max", "qwen3.8-flash"}
        if approval_mode == "auto_review" and global_counter_get(app.context.metadata, ["__job_counter", "add"])
        else {"qwen3.8-max"}
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("streaming", [False, True], ids=["non-stream", "stream"])
@pytest.mark.parametrize("use_tool", [False, True], ids=["chat", "add-tool"])
async def test_live_codex_demo_chat_and_add_tool(demo_app, streaming, use_tool):
    """Verify text delivery and actual ReMe add execution through each demo Job."""
    query = (
        "Use the add tool to calculate 21 + 21 and return only its result."
        if use_tool
        else "Write the integers from 1 through 20 in order, separated by spaces."
    )
    sys_prompt = "Use the add tool for addition. Otherwise follow the requested output format."
    previous_calls = global_counter_get(demo_app.context.metadata, ["__job_counter", "add"])
    async with asyncio.timeout(90):
        if streaming:
            chunks = [
                chunk
                async for chunk in demo_app.run_stream_job(
                    "stream_llm_demo",
                    query=query,
                    sys_prompt=sys_prompt,
                )
            ]
            errors = [
                chunk.chunk
                for chunk in chunks
                if chunk.chunk_type == ChunkEnum.ERROR and chunk.metadata.get("will_retry") is not True
            ]
            assert not errors, errors
            assert chunks[0].chunk_type == ChunkEnum.REPLY_START
            assert chunks[-1].chunk_type == ChunkEnum.DONE
            assert chunks[-1].done
            assert not any(chunk.chunk_type == ChunkEnum.APPROVAL for chunk in chunks)
            assert any(chunk.chunk_type == ChunkEnum.REPLY_END for chunk in chunks)
            assert chunks[-2].metadata.get("status") == "completed"
            content = [chunk.chunk for chunk in chunks if chunk.chunk_type == ChunkEnum.CONTENT and chunk.chunk]
            text = "".join(content).strip()
            counts = Counter(chunk.chunk_type.value for chunk in chunks)
            print(f"\nstream tool={use_tool}: chunks={dict(counts)}, answer={text!r}")
            if use_tool:
                assert counts[ChunkEnum.TOOL_CALL.value] > 0
                assert counts[ChunkEnum.TOOL_RESULT.value] > 0
        else:
            response = await demo_app.run_job("llm_demo", query=query, sys_prompt=sys_prompt)
            assert response.success, response.answer
            text = response.answer.strip()
            assert response.metadata["response"] == text
            print(f"\nnon-stream tool={use_tool}: answer={text!r}")
    assert text
    if use_tool:
        assert "42" in text
        assert global_counter_get(demo_app.context.metadata, ["__job_counter", "add"]) > previous_calls
    else:
        assert text.split() == [str(number) for number in range(1, 21)]


@pytest.mark.asyncio
async def test_live_codex_stream_demo_without_stream_queue(demo_app):
    """Without a queue, StreamLLMDemoStep uses the non-streaming wrapper reply."""
    step = StreamLLMDemoStep(app_context=demo_app.context, agent_wrapper="codex")
    async with asyncio.timeout(90):
        response = await step(query="Reply exactly FALLBACK_OK.")
    assert response.success, response.answer
    assert response.answer.strip() == "FALLBACK_OK"


@pytest.mark.asyncio
async def test_live_codex_native_shell_tool_needs_no_approval(demo_app):
    """The default policy also executes a native shell tool without an approval request."""
    wrapper = demo_app.context.components["agent_wrapper"]["codex"]
    async with asyncio.timeout(90):
        chunks = [
            chunk
            async for chunk in wrapper.reply_stream(
                "Run the shell command printf NATIVE_TOOL_OK, then return only the command output.",
            )
        ]
    assert any(
        chunk.chunk_type == ChunkEnum.TOOL_CALL and chunk.tool_call_name == "commandExecution" for chunk in chunks
    )
    assert any(
        chunk.chunk_type == ChunkEnum.TOOL_RESULT and chunk.tool_call_name == "commandExecution" for chunk in chunks
    )
    assert not any(chunk.chunk_type == ChunkEnum.APPROVAL for chunk in chunks)
    assert "NATIVE_TOOL_OK" in "".join(
        chunk.chunk for chunk in chunks if chunk.chunk_type == ChunkEnum.CONTENT and isinstance(chunk.chunk, str)
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("demo_app", ["auto_review"], indirect=True)
async def test_live_codex_automatic_review_remains_available(demo_app):
    """Explicitly selecting auto_review still invokes the configured Flash reviewer."""
    async with asyncio.timeout(90):
        chunks = [
            chunk
            async for chunk in demo_app.run_stream_job(
                "stream_llm_demo",
                query="Call the add tool to calculate 21 + 21, then return only its result.",
            )
        ]
    assert any(chunk.chunk_type == ChunkEnum.APPROVAL for chunk in chunks)
    assert global_counter_get(demo_app.context.metadata, ["__job_counter", "add"]) > 0
    assert "42" in "".join(chunk.chunk for chunk in chunks if chunk.chunk_type == ChunkEnum.CONTENT)
