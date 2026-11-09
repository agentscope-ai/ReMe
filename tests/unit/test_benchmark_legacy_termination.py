"""Offline regression coverage for the benchmark-local legacy termination policy."""

# pylint: disable=protected-access,redefined-outer-name

from copy import deepcopy
import base64
from importlib import import_module
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

from agentscope.event import ExceedMaxItersEvent, ReplyEndEvent, ReplyStartEvent
from agentscope.message import AssistantMsg, Base64Source, DataBlock, Usage, UserMsg
from PIL import Image
import pytest

from reme.components.agent_wrapper.as_agent_wrapper import AsAgentWrapper
from reme.components.application_context import ApplicationContext
from reme.components.file_store import LocalFileStore
from reme.config import resolve_app_config
from reme.enumeration import ChunkEnum
from reme.plugin import resolve_plugin_runtime
from reme.schema import collect_reply, replay_reply
from reme.steps.evolve._evolve import agent_reply_result_text


@pytest.fixture(params=["memlens", "mobilemem"])
def benchmark(request, monkeypatch):
    root = Path(__file__).resolve().parents[2]
    monkeypatch.syspath_prepend(str(root / f"plugins/{request.param}/src"))
    module = import_module(f"reme_{request.param}.benchmark_agent")
    assert Path(module.__file__).resolve().is_relative_to(root)
    return request.param, module.BenchmarkAsAgentWrapper


def wrapper_for(wrapper_type, tmp_path, monkeypatch, message):
    wrapper = wrapper_type(app_context=ApplicationContext(workspace_dir=str(tmp_path)), as_llm="")

    async def native_reply(_inputs, **_kwargs):
        yield ReplyStartEvent(name="agent", session_id="session-1", reply_id="reply-1")
        if message.finished_reason == "exceed_max_iters":
            yield ExceedMaxItersEvent(reply_id="reply-1", name="agent")
        if message.finished_reason is not None:
            yield ReplyEndEvent(session_id="session-1", reply_id="reply-1", finished_reason=message.finished_reason)
        yield message

    agent = SimpleNamespace(
        state=SimpleNamespace(session_id="session-1", context=[message]),
        reply_stream=native_reply,
    )
    wrapper.as_llm = SimpleNamespace(model=SimpleNamespace(generate_structured_output=AsyncMock()))
    monkeypatch.setattr(wrapper, "_build_agent", AsyncMock(return_value=(agent, "question")))
    monkeypatch.setattr(wrapper, "_dump_state", AsyncMock())
    monkeypatch.setattr(wrapper, "_record_token_usage", Mock())
    return wrapper


@pytest.mark.asyncio
@pytest.mark.parametrize("text", ["memory", ""])
async def test_real_image_memory_noop_accepts_legacy_exceed(benchmark, tmp_path, monkeypatch, text):
    name, wrapper_type = benchmark
    message = AssistantMsg(
        name="agent", content=text, finished_reason="exceed_max_iters", usage=Usage(input_tokens=10, output_tokens=2)
    )
    wrapper = wrapper_for(wrapper_type, tmp_path, monkeypatch, message)
    assert wrapper.backend == "agentscope"
    module = import_module(f"reme_{name}.auto_memory")
    step_type = getattr(module, "MemLensAutoMemoryStep" if name == "memlens" else "MobileMemAutoMemoryStep")
    store = LocalFileStore(embedding_store="", app_context=wrapper.app_context)
    step = step_type(app_context=wrapper.app_context, agent_wrapper=wrapper, file_store=store)
    monkeypatch.setattr(step, "_list_session_note", AsyncMock(return_value=None))
    picture = BytesIO()
    Image.new("RGB", (1, 1), (12, 34, 56)).save(picture, format="PNG")
    image = picture.getvalue()
    source = UserMsg(
        id="turn-1",
        name="user",
        content=[DataBlock(source=Base64Source(media_type="image/png", data=base64.b64encode(image).decode()))],
    )
    await step(messages=[source], session_id="history-1", include_images=True, include_source_sessions=True)
    assert step.context.response.success and step.context.response.answer == text
    assert step.context.response.metadata["modified"] is False
    assert (tmp_path / "session/dialog/history-1.jsonl").is_file()
    saved = list((tmp_path / "session/images/history-1").glob("*"))
    assert len(saved) == 1 and saved[0].read_bytes() == image
    inputs = wrapper._build_agent.call_args.args[0]
    blocks = [block for block in inputs.content if isinstance(block, DataBlock)]
    assert len(blocks) == 1 and base64.b64decode(blocks[0].source.data) == image
    wrapper._record_token_usage.assert_called_once()
    wrapper.as_llm.model.generate_structured_output.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("text", ["answer", "", "The maximum reasoning-acting iterations are exceeded."])
async def test_legacy_exceed_accepts_text_empty_and_sdk_fallback(benchmark, tmp_path, monkeypatch, text):
    _, wrapper_type = benchmark
    message = AssistantMsg(
        name="agent",
        content=text,
        finished_reason="exceed_max_iters",
        usage=Usage(input_tokens=10, output_tokens=2),
    )
    original = message.model_dump(mode="json")
    wrapper = wrapper_for(wrapper_type, tmp_path, monkeypatch, message)
    result = await wrapper.reply("question")
    assert result["status"] == "completed" and result["error"] is None
    assert result["result"] == text
    assert result["raw"]["message"] == original
    assert message.model_dump(mode="json") == original
    assert result["raw"]["benchmark_compat"] == {
        "policy": "agentscope-exceed-max-iters-legacy-v1",
        "legacy_source_commit": "d3e79918622173306be79ecda6494acadc7c4312",
        "native_status": "failed",
        "native_error": "AgentScope reply ended: exceed_max_iters",
        "native_termination_reason": "exceed_max_iters",
    }
    assert result["usage"] == {"input_tokens": 10, "output_tokens": 2, "total_tokens": 12}
    wrapper._record_token_usage.assert_called_once()
    wrapper.as_llm.model.generate_structured_output.assert_not_awaited()
    assert agent_reply_result_text(result) == text
    terminal = [event for event in result["events"] if event["chunk_type"] == ChunkEnum.REPLY_END]
    assert len(terminal) == 1
    assert terminal[0]["status"] == terminal[0]["chunk"]["status"] == "completed"
    assert terminal[0]["chunk"]["raw"] == result["raw"]
    assert any(event["chunk_type"] == ChunkEnum.ERROR for event in result["events"])
    assert await collect_reply(replay_reply(result)) == result


@pytest.mark.asyncio
@pytest.mark.parametrize("reason", ["completed", "error", "interrupted", None])
async def test_non_exceed_matches_core_outcome_exactly(benchmark, tmp_path, monkeypatch, reason):
    _, wrapper_type = benchmark
    message = AssistantMsg(
        name="agent",
        content="answer",
        finished_reason=reason,
        usage=Usage(input_tokens=10, output_tokens=2),
    )
    core = wrapper_for(AsAgentWrapper, tmp_path / "core", monkeypatch, message)
    compatible = wrapper_for(wrapper_type, tmp_path / "compatible", monkeypatch, message)
    core_agent = core._build_agent.return_value[0]
    compatible_agent = compatible._build_agent.return_value[0]
    assert await compatible._finish_reply(compatible_agent, message, {}) == await core._finish_reply(
        core_agent, message, {}
    )
    assert compatible._record_token_usage.call_count == core._record_token_usage.call_count == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("native_error", [{}, {"type": "upstream", "message": "provider unavailable"}])
async def test_native_error_even_empty_mapping_does_not_enable_compatibility(
    benchmark, tmp_path, monkeypatch, native_error
):
    _, wrapper_type = benchmark
    wrapper = wrapper_type(app_context=ApplicationContext(workspace_dir=str(tmp_path)), as_llm="")
    result = wrapper._reply_result(
        session_id="session-1",
        result="partial",
        status="failed",
        error="provider unavailable",
        raw={"message": {"finished_reason": "exceed_max_iters", "error": native_error}},
    )
    original = deepcopy(result)
    parent = AsyncMock(return_value=result)
    monkeypatch.setattr(AsAgentWrapper, "_finish_reply", parent)
    returned = await wrapper._finish_reply(None, None, {})
    assert returned is result and returned == original
    parent.assert_awaited_once_with(None, None, {})


@pytest.mark.asyncio
async def test_output_schema_exceed_retains_strict_core_outcome(benchmark, tmp_path, monkeypatch):
    _, wrapper_type = benchmark
    message = AssistantMsg(
        name="agent",
        content="answer",
        finished_reason="exceed_max_iters",
        usage=Usage(input_tokens=10, output_tokens=2),
    )
    wrapper = wrapper_for(wrapper_type, tmp_path, monkeypatch, message)
    result = await wrapper.reply("question", output_schema={})
    assert result["status"] == "failed"
    assert result["error"] == "AgentScope reply ended: exceed_max_iters"
    assert "benchmark_compat" not in result["raw"]
    wrapper.as_llm.model.generate_structured_output.assert_not_awaited()


def test_only_benchmark_config_selects_local_wrappers(benchmark):
    name, wrapper_type = benchmark
    runtime = resolve_plugin_runtime(resolve_app_config(config=f"{name}-benchmark", log_config=False))
    wrappers = runtime.config["components"]["agent_wrapper"]
    assert wrappers["default"]["backend"] == wrappers["bench"]["backend"] == f"{name}_benchmark_agent"
    assert runtime.registry.get("agent_wrapper", f"{name}_benchmark_agent") is wrapper_type
    assert runtime.registry.get("agent_wrapper", "agentscope") is AsAgentWrapper
    ordinary = resolve_plugin_runtime(resolve_app_config(config="benchmark", log_config=False))
    assert ordinary.config["components"]["agent_wrapper"]["default"]["backend"] == "agentscope"
    assert ordinary.config["components"]["agent_wrapper"]["bench"]["backend"] == "agentscope"
