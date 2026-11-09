"""Offline regression coverage for the benchmark-local legacy termination policy."""

# pylint: disable=missing-function-docstring,protected-access,redefined-outer-name

from copy import deepcopy
import base64
from importlib.metadata import EntryPoint
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

from agentscope.message import AssistantMsg, Base64Source, DataBlock, Usage, UserMsg
from PIL import Image
import pytest

from reme.components.agent_wrapper.as_agent_wrapper import AsAgentWrapper
from reme.components.application_context import ApplicationContext
from reme.components.file_store import LocalFileStore
from reme.config import resolve_app_config
from reme.plugin import resolve_plugin_runtime
from reme.steps.evolve._evolve import agent_reply_result_text


@pytest.fixture(params=["memlens", "mobilemem"])
def benchmark(request, monkeypatch):
    root = Path(__file__).resolve().parents[2]
    monkeypatch.syspath_prepend(str(root / f"plugins/{request.param}/src"))
    if request.param == "memlens":
        from reme_memlens.auto_memory import MemLensAutoMemoryStep as MemoryStep
        from reme_memlens.benchmark_agent import BenchmarkAsAgentWrapper
    else:
        from reme_mobilemem.auto_memory import MobileMemAutoMemoryStep as MemoryStep
        from reme_mobilemem.benchmark_agent import BenchmarkAsAgentWrapper
    return request.param, BenchmarkAsAgentWrapper, MemoryStep


@pytest.fixture
def wrapper(benchmark, tmp_path, monkeypatch):
    name, wrapper_type, _ = benchmark
    instance = wrapper_type(
        backend=f"{name}_benchmark_agent",
        app_context=ApplicationContext(workspace_dir=str(tmp_path)),
        as_llm="",
    )
    monkeypatch.setattr(instance, "_record_token_usage", Mock())
    return instance


@pytest.mark.asyncio
@pytest.mark.parametrize("text", ["answer", ""])
async def test_legacy_exceed_preserves_result_usage_and_diagnostics(wrapper, text):
    message = AssistantMsg(
        name="agent",
        content=text,
        finished_reason="exceed_max_iters",
        usage=Usage(input_tokens=10, output_tokens=2),
    )
    agent = SimpleNamespace(state=SimpleNamespace(session_id="session-1"))
    result = await wrapper._finish_reply(agent, message, {})
    assert result["status"] == "completed" and result["error"] is None
    assert agent_reply_result_text(result) == text
    assert result["raw"]["message"] == message.model_dump(mode="json")
    assert result["raw"]["benchmark_compat"]["native_status"] == "failed"
    assert result["raw"]["benchmark_compat"]["native_error"] == "AgentScope reply ended: exceed_max_iters"
    assert result["usage"] == {"input_tokens": 10, "output_tokens": 2, "total_tokens": 12}
    wrapper._record_token_usage.assert_called_once()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "reason,native_error,kwargs",
    [
        ("exceed_max_iters_extra", None, {}),
        ("exceed_max_iters", {}, {}),
        ("exceed_max_iters", {"message": "provider unavailable"}, {}),
        ("exceed_max_iters", None, {"output_schema": {}}),
    ],
)
async def test_nonlegacy_results_are_unchanged(wrapper, monkeypatch, reason, native_error, kwargs):
    native = wrapper._reply_result(
        session_id="session-1",
        result="partial",
        status="failed",
        error="native failure",
        raw={"message": {"finished_reason": reason, "error": native_error}},
    )
    original = deepcopy(native)
    monkeypatch.setattr(AsAgentWrapper, "_finish_reply", AsyncMock(return_value=native))
    assert await wrapper._finish_reply(None, None, kwargs) == original


@pytest.mark.asyncio
async def test_real_image_memory_accepts_empty_legacy_result(benchmark, wrapper, tmp_path, monkeypatch):
    _, _, step_type = benchmark
    assert wrapper.backend == "agentscope"
    agent = SimpleNamespace(state=SimpleNamespace(session_id="session-1"))
    message = AssistantMsg(name="agent", content="", finished_reason="exceed_max_iters")
    monkeypatch.setattr(wrapper, "reply", AsyncMock(return_value=await wrapper._finish_reply(agent, message, {})))
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
    assert step.context.response.success and step.context.response.answer == ""
    assert step.context.response.metadata["modified"] is False
    assert (tmp_path / "session/dialog/history-1.jsonl").is_file()
    saved = list((tmp_path / "session/images/history-1").glob("*"))
    assert len(saved) == 1 and saved[0].read_bytes() == image
    inputs = wrapper.reply.call_args.args[0]
    blocks = [block for block in inputs.content if isinstance(block, DataBlock)]
    assert len(blocks) == 1 and base64.b64decode(blocks[0].source.data) == image


def test_only_benchmark_config_selects_local_wrappers(benchmark, monkeypatch):
    name, wrapper_type, _ = benchmark
    entry = EntryPoint(name=name, value=f"reme_{name}", group="reme.plugins")
    monkeypatch.setattr(
        "reme.plugin.find_entry_points",
        lambda group, plugin: [entry] if (group, plugin) == (entry.group, entry.name) else [],
    )
    preset = Path(__file__).resolve().parents[2] / f"plugins/{name}/src/reme_{name}/benchmark.yaml"
    runtime = resolve_plugin_runtime(resolve_app_config(config=str(preset), log_config=False))
    wrappers = runtime.config["components"]["agent_wrapper"]
    assert wrappers["default"]["backend"] == wrappers["bench"]["backend"] == f"{name}_benchmark_agent"
    assert runtime.registry.get("agent_wrapper", f"{name}_benchmark_agent") is wrapper_type
    assert runtime.registry.get("agent_wrapper", "agentscope") is AsAgentWrapper
    ordinary = resolve_plugin_runtime(resolve_app_config(config="benchmark", log_config=False))
    assert ordinary.config["components"]["agent_wrapper"]["default"]["backend"] == "agentscope"
    assert ordinary.config["components"]["agent_wrapper"]["bench"]["backend"] == "agentscope"
