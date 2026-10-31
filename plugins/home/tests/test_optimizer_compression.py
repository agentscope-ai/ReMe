"""Compression recovery parity with dev_meta, without model or filesystem services."""

# pylint: disable=missing-function-docstring,missing-class-docstring,protected-access
# pylint: disable=redefined-outer-name,unused-argument

import asyncio
import json

from agentscope.agent import Agent, ContextConfig
from agentscope.message import AssistantMsg, TextBlock, UserMsg, Usage
from agentscope.state import AgentState
import jsonschema
import openai
import pytest
from reme_home.agent import compression, optimizer


class FailingCompressionModel:
    def __init__(self, ratio=0.9, failure=None):
        self.context_size = 90 / ratio
        self.failure = failure or RuntimeError("Failed to generate structured output for model.")
        self.calls = 0

    async def count_tokens(self, messages, tools=None):
        return sum(10 for message in messages if (message.get_text_content() or "").startswith("message-"))

    async def generate_structured_output(self, **kwargs):
        self.calls += 1
        raise self.failure


def make_agent(tmp_path, ratio=0.9, trigger=0.8, failure=None, summary="existing summary"):
    return optimizer.OptimizerAgent(
        name="optimizer",
        system_prompt="system",
        model=FailingCompressionModel(ratio, failure),
        state=AgentState(
            summary=summary,
            context=[UserMsg(name="user", content=f"message-{index}") for index in range(9)],
        ),
        context_config=ContextConfig(trigger_ratio=trigger, reserve_ratio=0.2),
        fallback_directory=tmp_path / "offloads",
        validation_cycle_timer=optimizer.ValidationCycleTimer(150),
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("trigger", "ratio", "expected"),
    [(0.8, 0.79, False), (0.8, 0.85, False), (0.8, 0.9, True), (0.65, 0.82, False), (0.65, 0.825, True)],
)
async def test_real_compression_failure_uses_dev_meta_emergency_threshold(tmp_path, trigger, ratio, expected):
    agent = make_agent(tmp_path, ratio, trigger)
    context_before, summary_before = agent.state.context, agent.state.summary
    await agent.compress_context()
    assert agent.validation_cycle_timer.context_compression_count == int(expected)
    assert len(list(tmp_path.rglob("context-*.json"))) == int(expected)
    assert agent.model.calls == int(ratio >= trigger)
    assert agent.context_config.compression_fallback_to_truncation  # The shared config was not mutated.
    if not expected:
        assert agent.state.context is context_before
        assert agent.state.summary is summary_before


@pytest.mark.asyncio
@pytest.mark.parametrize("summary", ["existing summary", [TextBlock(text="existing summary")]])
@pytest.mark.parametrize("reserve_ratio", [0.2, 0])
async def test_offload_retains_full_messages_summary_usage_and_recent_context(tmp_path, summary, reserve_ratio):
    agent = make_agent(tmp_path, summary=summary)
    agent.context_config.reserve_ratio = reserve_ratio
    agent.state.context[-1] = AssistantMsg(
        id=agent.state.reply_id,
        name=agent.name,
        content="message-8",
        usage=Usage(input_tokens=100, output_tokens=10),
    )
    original = [msg.model_dump(mode="json", exclude_none=False) for msg in agent.state.context]
    await agent.compress_context()
    (path,) = tmp_path.rglob("context-*.json")
    data = json.loads(path.read_text())
    assert data["messages"] == original[:7] if reserve_ratio else data["messages"] == original
    if reserve_ratio:
        assert [msg.get_text_content() for msg in agent.state.context] == ["message-7", "message-8"]
    else:
        assert len(agent.state.context) == 1
        assert agent.state.context[0].content == []  # SDK usage carrier after full offload.
    assert data["session_id"] == agent.state.session_id
    assert data["context_ratio"] == 0.9
    assert data["emergency_ratio"] == 0.9
    assert data["failure"]["type"] == "RuntimeError"
    text = agent.state.summary
    if isinstance(text, list):
        text = "\n".join(block.text for block in text)
    assert "existing summary" in text
    assert "without semantic summarization" in text
    assert str(path) in text
    assert agent._get_reply_usage().input_tokens == 100
    assert not list(path.parent.glob(".result-*"))


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "failure",
    [RuntimeError("The system prompt exceeds threshold"), ValueError("bad configuration"), asyncio.CancelledError()],
)
async def test_unrecoverable_failures_and_cancellation_propagate(tmp_path, failure):
    agent = make_agent(tmp_path, failure=failure)
    original = agent.state.context
    with pytest.raises(type(failure)):
        await agent.compress_context()
    assert agent.state.context is original
    assert not list(tmp_path.rglob("*.json"))
    assert agent.validation_cycle_timer.context_compression_count == 0


@pytest.mark.parametrize(
    ("failure", "expected"),
    [
        (RuntimeError("provider failed"), True),
        (RuntimeError("The system prompt exceeds threshold"), False),
        (jsonschema.ValidationError("invalid summary"), True),
        (openai.OpenAIError("provider unavailable"), True),
        (ValueError("programming error"), False),
        (OSError("disk full"), False),
    ],
)
def test_recoverable_error_classifier(failure, expected):
    assert compression.RecoverableCompressionAgent._is_recoverable_compression_error(failure) is expected


@pytest.mark.asyncio
async def test_override_config_and_successful_compression_use_native_pipeline(tmp_path, monkeypatch):
    agent = make_agent(tmp_path, ratio=0.85)
    config = ContextConfig(trigger_ratio=0.65, reserve_ratio=0.1)
    calls = []

    async def compress(self, context_config=None, instructions=None):
        calls.append((context_config, instructions))
        self.state.summary = "semantic summary"
        self.state.context = []

    monkeypatch.setattr(Agent, "_compress_context_impl", compress)
    await agent.compress_context(context_config=config)
    assert calls[0][0].trigger_ratio == 0.65
    assert calls[0][0].reserve_ratio == 0.1
    assert not calls[0][0].compression_fallback_to_truncation
    assert config.compression_fallback_to_truncation
    assert agent.state.summary == "semantic summary"
    assert agent.validation_cycle_timer.context_compression_count == 1
    assert not list(tmp_path.rglob("*.json"))


@pytest.mark.asyncio
async def test_offload_write_failure_keeps_context_and_does_not_count(tmp_path, monkeypatch):
    agent = make_agent(tmp_path)
    original = agent.state.context

    def fail(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(compression, "write_json", fail)
    with pytest.raises(OSError, match="disk full"):
        await agent.compress_context()
    assert agent.state.context is original
    assert agent.state.summary == "existing summary"
    assert agent.validation_cycle_timer.context_compression_count == 0


@pytest.mark.asyncio
async def test_cancellation_during_apply_finishes_consistent_offload_then_propagates(tmp_path, monkeypatch):
    agent = make_agent(tmp_path)
    started, release = asyncio.Event(), asyncio.Event()

    async def clear_cache(messages):
        started.set()
        await release.wait()

    monkeypatch.setattr(agent, "_clear_unreserved_read_cache", clear_cache)
    task = asyncio.create_task(agent.compress_context())
    await started.wait()
    task.cancel()
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert len(list(tmp_path.rglob("context-*.json"))) == 1
    assert "offloaded" in agent.state.summary
    assert len(agent.state.context) == 2


@pytest.mark.asyncio
async def test_missing_directory_or_unsplittable_context_fails_explicitly(tmp_path, monkeypatch):
    agent = make_agent(tmp_path)
    agent.fallback_directory = None
    with pytest.raises(RuntimeError, match="fallback_directory"):
        await agent.compress_context()
    agent.fallback_directory = tmp_path / "offloads"

    async def no_old_messages(*args):
        return [], agent.state.context

    monkeypatch.setattr(agent, "_split_context_for_compression", no_old_messages)
    with pytest.raises(RuntimeError, match="cannot remove any context"):
        await agent.compress_context()
    assert not list(tmp_path.rglob("*.json"))


def test_wrapper_uses_host_offload_directory(tmp_path):
    wrapper = optimizer.OptimizerAgentWrapper(name="optimizer")
    agent = wrapper._create_agent(
        options={"_context_fallback_directory": tmp_path / "logs/context-offloads"},
        name="optimizer",
        system_prompt="system",
        model=FailingCompressionModel(),
    )
    assert agent.fallback_directory == tmp_path / "logs/context-offloads"
