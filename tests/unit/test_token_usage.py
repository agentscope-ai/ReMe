"""Tests for unified agent token accounting."""

from agentscope.model._model_usage import ChatUsage
import pytest

from reme.components.agent_wrapper import AsAgentWrapper, BaseAgentWrapper
from reme.components.application_context import ApplicationContext
from reme.schema import TokenUsage
from reme.utils import global_counter_get_all


class _UsageWrapper(BaseAgentWrapper):
    async def reply(self, inputs, **kwargs):
        raise NotImplementedError


def test_claude_style_usage_normalizes_cache_into_complete_input():
    """Cache-excluded provider input is normalized into the complete input total."""
    usage = TokenUsage.from_provider(
        {
            "input_tokens": 10,
            "output_tokens": 4,
            "cache_read_input_tokens": 20,
            "cache_creation_input_tokens": 30,
        },
        input_includes_cache=False,
    )

    assert usage.model_dump() == {
        "input_tokens": 60,
        "output_tokens": 4,
        "cache_read_tokens": 20,
        "cache_write_tokens": 30,
        "reasoning_tokens": None,
        "total_tokens": 64,
    }


def test_codex_style_usage_does_not_double_count_cached_input():
    """Cache-inclusive provider input is preserved without adding cached tokens twice."""
    usage = TokenUsage.from_provider(
        {
            "input_tokens": 60,
            "output_tokens": 4,
            "cached_input_tokens": 20,
            "reasoning_output_tokens": 2,
        },
        input_includes_cache=True,
    )

    assert usage.input_tokens == 60
    assert usage.cache_read_tokens == 20
    assert usage.reasoning_tokens == 2
    assert usage.total_tokens == 64


def test_agentscope_anthropic_usage_includes_cache_tokens(tmp_path):
    """AgentScope uses one usage type, so provider identity comes from its model."""
    context = ApplicationContext(workspace_dir=str(tmp_path))
    wrapper = AsAgentWrapper(name="research", as_llm="", app_context=context)
    wrapper.as_llm = type(
        "AnthropicLLM",
        (),
        {"model": type("AnthropicModel", (), {"__module__": "agentscope.model._anthropic._model"})()},
    )()
    usage = ChatUsage(
        input_tokens=10,
        output_tokens=4,
        time=0.0,
        cache_input_tokens=20,
        cache_creation_input_tokens=30,
    )

    assert wrapper._agentscope_usage(usage).model_dump() == {  # pylint: disable=protected-access
        "input_tokens": 60,
        "output_tokens": 4,
        "cache_read_tokens": 20,
        "cache_write_tokens": 30,
        "reasoning_tokens": None,
        "total_tokens": 64,
    }


def test_combined_usage_marks_partially_reported_metrics_as_unknown():
    """A partial cache/reasoning sum must not be presented as a full total."""
    usage = TokenUsage.combine(
        [
            TokenUsage(input_tokens=10, output_tokens=4, cache_read_tokens=6),
            TokenUsage(input_tokens=5, output_tokens=2),
        ],
    )

    assert usage.model_dump() == {
        "input_tokens": 15,
        "output_tokens": 6,
        "cache_read_tokens": None,
        "cache_write_tokens": None,
        "reasoning_tokens": None,
        "total_tokens": 21,
    }


def test_token_counter_is_a_per_agent_metric_tree(tmp_path):
    """Recorded usage accumulates per agent, and optional metrics track reported calls."""
    context = ApplicationContext(workspace_dir=str(tmp_path))
    wrapper = _UsageWrapper(name="research", app_context=context)
    wrapper._record_token_usage(  # pylint: disable=protected-access
        TokenUsage(input_tokens=10, output_tokens=4, cache_read_tokens=6),
    )
    wrapper._record_token_usage(TokenUsage(input_tokens=3, output_tokens=2))  # pylint: disable=protected-access

    assert global_counter_get_all(context.metadata, ["__token_counter", "research"]) == {
        "value": 0,
        "children": {
            "input_tokens": {"value": 13, "children": {}},
            "output_tokens": {"value": 6, "children": {}},
            "total_tokens": {"value": 19, "children": {}},
            "cache_read_tokens": {"value": 6, "children": {}},
            "cache_read_tokens_reported_calls": {"value": 1, "children": {}},
        },
    }


@pytest.mark.asyncio
async def test_agentscope_stream_reply_does_not_record_token_usage(tmp_path, monkeypatch):
    """Only non-streaming AgentScope replies contribute to token accounting."""
    context = ApplicationContext(workspace_dir=str(tmp_path))
    wrapper = AsAgentWrapper(name="research", as_llm="", app_context=context)

    class FakeAgent:
        """Minimal AgentScope stream double."""

        state = type("State", (), {"session_id": "session-1"})()

        async def reply_stream(self, inputs):
            """Yield no events for the supplied input."""
            if inputs is None:
                yield None

    async def build_agent(inputs, **_kwargs):
        """Build the minimal stream double."""
        return FakeAgent(), inputs

    async def dump_state(_state):
        """Avoid durable state writes in this accounting test."""
        return None

    monkeypatch.setattr(wrapper, "_build_agent", build_agent)
    monkeypatch.setattr(wrapper, "_dump_state", dump_state)

    assert [chunk async for chunk in wrapper.reply_stream("hello")] == []
    assert "__token_counter" not in context.metadata
