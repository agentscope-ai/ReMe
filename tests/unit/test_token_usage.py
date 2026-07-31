"""Tests for unified agent token accounting."""

from types import SimpleNamespace

from agentscope.model._model_usage import ChatUsage
import pytest

from reme.components.agent_wrapper import AsAgentWrapper, BaseAgentWrapper
from reme.components.application_context import ApplicationContext
from reme.schema import TokenUsage
from reme.utils import global_counter_get_all


class _UsageWrapper(BaseAgentWrapper):
    async def reply(self, inputs, **kwargs):
        raise NotImplementedError


def test_provider_usage_keeps_only_input_and_output_tokens():
    """Provider-specific cache and reasoning details are not persisted."""
    usage = TokenUsage.from_provider(
        {
            "input_tokens": 10,
            "output_tokens": 4,
            "cache_read_input_tokens": 20,
            "cache_creation_input_tokens": 30,
            "reasoning_output_tokens": 2,
        },
    )

    assert usage.model_dump() == {
        "input_tokens": 10,
        "output_tokens": 4,
        "total_tokens": 14,
    }


def test_provider_usage_uses_reported_input_without_cache_adjustment():
    """Reported input is kept unchanged across backend-specific usage shapes."""
    usage = TokenUsage.from_provider(
        {
            "input_tokens": 60,
            "output_tokens": 4,
            "cached_input_tokens": 20,
            "reasoning_output_tokens": 2,
        },
    )

    assert usage.input_tokens == 60
    assert usage.total_tokens == 64


def test_agentscope_usage_keeps_portable_input_and_output(tmp_path):
    """AgentScope usage has the same input/output-only contract."""
    context = ApplicationContext(workspace_dir=str(tmp_path))
    wrapper = AsAgentWrapper(name="research", as_llm="", app_context=context)
    usage = ChatUsage(
        input_tokens=10,
        output_tokens=4,
        time=0.0,
        cache_input_tokens=20,
        cache_creation_input_tokens=30,
    )

    assert wrapper._agentscope_usage(usage).model_dump() == {  # pylint: disable=protected-access
        "input_tokens": 10,
        "output_tokens": 4,
        "total_tokens": 14,
    }


def test_combined_usage_sums_all_model_calls():
    """One wrapper invocation records aggregate input/output usage."""
    usage = TokenUsage.combine(
        [
            TokenUsage(input_tokens=10, output_tokens=4),
            TokenUsage(input_tokens=5, output_tokens=2),
        ],
    )

    assert usage.model_dump() == {
        "input_tokens": 15,
        "output_tokens": 6,
        "total_tokens": 21,
    }


def test_token_counter_is_a_per_agent_metric_tree(tmp_path):
    """Recorded usage accumulates input, output, and total tokens per agent."""
    context = ApplicationContext(workspace_dir=str(tmp_path))
    wrapper = _UsageWrapper(name="research", app_context=context)
    wrapper._record_token_usage(  # pylint: disable=protected-access
        TokenUsage(input_tokens=10, output_tokens=4),
    )
    wrapper._record_token_usage(TokenUsage(input_tokens=3, output_tokens=2))  # pylint: disable=protected-access

    assert global_counter_get_all(context.metadata, ["__token_counter", "research"]) == {
        "value": 0,
        "children": {
            "input_tokens": {"value": 13, "children": {}},
            "output_tokens": {"value": 6, "children": {}},
            "total_tokens": {"value": 19, "children": {}},
        },
    }


@pytest.mark.asyncio
async def test_agentscope_reply_records_final_message_usage(tmp_path, monkeypatch):
    """AgentScope 2.0.4.post1 reports aggregate usage on the final message."""
    context = ApplicationContext(workspace_dir=str(tmp_path))
    wrapper = AsAgentWrapper(name="research", as_llm="", app_context=context)
    message = SimpleNamespace(
        # Match the AgentScope 2.0.4.post1 accumulation test: 10/20 + 5/8.
        usage=SimpleNamespace(input_tokens=15, output_tokens=28),
        model_dump=lambda: {"text": "answer"},
        get_text_content=lambda: "answer",
    )
    agent = SimpleNamespace(
        state=SimpleNamespace(session_id="session-1", context=[message]),
        observe=lambda _inputs: _async_none(),
        reply=lambda: _async_value(message),
        reply_stream=_unexpected_reply_stream,
    )

    async def build_agent(inputs, **_kwargs):
        return agent, inputs

    monkeypatch.setattr(wrapper, "_build_agent", build_agent)
    monkeypatch.setattr(wrapper, "_dump_state", _async_none)

    result = await wrapper.reply("hello")

    assert result["usage"] == {"input_tokens": 15, "output_tokens": 28, "total_tokens": 43}
    assert (
        global_counter_get_all(context.metadata, ["__token_counter", "research"])["children"]["total_tokens"]["value"]
        == 43
    )


@pytest.mark.asyncio
async def test_agentscope_reply_without_usage_leaves_token_counters_unset(tmp_path, monkeypatch):
    """Replies without any usage information remain visibly unavailable."""
    context = ApplicationContext(workspace_dir=str(tmp_path))
    wrapper = AsAgentWrapper(name="research", as_llm="", app_context=context)
    message = SimpleNamespace(
        usage=None,
        model_dump=lambda: {"text": "answer"},
        get_text_content=lambda: "answer",
    )
    agent = SimpleNamespace(
        state=SimpleNamespace(session_id="session-1", context=[message]),
        observe=lambda _inputs: _async_none(),
        reply=lambda: _async_value(message),
        reply_stream=_unexpected_reply_stream,
    )

    async def build_agent(inputs, **_kwargs):
        return agent, inputs

    monkeypatch.setattr(wrapper, "_build_agent", build_agent)
    monkeypatch.setattr(wrapper, "_dump_state", _async_none)

    result = await wrapper.reply("hello")

    assert result["usage"] is None
    assert "__token_counter" not in context.metadata


async def _async_none(*_args, **_kwargs):
    return None


async def _async_value(value):
    return value


def _unexpected_reply_stream(*_args, **_kwargs):
    raise AssertionError("Non-streaming replies must use Agent.reply()")


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
