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


def test_provider_usage_accepts_prompt_and_completion_aliases():
    """Portable OpenAI-style aliases normalize to the common counters."""
    usage = TokenUsage.from_provider({"prompt_tokens": 3, "completion_tokens": 2})

    assert usage.model_dump() == {
        "input_tokens": 3,
        "output_tokens": 2,
        "total_tokens": 5,
    }


def test_total_tokens_is_always_derived_from_input_and_output():
    """Caller-supplied totals cannot violate the portable usage invariant."""
    usage = TokenUsage(input_tokens=3, output_tokens=2, total_tokens=999)

    assert usage.total_tokens == 5


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
@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("reported", [False, True])
async def test_agentscope_records_usage_in_both_return_modes(tmp_path, monkeypatch, streaming, reported):
    """Both APIs use the same final native message and record usage exactly once."""
    from agentscope.event import ReplyEndEvent
    from agentscope.message import AssistantMsg, Usage

    context = ApplicationContext(workspace_dir=str(tmp_path))
    wrapper = AsAgentWrapper(name="research", as_llm="", app_context=context)
    message = AssistantMsg(
        name="agent",
        content="answer",
        finished_reason="completed",
        usage=Usage(input_tokens=15, output_tokens=28) if reported else None,
    )

    async def native_stream(_inputs, **_kwargs):
        yield ReplyEndEvent(session_id="session-1", reply_id="reply", finished_reason="completed")
        yield message

    agent = SimpleNamespace(
        state=SimpleNamespace(session_id="session-1", context=[message]),
        reply_stream=native_stream,
    )

    async def build_agent(inputs, **_kwargs):
        return agent, inputs

    async def dump_state(_state):
        return None

    monkeypatch.setattr(wrapper, "_build_agent", build_agent)
    monkeypatch.setattr(wrapper._sessions, "save", dump_state)  # pylint: disable=protected-access
    if streaming:
        chunks = [chunk async for chunk in wrapper.reply_stream("hello")]
        result = chunks[-1].chunk
    else:
        result = await wrapper.reply("hello")
    assert result["usage"] == ({"input_tokens": 15, "output_tokens": 28, "total_tokens": 43} if reported else None)
    counters = global_counter_get_all(context.metadata, ["__token_counter", "research"])
    if reported:
        assert counters["children"]["total_tokens"]["value"] == 43
    else:
        assert counters is None
