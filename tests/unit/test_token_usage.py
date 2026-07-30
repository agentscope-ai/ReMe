"""Tests for unified agent token accounting."""

from reme.components.agent_wrapper import BaseAgentWrapper
from reme.components.application_context import ApplicationContext
from reme.schema import TokenUsage
from reme.utils import global_counter_get_all


class _UsageWrapper(BaseAgentWrapper):
    async def reply(self, inputs, **kwargs):
        raise NotImplementedError


def test_claude_style_usage_normalizes_cache_into_complete_input():
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


def test_token_counter_is_a_per_agent_metric_tree(tmp_path):
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
        },
    }
