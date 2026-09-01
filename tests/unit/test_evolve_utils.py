"""Tests for shared evolve helpers."""

# pylint: disable=protected-access

import os
import time

from agentscope.message import Msg
import pytest

from reme.steps.evolve._evolve import agent_reply_result_text, format_history
from reme.steps.evolve.auto_memory import AutoMemoryStep, _normalize_tags, _sanitize_msg_for_save


def test_agent_reply_result_text_uses_last_text_block():
    """Agent reply text should be the last text block in the final message."""
    reply = {
        "result": "intermediate text\nfinal text",
        "last_message": {
            "content": [
                {"type": "thinking", "thinking": "hidden"},
                {"type": "text", "text": "intermediate text"},
                {"type": "tool_call", "name": "write"},
                {"type": "text", "text": "final text"},
            ],
        },
    }

    assert agent_reply_result_text(reply) == "final text"


def test_agent_reply_result_text_falls_back_to_result():
    """Agent reply text falls back to result when no text block exists."""
    reply = {"result": " fallback text \n", "last_message": {"content": [{"type": "thinking"}]}}

    assert agent_reply_result_text(reply) == "fallback text"


def test_sanitize_msg_for_save_drops_tool_results_and_base64_data():
    """Saved history should keep conversation context without tool output pollution."""
    msg = Msg.model_validate(
        {
            "name": "assistant",
            "role": "assistant",
            "content": [
                {"type": "text", "text": "real conversation"},
                {
                    "type": "tool_call",
                    "id": "call-1",
                    "name": "memory_search",
                    "input": "{}",
                },
                {
                    "type": "tool_result",
                    "id": "call-1",
                    "name": "memory_search",
                    "output": "retrieved memory that should not become conversation history",
                },
                {
                    "type": "data",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": "abc",
                    },
                },
            ],
        },
    )

    sanitized = _sanitize_msg_for_save(msg)

    assert [block.type for block in sanitized.content] == ["text", "tool_call"]
    assert sanitized.content[0].text == "real conversation"
    assert sanitized.content[1].name == "memory_search"


def test_auto_memory_normalizes_frontmatter_tags():
    """Tags keep technical punctuation, reject phrases, de-duplicate, and stop at eight."""
    assert _normalize_tags(
        [
            "GPT-5",
            "C++",
            "C#",
            ".NET",
            "100",
            100,
            "memory system",
            "++",
            "ReMe",
            "reme",
            "tag7",
            "tag8",
            "tag9",
        ],
    ) == ["GPT-5", "C++", "C#", ".NET", "100", "ReMe", "tag7", "tag8"]
    assert _normalize_tags(None) == []
    assert _normalize_tags("GPT-5") == []
    assert _normalize_tags(["x" * 65, True, {}, "valid"]) == ["valid"]


def test_auto_memory_accepts_message_timestamp_aliases():
    """AutoMemoryStep preserves historical message timestamps from common benchmark fields."""
    top_level = AutoMemoryStep._to_msg(
        {
            "name": "user",
            "role": "user",
            "content": "first event",
            "time_created": "2023-01-19T08:00:00",
        },
    )
    nested = AutoMemoryStep._to_msg(
        {
            "name": "assistant",
            "role": "assistant",
            "content": "second event",
            "metadata": {"timestamp": "2023-01-20T09:30:00"},
        },
    )

    history = format_history([top_level, nested])

    assert top_level.created_at == "2023-01-19T08:00:00"
    assert nested.created_at == "2023-01-20T09:30:00"
    assert "[user @ 2023-01-19T08:00:00]" in history
    assert "[assistant @ 2023-01-20T09:30:00]" in history


def test_auto_memory_created_at_takes_precedence_over_aliases():
    """Explicit AgentScope ``created_at`` values are not overwritten by compatibility aliases."""
    msg = AutoMemoryStep._to_msg(
        {
            "name": "user",
            "role": "user",
            "content": "event",
            "created_at": "2023-02-01T00:00:00",
            "time_created": "2023-01-19T08:00:00",
            "metadata": {"timestamp": "2023-01-20T09:30:00"},
        },
    )

    assert msg.created_at == "2023-02-01T00:00:00"


def test_auto_memory_derives_latest_day_from_message_timestamps():
    """Historical imports should anchor the daily note date to the latest message time."""
    messages = [
        AutoMemoryStep._to_msg(
            {
                "name": "assistant",
                "role": "assistant",
                "content": "second event",
                "created_at": "2023-01-20T09:30:00",
            },
        ),
        AutoMemoryStep._to_msg(
            {
                "name": "user",
                "role": "user",
                "content": "first event",
                "metadata": {"timestamp": "2023-01-19T08:00:00"},
            },
        ),
    ]

    assert AutoMemoryStep._messages_day(messages) == "2023-01-20"


def test_auto_memory_converts_absolute_timestamps_to_workspace_day():
    """UTC timestamps are bucketed using the configured workspace timezone."""
    messages = [
        AutoMemoryStep._to_msg(
            {
                "name": "user",
                "role": "user",
                "content": "after midnight in Shanghai",
                "created_at": "2026-08-19T16:30:00Z",
            },
        ),
    ]

    assert AutoMemoryStep._messages_day(messages, "Asia/Shanghai") == "2026-08-20"
    assert AutoMemoryStep._messages_day(messages, "UTC") == "2026-08-19"


def test_auto_memory_uses_target_date_dst_when_timezone_is_unset():
    """The system timezone applies the target date's DST offset, not the current offset."""
    if not hasattr(time, "tzset"):
        pytest.skip("time.tzset is unavailable on this platform")
    original_timezone = os.environ.get("TZ")
    try:
        os.environ["TZ"] = "America/New_York"
        time.tzset()
        assert AutoMemoryStep._message_day("2026-01-01T04:30:00Z", None) == "2025-12-31"
    finally:
        if original_timezone is None:
            os.environ.pop("TZ", None)
        else:
            os.environ["TZ"] = original_timezone
        time.tzset()
