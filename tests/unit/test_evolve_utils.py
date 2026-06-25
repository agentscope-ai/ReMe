"""Tests for shared evolve helpers."""

from agentscope.message import Msg

from reme.steps.evolve._evolve import agent_reply_result_text
from reme.steps.evolve.auto_memory import _sanitize_msg_for_save


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
