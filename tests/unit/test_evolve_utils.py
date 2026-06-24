"""Tests for shared evolve helpers."""

from reme.steps.evolve._evolve import agent_reply_result_text


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
