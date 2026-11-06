"""Execution phases are presentation facts, never transcript content."""

import pytest

from reme.schema import AgentActivity, StreamChunk
from reme.utils.agent_activity import reduce_activity


@pytest.mark.parametrize(
    "payload, phase",
    [
        ({"chunk_type": "think", "chunk": "reason"}, "thinking"),
        ({"chunk_type": "content", "chunk": "answer"}, "answering"),
        ({"chunk_type": "tool_call", "tool_call_name": "read_conversation"}, "remembering"),
        ({"chunk_type": "tool_call", "tool_call_name": "web_search"}, "working"),
        ({"chunk_type": "tool_result", "status": "running"}, "working"),
        ({"chunk_type": "tool_result", "status": "failed"}, "thinking"),
        ({"chunk_type": "approval", "status": "waiting"}, "waiting"),
        ({"chunk_type": "approval", "status": "running"}, "answering"),
        ({"chunk_type": "reply_end", "status": "completed"}, "completed"),
        ({"chunk_type": "reply_end", "status": "failed"}, "failed"),
        ({"chunk_type": "reply_end", "status": "interrupted"}, "interrupted"),
        ({"chunk_type": "done", "done": True}, "answering"),
        ({"chunk_type": "error", "metadata": {"will_retry": True}}, "answering"),
    ],
)
def test_portable_events_update_activity_without_confusing_transport_or_tool_outcomes(payload, phase):
    """Only execution facts change phase, retaining delivery and turn identity."""
    previous = AgentActivity(session_id="selected", turn_id="turn", source="proactive", phase="answering")
    result = reduce_activity(previous, StreamChunk(**payload))
    assert result.phase == phase
    assert result.session_id == "selected"
    assert result.turn_id == "turn"
    assert result.source == "proactive"
    assert previous.phase == "answering"


def test_identical_text_deltas_have_identical_activity():
    """Token updates can be deduplicated without restarting presentation."""
    state = AgentActivity(session_id="s", turn_id="t", phase="thinking")
    first = reduce_activity(state, StreamChunk(chunk_type="content", chunk="one"))
    assert reduce_activity(first, StreamChunk(chunk_type="content", chunk="two")) == first
