"""Execution phases are presentation facts, never transcript content."""

import pytest

from reme.schema import AgentActivity, StreamChunk
from reme.utils.agent_activity import ActivityTracker


def test_task_activity_requires_a_run_identity_and_keeps_it_through_phase_changes():
    """A native turn is not the identity of a three-query task run."""
    for fields in ({}, {"run_id": "run"}, {"task_id": "research"}):
        with pytest.raises(ValueError, match="requires run_id and task_id"):
            AgentActivity(session_id="s", turn_id="t", source="task", phase="thinking", **fields)
    activity = AgentActivity(
        session_id="s",
        turn_id="t",
        source="task",
        phase="thinking",
        run_id="run",
        task_id="research",
    )
    updated = ActivityTracker(activity).update(StreamChunk(chunk_type="tool_call", tool_call_name="read_memory"))
    assert updated.phase == "remembering" and updated.run_id == "run" and updated.task_id == "research"


@pytest.mark.parametrize(
    "payload, phase",
    [
        ({"chunk_type": "think", "chunk": "reason"}, "thinking"),
        ({"chunk_type": "content", "chunk": "answer"}, "answering"),
        ({"chunk_type": "tool_call", "tool_call_name": "read_proactive_inbox"}, "remembering"),
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
    result = ActivityTracker(previous).update(StreamChunk(**payload))
    assert result.phase == phase
    assert result.session_id == "selected"
    assert result.turn_id == "turn"
    assert result.source == "proactive"
    assert previous.phase == "answering"


def test_identical_text_deltas_have_identical_activity():
    """Token updates can be deduplicated without restarting presentation."""
    state = AgentActivity(session_id="s", turn_id="t", phase="thinking")
    tracker = ActivityTracker(state)
    first = tracker.update(StreamChunk(chunk_type="content", chunk="one"))
    assert tracker.update(StreamChunk(chunk_type="content", chunk="two")) == first


@pytest.mark.parametrize("first", ["a", "b"])
def test_parallel_tools_remain_working_until_all_calls_finish(first):
    """Out-of-order completions retain the remaining tool's actual name and phase."""
    tracker = ActivityTracker(AgentActivity(session_id="s", turn_id="t", phase="thinking"))
    names = {"a": "read_memory", "b": "web_search"}
    for call_id, name in names.items():
        tracker.update(StreamChunk(chunk_type="tool_call", tool_call_id=call_id, tool_call_name=name))
    other = "b" if first == "a" else "a"
    state = tracker.update(StreamChunk(chunk_type="tool_result", tool_call_id=first, status="completed"))
    assert state.tool == names[other]
    assert state.phase == ("remembering" if other == "a" else "working")
    for duplicate in (first, "unknown"):
        assert tracker.update(StreamChunk(chunk_type="tool_result", tool_call_id=duplicate, status="failed")) == state
    assert tracker.update(StreamChunk(chunk_type="tool_result", tool_call_id=first, status="running")) == state
    assert tracker.update(StreamChunk(chunk_type="think", chunk="still processing")) == state
    final = tracker.update(StreamChunk(chunk_type="tool_result", tool_call_id=other, status="failed"))
    assert final.phase == "thinking" and final.tool is None


@pytest.mark.parametrize("outcome", ["completed", "failed", "interrupted"])
def test_terminal_outcomes_clear_unfinished_tools(outcome):
    """An interrupted tool cannot leak into a later turn, including reused call IDs."""
    tracker = ActivityTracker(AgentActivity(session_id="s", turn_id="t", phase="thinking"))
    tracker.update(StreamChunk(chunk_type="tool_call", tool_call_id="a", tool_call_name="read_memory"))
    result = tracker.update(StreamChunk(chunk_type="reply_end", status=outcome))
    assert result.phase == outcome and result.tool is None
    assert tracker.update(StreamChunk(chunk_type="think", chunk="late")).phase == outcome
    next_turn = ActivityTracker(AgentActivity(session_id="s", turn_id="next", phase="thinking"))
    assert (
        next_turn.update(StreamChunk(chunk_type="tool_call", tool_call_id="a", tool_call_name="web_search")).phase
        == "working"
    )


def test_unnamed_parallel_tools_still_block_the_thinking_phase():
    """An optional display name must not affect unfinished-call accounting."""
    tracker = ActivityTracker(AgentActivity(session_id="s", turn_id="t", phase="thinking"))
    for call_id in ("a", "b"):
        tracker.update(StreamChunk(chunk_type="tool_call", tool_call_id=call_id))
    tracker.update(StreamChunk(chunk_type="tool_result", tool_call_id="a", status="completed"))
    assert tracker.update(StreamChunk(chunk_type="think", chunk="reasoning")).phase == "working"
    assert (
        tracker.update(StreamChunk(chunk_type="tool_result", tool_call_id="b", status="completed")).phase == "thinking"
    )
