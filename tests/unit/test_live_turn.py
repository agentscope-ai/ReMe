"""Live snapshots are bounded projections, never a replacement for native history."""

# pylint: disable=missing-function-docstring

import asyncio

import pytest

from reme.components.agent_runtime._internal.live_turn import LiveTurn, LiveTurns
from reme.schema import StreamChunk


def test_live_blocks_compact_deltas_and_preserve_order_and_null_snapshots():
    turn = LiveTurn("session", "Hello")
    for _ in range(1000):
        turn.update(StreamChunk(turn_id="turn", block_id="text", chunk="你"))
    turn.update(
        StreamChunk(
            turn_id="turn",
            block_id="tool",
            tool_call_id="tool",
            chunk_type="tool_call",
            chunk={"path": "a.md"},
        ),
    )
    turn.update(StreamChunk(turn_id="turn", block_id="text", chunk=None, event="end"))
    message = turn.messages()[1]
    assert len(message["chunks"]) == 2
    assert message["chunks"][0]["chunk"] == "你" * 1000
    assert message["content"] == "你" * 1000
    assert [chunk["block_id"] for chunk in message["chunks"]] == ["text", "tool"]


def test_live_overlay_recovers_on_reconnect_and_yields_to_completed_native_history():
    live = LiveTurns()
    source = {
        "session_id": "session",
        "agent": {"status": "active"},
        "snapshot": {"messages": [], "revision": "native"},
    }
    with live.observe("Remy", "session", "Hello") as publish:
        publish(StreamChunk(turn_id="turn", block_id="text", chunk="Partial"))
        first = live.overlay("Remy", source)
        assert first == live.overlay("Remy", source)
        assert first["snapshot"]["revision"] != source["snapshot"]["revision"]
        assert not source["snapshot"]["messages"]
        publish(
            StreamChunk(
                chunk_type="reply_end",
                event="end",
                status="completed",
                chunk={"result": "Final", "raw": {"secret": "not forwarded"}},
            ),
        )
    recovered = live.overlay("Remy", source)
    assert recovered["snapshot"]["messages"][-1]["completed"]
    assert "not forwarded" not in str(recovered)
    native = {
        **source,
        "snapshot": {
            "messages": [
                {
                    "id": "native-reply",
                    "turn_id": "turn",
                    "role": "assistant",
                    "content": "Final",
                    "completed": True,
                    "chunks": [{"chunk_type": "content", "block_id": "text"}],
                },
            ],
        },
    }
    incomplete = {**native, "snapshot": {"messages": [{**native["snapshot"]["messages"][0], "chunks": []}]}}
    assert live.overlay("Remy", incomplete)["snapshot"]["messages"][-1]["content"] == "Partial"
    assert "Remy" in live.turns
    assert live.overlay("Remy", native) is native
    assert not live.turns


def test_cancellation_preserves_received_content_and_settles_only_the_live_turn():
    live = LiveTurns()
    with pytest.raises(asyncio.CancelledError):
        with live.observe("monitor", "session", "Hello") as publish:
            publish(StreamChunk(turn_id="turn", block_id="text", chunk="Partial"))
            raise asyncio.CancelledError()
    turn = live.turns["monitor"]
    assert turn.completed
    assert turn.messages()[1]["content"] == "Partial"
    assert turn.messages()[1]["chunks"][-1]["status"] == "interrupted"
    live.clear()
    assert not live.turns


@pytest.mark.asyncio
async def test_live_subscription_recovers_partial_output_without_polling_each_token(
    tmp_path,
):
    from .test_agent_runtime import running_runtime
    from reme.components.agent_runtime._internal.views import AgentViews
    from reme.schema import SessionSnapshot, SessionStatus
    from unittest.mock import AsyncMock

    async with running_runtime(tmp_path) as (runtime, _, wrapper):
        sid = await runtime.sessions.ensure("Remy", wrapper, {}, runtime.now())
        wrapper.read_session = AsyncMock(
            return_value=SessionSnapshot(session_id=sid, state=SessionStatus(session_id=sid, status="active")),
        )
        events = AgentViews(runtime).events("Remy", interval=60)
        try:
            assert (await anext(events))["type"] == "snapshot"
            with runtime.live_turns.observe("Remy", sid, "Hello") as publish:
                publish(StreamChunk(turn_id="live-turn", block_id="text", chunk="Partial"))
                user = await asyncio.wait_for(anext(events), 1)
                assert user["message"]["role"] == "user"
                reply = await asyncio.wait_for(anext(events), 1)
                assert reply["message"]["content"] == "Partial"
                assert wrapper.read_session.await_count == 1
                publish(StreamChunk(turn_id="live-turn", block_id="text", chunk=" output"))
                update = await asyncio.wait_for(anext(events), 1)
                assert update["message"]["content"] == "Partial output"
                assert wrapper.read_session.await_count == 1
                # Reading/reconnecting observes the same partial turn, without generating anything.
                restored = await runtime.read_agent("Remy")
                assert restored["snapshot"]["messages"][-1]["content"] == "Partial output"
                assert not wrapper.turns
        finally:
            await events.aclose()
        assert not runtime.live_turns.listeners
        assert "Remy" in runtime.live_turns.turns


@pytest.mark.asyncio
async def test_native_handoff_invalidates_every_subscribers_cached_snapshot(tmp_path):
    from .test_agent_runtime import running_runtime
    from reme.components.agent_runtime._internal.views import AgentViews
    from reme.schema import SessionSnapshot, SessionStatus, SessionMessage
    from unittest.mock import AsyncMock

    async with running_runtime(tmp_path) as (runtime, _, wrapper):
        sid = await runtime.sessions.ensure("Remy", wrapper, {}, runtime.now())
        snapshot = SessionSnapshot(session_id=sid, state=SessionStatus(session_id=sid, status="idle"))
        wrapper.read_session = AsyncMock(side_effect=lambda _: snapshot)
        first = AgentViews(runtime).events("Remy", interval=60)
        second = AgentViews(runtime).events("Remy", interval=60)
        try:
            await anext(first)
            await anext(second)
            with runtime.live_turns.observe("Remy", sid, "Hello") as publish:
                publish(StreamChunk(turn_id="turn", block_id="text", chunk="Final", event="end"))
                publish(StreamChunk(chunk_type="reply_end", event="end", status="completed", chunk={"result": "Final"}))
            snapshot.messages = [
                SessionMessage(
                    id="native",
                    turn_id="turn",
                    role="assistant",
                    content="Final",
                    chunks=[StreamChunk(block_id="text", chunk="Final", event="end")],
                ),
            ]
            await runtime.read_agent("Remy")
            assert not runtime.live_turns.turns
            for events in (first, second):
                update = await asyncio.wait_for(anext(events), 1)
                assert update["message"]["content"] == "Final"
                assert update["message"]["id"] == "native"
        finally:
            await first.aclose()
            await second.aclose()


def test_live_media_type_survives_a_null_block_end():
    turn = LiveTurn("session", "Show image")
    turn.update(
        StreamChunk(
            turn_id="turn",
            block_id="image",
            chunk_type="data",
            media_type="image/png",
            chunk="aGVsbG8=",
            event="start",
        ),
    )
    turn.update(StreamChunk(turn_id="turn", block_id="image", chunk_type="data", chunk=None, event="end"))
    image = turn.messages()[1]["chunks"][0]
    assert image["media_type"] == "image/png"
    assert image["chunk"] == "aGVsbG8="


@pytest.mark.parametrize("event", ["start", "delta"])
@pytest.mark.parametrize("chunk_type", ["content", "tool_result"])
def test_completed_live_blocks_ignore_late_starts_and_deltas(event, chunk_type):
    turn = LiveTurn("session", "Check")
    chunk = StreamChunk(
        turn_id="turn",
        block_id="block",
        tool_call_id="call" if chunk_type == "tool_result" else None,
        chunk_type=chunk_type,
        chunk="Final",
        event="end",
        status="completed",
    )
    turn.update(chunk)
    before = turn.messages()
    turn.update(chunk.model_copy(update={"chunk": "Stale", "event": event, "status": "running"}))
    assert turn.messages() == before


def test_tool_call_identity_wins_over_event_specific_block_ids():
    turn = LiveTurn("session", "Check")
    turn.update(
        StreamChunk(
            turn_id="turn",
            block_id="output",
            tool_call_id="call",
            chunk_type="tool_result",
            chunk="Final output",
            event="end",
        ),
    )
    before = turn.messages()
    for chunk_type in ("tool_call", "tool_result"):
        turn.update(
            StreamChunk(
                turn_id="turn",
                block_id="call",
                tool_call_id="call",
                chunk_type=chunk_type,
                chunk="Late",
                event="start",
            ),
        )
    assert turn.messages() == before


def test_native_handoff_waits_for_tool_results_even_if_the_call_id_is_present():
    live = LiveTurns()
    with live.observe("Remy", "session", "Check") as publish:
        publish(
            StreamChunk(
                turn_id="turn",
                chunk_type="tool_call",
                block_id="tool",
                tool_call_id="call",
                chunk={"command": "pwd"},
                event="start",
            ),
        )
        publish(
            StreamChunk(
                turn_id="turn",
                chunk_type="tool_result",
                block_id="result",
                tool_call_id="call",
                chunk="/workspace",
                event="end",
                status="completed",
            ),
        )
        publish(StreamChunk(chunk_type="reply_end", status="completed", chunk={"result": ""}, event="end"))
    call = {"chunk_type": "tool_call", "block_id": "tool", "tool_call_id": "call", "chunk": {"command": "pwd"}}
    value = {
        "session_id": "session",
        "agent": {"status": "idle"},
        "snapshot": {"messages": [{"turn_id": "turn", "role": "assistant", "completed": True, "chunks": [call]}]},
    }
    projected = live.overlay("Remy", value)
    assert "Remy" in live.turns
    assert projected["snapshot"]["messages"][-1]["chunks"][1]["chunk"] == "/workspace"
    value["snapshot"]["messages"][0]["chunks"] = [
        {"chunk_type": "tool_result", "block_id": "result", "tool_call_id": "call", "chunk": "/workspace"},
    ]
    assert live.overlay("Remy", value) is value
    assert not live.turns


def test_native_handoff_keeps_final_output_until_the_assistant_is_saved():
    live = LiveTurns()
    with live.observe("Remy", "session", "Hello") as publish:
        publish(StreamChunk(turn_id="turn", chunk_type="reply_end", status="completed", chunk={"result": "Final"}))
    value = {
        "session_id": "session",
        "agent": {"status": "idle"},
        "snapshot": {"messages": [{"turn_id": "turn", "role": "user", "completed": True}]},
    }
    live.overlay("Remy", value)
    assert "Remy" in live.turns
    value["snapshot"]["messages"].append(
        {"turn_id": "turn", "role": "assistant", "completed": True, "content": "Progress"},
    )
    live.overlay("Remy", value)
    assert "Remy" in live.turns
    value["snapshot"]["messages"][-1]["content"] = "Final"
    assert live.overlay("Remy", value) is value
