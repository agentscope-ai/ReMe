"""Portable outcomes retain native result details across agent backends."""

# pylint: disable=redefined-outer-name

import asyncio
import json
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import AsyncMock

from agentscope.message import AssistantMsg, Usage
from agentscope.event import ReplyStartEvent, ReplyEndEvent
from agentscope.model import StructuredResponse
from agentscope.model._model_usage import ChatUsage
from agentscope.types import ErrorInfo
from claude_agent_sdk import DeferredToolUse, ResultMessage
import pytest
from openai_codex.generated.v2_all import AgentMessageThreadItem, ItemCompletedNotification

from reme.components.agent_wrapper import AsAgentWrapper, CcAgentWrapper
from reme.components.application_context import ApplicationContext
from reme.utils import global_counter_get_all

from .test_codex_agent_wrapper import _Turn, _usage_event, system as _codex_system

system = _codex_system


def _agentscope(tmp_path, monkeypatch, message, model=None):
    """Use real SDK results with mocked model and persistence boundaries."""
    wrapper = AsAgentWrapper(app_context=ApplicationContext(workspace_dir=str(tmp_path)), as_llm="")

    async def reply_stream(_inputs, **_kwargs):
        yield ReplyStartEvent(name="agent", session_id="session-1", reply_id="reply-1")
        if message.finished_reason is not None:
            yield ReplyEndEvent(session_id="session-1", reply_id="reply-1", finished_reason=message.finished_reason)
        yield message

    agent = SimpleNamespace(
        state=SimpleNamespace(session_id="session-1", context=[message]),
        reply_stream=reply_stream,
    )
    wrapper.as_llm = SimpleNamespace(model=model)
    monkeypatch.setattr(wrapper, "_build_agent", AsyncMock(return_value=(agent, "hello")))
    monkeypatch.setattr(wrapper, "_dump_state", AsyncMock())
    return wrapper


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("reason", "status"),
    [
        ("completed", "completed"),
        ("error", "failed"),
        ("exceed_max_iters", "failed"),
        ("interrupted", "interrupted"),
        (None, "waiting"),
    ],
)
async def test_agentscope_outcome_preserves_native_message(tmp_path, monkeypatch, reason, status):
    """Native termination, content and cache counters survive portable normalization."""
    message = AssistantMsg(
        name="agent",
        content="final answer",
        finished_reason=reason,
        usage=Usage(input_tokens=10, output_tokens=2, cache_input_tokens=7),
    )
    if reason == "error":
        message.error = ErrorInfo(type="upstream", message="provider unavailable")
    model = SimpleNamespace(generate_structured_output=AsyncMock(return_value=StructuredResponse(content={})))
    wrapper = _agentscope(tmp_path, monkeypatch, message, model)
    result = await wrapper.reply("hello", **({"output_schema": {}} if status != "completed" else {}))

    assert set(result) == {"session_id", "result", "structured_output", "usage", "status", "error", "raw", "events"}
    assert result["status"] == status
    assert result["result"] == "final answer"
    assert result["raw"]["message"] == message.model_dump(mode="json")
    assert result["usage"] == {"input_tokens": 10, "output_tokens": 2, "total_tokens": 12}
    assert result["raw"]["message"]["usage"]["cache_input_tokens"] == 7
    assert result["error"] == (
        "provider unavailable"
        if reason == "error"
        else "AgentScope reply ended: exceed_max_iters" if reason == "exceed_max_iters" else None
    )
    model.generate_structured_output.assert_not_awaited()
    await _assert_roundtrip(result)


@pytest.mark.asyncio
@pytest.mark.parametrize("interrupted", [False, True])
async def test_agentscope_preserves_extra_structured_response(tmp_path, monkeypatch, interrupted):
    """The extra model call keeps its metadata and usage without overwriting the reply."""
    message = AssistantMsg(
        name="agent",
        content="answer",
        finished_reason="completed",
        usage=Usage(input_tokens=10, output_tokens=2),
    )
    structured = StructuredResponse(
        content={},
        usage=ChatUsage(input_tokens=3, output_tokens=1, time=0.0, cache_input_tokens=2),
        metadata={"provider": "extra details"},
        finished_reason="interrupted" if interrupted else "completed",
    )
    model = SimpleNamespace(generate_structured_output=AsyncMock(return_value=structured))
    wrapper = _agentscope(tmp_path, monkeypatch, message, model)
    result = await wrapper.reply("hello", output_schema={})

    assert result["status"] == ("interrupted" if interrupted else "completed")
    assert result["structured_output"] == {}
    assert result["usage"] == {"input_tokens": 13, "output_tokens": 3, "total_tokens": 16}
    assert result["raw"]["message"] == message.model_dump(mode="json")
    assert result["raw"]["structured_response"]["metadata"] == structured.metadata
    assert result["raw"]["structured_response"]["usage"]["cache_input_tokens"] == 2
    await _assert_roundtrip(result)


@pytest.mark.asyncio
async def test_agentscope_propagates_cancellation_even_if_sdk_returns_a_message(tmp_path, monkeypatch):
    """An SDK interruption fallback cannot turn caller cancellation into a normal return."""
    message = AssistantMsg(
        name="agent",
        content="interrupted",
        finished_reason="interrupted",
        usage=Usage(input_tokens=10, output_tokens=2),
    )
    wrapper = _agentscope(tmp_path, monkeypatch, message)
    agent = wrapper._build_agent.return_value[0]  # pylint: disable=protected-access
    entered = asyncio.Event()

    async def native_reply(_inputs, **_kwargs):
        entered.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            yield ReplyEndEvent(session_id="session-1", reply_id="reply-1", finished_reason="interrupted")
            yield message

    agent.reply_stream = native_reply
    task = asyncio.create_task(wrapper.reply("hello"))
    await asyncio.wait_for(entered.wait(), timeout=2)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    wrapper._dump_state.assert_awaited_once_with(agent.state)  # pylint: disable=protected-access
    counters = global_counter_get_all(wrapper.app_context.metadata, ["__token_counter", wrapper.name])
    assert counters["children"]["total_tokens"]["value"] == 12


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("overrides", "status"),
    [
        ({}, "completed"),
        ({"is_error": True, "subtype": "error_max_turns", "errors": ["turn limit"]}, "failed"),
        ({"terminal_reason": "aborted_tools"}, "interrupted"),
        ({"deferred_tool_use": DeferredToolUse(id="tool-1", name="external", input={})}, "waiting"),
    ],
)
async def test_claude_outcome_preserves_native_result(tmp_path, monkeypatch, overrides, status):
    """Fees, cache usage, permission denials and terminal reasons survive the common shape."""
    message = ResultMessage(
        subtype="success",
        duration_ms=20,
        duration_api_ms=10,
        is_error=False,
        num_turns=2,
        session_id="session-1",
        result="answer",
        structured_output=False,
        total_cost_usd=0.01,
        usage={"input_tokens": 10, "output_tokens": 2, "cache_read_input_tokens": 7},
        permission_denials=[{"tool_name": "Write", "tool_use_id": "tool-1"}],
    )
    message = replace(message, **overrides)

    async def query(**_kwargs):
        yield message
        if message.is_error:
            raise RuntimeError("Claude Code returned an error result: turn limit")

    monkeypatch.setattr("claude_agent_sdk.query", query)
    wrapper = CcAgentWrapper(app_context=ApplicationContext(workspace_dir=str(tmp_path)))
    result = await wrapper.reply("hello")

    assert set(result) == {"session_id", "result", "structured_output", "usage", "status", "error", "raw", "events"}
    assert result["status"] == status
    assert result["structured_output"] is False
    assert result["error"] == ("turn limit" if status == "failed" else None)
    assert result["raw"]["total_cost_usd"] == 0.01
    assert result["raw"]["usage"]["cache_read_input_tokens"] == 7
    assert result["raw"]["permission_denials"] == message.permission_denials
    assert result["raw"]["terminal_reason"] == message.terminal_reason
    await _assert_roundtrip(result)


@pytest.mark.asyncio
@pytest.mark.parametrize("cancelled", [False, True])
async def test_claude_does_not_hide_transport_errors_or_cancellation(tmp_path, monkeypatch, cancelled):
    """An unrelated failure after a result is still an exception, including cancellation."""

    async def query(**_kwargs):
        yield ResultMessage(
            subtype="error_during_execution",
            duration_ms=1,
            duration_api_ms=1,
            is_error=True,
            num_turns=1,
            session_id="session-1",
            errors=["native error"],
        )
        if cancelled:
            raise asyncio.CancelledError()
        raise RuntimeError("transport disconnected")

    monkeypatch.setattr("claude_agent_sdk.query", query)
    wrapper = CcAgentWrapper(app_context=ApplicationContext(workspace_dir=str(tmp_path)))
    with pytest.raises(asyncio.CancelledError if cancelled else RuntimeError, match=None if cancelled else "transport"):
        await wrapper.reply("hello")


@pytest.mark.asyncio
@pytest.mark.parametrize("status", ["completed", "failed", "interrupted"])
async def test_reply_preserves_native_outcomes_items_and_usage(system, monkeypatch, status):
    """Failed and interrupted turns retain completed items and reported usage."""
    from openai_codex.generated.v2_all import CommandExecutionThreadItem, TurnError, TurnStatus

    first, _, _, _ = system
    original = _Turn.stream
    command = CommandExecutionThreadItem(
        type="commandExecution",
        id="command-1",
        command="pwd",
        command_actions=[],
        source="agent",
        cwd="/tmp",
        status="completed",
        aggregated_output="/tmp",
        exit_code=0,
    )

    async def stream(turn):
        yield _usage_event((10, 2), (10, 2), turn_id=turn.id)
        yield SimpleNamespace(
            method="item/completed",
            payload=ItemCompletedNotification(threadId=turn.thread_id, turnId=turn.id, item=command, completedAtMs=0),
        )
        async for event in original(turn):
            if event.method == "turn/completed":
                event.payload.turn.status = TurnStatus(status)
                if status == "failed":
                    event.payload.turn.error = TurnError(message="native failure", additional_details="details")
            yield event

    monkeypatch.setattr(_Turn, "stream", stream)
    result = await first.reply("answer")

    assert set(result) == {"session_id", "result", "structured_output", "usage", "status", "error", "raw", "events"}
    assert result["status"] == status
    assert result["result"] == "answer"
    assert result["usage"] == {"input_tokens": 10, "output_tokens": 2, "total_tokens": 12}
    assert result["error"] == ("native failure" if status == "failed" else None)
    assert result["raw"]["status"] == status
    assert result["raw"]["items"][0]["aggregatedOutput"] == "/tmp"
    assert result["raw"]["usage"]["total"]["inputTokens"] == 10
    if status == "failed":
        assert result["raw"]["error"]["additionalDetails"] == "details"
    await _assert_roundtrip(result)


@pytest.mark.asyncio
async def test_reply_selects_final_answer_and_preserves_other_messages(system, monkeypatch):
    """Codex message phases select final text without discarding other messages."""
    first, _, _, _ = system
    original = _Turn.stream

    async def stream(turn):
        for index, phase in enumerate((None, "final_answer", "commentary")):
            yield SimpleNamespace(
                method="item/completed",
                payload=ItemCompletedNotification(
                    threadId=turn.thread_id,
                    turnId=turn.id,
                    completedAtMs=0,
                    item=AgentMessageThreadItem(type="agentMessage", id=f"item-{index}", text=str(phase), phase=phase),
                ),
            )
        async for event in original(turn):
            yield event

    monkeypatch.setattr(_Turn, "stream", stream)
    result = await first.reply("legacy fallback")

    assert result["result"] == "final_answer"
    assert len(result["raw"]["items"]) == 4


@pytest.mark.asyncio
async def test_invalid_structured_reply_retains_native_result(system):
    """A JSON parsing failure preserves the provider's successful native turn."""
    first, _, _, _ = system
    result = await first.reply("invalid JSON", output_schema={"type": "object"})

    assert result["status"] == "failed"
    assert "invalid JSON" in result["error"]
    assert result["result"] == "invalid JSON"
    assert result["raw"]["status"] == "completed"
    assert result["raw"]["final_response"] == "invalid JSON"


async def _assert_roundtrip(reply):
    """JSON storage, replay and collection preserve every event and outcome field."""
    from reme.schema import collect_reply, replay_reply

    stored = json.loads(json.dumps(reply))
    replayed = [chunk async for chunk in replay_reply(stored)]
    assert [chunk.model_dump(mode="json") for chunk in replayed] == stored["events"]

    async def stream():
        for chunk in replayed:
            yield chunk

    assert await collect_reply(stream()) == stored
    replayed[0].metadata["mutated"] = True
    assert "mutated" not in stored["events"][0]["metadata"]


@pytest.mark.asyncio
async def test_claude_preserves_unprojected_native_messages_and_final_outcome(tmp_path, monkeypatch):
    """Unknown blocks, signatures and tool results stay available beyond the display projection."""
    from claude_agent_sdk import AssistantMessage, SystemMessage, ThinkingBlock, ToolResultBlock, UserMessage

    native = [
        SystemMessage(subtype="future_event", data={"future": [1, False, None]}),
        AssistantMessage(content=[ThinkingBlock(thinking="check", signature="signature")], model="claude"),
        UserMessage(content=[ToolResultBlock(tool_use_id="hidden", content=[{"type": "text", "text": "result"}])]),
        ResultMessage(
            subtype="success",
            duration_ms=1,
            duration_api_ms=1,
            is_error=False,
            num_turns=1,
            session_id="session",
            result="final",
            structured_output=[],
        ),
    ]
    calls = []

    async def query(**kwargs):
        calls.append(kwargs)
        for message in native:
            yield message

    monkeypatch.setattr("claude_agent_sdk.query", query)
    wrapper = CcAgentWrapper(app_context=ApplicationContext(workspace_dir=str(tmp_path)))
    reply = await wrapper.reply("hello", output_schema={})
    assert len(calls) == 1
    assert [event["metadata"]["native"] for event in reply["events"] if "native" in event["metadata"]] == [
        wrapper._serialize(message) for message in native  # pylint: disable=protected-access
    ]
    assert reply["structured_output"] == []
    await _assert_roundtrip(reply)


@pytest.mark.asyncio
async def test_collector_preserves_empty_snapshots_unknown_fields_and_transport_marker():
    """The trace keeps block operations and future extension fields exactly, including DONE."""
    from reme.enumeration import ChunkEnum
    from reme.schema import StreamChunk, collect_reply

    events = [
        StreamChunk(chunk="draft", block_id="text", metadata={"native": {"signature": "signed"}}),
        StreamChunk(chunk="", event="end", block_id="text", future_field={"x": [False, None]}),
        StreamChunk(chunk=None, event="end", block_id="text"),
        StreamChunk(
            chunk_type=ChunkEnum.REPLY_END,
            event="end",
            status="waiting",
            chunk={
                "session_id": "session",
                "result": "",
                "structured_output": False,
                "usage": None,
                "status": "waiting",
                "error": None,
                "raw": {"future": 1},
            },
        ),
        StreamChunk(chunk_type=ChunkEnum.DONE, done=True),
    ]

    async def stream():
        for event in events:
            yield event

    reply = await collect_reply(stream())
    assert reply["events"] == [event.model_dump(mode="json") for event in events]
    await _assert_roundtrip(reply)


@pytest.mark.asyncio
@pytest.mark.parametrize("duplicate", [False, True])
async def test_collector_rejects_missing_or_duplicate_outcomes(duplicate):
    """Transport termination cannot manufacture a successful reply."""
    from reme.enumeration import ChunkEnum
    from reme.schema import StreamChunk, collect_reply

    async def stream():
        if duplicate:
            for _ in range(2):
                yield StreamChunk(chunk_type=ChunkEnum.REPLY_END, chunk={"result": "answer"})
        yield StreamChunk(chunk_type=ChunkEnum.DONE, done=True)

    with pytest.raises(ValueError, match="reply_end"):
        await collect_reply(stream())


@pytest.mark.asyncio
async def test_codex_terminal_snapshot_supplies_items_missing_from_notifications(system, monkeypatch):
    """The final native turn can complete or correct the item notifications."""
    first, _, _, _ = system
    original = _Turn.stream

    async def stream(turn):
        async for event in original(turn):
            if event.method == "item/completed":
                continue
            if event.method == "turn/completed":
                event.payload.turn.items[0].root.text = "authoritative final answer"
            yield event

    monkeypatch.setattr(_Turn, "stream", stream)
    reply = await first.reply("draft")
    assert reply["result"] == "authoritative final answer"
    assert reply["raw"]["items"][0]["text"] == "authoritative final answer"
    await _assert_roundtrip(reply)
