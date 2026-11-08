"""Regression coverage for information preserved across agent stream adapters."""

# pylint: disable=missing-function-docstring,protected-access

import asyncio
from types import SimpleNamespace

import pytest
from agentscope.event import DataBlockDeltaEvent, ReplyEndEvent, TextBlockEndEvent, TextBlockStartEvent
from claude_agent_sdk import AssistantMessage, ResultMessage, StreamEvent, TextBlock
from openai_codex.generated.v2_all import (
    AgentMessageThreadItem,
    ItemCompletedNotification,
    PlanThreadItem,
    ReasoningThreadItem,
    WebSearchThreadItem,
)

from reme.components.agent_wrapper.as_agent_wrapper import AsAgentWrapper
from reme.components.agent_wrapper.cc_agent_wrapper import CcAgentWrapper
from reme.components.agent_wrapper.codex_agent_wrapper import CodexAgentWrapper
from reme.components.agent_wrapper.codex_session_history import item_chunks
from reme.components.application_context import ApplicationContext
from reme.components.client.http_client import HttpClient
from reme.components.runtime_context import RuntimeContext
from reme.enumeration import ChunkEnum
from reme.schema import StreamChunk
from reme.steps.common.chat import ChatStep
from reme.utils.common_utils import call_action


@pytest.mark.parametrize(
    "item",
    [
        AgentMessageThreadItem(type="agentMessage", id="answer", text="corrected", phase="final_answer"),
        PlanThreadItem(type="plan", id="plan", text="updated plan"),
        ReasoningThreadItem(type="reasoning", id="thought", summary=["one", "two"], content=["raw"]),
        WebSearchThreadItem(type="webSearch", id="search", query="ReMe"),
    ],
)
def test_codex_completed_items_match_history_and_retain_final_snapshots(item):
    payload = ItemCompletedNotification(threadId="session", turnId="turn", item=item, completedAtMs=0)
    chunks = CodexAgentWrapper._event_to_chunks(SimpleNamespace(method="item/completed", payload=payload), "session")
    assert chunks == item_chunks(item, session_id="session", turn_id="turn")
    assert chunks and all(chunk.event == "end" and chunk.turn_id == "turn" for chunk in chunks)
    if item.type == "agentMessage":
        assert chunks[0].chunk == "corrected"
        assert chunks[0].metadata["phase"] == "final_answer"
    elif item.type == "webSearch":
        assert chunks[0].tool_call_id == "search"
        assert chunks[0].status == "completed"
        assert chunks[0].chunk["query"] == "ReMe"
    elif item.type == "reasoning":
        assert [chunk.block_id for chunk in chunks] == ["thought:summary:0", "thought:summary:1", "thought:content:0"]
        assert [chunk.chunk for chunk in chunks] == ["one", "two", "raw"]


def test_codex_reasoning_deltas_correlate_with_their_final_sections():
    chunks = []
    for method, index in [("item/reasoning/summaryTextDelta", 1), ("item/reasoning/textDelta", 0)]:
        payload = SimpleNamespace(
            item_id="thought",
            turn_id="turn",
            delta="text",
            summary_index=index,
            content_index=index,
        )
        chunks.extend(CodexAgentWrapper._event_to_chunks(SimpleNamespace(method=method, payload=payload), "session"))
    assert [chunk.block_id for chunk in chunks] == ["thought:summary:1", "thought:content:0"]
    assert chunks[0].metadata["kind"] != chunks[1].metadata["kind"]


def test_agentscope_block_boundaries_final_correction_and_url_data():
    start = AsAgentWrapper._event_to_chunk(TextBlockStartEvent(reply_id="reply", block_id="block"))
    end = AsAgentWrapper._event_to_chunk(TextBlockEndEvent(reply_id="reply", block_id="block", text="corrected"))
    url = AsAgentWrapper._event_to_chunk(
        DataBlockDeltaEvent(
            reply_id="reply",
            block_id="image",
            media_type="image/png",
            url="https://example.com/a.png",
        ),
    )
    assert start.event == "start" and start.chunk is None
    assert end.event == "end" and end.chunk == "corrected"
    assert end.turn_id == "reply"
    assert url.chunk == {"url": "https://example.com/a.png"}
    assert url.media_type == "image/png"
    interrupted = AsAgentWrapper._event_to_chunk(
        ReplyEndEvent(session_id="session", reply_id="reply", finished_reason="interrupted"),
    )
    assert interrupted.status == "interrupted"
    assert interrupted.metadata["finished_reason"] == "interrupted"


def test_claude_indices_are_scoped_by_message_and_signatures_are_retained():
    states = {}
    chunks = [
        CcAgentWrapper._raw_event_to_chunk(
            {"type": "content_block_start", "index": index, "content_block": {"type": "text", "text": ""}},
            message_id="message",
            block_states=states,
        )
        for index in [0, 1]
    ]
    assert [chunk.block_id for chunk in chunks] == ["message:0", "message:1"]
    signature = CcAgentWrapper._raw_event_to_chunk(
        {"type": "content_block_delta", "index": 1, "delta": {"type": "signature_delta", "signature": "signed"}},
        message_id="message",
        block_states=states,
    )
    assert signature.chunk["delta"]["signature"] == "signed"
    assert signature.chunk_type == ChunkEnum.DATA
    unknown = CcAgentWrapper._raw_event_to_chunk(
        {"type": "content_block_start", "index": 2, "content_block": {"type": "future_type", "value": 1}},
        message_id="message",
        block_states=states,
    )
    assert unknown.chunk["content_block"]["value"] == 1


@pytest.mark.asyncio
async def test_claude_multiple_model_messages_have_one_invocation_end(tmp_path, monkeypatch):
    async def query(**_kwargs):
        for message_id, text in [("first", "checking"), ("second", "answer")]:
            for event in [
                {"type": "message_start", "message": {"id": message_id}},
                {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
                {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": text}},
                {"type": "content_block_stop", "index": 0},
                {"type": "message_stop"},
            ]:
                yield StreamEvent(uuid=message_id, session_id="session", event=event)
            yield AssistantMessage(content=[TextBlock(text=text + "!")], model="claude", message_id=message_id)
        yield ResultMessage(
            subtype="success",
            duration_ms=1,
            duration_api_ms=1,
            is_error=False,
            num_turns=2,
            session_id="session",
        )

    monkeypatch.setattr("claude_agent_sdk.query", query)
    wrapper = CcAgentWrapper(app_context=ApplicationContext(workspace_dir=str(tmp_path)))
    chunks = [chunk async for chunk in wrapper.reply_stream("hello")]
    assert sum(chunk.chunk_type == ChunkEnum.REPLY_START for chunk in chunks) == 1
    assert sum(chunk.chunk_type == ChunkEnum.REPLY_END for chunk in chunks) == 1
    assert chunks[-1].chunk_type == ChunkEnum.REPLY_END
    assert chunks[-1].status == "completed"
    assert len({chunk.turn_id for chunk in chunks}) == 1
    snapshots = [
        chunk
        for chunk in chunks
        if chunk.chunk_type == ChunkEnum.CONTENT and chunk.event == "end" and chunk.chunk is not None
    ]
    assert [(chunk.block_id, chunk.chunk) for chunk in snapshots] == [("first:0", "checking!"), ("second:0", "answer!")]


@pytest.mark.asyncio
async def test_claude_interleaved_child_events_keep_their_own_block_state(tmp_path, monkeypatch):
    async def query(**_kwargs):
        for parent, event in [
            (None, {"type": "message_start", "message": {"id": "parent"}}),
            (None, {"type": "content_block_start", "index": 0, "content_block": {"type": "text"}}),
            ("tool", {"type": "message_start", "message": {"id": "child"}}),
            ("tool", {"type": "content_block_start", "index": 0, "content_block": {"type": "text"}}),
            (None, {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "parent"}}),
            ("tool", {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "child"}}),
        ]:
            yield StreamEvent(uuid="event", session_id="session", parent_tool_use_id=parent, event=event)
        yield ResultMessage(
            subtype="success",
            duration_ms=1,
            duration_api_ms=1,
            is_error=False,
            num_turns=1,
            session_id="session",
        )

    monkeypatch.setattr("claude_agent_sdk.query", query)
    wrapper = CcAgentWrapper(app_context=ApplicationContext(workspace_dir=str(tmp_path)))
    chunks = [chunk async for chunk in wrapper.reply_stream("hello")]
    deltas = [chunk for chunk in chunks if chunk.chunk_type == ChunkEnum.CONTENT and chunk.event == "delta"]
    assert [(chunk.block_id, chunk.chunk) for chunk in deltas] == [("parent:0", "parent"), ("child:0", "child")]
    assert deltas[1].metadata["parent_tool_use_id"] == "tool"


@pytest.mark.asyncio
async def test_chat_aggregates_snapshots_without_duplicate_text_and_records_terminal_failure(tmp_path, monkeypatch):
    async def reply_stream(_self, _inputs, **_kwargs):
        for event in [
            StreamChunk(block_id="answer", chunk="draft"),
            StreamChunk(block_id="answer", event="end", chunk="corrected"),
            StreamChunk(block_id="answer", event="end", chunk="corrected"),
            StreamChunk(block_id="removed", chunk="discard this"),
            StreamChunk(block_id="removed", event="end", chunk=""),
            StreamChunk(
                chunk_type=ChunkEnum.REPLY_END,
                event="end",
                status="failed",
                metadata={"error": "upstream failed"},
                chunk={
                    key: value
                    for key, value in _self._reply_result(
                        session_id="session",
                        result="corrected",
                        raw={},
                        status="failed",
                        error="upstream failed",
                    ).items()
                    if key != "events"
                },
            ),
        ]:
            yield event

    monkeypatch.setattr(AsAgentWrapper, "reply_stream", reply_stream)
    context = ApplicationContext(workspace_dir=str(tmp_path))
    agent = AsAgentWrapper(as_llm="", app_context=context)
    runtime = RuntimeContext(query="hello", stream_queue=asyncio.Queue())
    response = await ChatStep(agent_wrapper=agent, app_context=context)(runtime)
    assert response.answer == "corrected"
    assert not response.success
    assert runtime.stream_error == "upstream failed"


@pytest.mark.asyncio
async def test_http_text_consumers_reconcile_snapshots_and_skip_null_or_diagnostic_events(monkeypatch):
    async def stream(_self, _action, _payload):
        for chunk in [
            StreamChunk(block_id="answer", chunk="draft"),
            StreamChunk(block_id="answer", event="end", chunk="corrected"),
            StreamChunk(block_id="answer", event="end", chunk="corrected"),
            StreamChunk(block_id="answer", event="end", chunk=None),
            StreamChunk(chunk_type=ChunkEnum.DATA, chunk={"internal": True}, metadata={"visibility": "diagnostic"}),
        ]:
            yield chunk

    monkeypatch.setattr(HttpClient, "_iter_stream_chunks", stream)
    assert await call_action("chat") == "corrected"
    display = "".join([text async for text in HttpClient()._execute("chat", {})])
    assert display == "draft\ncorrected"


@pytest.mark.asyncio
async def test_agentscope_closing_stream_records_reported_usage_and_closes_sdk(tmp_path, monkeypatch):
    from agentscope.event import ModelCallEndEvent
    from unittest.mock import AsyncMock
    from reme.utils import global_counter_get_all

    closed = []

    async def native_stream(_inputs, **_kwargs):
        try:
            yield ModelCallEndEvent(reply_id="reply", input_tokens=3, output_tokens=2)
            await asyncio.Event().wait()
        finally:
            closed.append(True)

    agent = SimpleNamespace(state=SimpleNamespace(session_id="session"), reply_stream=native_stream)
    wrapper = AsAgentWrapper(app_context=ApplicationContext(workspace_dir=str(tmp_path)), as_llm="")
    monkeypatch.setattr(wrapper, "_build_agent", AsyncMock(return_value=(agent, "hello")))
    monkeypatch.setattr(wrapper, "_dump_state", AsyncMock())
    stream = wrapper.reply_stream("hello")
    await anext(stream)
    await stream.aclose()
    assert closed == [True]
    counters = global_counter_get_all(wrapper.app_context.metadata, ["__token_counter", wrapper.name])
    assert counters["children"]["total_tokens"]["value"] == 5
