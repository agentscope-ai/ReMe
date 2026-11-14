"""AgentScope state persistence must follow JSONL record boundaries."""

from pathlib import Path

import pytest
from agentscope.message import AssistantMsg, TextBlock, ToolCallBlock, UserMsg
from agentscope.state import AgentState

from reme.utils import AsStateHandler


@pytest.mark.asyncio
@pytest.mark.parametrize("separator", ["ordinary", "\n", "\r\n", "\u0085", "\u2028", "\u2029"])
@pytest.mark.parametrize("field", ["summary", "context"])
async def test_state_round_trip_preserves_text_separators(tmp_path: Path, separator: str, field: str):
    """Valid Unicode JSON string content never creates an extra JSONL record."""
    text = f"before{separator}after"
    state = AgentState(
        session_id="unicode-session",
        reply_id="saved-reply",
        cur_iter=3,
        summary=text if field == "summary" else "ordinary summary",
        context=[UserMsg(name="user", content=text if field == "context" else "ordinary message")],
    )
    handler = AsStateHandler.for_session(tmp_path, state.session_id)

    await handler.dump(state)
    restored = await handler.load()

    assert restored.model_dump() == state.model_dump()


@pytest.mark.asyncio
@pytest.mark.parametrize("newline", ["\n", "\r\n"])
async def test_state_load_accepts_lf_and_crlf_records(tmp_path: Path, newline: str):
    """Existing LF and CRLF files retain their header and context records."""
    state = AgentState(session_id="existing-session", summary="summary", context=[UserMsg(name="user", content="text")])
    handler = AsStateHandler.for_session(tmp_path, state.session_id)
    await handler.dump(state)
    content = handler.path.read_text(encoding="utf-8")
    handler.path.write_bytes(content.replace("\n", newline).encode("utf-8"))

    assert (await handler.load()).model_dump() == state.model_dump()


@pytest.mark.asyncio
async def test_state_load_accepts_empty_file(tmp_path: Path):
    """An empty state file continues to load as a fresh AgentState."""
    handler = AsStateHandler.for_session(tmp_path, "empty-session")
    handler.path.write_text("", encoding="utf-8")

    restored = await handler.load()
    assert restored.summary == ""
    assert restored.context == []
    assert restored.cur_iter == 0
    assert restored.session_id


@pytest.mark.asyncio
async def test_state_round_trip_preserves_sdk_resume_context(tmp_path):
    """Parked replies, middleware state and activated tools survive restart."""
    state = AgentState(session_id="state-context")
    state.reply_context.structured_schema = {"type": "object", "required": ["ok"]}
    state.reply_context.structured_output = {"ok": True}
    state.tool_context.activated_groups = ["group"]
    state.middle_context = {"probe": {"round": 2}}
    state.context.append(
        AssistantMsg(
            name="agent",
            id=state.reply_id,
            content=[ToolCallBlock(id="tool", name="example", input="{}", state="asking")],
        ),
    )
    handler = AsStateHandler.for_session(tmp_path, state.session_id)
    await handler.dump(state)
    restored = await handler.load()
    assert restored.model_dump(mode="json") == state.model_dump(mode="json")
    assert restored.has_awaiting_tool_calls("agent")


@pytest.mark.asyncio
async def test_legacy_header_remains_readable(tmp_path):
    """Previously saved summary/context files load without rewriting user data."""
    header = UserMsg(
        name="__state__",
        content="summary",
        metadata={"session_id": "legacy", "reply_id": "reply", "cur_iter": 2},
    )
    message = UserMsg(name="user", content="before\u2028after")
    handler = AsStateHandler.for_session(tmp_path, "legacy")
    original = header.model_dump_json() + "\n" + message.model_dump_json() + "\n"
    handler.path.write_text(original, encoding="utf-8")
    restored = await handler.load()
    assert restored.summary == "summary" and restored.reply_id == "reply" and restored.cur_iter == 2
    assert restored.context == [message]
    assert handler.path.read_text(encoding="utf-8") == original


@pytest.mark.asyncio
@pytest.mark.parametrize("summary", [[], [TextBlock(text="summary", id="summary-block")]])
async def test_state_preserves_text_only_summary_blocks(tmp_path, summary):
    """A native block-list summary keeps its type and IDs, including when empty."""
    state = AgentState(session_id="block-summary", summary=summary)
    handler = AsStateHandler.for_session(tmp_path, state.session_id)
    await handler.dump(state)
    assert (await handler.load()).model_dump(mode="json") == state.model_dump(mode="json")
