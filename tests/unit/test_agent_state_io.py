"""AgentScope state persistence must follow JSONL record boundaries."""

from pathlib import Path
import json

import pytest
from agentscope.message import AssistantMsg, TextBlock, ToolCallBlock, UserMsg
from agentscope.state import AgentState

from reme.utils import AsStateHandler
from reme.utils.agent_state_io import AsSessionData
from reme.schema import SessionMessage


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
async def test_state_load_rejects_empty_file(tmp_path: Path):
    """A truncated snapshot must not silently become an empty conversation."""
    handler = AsStateHandler.for_session(tmp_path, "empty-session")
    handler.path.write_text("", encoding="utf-8")

    with pytest.raises(ValueError):
        await handler.load()
    assert handler.path.read_bytes() == b""


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
@pytest.mark.parametrize("summary", [[], [TextBlock(text="summary", id="summary-block")]])
async def test_state_preserves_text_only_summary_blocks(tmp_path, summary):
    """A native block-list summary keeps its type and IDs, including when empty."""
    state = AgentState(session_id="block-summary", summary=summary)
    handler = AsStateHandler.for_session(tmp_path, state.session_id)
    await handler.dump(state)
    assert (await handler.load()).model_dump(mode="json") == state.model_dump(mode="json")


@pytest.mark.asyncio
async def test_display_history_appends_only_changed_messages_and_keeps_sdk_snapshot_small(tmp_path):
    """Long display history never gets reserialized into each new native snapshot."""
    handler = AsStateHandler.for_session(tmp_path, "history")
    session = AsSessionData(
        AgentState(session_id="history"),
        [SessionMessage(id=str(index), role="user", content="old-history-" + "x" * 1000) for index in range(100)],
    )
    await handler.dump_session(session)
    before = handler.history_path().read_bytes()
    reloaded = await handler.load_session()
    reloaded.messages.append(SessionMessage(id="next", role="assistant", content="new answer"))
    await handler.dump_session(reloaded)
    after = handler.history_path().read_bytes()
    assert after.startswith(before) and len(after) - len(before) < 1000
    assert "old-history-" not in handler.path.read_text(encoding="utf-8")
    assert (await handler.load_session()).messages == reloaded.messages
    reloaded.messages[-1].content = "corrected answer"
    await handler.dump_session(reloaded)
    assert (await handler.load_session()).messages == reloaded.messages


@pytest.mark.asyncio
async def test_failed_snapshot_publication_does_not_commit_history_and_retry_recovers(tmp_path, monkeypatch):
    """A crash between history fsync and state replacement leaves the old prefix authoritative."""
    handler = AsStateHandler.for_session(tmp_path, "crash")
    session = AsSessionData(AgentState(session_id="crash"), [SessionMessage(id="one", role="user", content="saved")])
    await handler.dump_session(session)
    before = handler.path.read_bytes()
    session.messages.append(SessionMessage(id="two", role="assistant", content="retry"))
    original = handler._write  # pylint: disable=protected-access

    def fail(_content):
        raise OSError("publication failed")

    monkeypatch.setattr(handler, "_write", fail)
    with pytest.raises(OSError, match="publication failed"):
        await handler.dump_session(session)
    assert handler.path.read_bytes() == before
    restored = await handler.load_session()
    assert [message.content for message in restored.messages] == ["saved"]
    monkeypatch.setattr(handler, "_write", original)
    restored.messages.append(session.messages[-1])
    await handler.dump_session(restored)
    assert (await handler.load_session()).messages == session.messages
    assert handler.history_path().read_text().count('"content": "retry"') == 1


@pytest.mark.asyncio
async def test_forks_write_independent_display_histories(tmp_path):
    """A fork writes its own paired history without changing the source."""
    handler = AsStateHandler.for_session(tmp_path, "source")
    message = SessionMessage(id="one", role="user", content="source")
    await handler.dump_session(AsSessionData(AgentState(session_id="source"), [message]))
    session = await handler.load_session()
    assert session.messages == [message]
    fork = AsStateHandler.for_session(tmp_path, "fork")
    session.state.session_id = "fork"
    await fork.dump_session(session)
    session.messages[0].content = "forked"
    await fork.dump_session(session)
    assert (await handler.load_session()).messages == [message]
    assert (await fork.load_session()).messages[0].content == "forked"


@pytest.mark.asyncio
async def test_committed_history_corruption_fails_without_rewriting_source(tmp_path):
    """Only bytes beyond the committed prefix may be discarded by a writer."""
    handler = AsStateHandler.for_session(tmp_path, "corrupt")
    await handler.dump_session(AsSessionData(AgentState(session_id="corrupt")))
    history = handler.history_path()
    history.write_bytes(b"{broken}\n")
    original = history.read_bytes()
    with pytest.raises((ValueError, json.JSONDecodeError)):
        await handler.load_session()
    assert history.read_bytes() == original


@pytest.mark.asyncio
async def test_history_symlinks_cannot_redirect_session_writes(tmp_path):
    """Paired logs keep the same containment boundary as native snapshots."""
    directory = tmp_path / "sessions"
    directory.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (directory / "history").symlink_to(outside, target_is_directory=True)
    handler = AsStateHandler.for_session(directory, "safe")
    with pytest.raises(ValueError, match="history path"):
        await handler.dump_session(AsSessionData(AgentState(session_id="safe")))
    assert not list(outside.iterdir()) and not handler.path.exists()
