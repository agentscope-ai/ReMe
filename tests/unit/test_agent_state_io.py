"""AgentScope state persistence must follow JSONL record boundaries."""

from pathlib import Path

import pytest
from agentscope.message import UserMsg
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
