"""Incremental conversation rounds, bounded text and durable recovery before cursor commits."""

# pylint: disable=missing-function-docstring,protected-access

import json
from unittest.mock import AsyncMock

import pytest

from reme.schema import SessionMessage
from reme.steps.proactive.tools import ReadConversationStep


def dialogue():
    return [
        SessionMessage(id="q1", role="user", content="old question", turn_id="turn-1"),
        SessionMessage(id="a1", role="assistant", content="old answer", turn_id="turn-1"),
        SessionMessage(id="q2", role="user", content="new question", turn_id="turn-2"),
        SessionMessage(
            id="a2",
            role="assistant",
            content="new answer",
            turn_id="turn-2",
            created_at="2026-11-08T12:00:00+08:00",
            chunks=[{"chunk_type": "think", "chunk": "private"}],
        ),
        SessionMessage(id="reminder", role="assistant", source="proactive", content="not a chat round"),
    ]


@pytest.mark.asyncio
async def test_reads_completed_text_once_and_never_returns_tools_or_proactive_messages(pa_workspace):
    env = pa_workspace
    env.snapshot.messages = dialogue()
    step = ReadConversationStep(app_context=env.context)
    env.agent.read_session.assert_not_awaited()
    result = (await step()).answer
    assert isinstance(result, str) and "old question" in result and "new answer" in result
    assert "[user]" in result and "[assistant]" in result and "2026-11-08T12:00:00+08:00" in result
    assert "private" not in result and "not a chat round" not in result
    assert (await ReadConversationStep(app_context=env.context)()).answer == ""


@pytest.mark.asyncio
async def test_latest_round_limit_keeps_question_and_answer_together(pa_workspace):
    env = pa_workspace
    env.snapshot.messages = dialogue()
    result = (await ReadConversationStep(app_context=env.context)(max_turns=1)).answer
    assert "new question" in result and "new answer" in result
    assert "old question" not in result and "old answer" not in result


@pytest.mark.asyncio
async def test_total_budget_truncates_bodies_and_preserves_both_roles(pa_workspace):
    env = pa_workspace
    env.snapshot.messages = [
        SessionMessage(id="q", role="user", content="q" * 500),
        SessionMessage(id="a", role="assistant", content="a" * 500),
    ]
    result = (await ReadConversationStep(app_context=env.context)(max_chars=300)).answer
    assert len(result) <= 300 and "[user]" in result and "[assistant]" in result and "[已截断]" in result


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "kwargs",
    [{"max_turns": value} for value in (0, True, 101)] + [{"max_chars": 0}, {"max_message_chars": False}],
)
async def test_invalid_budgets_fail_before_reading(pa_workspace, kwargs):
    with pytest.raises(ValueError):
        await ReadConversationStep(app_context=pa_workspace.context)(**kwargs)
    pa_workspace.agent.read_session.assert_not_awaited()


@pytest.mark.asyncio
async def test_active_last_round_is_consumed_only_after_completion(pa_workspace):
    env = pa_workspace
    env.snapshot.messages = dialogue()
    env.snapshot.state.status = "active"
    first = (await ReadConversationStep(app_context=env.context)()).answer
    assert "old answer" in first and "new question" not in first
    env.snapshot.state.status = "idle"
    second = (await ReadConversationStep(app_context=env.context)()).answer
    assert "new question" in second and "new answer" in second and "old question" not in second


@pytest.mark.asyncio
async def test_native_failure_and_checkpoint_failure_preserve_cursors(pa_workspace, monkeypatch):
    env = pa_workspace
    env.snapshot.messages = dialogue()
    before = env.runtime.session_ids_path.read_bytes()
    env.agent.read_session.side_effect = OSError("unavailable")
    with pytest.raises(OSError, match="unavailable"):
        await ReadConversationStep(app_context=env.context)()
    assert env.runtime.session_ids_path.read_bytes() == before
    env.agent.read_session.side_effect = None
    monkeypatch.setattr(env.runtime, "checkpoint_proactive", AsyncMock(side_effect=OSError("disk full")))
    with pytest.raises(OSError, match="disk full"):
        await ReadConversationStep(app_context=env.context)(pa_round_id="round")
    assert env.runtime.session_ids_path.read_bytes() == before


@pytest.mark.asyncio
async def test_consumed_input_is_recoverable_before_cursor_changes(pa_workspace):
    env = pa_workspace
    env.snapshot.messages = dialogue()
    async with env.runtime.proactive_round() as current:
        result = (await ReadConversationStep(app_context=env.context)(pa_round_id=current.id)).answer
    entries = [json.loads(line) for line in env.runtime.proactive_context_path.read_text().splitlines()]
    assert entries[-1]["value"] == result and entries[-1]["kind"] == "conversation"
    assert json.loads(env.runtime.session_ids_path.read_text())[-1]["last_read_message_id"] == "a2"


@pytest.mark.asyncio
async def test_path_restrictions_and_empty_history(pa_workspace):
    env = pa_workspace
    with pytest.raises(PermissionError):
        await ReadConversationStep(app_context=env.context)(_allowed_paths=[])
    env.agent.read_session.assert_not_awaited()
    assert (await ReadConversationStep(app_context=env.context)()).answer == ""
