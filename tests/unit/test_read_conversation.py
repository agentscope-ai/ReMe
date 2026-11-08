"""Incremental conversation rounds, bounded text and durable recovery before cursor commits."""

# pylint: disable=missing-function-docstring,protected-access

import json
from unittest.mock import AsyncMock

import pytest

from reme.schema import SessionMessage
from reme.config import resolve_app_config
from reme.components.job import BaseJob
from reme.steps.proactive import tools


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
    env.agent.read_session.assert_not_awaited()
    result = await env.runtime.read_conversation()
    assert isinstance(result, str) and "old question" in result and "new answer" in result
    assert "[user]" in result and "[assistant]" in result and "2026-11-08T12:00:00+08:00" in result
    assert "private" not in result and "not a chat round" not in result
    assert await env.runtime.read_conversation() == ""


@pytest.mark.asyncio
async def test_latest_round_limit_keeps_question_and_answer_together(pa_workspace):
    env = pa_workspace
    env.snapshot.messages = dialogue()
    result = await env.runtime.read_conversation(max_turns=1)
    assert "new question" in result and "new answer" in result
    assert "old question" not in result and "old answer" not in result


@pytest.mark.asyncio
async def test_default_limit_returns_ten_complete_rounds(pa_workspace):
    env = pa_workspace
    env.snapshot.messages = [
        SessionMessage(id=f"{role}-{index}", role=role, content=f"{role}-{index:02d}", turn_id=f"turn-{index}")
        for index in range(12)
        for role in ("user", "assistant")
    ]
    result = await env.runtime.read_conversation()
    assert result.count("[user]") == result.count("[assistant]") == 10
    assert "user-00" not in result and "user-01" not in result
    assert "user-02" in result and "assistant-11" in result


@pytest.mark.asyncio
async def test_total_budget_truncates_bodies_and_preserves_both_roles(pa_workspace):
    env = pa_workspace
    env.snapshot.messages = [
        SessionMessage(id="q", role="user", content="q" * 500),
        SessionMessage(id="a", role="assistant", content="a" * 500),
    ]
    result = await env.runtime.read_conversation(max_chars=300)
    assert len(result) <= 300 and "[user]" in result and "[assistant]" in result and "[已截断]" in result


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "kwargs",
    [{"max_turns": value} for value in (0, True, 101)] + [{"max_chars": 0}, {"max_message_chars": False}],
)
async def test_invalid_budgets_fail_before_reading(pa_workspace, kwargs):
    with pytest.raises(ValueError):
        await pa_workspace.runtime.read_conversation(**kwargs)
    pa_workspace.agent.read_session.assert_not_awaited()


@pytest.mark.asyncio
async def test_active_last_round_is_consumed_only_after_completion(pa_workspace):
    env = pa_workspace
    env.snapshot.messages = dialogue()
    env.snapshot.state.status = "active"
    first = await env.runtime.read_conversation()
    assert "old answer" in first and "new question" not in first
    env.snapshot.state.status = "idle"
    second = await env.runtime.read_conversation()
    assert "new question" in second and "new answer" in second and "old question" not in second


@pytest.mark.asyncio
async def test_native_failure_and_checkpoint_failure_preserve_cursors(pa_workspace, monkeypatch):
    env = pa_workspace
    env.snapshot.messages = dialogue()
    before = env.runtime.session_ids_path.read_bytes()
    env.agent.read_session.side_effect = OSError("unavailable")
    with pytest.raises(OSError, match="unavailable"):
        await env.runtime.read_conversation()
    assert env.runtime.session_ids_path.read_bytes() == before
    env.agent.read_session.side_effect = None
    monkeypatch.setattr(env.runtime, "checkpoint_proactive", AsyncMock(side_effect=OSError("disk full")))
    with pytest.raises(OSError, match="disk full"):
        await env.runtime.read_conversation(round_id="round")
    assert env.runtime.session_ids_path.read_bytes() == before


@pytest.mark.asyncio
async def test_consumed_input_is_recoverable_before_cursor_changes(pa_workspace):
    env = pa_workspace
    env.snapshot.messages = dialogue()
    async with env.runtime.proactive_round() as current:
        result = await env.runtime.read_conversation(round_id=current.id)
    entries = [json.loads(line) for line in env.runtime.proactive_context_path.read_text().splitlines()]
    assert entries[-1]["value"] == result and entries[-1]["kind"] == "conversation"
    assert json.loads(env.runtime.session_ids_path.read_text())[-1]["last_read_message_id"] == "a2"


@pytest.mark.asyncio
async def test_registered_conversation_tool_consumes_through_a_bound_job(pa_workspace):
    env = pa_workspace
    jobs = resolve_app_config(config="proactive", log_config=False)["jobs"]
    config = jobs["read_conversation"]
    assert not config["enable_serve"] and config["parameters"]["properties"]["max_turns"]["default"] == 10
    env.snapshot.messages = dialogue()
    job = BaseJob(name="read_conversation", app_context=env.context, **config)
    async with job, env.runtime.proactive_round() as current:
        options = {"pa_caller": "proactive", "pa_round_id": current.id}
        response = await job(**options)
        assert response.success and "new answer" in response.answer
        assert (await job(**options)).answer == ""
    entries = [json.loads(line) for line in env.runtime.proactive_context_path.read_text().splitlines()]
    assert entries[-1]["kind"] == "conversation" and "new answer" in entries[-1]["value"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "caller,round_id,scope",
    [
        ("chat", None, None),
        ("proactive", "wrong-round", None),
        ("proactive", None, []),
    ],
)
async def test_conversation_tool_checks_round_and_permissions(pa_workspace, caller, round_id, scope):
    env = pa_workspace
    async with env.runtime.proactive_round() as current:
        with pytest.raises((RuntimeError, PermissionError)):
            await tools.ReadConversationStep(app_context=env.context)(
                pa_caller=caller,
                pa_round_id=round_id or current.id,
                _allowed_paths=scope,
            )
    env.agent.read_session.assert_not_awaited()


@pytest.mark.asyncio
async def test_conversation_tool_rejects_a_closed_round(pa_workspace):
    env = pa_workspace
    with pytest.raises(RuntimeError, match="live proactive round"):
        await tools.ReadConversationStep(app_context=env.context)(pa_caller="proactive", pa_round_id="old-round")
    env.agent.read_session.assert_not_awaited()
