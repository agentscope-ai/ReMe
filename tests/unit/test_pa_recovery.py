"""Regression coverage for consumed inputs, durable results and rebuilt projections."""

# pylint: disable=missing-function-docstring,protected-access

import asyncio
from unittest.mock import AsyncMock, Mock

import frontmatter
import pytest

from reme.schema import SessionMessage
from reme.steps.proactive import ProactiveAgentStep, TempAgentStep
from reme.steps.proactive.tools import ReadConversationStep, ReadCronResultsStep

from .pa_helpers import bind_pa, configure, snapshot, task


@pytest.mark.asyncio
async def test_failed_consumption_keeps_sources_and_handoff_without_replaying_inputs_after_restart(pa_workspace):
    env = pa_workspace
    configure(env, [task()])
    result_path = snapshot(env, content="unique-cron-finding")
    await env.runtime.write_proactive_inbox("unique-feedback")
    source = env.runtime.proactive_inbox_path.read_bytes()
    await env.runtime.save_proactive_handoff("Previous unfinished work")
    env.snapshot.messages = [SessionMessage(id="user-1", role="user", content="unique-conversation")]
    options = []

    async def reply(_query, **kwargs):
        options.append(kwargs)
        binding = kwargs["injected_job_kwargs"]
        conversation = await ReadConversationStep(app_context=env.context)(**binding)
        cron = await ReadCronResultsStep(app_context=env.context)(**binding)
        feedback = await env.runtime.read_proactive_inbox(round_id=binding["pa_round_id"])
        if len(options) == 1:
            assert conversation.answer and cron.answer and feedback
            raise RuntimeError("failed after consuming inputs")
        assert not conversation.answer and not cron.answer and not feedback
        prompt = kwargs["system_prompt"]
        assert "Previous unfinished work" in prompt
        assert all(value not in prompt for value in ["unique-conversation", "unique-feedback", "unique-cron-finding"])
        await env.runtime.send_chat_message(
            "",
            kind="normal",
            round_id=binding["pa_round_id"],
            session_id=binding["session_id"],
        )
        return {"status": "completed", "result": ""}

    env.agent.reply = AsyncMock(side_effect=reply)
    with pytest.raises(RuntimeError, match="failed after consuming"):
        await ProactiveAgentStep(app_context=env.context)()
    assert env.runtime.proactive_handoff_path.read_text() == "Previous unfinished work\n"
    assert env.runtime.proactive_inbox_path.read_bytes() == source
    post = frontmatter.load(result_path)
    assert post.content == "unique-cron-finding" and post["status"] == "read"
    assert not (env.runtime.component_metadata_path / "proactive_context.jsonl").exists()
    env.runtime = bind_pa(env.context, env.agent)
    env.runtime.now = lambda: env.now[0]
    env.runtime._initialize_files()
    env.runtime.session_id = "session-1"
    assert env.runtime.proactive_handoff_path.read_text() == "Previous unfinished work\n"
    await ProactiveAgentStep(app_context=env.context)()
    assert env.runtime.proactive_handoff_path.read_text() == ""


@pytest.mark.asyncio
async def test_unused_legacy_recovery_is_ignored_and_preserved_when_preparation_fails(pa_workspace):
    env = pa_workspace
    journal = env.runtime.component_metadata_path / "proactive_context.jsonl"
    journal.write_bytes(b"Invalid legacy recovery that must remain untouched\n")
    original = journal.read_bytes()
    await env.runtime.save_proactive_handoff("Previous unfinished work")
    env.agent.create_session.side_effect = ConnectionError("offline")
    with pytest.raises(ConnectionError):
        await ProactiveAgentStep(app_context=env.context)()
    assert journal.read_bytes() == original
    assert env.runtime.proactive_handoff_path.read_text() == "Previous unfinished work\n"


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", [asyncio.CancelledError, OSError], ids=["cancelled", "temporary-outage"])
async def test_long_temporary_result_is_saved_before_delivery_and_recovered_without_model_reexecution(
    pa_workspace,
    monkeypatch,
    failure,
):
    env = pa_workspace
    temporary = await env.runtime.temp_tasks.create("research")
    content = "完整发现与依据。" * 2000 + "[来源](https://example.test/report)"
    env.agent.reply = AsyncMock(return_value={"session_id": "fork-1", "status": "completed", "result": content})
    submit = env.runtime.reminders.submit_immediate

    async def interrupted(*_args, **_kwargs):
        saved = env.runtime.temp_tasks.get(temporary.id)
        assert saved.result_path and saved.status == "sending"
        assert frontmatter.load(env.runtime.workspace_path / saved.result_path).content == content
        raise failure("delivery unavailable")

    monkeypatch.setattr(env.runtime.reminders, "submit_immediate", interrupted)
    if failure is asyncio.CancelledError:
        with pytest.raises(asyncio.CancelledError):
            await TempAgentStep(app_context=env.context)._run(temporary)
    else:
        await TempAgentStep(app_context=env.context)._run(temporary)
    saved = env.runtime.temp_tasks.get(temporary.id)
    assert saved.status == "sending"
    monkeypatch.setattr(env.runtime.reminders, "submit_immediate", submit)
    env.agent.reply = AsyncMock(side_effect=AssertionError("Do not repeat completed work"))
    await TempAgentStep(app_context=env.context)._run(saved)
    completed = env.runtime.temp_tasks.get(temporary.id)
    assert completed.status == "completed" and completed.error is None
    assert env.runtime.reminders.read()[0].content == content
    env.agent.reply.assert_not_awaited()


@pytest.mark.asyncio
async def test_revoked_temporary_worker_cannot_publish_its_late_result(pa_workspace):
    env = pa_workspace
    temporary = await env.runtime.temp_tasks.create("old request")

    async def reply(*_args, **_kwargs):
        await env.runtime.temp_tasks.manage(temporary.id, "update", query="new request")
        return {"session_id": "old-worker", "status": "completed", "result": "stale result"}

    env.agent.reply = AsyncMock(side_effect=reply)
    with pytest.raises(asyncio.CancelledError):
        await TempAgentStep(app_context=env.context)._run(temporary)
    assert not env.runtime.temp_tasks.document_path(temporary.id).exists()
    assert not env.runtime.reminders.read()


@pytest.mark.asyncio
async def test_reminder_projection_is_reused_and_rebuilt_after_external_replacement(pa_workspace, monkeypatch):
    env = pa_workspace
    temporary = await env.runtime.temp_tasks.create("work")
    await env.runtime.temp_tasks.patch(temporary.id, temporary.revision, status="running")
    await env.runtime.temp_tasks.send(temporary.id, temporary.revision, "first result")
    reminders = env.runtime.reminders
    reminders.read()
    rebuild = Mock(wraps=reminders._read_records)
    monkeypatch.setattr(reminders, "_read_records", rebuild)
    reminders.read()[0].content = "caller mutation"
    assert reminders.read()[0].content == "first result"
    rebuild.assert_not_called()
    record = reminders.read()[0].model_copy(update={"content": "external result"})
    env.runtime.files.write_text(env.runtime.chat_inbox_path, record.model_dump_json() + "\n")
    assert reminders.read()[0].content == "external result"
    reminders.read()
    rebuild.assert_called_once()
