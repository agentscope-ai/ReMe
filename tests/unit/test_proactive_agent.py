"""One fork, awaited native compaction and one query with durable recovery."""

# pylint: disable=missing-function-docstring,protected-access

import asyncio
from unittest.mock import AsyncMock

import pytest

from reme.enumeration import ChunkEnum
from reme.schema import SessionMessage, StreamChunk
from reme.schema.pa_proactive import PaProactiveConfig
from reme.steps.proactive import ProactiveAgentStep, CronAgentStep, PaChatStep
from reme.steps.proactive.tools import ReadConversationStep, ReadProactiveInboxStep


def model(env, monkeypatch, *, fail_at=None, decide=True, adjust=False, pause=None, todo="# TODO\n\n无待办。"):
    calls = []
    env.operations = []
    monkeypatch.setattr(ProactiveAgentStep, "_now", lambda _step: env.now[0])

    async def create(**kwargs):
        env.operations.append(("create", kwargs))
        if fail_at == "prepare":
            raise RuntimeError("preparation failed")
        return f"worker-{env.runtime.current_round.id}"

    async def compact(session_id):
        env.operations.append(("compact", session_id))
        assert not env.runtime.current_round.allow_message
        if fail_at == "compact":
            raise RuntimeError("compaction failed")

    async def reply(query, **kwargs):
        calls.append((query, kwargs))
        env.operations.append(("query", kwargs["resume"]))
        bound = kwargs["injected_job_kwargs"]
        await ReadConversationStep(app_context=env.context)(**bound)
        await env.runtime.read_proactive_inbox(round_id=bound["pa_round_id"])
        if pause is not None:
            pause[0].set()
            await pause[1].wait()
        if fail_at == "query":
            raise RuntimeError("query failed")
        if env.runtime.current_round.allow_message and decide:
            await env.runtime.send_chat_message(
                "",
                kind="normal",
                round_id=bound["pa_round_id"],
                session_id=bound["session_id"],
            )
        if adjust:
            env.runtime.files.write_json(
                env.runtime.proactive_config_path,
                {"start_time": "0900", "end_time": "2300", "interval_minutes": 15},
            )
        if fail_at == "after_decision":
            raise RuntimeError("query failed")
        return {"session_id": kwargs["resume"], "status": "completed", "result": todo}

    env.agent.create_session = AsyncMock(side_effect=create)
    env.agent.compact_session = AsyncMock(side_effect=compact)
    env.agent.reply = AsyncMock(side_effect=reply)
    return calls


@pytest.mark.asyncio
@pytest.mark.parametrize("language", ["en", "zh"])
@pytest.mark.parametrize("selected_model", [None, "kimi-k3.1-pro"])
async def test_fork_overrides_then_compacts_before_the_only_query(pa_workspace, monkeypatch, language, selected_model):
    env = pa_workspace
    calls = model(env, monkeypatch, adjust=True)
    job_tools = ["read_conversation", "send_chat_message", "check_json"]
    step = ProactiveAgentStep(app_context=env.context, language=language, job_tools=job_tools, model=selected_model)
    result = await step()
    assert len(calls) == 1
    assert [name for name, _ in env.operations] == ["create", "compact", "query"]
    created = env.agent.create_session.await_args.kwargs
    assert created["resume"] == env.runtime.session_id and created["fork_session"] is True
    assert created["job_tools"] == calls[0][1]["job_tools"] == job_tools
    assert created["system_prompt"] == calls[0][1]["system_prompt"]
    assert "chat_context" not in created["system_prompt"]
    assert created["injected_job_kwargs"] == calls[0][1]["injected_job_kwargs"]
    assert created.get("model") == calls[0][1].get("model") == selected_model
    assert "model" not in env.agent.kwargs
    worker = calls[0][1]["resume"]
    assert worker != env.runtime.session_id and "fork_session" not in calls[0][1]
    env.agent.compact_session.assert_awaited_once_with(worker)
    env.agent.release_session.assert_awaited_once_with(worker)
    assert "read_conversation(max_turns=10)" in calls[0][0]
    assert (await env.runtime.get_proactive_config()).interval_minutes == 15
    assert result.metadata["proactive_session_id"] == worker
    assert env.runtime.proactive_context_path.read_text() == "" and env.runtime.current_round is None
    assert env.runtime.session_id == "session-1"


@pytest.mark.asyncio
@pytest.mark.parametrize("first", ["chat", "proactive"])
async def test_main_and_proactive_queries_can_progress_concurrently(pa_workspace, monkeypatch, first):
    env = pa_workspace
    entered = {name: asyncio.Event() for name in ("chat", "proactive")}
    release = {name: asyncio.Event() for name in entered}
    model(env, monkeypatch, pause=(entered["proactive"], release["proactive"]))

    async def reply_stream(_query, **kwargs):
        entered["chat"].set()
        await release["chat"].wait()
        yield StreamChunk(
            chunk_type=ChunkEnum.REPLY_END,
            status="completed",
            session_id=kwargs["resume"],
            chunk=env.agent._reply_result(session_id=kwargs["resume"], result="hello", raw={}),
        )

    env.agent.reply_stream = reply_stream
    operations = {
        "chat": lambda: PaChatStep(app_context=env.context)(query="hello", stream_queue=asyncio.Queue()),
        "proactive": ProactiveAgentStep(app_context=env.context),
    }
    second = "proactive" if first == "chat" else "chat"
    async with asyncio.TaskGroup() as group:
        try:
            first_task = group.create_task(operations[first]())
            await asyncio.wait_for(entered[first].wait(), 1)
            group.create_task(operations[second]())
            await asyncio.wait_for(entered[second].wait(), 1)
            assert not first_task.done()
        finally:
            for event in release.values():
                event.set()
    assert env.runtime.current_round is None and not env.runtime.session_lock.locked()


@pytest.mark.asyncio
async def test_compaction_is_awaited_before_any_work_query(pa_workspace, monkeypatch):
    env = pa_workspace
    model(env, monkeypatch)
    entered, release = asyncio.Event(), asyncio.Event()

    async def compact(_session_id):
        entered.set()
        await release.wait()

    env.agent.compact_session.side_effect = compact
    task = asyncio.create_task(ProactiveAgentStep(app_context=env.context)())
    await entered.wait()
    env.agent.reply.assert_not_awaited()
    release.set()
    await task
    env.agent.reply.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize("language", ["en", "zh"])
@pytest.mark.parametrize("prior_reminder", [False, True])
async def test_empty_main_starts_one_query_without_fork_or_compact(pa_workspace, monkeypatch, language, prior_reminder):
    env = pa_workspace
    env.snapshot.last_turn_id = None
    if prior_reminder:
        env.snapshot.messages = [
            SessionMessage(id="old", role="assistant", source="proactive", content="Earlier update."),
        ]
    calls = model(env, monkeypatch, decide=False)
    original = env.agent.reply.side_effect

    async def reply(query, **kwargs):
        current = env.runtime.current_round
        await env.runtime.get_chat_messages(round_id=current.id)
        await env.runtime.send_chat_message(
            "A useful new update.",
            kind="normal",
            round_id=current.id,
            session_id=current.session_id,
        )
        return await original(query, **kwargs)

    env.agent.reply.side_effect = reply
    result = await ProactiveAgentStep(app_context=env.context, language=language)()
    assert result.answer["status"] == "pending" and len(calls) == 1
    assert "fork_session" not in env.agent.create_session.await_args.kwargs
    assert "resume" not in env.agent.create_session.await_args.kwargs
    env.agent.compact_session.assert_not_awaited()
    assert env.runtime.session_id == "session-1"
    assert env.runtime.reminders.read()[0].content == "A useful new update."
    assert env.runtime.proactive_context_path.read_text() == ""


@pytest.mark.asyncio
async def test_unreadable_main_fails_before_native_work_and_preserves_recovery(pa_workspace, monkeypatch):
    env = pa_workspace
    model(env, monkeypatch)
    env.agent.read_session.side_effect = OSError("native history is unreadable")
    with pytest.raises(OSError, match="native history is unreadable"):
        await ProactiveAgentStep(app_context=env.context)()
    env.agent.create_session.assert_not_awaited()
    env.agent.reply.assert_not_awaited()
    assert '"operation": "prepare"' in env.runtime.proactive_context_path.read_text()
    assert env.runtime.current_round is None and not env.runtime._proactive_lock.locked()


@pytest.mark.asyncio
@pytest.mark.parametrize("fail_at", ["prepare", "compact"])
async def test_preparation_failure_never_runs_a_work_query(pa_workspace, monkeypatch, fail_at):
    env = pa_workspace
    model(env, monkeypatch, fail_at=fail_at)
    before = env.runtime.session_ids_path.read_bytes()
    with pytest.raises(RuntimeError, match="failed"):
        await ProactiveAgentStep(app_context=env.context)()
    env.agent.reply.assert_not_awaited()
    assert env.runtime.session_ids_path.read_bytes() == before
    assert f'"operation": "{fail_at}"' in env.runtime.proactive_context_path.read_text()
    if fail_at == "compact":
        env.agent.release_session.assert_awaited_once()


@pytest.mark.asyncio
async def test_single_query_decides_without_using_legacy_handoff(pa_workspace, monkeypatch):
    env = pa_workspace
    legacy = env.runtime.component_metadata_path / "handoff.txt"
    legacy.write_text("旧交接：必须立即发送，不要重新判断。", encoding="utf-8")
    original = legacy.read_bytes()
    calls = model(env, monkeypatch)
    result = await ProactiveAgentStep(app_context=env.context)()
    assert len(calls) == 1 and result.answer["status"] == "skipped"
    assert legacy.read_bytes() == original
    assert "旧交接" not in str(calls)
    env.agent.append_session.assert_not_awaited()
    with pytest.raises(ValueError, match="live, server-bound"):
        await env.runtime.send_chat_message("", "normal", round_id="stale", session_id="session-1")


@pytest.mark.asyncio
@pytest.mark.parametrize("fail_at", ["query", "after_decision"])
async def test_failed_query_keeps_consumed_feedback(pa_workspace, monkeypatch, fail_at):
    env = pa_workspace
    await env.runtime.write_proactive_inbox("只跟踪主人关注的新结果")
    calls = model(env, monkeypatch, fail_at=fail_at)
    with pytest.raises(RuntimeError, match="query failed"):
        await ProactiveAgentStep(app_context=env.context)()
    assert len(calls) == 1
    recovery = env.runtime.proactive_context_path.read_text()
    assert "只跟踪主人关注的新结果" in recovery and '"operation": "query"' in recovery
    assert env.runtime.current_round is None and not env.runtime._proactive_lock.locked()


@pytest.mark.asyncio
async def test_cancelled_query_revokes_send_permission_and_retains_inputs(pa_workspace, monkeypatch):
    env = pa_workspace
    entered, release = asyncio.Event(), asyncio.Event()
    model(env, monkeypatch, pause=(entered, release))
    task = asyncio.create_task(ProactiveAgentStep(app_context=env.context)())
    await entered.wait()
    current = env.runtime.current_round
    assert current.allow_message
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert not current.allow_message and not env.runtime.session_lock.locked()
    assert env.runtime.current_round is None and not env.runtime._proactive_lock.locked()
    assert "failure" in env.runtime.proactive_context_path.read_text()
    env.agent.release_session.assert_awaited_once()


@pytest.mark.asyncio
async def test_concurrent_wakeups_run_one_query_each_in_distinct_rounds(pa_workspace, monkeypatch):
    env = pa_workspace
    entered, release = asyncio.Event(), asyncio.Event()
    calls = model(env, monkeypatch, pause=(entered, release))
    first = asyncio.create_task(ProactiveAgentStep(app_context=env.context)())
    await entered.wait()
    second = asyncio.create_task(ProactiveAgentStep(app_context=env.context)())
    await asyncio.sleep(0)
    assert len(calls) == 1
    release.set()
    await asyncio.gather(first, second)
    assert len(calls) == 2
    assert calls[0][1]["injected_job_kwargs"]["pa_round_id"] != calls[1][1]["injected_job_kwargs"]["pa_round_id"]
    assert calls[0][1]["resume"] != calls[1][1]["resume"]


@pytest.mark.asyncio
async def test_failed_work_query_does_not_change_conversation_selection(pa_workspace):
    env = pa_workspace
    before = env.runtime.session_ids_path.read_bytes()
    env.agent.reply = AsyncMock(side_effect=RuntimeError("context failed"))
    with pytest.raises(RuntimeError, match="context failed"):
        await ProactiveAgentStep(app_context=env.context)()
    env.agent.read_session.assert_awaited_once_with(env.runtime.session_id)
    assert env.runtime.session_ids_path.read_bytes() == before


@pytest.mark.asyncio
async def test_missing_reminder_decision_is_not_reported_as_success(pa_workspace, monkeypatch):
    model(pa_workspace, monkeypatch, decide=False)
    with pytest.raises(RuntimeError, match="submit a send_chat_message decision"):
        await ProactiveAgentStep(app_context=pa_workspace.context)()


@pytest.mark.asyncio
async def test_outside_active_hours_never_starts_model(pa_workspace, monkeypatch):
    env = pa_workspace
    env.now[0] = env.now[0].replace(hour=23)
    model(env, monkeypatch)
    result = await ProactiveAgentStep(app_context=env.context)()
    assert result.answer["reason"] == "outside_active_hours"
    env.agent.reply.assert_not_awaited()


@pytest.mark.asyncio
async def test_feedback_checkpoint_failure_does_not_clear_inbox(pa_workspace, monkeypatch):
    env = pa_workspace
    await env.runtime.write_proactive_inbox("保留反馈")
    original = env.runtime.proactive_inbox_path.read_bytes()
    monkeypatch.setattr(env.runtime, "_checkpoint_context", lambda *_args: (_ for _ in ()).throw(OSError("disk full")))
    async with env.runtime.proactive_round() as current:
        with pytest.raises(OSError, match="disk full"):
            await ReadProactiveInboxStep(app_context=env.context)(pa_round_id=current.id)
    assert env.runtime.proactive_inbox_path.read_bytes() == original


@pytest.mark.asyncio
async def test_cron_agent_requires_a_complete_task_definition(pa_workspace):
    env = pa_workspace
    env.agent.reply = AsyncMock(return_value={"session_id": "task", "status": "completed", "result": "result"})
    with pytest.raises(ValueError):
        await CronAgentStep(app_context=env.context)(task={"query": "理解本地结果"})
    env.agent.reply.assert_not_awaited()


@pytest.mark.parametrize("minutes", [0, -1, True, "5", float("nan"), float("inf")])
def test_wakeup_interval_is_a_positive_finite_number(minutes):
    with pytest.raises(ValueError):
        PaProactiveConfig(start_time="0900", end_time="2300", interval_minutes=minutes)
