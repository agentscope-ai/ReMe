"""Reminder fork isolation, explicit delivery, consumption and cancellation boundaries."""

# pylint: disable=protected-access,missing-function-docstring,redefined-outer-name

import asyncio
from datetime import timedelta
from unittest.mock import AsyncMock

import pytest

from reme.components import RuntimeContext
from reme.schema import SessionMessage
from reme.steps.proactive.agent import ChatAgentStep, InboxChatStep, NewChatSessionStep
from reme.steps.proactive.tools.inbox import ReadInboxStep, SendChatMessageStep
from .test_agent_runtime import completed_result, create_agent, running_runtime


async def prepare(runtime, context, priority="P1"):
    await NewChatSessionStep(app_context=context)()
    await create_agent(runtime)
    result, session = completed_result(runtime)
    result.notifications[0].priority = priority
    await runtime.run("monitor", AsyncMock(return_value=(result, session)))
    await runtime.outbox.sync()
    runtime.notifications.last_activity_at -= timedelta(seconds=2000)
    return {"priority": priority, "msg_ids": list(runtime.notifications.pending)}


def append_message(session_id, content, *, message_id, **kwargs):
    return SessionMessage(
        id=message_id,
        role="assistant",
        content=content,
        source="proactive",
        origin_session_id=session_id,
        delivery=kwargs["delivery"],
    )


@pytest.mark.parametrize("content", ["有新的进展，实验已经完成。", "", "   "])
@pytest.mark.asyncio
async def test_fork_sends_only_explicit_content_and_consumes_decisions(tmp_path, content):
    async with running_runtime(tmp_path) as (runtime, context, wrapper):
        arguments = await prepare(runtime, context)
        source = (await runtime.sessions.current("Remy")).session_id
        activity = runtime.notifications.last_activity_at
        wrapper.append_session = AsyncMock(side_effect=append_message)
        captured = {}

        async def reply(_hint, **options):
            captured.update(options)
            assert not runtime.chat_lock.locked()
            bound = options["injected_job_kwargs"]
            result = await ReadInboxStep(app_context=context)(**bound)
            turn = runtime.notifications.current_turn(bound["_inbox_turn"])
            assert [item["id"] for item in turn.items] == arguments["msg_ids"]
            assert result.answer == {
                "items": [
                    {
                        "agent_name": "monitor",
                        "created_at": turn.items[0]["created_at"],
                        "message": "Experiment finished",
                        "paths": ["daily/experiment.md"],
                    },
                ],
            }
            assert list(runtime.notifications.pending) == arguments["msg_ids"]
            send = SendChatMessageStep(app_context=context)
            first = await send(content=content, **bound)
            second = await SendChatMessageStep(app_context=context)(content=content, **bound)
            assert first.answer == second.answer
            return {"status": "completed", "result": "Internal process and final reply", "session_id": "fork"}

        wrapper.reply = AsyncMock(side_effect=reply)
        response = await InboxChatStep(app_context=context)(**arguments)
        assert response.answer == {"sent": bool(content.strip())}
        assert captured["resume"] == source
        assert captured["fork_session"] and captured["ephemeral"] and captured["sandbox"] == "read-only"
        assert captured["job_tools"] == ["read_inbox", "send_chat_msg"]
        assert captured["config"]["web_search"] == "disabled"
        assert not runtime.notifications.pending
        assert runtime.notifications.active_turn is None
        assert runtime.notifications.last_activity_at == activity
        assert (await runtime.sessions.list("Remy"))["total"] == 1
        if content.strip():
            wrapper.append_session.assert_awaited_once()
            assert wrapper.append_session.call_args.args == (source, content.strip())
            assert wrapper.append_session.call_args.kwargs["role"] == "assistant"
            revision = (await wrapper.read_session(source)).revision
            assert wrapper.append_session.call_args.kwargs["expected_revision"] == revision
        else:
            wrapper.append_session.assert_not_awaited()
        with pytest.raises(ValueError, match="active reminder fork"):
            await ReadInboxStep(app_context=context)(**captured["injected_job_kwargs"])


@pytest.mark.parametrize(
    "supporting_paths,expected_paths",
    [
        ([], []),
        (["daily/experiment.md", "daily/experiment.md"], ["daily/experiment.md"]),
        (
            ["daily/other.md", "daily/experiment.md", "daily/other.md"],
            ["daily/other.md", "daily/experiment.md"],
        ),
    ],
)
@pytest.mark.asyncio
async def test_read_inbox_exposes_unique_supporting_paths_without_changing_internal_snapshot(
    tmp_path,
    supporting_paths,
    expected_paths,
):
    async with running_runtime(tmp_path) as (runtime, context, _):
        await NewChatSessionStep(app_context=context)()
        await create_agent(runtime)
        result, session = completed_result(runtime)
        reminder = result.notifications[0]
        reminder.paths = [reminder.paths[0], *supporting_paths, reminder.paths[0]]
        await runtime.run("monitor", AsyncMock(return_value=(result, session)))
        await runtime.outbox.sync()
        ids = list(runtime.notifications.pending)
        session_id = (await runtime.sessions.current("Remy")).session_id
        async with runtime.notifications.turn(session_id, "P0", ids) as turn:
            bound = {"_agent_name": "Remy", "_inbox_turn": turn.id}
            tool = ReadInboxStep(app_context=context)
            response = await tool(**bound)
            assert response.answer["items"][0]["paths"] == expected_paths
            assert turn.items[0]["paths"] == reminder.paths
            assert turn.items[0]["id"] == ids[0] and turn.items[0]["read_at"] is None
            response.answer["items"][0]["paths"].append("local-only.md")
            response.answer["items"][0]["message"] = "Local edit"
            reread = (await tool(**bound)).answer["items"][0]
            assert reread["paths"] == expected_paths and reread["message"] == reminder.message
            assert list(runtime.notifications.pending) == ids


@pytest.mark.parametrize("content", ["", "Old reminder"])
@pytest.mark.asyncio
async def test_native_content_changes_leave_stale_reminder_decisions_pending(tmp_path, content):
    async with running_runtime(tmp_path) as (runtime, context, wrapper):
        arguments = await prepare(runtime, context)
        snapshot = await wrapper.read_session((await runtime.sessions.current("Remy")).session_id)

        async def append(_session_id, _content, *, expected_revision, **_kwargs):
            if expected_revision != snapshot.revision:
                raise RuntimeError("Session changed before appending")
            raise AssertionError("A stale decision must not append")

        async def reply(_hint, **options):
            bound = options["injected_job_kwargs"]
            await ReadInboxStep(app_context=context)(**bound)
            snapshot.messages.append(SessionMessage(id="stop", role="user", content="Stop reminders"))
            await SendChatMessageStep(app_context=context)(content=content, **bound)

        wrapper.read_session = AsyncMock(return_value=snapshot)
        wrapper.append_session = AsyncMock(side_effect=append)
        wrapper.reply = AsyncMock(side_effect=reply)
        with pytest.raises(RuntimeError, match="changed"):
            await InboxChatStep(app_context=context)(**arguments)
        assert list(runtime.notifications.pending) == arguments["msg_ids"]


@pytest.mark.parametrize("stage", ["before_read", "after_read", "plain_reply", "append_failure"])
@pytest.mark.asyncio
async def test_failed_or_missing_decisions_preserve_messages_for_retry(tmp_path, stage):
    async with running_runtime(tmp_path) as (runtime, context, wrapper):
        arguments = await prepare(runtime, context)
        wrapper.append_session = AsyncMock(side_effect=OSError("append failed"))

        async def reply(_hint, **options):
            if stage == "before_read":
                raise OSError("fork failed")
            bound = options["injected_job_kwargs"]
            await ReadInboxStep(app_context=context)(**bound)
            if stage == "after_read":
                raise OSError("fork failed after reading")
            if stage == "append_failure":
                await SendChatMessageStep(app_context=context)(content="Update", **bound)
            return {"status": "completed", "result": "Ordinary text is not a reminder"}

        wrapper.reply = AsyncMock(side_effect=reply)
        with pytest.raises((OSError, RuntimeError)):
            await InboxChatStep(app_context=context)(**arguments)
        assert list(runtime.notifications.pending) == arguments["msg_ids"]
        assert runtime.notifications.active_turn is None
        if stage != "append_failure":
            wrapper.append_session.assert_not_awaited()


@pytest.mark.asyncio
async def test_fork_allows_foreground_chat_and_rejects_a_rotated_session(tmp_path):
    async with running_runtime(tmp_path) as (runtime, context, wrapper):
        arguments = await prepare(runtime, context)
        wrapper.append_session = AsyncMock(side_effect=append_message)

        async def reply(_hint, **options):
            bound = options["injected_job_kwargs"]
            await ReadInboxStep(app_context=context)(**bound)
            async with asyncio.timeout(1):
                response = await ChatAgentStep(app_context=context)(
                    RuntimeContext(stream_queue=asyncio.Queue()),
                    query="Another topic",
                )
            assert response.success
            await NewChatSessionStep(app_context=context)()
            await SendChatMessageStep(app_context=context)(content="Stale reminder", **bound)

        wrapper.reply = AsyncMock(side_effect=reply)
        with pytest.raises(RuntimeError, match="Chat session changed"):
            await InboxChatStep(app_context=context)(**arguments)
        wrapper.append_session.assert_not_awaited()
        assert list(runtime.notifications.pending) == arguments["msg_ids"]


@pytest.mark.asyncio
async def test_failed_session_read_releases_selected_inbox_claim(tmp_path):
    async with running_runtime(tmp_path) as (runtime, context, wrapper):
        arguments = await prepare(runtime, context)
        session = (await runtime.sessions.current("Remy")).session_id
        wrapper.read_session = AsyncMock(side_effect=OSError("native session unavailable"))
        with pytest.raises(OSError, match="unavailable"):
            async with runtime.notifications.turn(session, arguments["priority"], arguments["msg_ids"]):
                raise AssertionError("session selection must fail before inference")
        snapshot = await runtime.inbox.snapshot("Remy")
        assert snapshot["pending_count"] == 1 and snapshot["in_flight_count"] == 0
        assert runtime.notifications.active_turn is None


@pytest.mark.parametrize("method", ["read_session", "get_session_status", "session_exists"])
@pytest.mark.asyncio
async def test_inbox_timeout_covers_native_session_setup_and_releases_claim(tmp_path, method):
    async with running_runtime(tmp_path) as (runtime, context, wrapper):
        arguments = await prepare(runtime, context)

        async def unavailable(_session_id):
            await asyncio.Event().wait()

        setattr(wrapper, method, AsyncMock(side_effect=unavailable))
        with pytest.raises(TimeoutError):
            async with asyncio.timeout(1) as outer:
                await InboxChatStep(app_context=context, timeout_seconds=0.01)(**arguments)
        assert not outer.expired()
        snapshot = await runtime.inbox.snapshot("Remy")
        assert snapshot["pending_count"] == 1 and snapshot["in_flight_count"] == 0
        assert runtime.notifications.active_turn is None


@pytest.mark.asyncio
async def test_priority_wakeup_can_overtake_older_low_priority_reminders(tmp_path):
    async with running_runtime(tmp_path) as (runtime, context, wrapper):
        low = await prepare(runtime, context, "P2")
        await runtime.create({"agent_name": "urgent", "agent_type": "proactive", "task_memory": "Urgent observations"})
        await runtime.append(
            "urgent",
            {"id": "urgent", "event_ts": runtime.now(), "source": "test", "description": "Urgent"},
        )
        result, session = completed_result(runtime, "urgent")
        result.notifications[0].priority = "P0"
        await runtime.run("urgent", AsyncMock(return_value=(result, session)))
        await runtime.outbox.sync()
        urgent = [identifier for identifier in runtime.notifications.pending if identifier not in low["msg_ids"]]

        async def reply(_hint, **options):
            bound = options["injected_job_kwargs"]
            items = (await ReadInboxStep(app_context=context)(**bound)).answer["items"]
            assert [item["agent_name"] for item in items] == ["urgent"]
            turn = runtime.notifications.current_turn(bound["_inbox_turn"])
            assert [item["id"] for item in turn.items] == urgent
            await SendChatMessageStep(app_context=context)(content="", **bound)
            return {"status": "completed"}

        wrapper.reply = AsyncMock(side_effect=reply)
        await InboxChatStep(app_context=context)(priority="P0", msg_ids=urgent)
        assert list(runtime.notifications.pending) == low["msg_ids"]
        assert (await runtime.inbox.snapshot("Remy"))["pending_count"] == 1
        assert runtime.notifications.active_turn is None


@pytest.mark.asyncio
async def test_cancelled_fork_releases_scope_and_preserves_uncommitted_messages(tmp_path):
    async with running_runtime(tmp_path) as (runtime, context, wrapper):
        arguments = await prepare(runtime, context)
        reading = asyncio.Event()
        released = asyncio.Event()

        async def reply(_hint, **options):
            try:
                await ReadInboxStep(app_context=context)(**options["injected_job_kwargs"])
                reading.set()
                await asyncio.Event().wait()
            finally:
                released.set()

        wrapper.reply = AsyncMock(side_effect=reply)

        async def run_job(_name, **kwargs):
            return await InboxChatStep(app_context=context)(**kwargs)

        await runtime.notifications.tick(run_job)
        async with asyncio.timeout(1):
            await reading.wait()
            await runtime.notifications.close()
        assert released.is_set() and runtime.notifications.worker is None
        assert runtime.notifications.active_turn is None
        assert list(runtime.notifications.pending) == arguments["msg_ids"]


@pytest.mark.parametrize("priority, threshold", [("P0", 0), ("P1", 10), ("P2", 1800)])
@pytest.mark.asyncio
async def test_priority_idle_thresholds_and_busy_chat(tmp_path, priority, threshold):
    async with running_runtime(tmp_path) as (runtime, context, wrapper):
        arguments = await prepare(runtime, context, priority)
        now = runtime.now()
        runtime.now = lambda: now
        step = InboxChatStep(app_context=context)
        async with runtime.chat_lock:
            assert (await step(**arguments)).answer == {"skipped": "busy"}
        if threshold:
            runtime.notifications.last_activity_at = now - timedelta(seconds=threshold - 0.001)
            assert (await step(**arguments)).answer == {"skipped": "not_idle"}
        runtime.notifications.last_activity_at = now - timedelta(seconds=threshold)

        async def reply(_hint, **options):
            bound = options["injected_job_kwargs"]
            await ReadInboxStep(app_context=context)(**bound)
            await SendChatMessageStep(app_context=context)(content="Update", **bound)
            return {"status": "completed"}

        wrapper.reply = AsyncMock(side_effect=reply)
        wrapper.append_session = AsyncMock(side_effect=append_message)
        assert (await step(**arguments)).answer == {"sent": True}
        assert wrapper.append_session.call_args.kwargs["delivery"] == ("immediate" if priority == "P0" else "idle")


@pytest.mark.asyncio
async def test_tools_cannot_read_other_priorities_or_consume_later_arrivals(tmp_path):
    async with running_runtime(tmp_path) as (runtime, context, wrapper):
        arguments = await prepare(runtime, context)
        wrapper.append_session = AsyncMock(side_effect=append_message)

        async def reply(_hint, **options):
            bound = options["injected_job_kwargs"]
            with pytest.raises(ValueError, match="read_inbox"):
                await SendChatMessageStep(app_context=context)(content="Update", **bound)
            with pytest.raises(ValueError, match="accepts no arguments"):
                await ReadInboxStep(app_context=context)(priority="P2", **bound)
            await ReadInboxStep(app_context=context)(**bound)
            await runtime.create(
                {"agent_name": "other", "agent_type": "proactive", "task_memory": "Other observations"},
            )
            await runtime.append(
                "other",
                {"id": "new", "event_ts": runtime.now(), "source": "test", "description": "New"},
            )
            await runtime.run("other", AsyncMock(return_value=completed_result(runtime, "other")))
            await runtime.outbox.sync()
            await SendChatMessageStep(app_context=context)(content="", **bound)
            with pytest.raises(ValueError, match="already been submitted"):
                await SendChatMessageStep(app_context=context)(content="Different decision", **bound)
            return {"status": "completed"}

        wrapper.reply = AsyncMock(side_effect=reply)
        await InboxChatStep(app_context=context)(**arguments)
        assert len(runtime.notifications.pending) == 1
        assert not set(arguments["msg_ids"]) & runtime.notifications.pending.keys()
        wrapper.append_session.assert_not_awaited()


@pytest.mark.parametrize("change", ["disabled", "window_closed"])
@pytest.mark.asyncio
async def test_delivery_rechecks_sender_settings_after_the_fork_starts(tmp_path, change):
    async with running_runtime(tmp_path) as (runtime, context, wrapper):
        arguments = await prepare(runtime, context)
        wrapper.append_session = AsyncMock(side_effect=append_message)

        async def reply(_hint, **options):
            bound = options["injected_job_kwargs"]
            await ReadInboxStep(app_context=context)(**bound)
            if change == "disabled":
                await runtime.update("monitor", {"enabled": False})
            else:
                await runtime.update("monitor", {"notify_window": {"start": "09:00", "end": "10:00"}})
                now = runtime.now().replace(hour=10, minute=0, second=0)
                runtime.now = lambda: now
            await SendChatMessageStep(app_context=context)(content="Update", **bound)

        wrapper.reply = AsyncMock(side_effect=reply)
        with pytest.raises(RuntimeError, match="no longer eligible"):
            await InboxChatStep(app_context=context)(**arguments)
        wrapper.append_session.assert_not_awaited()
        assert list(runtime.notifications.pending) == arguments["msg_ids"]
        assert runtime.notifications.active_turn is None


@pytest.mark.asyncio
async def test_cancelling_delivery_waiting_for_foreground_chat_does_not_hang(tmp_path):
    async with running_runtime(tmp_path) as (runtime, context, wrapper):
        arguments = await prepare(runtime, context)
        sending = asyncio.Event()
        lock_held = asyncio.Event()
        release_lock = asyncio.Event()

        async def foreground():
            async with runtime.chat_lock:
                lock_held.set()
                await release_lock.wait()

        async def reply(_hint, **options):
            bound = options["injected_job_kwargs"]
            await ReadInboxStep(app_context=context)(**bound)
            await lock_held.wait()
            sending.set()
            await SendChatMessageStep(app_context=context)(content="Update", **bound)

        wrapper.reply = AsyncMock(side_effect=reply)
        wrapper.append_session = AsyncMock(side_effect=append_message)
        task = asyncio.create_task(InboxChatStep(app_context=context)(**arguments))
        while runtime.notifications.active_turn is None:
            await asyncio.sleep(0)
        foreground_task = asyncio.create_task(foreground())
        try:
            async with asyncio.timeout(1):
                await sending.wait()
                task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await task
            assert runtime.notifications.active_turn is None
            assert list(runtime.notifications.pending) == arguments["msg_ids"]
            wrapper.append_session.assert_not_awaited()
        finally:
            release_lock.set()
            await foreground_task


@pytest.mark.asyncio
async def test_codex_ephemeral_fork_matches_the_native_sdk_contract(monkeypatch):
    from types import SimpleNamespace
    from openai_codex import AsyncCodex
    from reme.components.agent_wrapper import CodexAgentWrapper

    options = CodexAgentWrapper()._thread_options({"fork_session": True, "ephemeral": True})
    assert options["ephemeral"] is True
    codex = AsyncCodex()
    transport = SimpleNamespace(
        thread_fork=AsyncMock(return_value=SimpleNamespace(thread=SimpleNamespace(id="fork-session"))),
    )
    monkeypatch.setattr(codex, "_ensure_initialized", AsyncMock())
    monkeypatch.setattr(codex, "_client", transport)

    fork = await codex.thread_fork("source-session", **options)

    assert fork.id == "fork-session"
    source, params = transport.thread_fork.call_args.args
    assert source == "source-session" and params.ephemeral is True
    assert params.exclude_turns is True


@pytest.mark.asyncio
@pytest.mark.parametrize("content", ["Outdated update", ""])
async def test_fork_rechecks_original_reminders_before_sending_or_consuming(tmp_path, content):
    async with running_runtime(tmp_path) as (runtime, context, wrapper):
        arguments = await prepare(runtime, context)
        wrapper.append_session = AsyncMock(side_effect=append_message)

        async def reply(_hint, **options):
            bound = options["injected_job_kwargs"]
            await ReadInboxStep(app_context=context)(**bound)
            ref = next(iter(runtime.notifications.pending.values()))
            path = runtime.files.path(ref.file)
            runtime.files.write(path, runtime.files.read(path).replace("Experiment finished", "Experiment cancelled"))
            await SendChatMessageStep(app_context=context)(content=content, **bound)

        wrapper.reply = AsyncMock(side_effect=reply)
        with pytest.raises(RuntimeError, match="reminders changed"):
            await InboxChatStep(app_context=context)(**arguments)
        wrapper.append_session.assert_not_awaited()
        assert list(runtime.notifications.pending) == arguments["msg_ids"]
