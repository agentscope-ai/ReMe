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
    runtime.inbox.last_activity_at -= timedelta(seconds=2000)
    return {"priority": priority, "msg_ids": list(runtime.inbox.pending)}


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
        source = (await runtime.sessions.current("chat")).session_id
        activity = runtime.inbox.last_activity_at
        wrapper.append_session = AsyncMock(side_effect=append_message)
        captured = {}

        async def reply(_hint, **options):
            captured.update(options)
            assert not runtime.chat_lock.locked()
            bound = options["injected_job_kwargs"]
            result = await ReadInboxStep(app_context=context)(priority="P1", **bound)
            assert [item["id"] for item in result.answer["items"]] == arguments["msg_ids"]
            assert list(runtime.inbox.pending) == arguments["msg_ids"]
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
        assert not runtime.inbox.pending
        assert runtime.inbox.active_turn is None
        assert runtime.inbox.last_activity_at == activity
        assert (await runtime.sessions.list("chat"))["total"] == 1
        if content.strip():
            wrapper.append_session.assert_awaited_once()
            assert wrapper.append_session.call_args.args == (source, content.strip())
            assert wrapper.append_session.call_args.kwargs["role"] == "assistant"
        else:
            wrapper.append_session.assert_not_awaited()
        with pytest.raises(ValueError, match="active reminder fork"):
            await ReadInboxStep(app_context=context)(priority="P1", **captured["injected_job_kwargs"])


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
            await ReadInboxStep(app_context=context)(priority="P1", **bound)
            if stage == "after_read":
                raise OSError("fork failed after reading")
            if stage == "append_failure":
                await SendChatMessageStep(app_context=context)(content="Update", **bound)
            return {"status": "completed", "result": "Ordinary text is not a reminder"}

        wrapper.reply = AsyncMock(side_effect=reply)
        with pytest.raises((OSError, RuntimeError)):
            await InboxChatStep(app_context=context)(**arguments)
        assert list(runtime.inbox.pending) == arguments["msg_ids"]
        assert runtime.inbox.active_turn is None
        if stage != "append_failure":
            wrapper.append_session.assert_not_awaited()


@pytest.mark.asyncio
async def test_fork_allows_foreground_chat_and_rejects_a_rotated_session(tmp_path):
    async with running_runtime(tmp_path) as (runtime, context, wrapper):
        arguments = await prepare(runtime, context)
        wrapper.append_session = AsyncMock(side_effect=append_message)

        async def reply(_hint, **options):
            bound = options["injected_job_kwargs"]
            await ReadInboxStep(app_context=context)(priority="P1", **bound)
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
        assert list(runtime.inbox.pending) == arguments["msg_ids"]
        assert runtime.inbox.active_turn is None


@pytest.mark.asyncio
async def test_cancelled_fork_releases_scope_and_preserves_uncommitted_messages(tmp_path):
    async with running_runtime(tmp_path) as (runtime, context, wrapper):
        arguments = await prepare(runtime, context)
        reading = asyncio.Event()
        released = asyncio.Event()

        async def reply(_hint, **options):
            try:
                await ReadInboxStep(app_context=context)(priority="P1", **options["injected_job_kwargs"])
                reading.set()
                await asyncio.Event().wait()
            finally:
                released.set()

        wrapper.reply = AsyncMock(side_effect=reply)

        async def run_job(_name, **kwargs):
            return await InboxChatStep(app_context=context)(**kwargs)

        await runtime.inbox.tick(run_job, "inbox_chat")
        async with asyncio.timeout(1):
            await reading.wait()
            await runtime.inbox.close()
        assert released.is_set() and runtime.inbox.worker is None
        assert runtime.inbox.active_turn is None
        assert list(runtime.inbox.pending) == arguments["msg_ids"]


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
            runtime.inbox.last_activity_at = now - timedelta(seconds=threshold - 0.001)
            assert (await step(**arguments)).answer == {"skipped": "not_idle"}
        runtime.inbox.last_activity_at = now - timedelta(seconds=threshold)

        async def reply(_hint, **options):
            bound = options["injected_job_kwargs"]
            await ReadInboxStep(app_context=context)(priority=priority, **bound)
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
            with pytest.raises(ValueError, match="priority"):
                await ReadInboxStep(app_context=context)(priority="P2", **bound)
            await ReadInboxStep(app_context=context)(priority="P1", **bound)
            await runtime.create({"name": "other", "trigger_type": "event", "description": "Other observations"})
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
        assert len(runtime.inbox.pending) == 1
        assert not set(arguments["msg_ids"]) & runtime.inbox.pending.keys()
        wrapper.append_session.assert_not_awaited()


@pytest.mark.parametrize("change", ["disabled", "window_closed"])
@pytest.mark.asyncio
async def test_delivery_rechecks_sender_settings_after_the_fork_starts(tmp_path, change):
    async with running_runtime(tmp_path) as (runtime, context, wrapper):
        arguments = await prepare(runtime, context)
        wrapper.append_session = AsyncMock(side_effect=append_message)

        async def reply(_hint, **options):
            bound = options["injected_job_kwargs"]
            await ReadInboxStep(app_context=context)(priority="P1", **bound)
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
        assert list(runtime.inbox.pending) == arguments["msg_ids"]
        assert runtime.inbox.active_turn is None


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
            await ReadInboxStep(app_context=context)(priority="P1", **bound)
            await lock_held.wait()
            sending.set()
            await SendChatMessageStep(app_context=context)(content="Update", **bound)

        wrapper.reply = AsyncMock(side_effect=reply)
        wrapper.append_session = AsyncMock(side_effect=append_message)
        task = asyncio.create_task(InboxChatStep(app_context=context)(**arguments))
        while runtime.inbox.active_turn is None:
            await asyncio.sleep(0)
        foreground_task = asyncio.create_task(foreground())
        try:
            async with asyncio.timeout(1):
                await sending.wait()
                task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await task
            assert runtime.inbox.active_turn is None
            assert list(runtime.inbox.pending) == arguments["msg_ids"]
            wrapper.append_session.assert_not_awaited()
        finally:
            release_lock.set()
            await foreground_task


def test_codex_ephemeral_fork_uses_the_native_paginated_protocol():
    import inspect
    from openai_codex import AsyncCodex
    from reme.components.agent_wrapper import CodexAgentWrapper

    options = CodexAgentWrapper()._thread_options({"fork_session": True, "ephemeral": True})
    assert options["include_turns"] is False
    assert options["ephemeral"] is True
    inspect.signature(AsyncCodex.thread_fork).bind(None, "source-session", **options)


@pytest.mark.asyncio
@pytest.mark.parametrize("content", ["Outdated update", ""])
async def test_fork_rechecks_original_reminders_before_sending_or_consuming(tmp_path, content):
    async with running_runtime(tmp_path) as (runtime, context, wrapper):
        arguments = await prepare(runtime, context)
        wrapper.append_session = AsyncMock(side_effect=append_message)

        async def reply(_hint, **options):
            bound = options["injected_job_kwargs"]
            await ReadInboxStep(app_context=context)(priority="P1", **bound)
            ref = next(iter(runtime.inbox.pending.values()))
            path = runtime.files.path(ref.file)
            runtime.files.write(path, runtime.files.read(path).replace("Experiment finished", "Experiment cancelled"))
            await SendChatMessageStep(app_context=context)(content=content, **bound)

        wrapper.reply = AsyncMock(side_effect=reply)
        with pytest.raises(RuntimeError, match="reminders changed"):
            await InboxChatStep(app_context=context)(**arguments)
        wrapper.append_session.assert_not_awaited()
        assert list(runtime.inbox.pending) == arguments["msg_ids"]
