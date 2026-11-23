"""File ownership, uncertain commits and reminder recovery across real failure boundaries."""

# pylint: disable=protected-access,missing-function-docstring

import asyncio
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, Mock
from zoneinfo import ZoneInfo

import pytest
import frontmatter

from reme.schema import Event, Response, SessionMessage
from reme.steps.proactive.agent import ProcessChatRemindersStep, ProactiveAgentStep
from reme.steps.file_io import SaveStep, get_path_lock
from reme.steps.proactive.tools.chat_reminders import SubmitChatReminderStep
from .test_proactive_runtime import completed_result, create_agent, running_runtime
from .test_inbox_fork import prepare
from .test_proactive_run import REMIND, SKIP, install_round


@pytest.mark.asyncio
@pytest.mark.parametrize("filename", ["inbox_read.jsonl", "inbox_unread.jsonl"])
async def test_real_receipt_write_failure_preserves_markdown_and_recovers_delivery(tmp_path, monkeypatch, filename):
    async with running_runtime(tmp_path) as (runtime, context, wrapper):
        await create_agent(runtime)
        now = runtime.now()
        monkeypatch.setattr(runtime, "now", lambda: now)
        step = ProactiveAgentStep(app_context=context)
        paths = step._paths(await runtime.definition("monitor"), now)
        install_round(runtime, wrapper, monkeypatch, paths)
        replace = runtime.inbox._replace

        def fail(path, raw):
            if path.parent.parent.name == "monitor" and path.name == filename:
                raise OSError("receipt disk failure")
            return replace(path, raw)

        with monkeypatch.context() as patch:
            patch.setattr(runtime.inbox, "_replace", fail)
            with pytest.raises(OSError, match="receipt disk failure"):
                await step(agent_name="monitor")
        committed = filename == "inbox_unread.jsonl"
        assert runtime.inbox._failed and "monitor" not in runtime.workers
        assert paths["record_path"].read_text() == REMIND.format(priority="P1", body="\nNew progress\n")
    async with running_runtime(tmp_path) as (runtime, _, _):
        snapshot = await runtime.inbox.snapshot("monitor")
        assert snapshot["pending_count"] == (0 if committed else 1)
        assert (snapshot["receipt"] is not None) is committed
        assert bool(runtime.notifications.pending) is committed


@pytest.mark.asyncio
async def test_repeat_local_time_reserves_path_of_unregistered_receipt(tmp_path, monkeypatch):
    first_time = datetime(2026, 11, 1, 1, 30, tzinfo=ZoneInfo("America/New_York"))
    async with running_runtime(tmp_path) as (runtime, context, wrapper):
        monkeypatch.setattr(runtime, "now", lambda: first_time)
        await create_agent(runtime)
        agent = await runtime.definition("monitor")
        first_step = ProactiveAgentStep(app_context=context)
        first_paths = first_step._paths(agent, first_time)
        install_round(runtime, wrapper, monkeypatch, first_paths)
        with monkeypatch.context() as patch:
            patch.setattr(runtime.outbox.storage, "write", Mock(side_effect=OSError("outbox write failed")))
            first = await first_step(agent_name="monitor")
        first_path = first_paths["record_path"]
        first_ref = first.answer["notifications"][0]["id"]
        snapshot = await runtime.inbox.snapshot("monitor")
        assert snapshot["receipt"].metadata["reminder_path"] == first_ref
        assert not runtime.outbox.has_registration("monitor", first_path)
        moved = first_path.rename(tmp_path / first_path.name)

        second_time = first_time.replace(fold=1)
        monkeypatch.setattr(runtime, "now", lambda: second_time)
        await runtime.append_events(
            "monitor",
            [Event(id="event-2", event_ts=second_time, source="experiment", description="Second result")],
        )
        second_path = first_path.with_stem(f"{first_path.stem}_1")
        second_ref = second_path.relative_to(tmp_path).as_posix()
        second_text = REMIND.format(priority="P1", body="\n").replace("Verified result", "Second result")
        install_round(runtime, wrapper, monkeypatch, {"record_path": second_path}, text=second_text)
        reply_stream = wrapper.reply_stream

        async def recover_during_round(query, **options):
            assert second_ref in query
            async for chunk in reply_stream(query, **options):
                await runtime.outbox.sync()
                assert not runtime.notifications.pending
                yield chunk

        monkeypatch.setattr(wrapper, "reply_stream", recover_during_round)
        second = await ProactiveAgentStep(app_context=context)(agent_name="monitor")
        assert second.answer["notifications"][0]["id"] == second_ref
        assert (await runtime.inbox.snapshot("monitor"))["pending_count"] == 0
        assert not first_path.exists()

        moved.rename(first_path)
        await runtime.outbox.sync()
        mailbox = await runtime.outbox.list(state="pending")
        assert {item["id"]: item["message"] for item in mailbox["items"]} == {
            first_ref: "Verified result",
            second_ref: "Second result",
        }


@pytest.mark.asyncio
async def test_full_cron_queue_starts_its_existing_consumer(tmp_path, monkeypatch):
    async with running_runtime(tmp_path) as (runtime, _, _):
        runtime.inbox.max_pending_items = 1
        await runtime.create(
            {"agent_name": "timer", "task_description": "Report", "cron": "* * * * *"},
        )
        agent = await runtime.definition("timer")
        start = runtime.now().replace(second=0, microsecond=0)
        monkeypatch.setattr(runtime, "now", lambda: start)
        await runtime._schedule_cron(agent)
        first = runtime.execution("timer").schedule[1]
        monkeypatch.setattr(runtime, "now", lambda: first)
        await runtime._schedule_cron(agent)
        monkeypatch.setattr(runtime, "now", lambda: first + timedelta(minutes=1))

        async def job(_name, agent_name):
            answer = await runtime.run(
                agent_name,
                AsyncMock(return_value=(None, "session")),
            )
            return Response(answer=answer)

        async with asyncio.timeout(2):
            await runtime._tick_tasks(job)
            await asyncio.gather(*runtime.workers.values())
        assert (await runtime.inbox.snapshot("timer"))["pending_count"] == 0
        assert runtime.execution("timer").schedule[1] == first + timedelta(minutes=1)


@pytest.mark.asyncio
async def test_waiting_append_releases_management_lock_for_consumption(tmp_path):
    async with running_runtime(tmp_path) as (runtime, _, _):
        await create_agent(runtime)
        runtime.inbox.max_pending_items = 1
        append = asyncio.create_task(
            runtime.append_events(
                "monitor",
                [Event(id="next", event_ts=runtime.now(), source="test", description="Next")],
            ),
        )
        try:
            async with asyncio.timeout(2):
                await runtime.run("monitor", AsyncMock(return_value=(None, "session")))
                await append
            assert (await runtime.inbox.snapshot("monitor"))["pending_count"] == 1
        finally:
            append.cancel()
            await asyncio.gather(append, return_exceptions=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("partial", [False, True])
async def test_recovered_native_reminder_finishes_ack_without_regenerating(tmp_path, monkeypatch, partial):
    async with running_runtime(tmp_path) as (runtime, context, wrapper):
        args = await prepare(runtime, context)
        if partial:
            await runtime.create(
                {
                    "agent_name": "other",
                    "task_description": "Observe",
                },
            )
            await runtime.append_events(
                "other",
                [Event(event_ts=runtime.now(), source="test", description="Finished")],
            )
            result, session = completed_result(runtime, "other", priority="P1")
            await runtime.run("other", AsyncMock(return_value=(result, session)))
            await runtime.outbox.sync()
            args["msg_ids"] = list(runtime.notifications.pending)
        source = (await runtime.sessions.current("Remy")).session_id
        snapshot = await wrapper.read_session(source)
        wrapper.read_session = AsyncMock(return_value=snapshot)

        async def append(_session, content, *, message_id, **_kwargs):
            message = SessionMessage(id=f"reme-proactive-{message_id}", role="assistant", content=content)
            snapshot.messages.append(message)
            return message

        wrapper.append_session = AsyncMock(side_effect=append)
        async with runtime.notifications.turn(source, "P1", args["msg_ids"]) as turn:
            await runtime.notifications.read_turn(turn)

            write = runtime.outbox._write
            acknowledged = 0

            def fail_ack(path, item, **kwargs):
                nonlocal acknowledged
                if item.read_at is not None:
                    if not partial or acknowledged:
                        raise OSError("ack failed")
                    acknowledged += 1
                return write(path, item, **kwargs)

            with monkeypatch.context() as patch:
                patch.setattr(runtime.outbox, "_write", fail_ack)
                with pytest.raises(OSError, match="ack failed"):
                    await runtime.notifications.send(turn, "The experiment has finished.")
        await runtime.outbox.refresh()
        wrapper.reply = AsyncMock(side_effect=AssertionError("Accepted decisions must not be regenerated"))
        response = await ProcessChatRemindersStep(app_context=context)(**args)
        assert response.answer == {"sent": True}
        assert not runtime.notifications.pending
        wrapper.reply.assert_not_awaited()
        wrapper.append_session.assert_awaited_once()


@pytest.mark.asyncio
async def test_delete_cancels_an_append_waiting_for_capacity(tmp_path):
    async with running_runtime(tmp_path) as (runtime, _, _):
        await create_agent(runtime)
        runtime.inbox.max_pending_items = 1
        append = asyncio.create_task(
            runtime.append_events(
                "monitor",
                [Event(id="next", event_ts=runtime.now(), source="test", description="Next")],
            ),
        )
        try:
            async with asyncio.timeout(2):
                while append not in runtime.agent_tasks:
                    await asyncio.sleep(0)
                await runtime.delete("monitor")
            assert append.cancelled()
            assert not runtime.agent_tasks
        finally:
            append.cancel()
            await asyncio.gather(append, return_exceptions=True)


@pytest.mark.asyncio
async def test_silence_rechecks_native_revision_after_outbox_refresh(tmp_path, monkeypatch):
    async with running_runtime(tmp_path) as (runtime, context, wrapper):
        args = await prepare(runtime, context)
        source = (await runtime.sessions.current("Remy")).session_id
        snapshot = await wrapper.read_session(source)
        wrapper.read_session = AsyncMock(return_value=snapshot)
        refresh = runtime.outbox.refresh

        async def changed():
            snapshot.messages.append(SessionMessage(id="new-user-input", role="user", content="Tell me immediately"))
            return await refresh()

        async with runtime.notifications.turn(source, "P1", args["msg_ids"]) as turn:
            await runtime.notifications.read_turn(turn)
            monkeypatch.setattr(runtime.outbox, "refresh", changed)
            with pytest.raises(RuntimeError, match="Session changed"):
                await runtime.notifications.send(turn, "")
        assert list(runtime.notifications.pending) == args["msg_ids"]


@pytest.mark.asyncio
async def test_full_receiver_consumes_both_reminders_without_waiting_for_more_capacity(tmp_path):
    async with running_runtime(tmp_path) as (runtime, context, wrapper):
        for name in ("first", "second"):
            await runtime.create({"agent_name": name, "task_description": "Report"})
            await runtime.append_events(name, [Event(event_ts=runtime.now(), source="test", description="Finished")])
            await runtime.run(name, AsyncMock(return_value=completed_result(runtime, name)))
        runtime.inbox.max_pending_items = 1
        runtime.notifications.config.p0_delay_seconds = 0

        async def decide(_hint, **options):
            bound = options["injected_job_kwargs"]
            await SubmitChatReminderStep(app_context=context)(content="", **bound)
            return {"status": "completed"}

        async def run_job(name, **kwargs):
            assert name == "process_chat_reminders"
            return await ProcessChatRemindersStep(app_context=context)(**kwargs)

        wrapper.reply = AsyncMock(side_effect=decide)
        jobs = AsyncMock(side_effect=run_job)
        async with asyncio.timeout(2):
            for _ in range(2):
                await runtime.tick(jobs)
                if worker := runtime.notifications.worker:
                    await worker
        assert jobs.await_count == 2
        assert not runtime.notifications.active_turn
        assert len(runtime.outbox.records) == 2
        assert all(item.read_at is not None for item in runtime.outbox.records.values())


@pytest.mark.asyncio
@pytest.mark.parametrize("draft", [REMIND.format(priority="P1", body="\nNew progress\n"), SKIP, None])
async def test_round_rejects_a_record_changed_after_validation(tmp_path, monkeypatch, draft):
    async with running_runtime(tmp_path) as (runtime, context, wrapper):
        await create_agent(runtime)
        now = runtime.now()
        monkeypatch.setattr(runtime, "now", lambda: now)
        step = ProactiveAgentStep(app_context=context)
        paths = step._paths(await runtime.definition("monitor"), now)
        path = paths["record_path"]
        install_round(runtime, wrapper, monkeypatch, paths, text=draft, reply_text="NO_KEY_INFORMATION")

        async def edit_after_validation(*_args):
            response = await SaveStep(app_context=context)(
                path=path.relative_to(tmp_path).as_posix(),
                content=(draft or SKIP) + "User edited record\n",
            )
            assert response.success

        monkeypatch.setattr(wrapper, "get_tool_calls", edit_after_validation)
        with pytest.raises(RuntimeError, match="changed outside this round"):
            await step(agent_name="monitor")
        assert path.read_text() == (draft or SKIP) + "User edited record\n"
    async with running_runtime(tmp_path) as (runtime, _, _):
        assert (await runtime.inbox.snapshot("monitor"))["pending_count"] == 1


@pytest.mark.asyncio
async def test_round_keeps_the_record_locked_until_delivery(tmp_path, monkeypatch):
    async with running_runtime(tmp_path) as (runtime, context, wrapper):
        await create_agent(runtime)
        now = runtime.now()
        monkeypatch.setattr(runtime, "now", lambda: now)
        step = ProactiveAgentStep(app_context=context)
        paths = step._paths(await runtime.definition("monitor"), now)
        install_round(runtime, wrapper, monkeypatch, paths)
        path = paths["record_path"]
        lock = await get_path_lock(path)
        ack = runtime.inbox.ack
        edit = None

        async def edit_record():
            async with lock:
                document = frontmatter.loads(path.read_text())
                assert runtime.outbox.records[path.relative_to(tmp_path).as_posix()].delivered_at
                document["reminder_priority"] = "P0"
                document.content = "User edited record"
                runtime.files.write(path, frontmatter.dumps(document))

        async def acknowledge(*args, **kwargs):
            nonlocal edit
            assert lock.locked()
            edit = asyncio.create_task(edit_record())
            await asyncio.sleep(0)
            assert not edit.done()
            return await ack(*args, **kwargs)

        monkeypatch.setattr(runtime.inbox, "ack", acknowledge)
        try:
            response = await step(agent_name="monitor")
            assert response.answer["notifications"][0]["priority"] == "P1"
        finally:
            if edit is not None:
                await edit
        assert frontmatter.loads(path.read_text()).content == "User edited record"


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["handoff", "update"])
async def test_outbox_operations_share_the_file_jobs_path_lock(tmp_path, operation):
    async with running_runtime(tmp_path) as (runtime, _, _):
        await create_agent(runtime)
        result, session = completed_result(runtime)
        path = runtime.files.path(result.id)
        if operation == "update":
            await runtime.run("monitor", AsyncMock(return_value=(result, session)))
            await runtime.outbox.sync()
            ref = next(iter(runtime.outbox.records))
            action = runtime.outbox.update(ref, read_at=runtime.now())
        else:
            action = runtime.run("monitor", AsyncMock(return_value=(result, session)))
        task = None
        try:
            async with await get_path_lock(path):
                task = asyncio.create_task(action)
                await asyncio.sleep(0.02)
                assert not task.done()
            await asyncio.wait_for(task, 2)
        finally:
            if task is not None:
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
