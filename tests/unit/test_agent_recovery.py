"""File ownership, uncertain commits and reminder recovery across real failure boundaries."""

# pylint: disable=protected-access,missing-function-docstring

import asyncio
from datetime import datetime, timedelta
from unittest.mock import AsyncMock

import pytest
import frontmatter

from reme.enumeration import ChunkEnum
from reme.schema import Response, SessionMessage, StreamChunk
from reme.schema.proactive_agent import ProactiveRunResult
from reme.steps.proactive.agent import InboxChatStep, ProactiveAgentStep
from reme.steps.file_io import SaveStep, get_path_lock
from reme.steps.proactive.tools.inbox import ReadInboxStep, SendChatMessageStep
from .test_agent_runtime import completed_result, create_agent, running_runtime
from .test_inbox_fork import prepare
from .test_proactive_run import install_round


@pytest.mark.asyncio
@pytest.mark.parametrize("model_writes", [False, True])
async def test_external_summary_edit_survives_failed_or_conflicting_round(tmp_path, monkeypatch, model_writes):
    async with running_runtime(tmp_path) as (runtime, context, wrapper):
        await create_agent(runtime)
        now = runtime.now()
        monkeypatch.setattr(runtime, "now", lambda: now)
        step = ProactiveAgentStep(app_context=context)
        paths = step._paths(await runtime.definition("monitor"), now)
        summary = paths["today_summary_path"]
        runtime.files.write(summary, "original\n")
        if model_writes:
            install_round(runtime, wrapper, monkeypatch, paths, hook=lambda: summary.write_text("my edit\n"))
        else:

            async def fail(*_args, **_kwargs):
                summary.write_text("my edit\n")
                raise OSError("model failed without writing")
                yield StreamChunk(chunk_type=ChunkEnum.CONTENT, chunk="unreachable")  # pylint: disable=unreachable

            monkeypatch.setattr(wrapper, "reply_stream", fail)
        with pytest.raises(RuntimeError if model_writes else OSError):
            await step(agent_name="monitor")
        assert summary.read_text() == "my edit\n"
        assert not paths["record_path"].exists() and not paths["reminder_path"].exists()
        assert not list(tmp_path.rglob(".round-*.md"))
        assert (await runtime.inbox.snapshot("monitor"))["pending_count"] == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("filename", ["inbox_read.jsonl", "inbox_unread.jsonl"])
async def test_real_receipt_write_failure_restores_only_uncommitted_files(tmp_path, monkeypatch, filename):
    async with running_runtime(tmp_path) as (runtime, context, wrapper):
        await create_agent(runtime)
        now = runtime.now()
        monkeypatch.setattr(runtime, "now", lambda: now)
        step = ProactiveAgentStep(app_context=context)
        paths = step._paths(await runtime.definition("monitor"), now)
        runtime.files.write(paths["today_summary_path"], "original\n")
        install_round(runtime, wrapper, monkeypatch, paths)
        restore = AsyncMock(wraps=step._restore_files)
        monkeypatch.setattr(step, "_restore_files", restore)
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
        assert restore.await_count == (0 if committed else 1)
        assert paths["record_path"].exists() is committed
        assert paths["reminder_path"].exists() is committed
        assert (paths["today_summary_path"].read_text() == "original\n") is not committed
        assert not list(tmp_path.rglob(".round-*.md"))
    async with running_runtime(tmp_path) as (runtime, _, _):
        snapshot = await runtime.inbox.snapshot("monitor")
        assert snapshot["pending_count"] == (0 if committed else 1)
        assert (snapshot["receipt"] is not None) is committed


@pytest.mark.asyncio
async def test_rollback_preserves_external_edit_after_our_publication(tmp_path, monkeypatch):
    async with running_runtime(tmp_path) as (runtime, context, wrapper):
        await create_agent(runtime)
        now = runtime.now()
        monkeypatch.setattr(runtime, "now", lambda: now)
        step = ProactiveAgentStep(app_context=context)
        paths = step._paths(await runtime.definition("monitor"), now)
        summary = paths["today_summary_path"]
        runtime.files.write(summary, "original\n")
        install_round(runtime, wrapper, monkeypatch, paths)

        async def fail(*_args, **_kwargs):
            summary.write_text("my edit after publication\n")
            raise OSError("ack failed")

        monkeypatch.setattr(runtime.inbox, "ack", fail)
        with pytest.raises(OSError, match="ack failed"):
            await step(agent_name="monitor")
        assert summary.read_text() == "my edit after publication\n"
        assert not paths["record_path"].exists() and not paths["reminder_path"].exists()


@pytest.mark.asyncio
async def test_full_cron_queue_starts_its_existing_consumer(tmp_path, monkeypatch):
    async with running_runtime(tmp_path) as (runtime, _, _):
        runtime.inbox.max_pending_items = 1
        await runtime.create(
            {"agent_name": "timer", "task_description": "Report", "agent_type": "cron", "cron": "* * * * *"},
        )
        agent = await runtime.definition("timer")
        start = runtime.now().replace(second=0, microsecond=0)
        monkeypatch.setattr(runtime, "now", lambda: start)
        await runtime._schedule_cron(agent)
        first = runtime.cron["timer"][1]
        monkeypatch.setattr(runtime, "now", lambda: first)
        await runtime._schedule_cron(agent)
        monkeypatch.setattr(runtime, "now", lambda: first + timedelta(minutes=1))

        async def job(_name, agent_name):
            answer = await runtime.run(
                agent_name,
                AsyncMock(return_value=(ProactiveRunResult(notifications=[]), "session")),
            )
            return Response(answer=answer)

        async with asyncio.timeout(2):
            await runtime._tick_tasks(job)
            await asyncio.gather(*runtime.workers.values())
        assert (await runtime.inbox.snapshot("timer"))["pending_count"] == 0
        assert runtime.cron["timer"][1] == first + timedelta(minutes=1)


@pytest.mark.asyncio
async def test_waiting_append_releases_management_lock_for_consumption(tmp_path):
    async with running_runtime(tmp_path) as (runtime, _, _):
        await create_agent(runtime)
        runtime.inbox.max_pending_items = 1
        append = asyncio.create_task(
            runtime.append(
                "monitor",
                {"id": "next", "event_ts": runtime.now(), "source": "test", "description": "Next"},
            ),
        )
        try:
            async with asyncio.timeout(2):
                await runtime.run("monitor", AsyncMock(return_value=(ProactiveRunResult(notifications=[]), "session")))
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
                    "agent_type": "proactive",
                },
            )
            await runtime.append(
                "other",
                {
                    "event_ts": runtime.now(),
                    "source": "test",
                    "description": "Finished",
                },
            )
            result, session = completed_result(runtime, "other")
            result.notifications[0].priority = "P1"
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
        response = await InboxChatStep(app_context=context)(**args)
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
            runtime.append(
                "monitor",
                {"id": "next", "event_ts": runtime.now(), "source": "test", "description": "Next"},
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
            await runtime.create({"agent_name": name, "agent_type": "proactive", "task_description": "Report"})
            await runtime.append(name, {"event_ts": runtime.now(), "source": "test", "description": "Finished"})
            await runtime.run(name, AsyncMock(return_value=completed_result(runtime, name)))
        runtime.inbox.max_pending_items = 1
        runtime.notifications.config.p0_delay_seconds = 0

        async def decide(_hint, **options):
            bound = options["injected_job_kwargs"]
            await ReadInboxStep(app_context=context)(**bound)
            await SendChatMessageStep(app_context=context)(content="", **bound)
            return {"status": "completed"}

        async def run_job(name, **kwargs):
            assert name == "process_chat_reminders"
            return await InboxChatStep(app_context=context)(**kwargs)

        wrapper.reply = AsyncMock(side_effect=decide)
        jobs = AsyncMock(side_effect=run_job)
        async with asyncio.timeout(2):
            for _ in range(2):
                await runtime.tick(jobs)
                if worker := runtime.notifications.worker:
                    await worker
        assert jobs.await_count == 2
        assert (await runtime.inbox.snapshot("Remy"))["pending_count"] == 0
        assert len(runtime.outbox.records) == 2
        assert all(item.read_at is not None for item in runtime.outbox.records.values())


@pytest.mark.asyncio
@pytest.mark.parametrize("stage", ["before_save", "while_save_waits", "after_save"])
@pytest.mark.parametrize("ack_fails", [False, True])
async def test_user_reminder_edits_survive_outbox_save_and_receipt_failure(tmp_path, monkeypatch, stage, ack_fails):
    async with running_runtime(tmp_path) as (runtime, context, wrapper):
        await create_agent(runtime)
        now = runtime.now()
        monkeypatch.setattr(runtime, "now", lambda: now)
        step = ProactiveAgentStep(app_context=context)
        paths = step._paths(await runtime.definition("monitor"), now)
        install_round(runtime, wrapper, monkeypatch, paths)
        save = runtime.outbox.save

        async def edit_reminder():
            path = paths["reminder_path"]
            document = frontmatter.loads(path.read_text())
            assert ("id" in document) is (stage == "after_save")
            document.content = "User edited reminder"
            response = await SaveStep(app_context=context)(
                path=path.relative_to(tmp_path).as_posix(),
                content=frontmatter.dumps(document),
                expected_mtime=datetime.fromtimestamp(path.stat().st_mtime).isoformat(),
            )
            assert response.success

        async def edit_during_save(*args, **kwargs):
            if stage == "while_save_waits":
                async with await get_path_lock(paths["reminder_path"]):
                    edit = asyncio.create_task(edit_reminder())
                    await asyncio.sleep(0)
                    publication = asyncio.create_task(save(*args, **kwargs))
                    await asyncio.sleep(0.02)
                    assert not publication.done()
                _, refs = await asyncio.gather(edit, publication)
                return refs
            if stage == "before_save":
                await edit_reminder()
            refs = await save(*args, **kwargs)
            if stage == "after_save":
                await edit_reminder()
            return refs

        monkeypatch.setattr(runtime.outbox, "save", edit_during_save)
        ack = AsyncMock(wraps=runtime.inbox.ack)
        monkeypatch.setattr(runtime.inbox, "ack", ack)
        if ack_fails:
            replace = runtime.inbox._replace

            def fail_receipt(path, raw):
                if path.parent.parent.name == "monitor" and path.name == "inbox_read.jsonl":
                    raise OSError("receipt disk failure")
                return replace(path, raw)

            monkeypatch.setattr(runtime.inbox, "_replace", fail_receipt)
        if stage != "after_save":
            with pytest.raises(RuntimeError, match="changed outside this round"):
                await step(agent_name="monitor")
            ack.assert_not_awaited()
        elif ack_fails:
            with pytest.raises(OSError, match="receipt disk failure"):
                await step(agent_name="monitor")
        else:
            assert (await step(agent_name="monitor")).success
        assert paths["reminder_path"].exists()
        assert frontmatter.loads(paths["reminder_path"].read_text()).content == "User edited reminder"
    async with running_runtime(tmp_path) as (runtime, _, _):
        committed = stage == "after_save" and not ack_fails
        assert (await runtime.inbox.snapshot("monitor"))["pending_count"] == (0 if committed else 1)
        assert frontmatter.loads(paths["reminder_path"].read_text()).content == "User edited reminder"


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["save", "update"])
async def test_outbox_writers_share_the_file_jobs_path_lock(tmp_path, operation):
    async with running_runtime(tmp_path) as (runtime, _, _):
        await create_agent(runtime)
        result, session = completed_result(runtime)
        path = runtime.files.path(result.notifications[0].paths[0])
        if operation == "update":
            await runtime.run("monitor", AsyncMock(return_value=(result, session)))
            await runtime.outbox.sync()
            ref = next(iter(runtime.outbox.committed))
            action = runtime.outbox.update(ref, user_read_at=runtime.now())
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
