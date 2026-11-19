"""File ownership, uncertain commits and reminder recovery across real failure boundaries."""

# pylint: disable=protected-access,missing-function-docstring

import asyncio
from datetime import timedelta
from unittest.mock import AsyncMock

import pytest

from reme.enumeration import ChunkEnum
from reme.schema import Response, SessionMessage, StreamChunk
from reme.schema.proactive_agent import ProactiveRunResult
from reme.steps.proactive.agent import InboxChatStep, ProactiveAgentStep
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
            await runtime.create({"agent_name": "other", "task_description": "Observe", "agent_type": "proactive"})
            await runtime.append("other", {"event_ts": runtime.now(), "source": "test", "description": "Finished"})
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

            async def fail_consume(identifiers):
                if partial:
                    ref = runtime.notifications.pending[identifiers[0]]
                    await runtime.outbox.update(ref, read_at=runtime.now())
                raise OSError("ack failed")

            with monkeypatch.context() as patch:
                patch.setattr(runtime.notifications, "consume", fail_consume)
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
