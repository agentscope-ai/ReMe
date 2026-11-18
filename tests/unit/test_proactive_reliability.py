"""Regression coverage for real reminder failures and safe recovery boundaries."""

# pylint: disable=missing-function-docstring,protected-access

import asyncio
from datetime import timedelta
from unittest.mock import AsyncMock, Mock

import frontmatter
import pytest

from reme.components import ApplicationContext, RuntimeContext
from reme.components.agent_wrapper import CodexAgentWrapper
from reme.components.job import CronJob
from reme.enumeration import ChunkEnum
from reme.schema import Response, StreamChunk
from reme.schema.proactive_agent import ProactiveRunResult
from reme.steps.proactive.agent import ChatAgentStep, InboxChatStep, ProactiveAgentStep
from reme.steps.proactive.tools.inbox import ReadInboxStep, SendChatMessageStep
from .test_agent_runtime import inbox_path, completed_result, create_agent, running_runtime
from .test_inbox_fork import append_message, prepare


@pytest.mark.asyncio
async def test_latest_foreground_instruction_invalidates_same_session_reminder(tmp_path):
    async with running_runtime(tmp_path) as (runtime, context, wrapper):
        arguments = await prepare(runtime, context)
        wrapper.append_session = AsyncMock(side_effect=append_message)

        async def reply(_hint, **options):
            bound = options["injected_job_kwargs"]
            await ReadInboxStep(app_context=context)(**bound)
            before = await runtime.sessions.current("Remy")
            await ChatAgentStep(app_context=context)(
                RuntimeContext(stream_queue=asyncio.Queue()),
                query="Cancel the reminder",
            )
            assert (await runtime.sessions.current("Remy")).session_id == before.session_id
            await SendChatMessageStep(app_context=context)(content="Outdated decision", **bound)

        wrapper.reply = AsyncMock(side_effect=reply)
        with pytest.raises(RuntimeError, match="conversation changed"):
            await InboxChatStep(app_context=context)(**arguments)
        wrapper.append_session.assert_not_awaited()
        assert list(runtime.notifications.pending) == arguments["msg_ids"]


@pytest.mark.asyncio
async def test_inbox_timeout_revokes_tools_without_consuming_messages(tmp_path):
    async with running_runtime(tmp_path) as (runtime, context, wrapper):
        arguments = await prepare(runtime, context)
        cancelled = asyncio.Event()

        async def reply(*_args, **_kwargs):
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.set()

        wrapper.reply = AsyncMock(side_effect=reply)
        with pytest.raises(TimeoutError):
            await InboxChatStep(app_context=context, timeout_seconds=0.01)(**arguments)
        assert cancelled.is_set() and runtime.notifications.active_turn is None
        assert list(runtime.notifications.pending) == arguments["msg_ids"]


@pytest.mark.asyncio
async def test_incompatible_sdk_blocks_retry_and_is_visible_in_agent_views(tmp_path):
    async with running_runtime(tmp_path) as (runtime, context, _):
        arguments = await prepare(runtime, context)
        run = AsyncMock(
            return_value=Response(
                success=False,
                answer="unexpected SDK argument",
                metadata={"reme.job_error": {"type": "TypeError"}},
            ),
        )
        await runtime.notifications._wake(run, arguments["priority"], arguments["msg_ids"])
        await runtime.notifications.tick(run)
        assert run.await_count == 1 and runtime.notifications.blocked
        chat = (await runtime.list_views())["items"][0]
        assert chat["status"] == "error" and chat["error"] == "unexpected SDK argument"
        assert list(runtime.notifications.pending) == arguments["msg_ids"]


@pytest.mark.asyncio
async def test_transient_inbox_failure_backs_off_and_success_clears_health(tmp_path):
    async with running_runtime(tmp_path) as (runtime, context, _):
        arguments = await prepare(runtime, context)
        now = runtime.now()
        runtime.now = lambda: now
        run = AsyncMock(side_effect=ConnectionError("offline"))
        for attempt in range(4):
            await runtime.notifications._wake(run, arguments["priority"], arguments["msg_ids"])
            assert runtime.notifications.retry_at == now + timedelta(seconds=5 * 2**attempt)
        assert not runtime.notifications.blocked
        await runtime.notifications._wake(AsyncMock(return_value=Response()), "P1", arguments["msg_ids"])
        assert (
            runtime.notifications.error is None
            and runtime.notifications.retry_at is None
            and runtime.notifications.failures == 0
        )


@pytest.mark.asyncio
async def test_missing_reminder_isolated_across_restart_and_recovers_when_restored(tmp_path):
    async with running_runtime(tmp_path) as (runtime, _, _):
        await create_agent(runtime)
        await runtime.run("monitor", AsyncMock(return_value=completed_result(runtime)))
        await runtime.outbox.sync()
        ref = next(iter(runtime.notifications.pending.values()))
        path = tmp_path / ref.file
        original = path.read_bytes()
        path.unlink()
        await runtime.create({"agent_name": "other", "agent_type": "proactive", "task_description": "Other"})
        await runtime.append(
            "other",
            {"id": "other", "event_ts": runtime.now(), "source": "test", "description": "Other"},
        )
        await runtime.run("other", AsyncMock(return_value=completed_result(runtime, "other")))
        await runtime.outbox.sync()
        assert ref.file in runtime.outbox.issues and len(runtime.notifications.pending) == 1
        assert next(iter(runtime.outbox.owners.values())) == "other"
    async with running_runtime(tmp_path) as (runtime, _, _):
        assert ref.file in runtime.outbox.issues and len(runtime.notifications.pending) == 1
        path.write_bytes(original)
        await runtime.outbox.sync()
        assert not runtime.outbox.issues and len(runtime.notifications.pending) == 2


@pytest.mark.parametrize("read", [False, True])
@pytest.mark.asyncio
async def test_corrupt_inbox_views_rebuild_from_canonical_reminders(tmp_path, read):
    async with running_runtime(tmp_path) as (runtime, _, _):
        await create_agent(runtime)
        await runtime.run("monitor", AsyncMock(return_value=completed_result(runtime)))
        await runtime.outbox.sync()
        ref = next(iter(runtime.notifications.pending.values()))
        if read:
            await runtime.notifications.consume([ref.id])
        runtime.files.write(inbox_path(runtime, read), "{corrupt}\n")
        await runtime.outbox.sync()
        assert (ref.id in runtime.notifications.read_refs) is read
        assert (ref.id in runtime.notifications.pending) is not read
        assert (runtime.outbox.records[ref].read_at is not None) is read


@pytest.mark.asyncio
async def test_missing_result_blocks_temporary_collection_before_source_deletion(tmp_path):
    async with running_runtime(tmp_path) as (runtime, _, _):
        start = runtime.now()
        await runtime.create(
            {"agent_name": "experiment", "task_description": "Observe", "agent_type": "proactive", "temporary": True},
        )
        await runtime.append("experiment", {"event_ts": start, "source": "lab", "description": "Finished"})
        await runtime.run("experiment", AsyncMock(return_value=completed_result(runtime, "experiment")))
        await runtime.outbox.sync()
        ref = next(iter(runtime.notifications.pending.values()))
        runtime.files.path(ref.file).unlink()
        directory = runtime.agents.path("experiment")
        runtime.now = lambda: start + timedelta(days=8)
        await runtime.tick(AsyncMock())
        assert directory.is_dir() and (directory / "TASK.md").is_file()
        assert ref.file in runtime.outbox.issues
        assert (await runtime.inbox.snapshot("experiment"))["receipt"].metadata["reminders"] == [ref.model_dump()]


@pytest.mark.asyncio
async def test_temporary_lifetime_survives_priority_consumption_and_restart(tmp_path, monkeypatch):
    async with running_runtime(tmp_path) as (runtime, _, _):
        now = runtime.now()
        runtime.now = lambda: now - timedelta(days=8)
        await runtime.create(
            {"agent_name": "experiment", "task_description": "Observe", "agent_type": "proactive", "temporary": True},
        )
        await runtime.append("experiment", {"id": "old", "event_ts": now, "source": "lab", "description": "Old"})
        recent = now - timedelta(days=1)
        runtime.now = lambda: recent
        await runtime.append("experiment", {"id": "new", "event_ts": now, "source": "lab", "description": "New"})
        claim = runtime.inbox.claim

        async def urgent(name, limit=None, **_kwargs):
            return await claim(name, limit, item_ids=["new"])

        monkeypatch.setattr(runtime.inbox, "claim", urgent)
        await runtime.run("experiment", AsyncMock(return_value=(ProactiveRunResult(notifications=[]), "session")))
    async with running_runtime(tmp_path) as (runtime, _, _):
        directory = runtime.agents.path("experiment")
        runtime.now = lambda: now
        await runtime.run("experiment", AsyncMock(return_value=(ProactiveRunResult(notifications=[]), "session")))
        await runtime.tick(AsyncMock())
        assert directory.is_dir()
        receipt = (await runtime.inbox.snapshot("experiment"))["receipt"]
        assert receipt.metadata["last_received_at"] == recent.isoformat()


@pytest.mark.asyncio
async def test_session_setup_preserves_workspace_documents(tmp_path):
    async with running_runtime(tmp_path) as (runtime, context, _):
        await create_agent(runtime)
        agent = await runtime.definition("monitor")
        step = ProactiveAgentStep(app_context=context)
        now = runtime.now()
        documents = (
            ("Remy", "MEMORY.md", ChatAgentStep(app_context=context).session_options()),
            ("monitor", "TASK.md", step.session_options(agent, step._paths(agent, now), now)),
        )
        for name, filename, item in documents:
            path = runtime.agents.path(name) / filename
            assert "read_only_paths" not in item and "agent_memory" not in item["job_tools"]
            path.write_text("Confirmed fact\n", encoding="utf-8")
            await runtime.sessions.ensure(name, runtime.agent_wrapper, item, runtime.now())
            assert path.read_text() == "Confirmed fact\n"


def test_native_memory_updates_use_standard_workspace_permissions(tmp_path):
    wrapper = CodexAgentWrapper(cwd=tmp_path)
    arguments = {
        "sandbox": "workspace-write",
        "config": {"sandbox_workspace_write": {"network_access": False}},
    }
    options = wrapper._thread_options(arguments)
    assert options["sandbox"].value == "workspace-write"
    assert options["config"] == arguments["config"]
    assert wrapper._turn_options(arguments)["sandbox"].value == "workspace-write"


@pytest.mark.parametrize("phase", ["session_setup", "inference"])
@pytest.mark.asyncio
async def test_proactive_timeout_releases_claim_and_preserves_inputs(tmp_path, phase):
    async with running_runtime(tmp_path) as (runtime, context, wrapper):
        await create_agent(runtime)
        runtime.files.write(tmp_path / "daily" / runtime.now().date().isoformat() / "monitor.md", "# Progress\n")

        async def never_finish(*_args, **_kwargs):
            yield StreamChunk(chunk_type=ChunkEnum.CONTENT, chunk="Working")
            await asyncio.Event().wait()

        if phase == "session_setup":
            create_session = wrapper.create_session

            async def delayed_session(**kwargs):
                await asyncio.sleep(0.05)
                return await create_session(**kwargs)

            wrapper.create_session = delayed_session
        else:
            wrapper.reply_stream = never_finish
        with pytest.raises(TimeoutError):
            await ProactiveAgentStep(app_context=context, timeout_seconds=0.01)(agent_name="monitor")
        snapshot = await runtime.inbox.snapshot("monitor")
        assert snapshot["pending_count"] == 1 and snapshot["in_flight_count"] == 0
        assert not runtime.workers
        if phase == "session_setup":
            assert not wrapper.turns


@pytest.mark.asyncio
async def test_cron_logs_unsuccessful_collection_without_an_exception(tmp_path):
    context = ApplicationContext(workspace_dir=str(tmp_path))
    job = CronJob(app_context=context, name="collector", cron="* * * * *")
    job._stop_event = asyncio.Event()
    job._next_fire_delay = lambda: 0
    job._wait_or_stop = AsyncMock()
    job.logger = Mock()

    async def partial():
        job._stop_event.set()
        return Response(success=False, answer={"errors": {"screenshot": "capture denied"}})

    job._execute_steps = partial
    await job()
    job.logger.error.assert_called_once()
    assert "capture denied" in job.logger.error.call_args.args[0]


@pytest.mark.asyncio
async def test_canonical_acknowledgement_rebuilds_reference_views(tmp_path):
    async with running_runtime(tmp_path) as (runtime, _, _):
        await create_agent(runtime)
        await runtime.run("monitor", AsyncMock(return_value=completed_result(runtime)))
        await runtime.outbox.sync()
        ref = next(iter(runtime.notifications.pending.values()))
        path = runtime.files.path(ref.file)
        document = frontmatter.loads(runtime.files.read(path))
        document["read_at"] = runtime.now().isoformat()
        runtime.files.write(path, frontmatter.dumps(document))
        await runtime.outbox.sync()
        assert not runtime.notifications.pending
        document = frontmatter.loads(runtime.files.read(runtime.files.path(ref.file)))
        assert ref.file.endswith(".md")
        assert document.content == "Experiment finished"
        assert set(document.metadata) == {
            "action",
            "id",
            "created_at",
            "priority",
            "paths",
            "delivered_at",
            "read_at",
            "user_read_at",
            "chat_message_ids",
        }
        assert document["user_read_at"] is None
        assert document["read_at"] and document["delivered_at"]
        assert not list((runtime.agents.path("monitor") / "reminder").glob("*.jsonl"))
        inbox_path(runtime).unlink()
        inbox_path(runtime, True).unlink()
        await runtime.outbox.sync()
        assert not runtime.notifications.pending
        assert ref.id in runtime.notifications.read_refs


@pytest.mark.asyncio
async def test_watcher_observes_external_acknowledgements(tmp_path):
    async with running_runtime(tmp_path, watch=True) as (runtime, _, _):
        await create_agent(runtime)
        await runtime.run("monitor", AsyncMock(return_value=completed_result(runtime)))
        async with asyncio.timeout(5):
            while not runtime.notifications.pending:
                await asyncio.sleep(0.02)
        ref = next(iter(runtime.notifications.pending.values()))
        path = runtime.files.path(ref.file)
        document = frontmatter.loads(runtime.files.read(path))
        document["read_at"] = runtime.now().isoformat()
        runtime.files.write(path, frontmatter.dumps(document))
        async with asyncio.timeout(5):
            while runtime.notifications.pending:
                await asyncio.sleep(0.02)
        assert runtime.outbox.records[ref].read_at is not None
