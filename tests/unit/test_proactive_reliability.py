"""Regression coverage for real reminder failures and safe recovery boundaries."""

# pylint: disable=missing-function-docstring,protected-access

import asyncio
from datetime import timedelta
from unittest.mock import AsyncMock, Mock

import pytest

from reme.components import ApplicationContext, RuntimeContext
from reme.components.agent_wrapper import CodexAgentWrapper
from reme.components.job import CronJob
from reme.schema import Response
from reme.steps.proactive.agent import ChatAgentStep, InboxChatStep, ProactiveAgentStep
from reme.steps.proactive.tools.inbox import ReadInboxStep, SendChatMessageStep
from reme.steps.proactive.tools.memory import AgentMemoryStep
from .test_agent_runtime import completed_result, create_agent, running_runtime
from .test_inbox_fork import append_message, prepare


@pytest.mark.asyncio
async def test_latest_foreground_instruction_invalidates_same_session_reminder(tmp_path):
    async with running_runtime(tmp_path) as (runtime, context, wrapper):
        arguments = await prepare(runtime, context)
        wrapper.append_session = AsyncMock(side_effect=append_message)

        async def reply(_hint, **options):
            bound = options["injected_job_kwargs"]
            await ReadInboxStep(app_context=context)(priority="P1", **bound)
            before = await runtime.sessions.current("chat")
            await ChatAgentStep(app_context=context)(
                RuntimeContext(stream_queue=asyncio.Queue()),
                query="Cancel the reminder",
            )
            assert (await runtime.sessions.current("chat")).session_id == before.session_id
            await SendChatMessageStep(app_context=context)(content="Outdated decision", **bound)

        wrapper.reply = AsyncMock(side_effect=reply)
        with pytest.raises(RuntimeError, match="conversation changed"):
            await InboxChatStep(app_context=context)(**arguments)
        wrapper.append_session.assert_not_awaited()
        assert list(runtime.inbox.pending) == arguments["msg_ids"]


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
        assert cancelled.is_set() and runtime.inbox.active_turn is None
        assert list(runtime.inbox.pending) == arguments["msg_ids"]


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
        await runtime.inbox._wake(run, "inbox_chat", arguments["priority"], arguments["msg_ids"])
        await runtime.inbox.tick(run, "inbox_chat")
        assert run.await_count == 1 and runtime.inbox.blocked
        chat = (await runtime.list_views())["items"][0]
        assert chat["status"] == "error" and chat["error"] == "unexpected SDK argument"
        assert list(runtime.inbox.pending) == arguments["msg_ids"]


@pytest.mark.asyncio
async def test_transient_inbox_failure_backs_off_and_success_clears_health(tmp_path):
    async with running_runtime(tmp_path) as (runtime, context, _):
        arguments = await prepare(runtime, context)
        now = runtime.now()
        runtime.now = lambda: now
        run = AsyncMock(side_effect=ConnectionError("offline"))
        for attempt in range(4):
            await runtime.inbox._wake(run, "inbox_chat", arguments["priority"], arguments["msg_ids"])
            assert runtime.inbox.retry_at == now + timedelta(seconds=5 * 2**attempt)
        assert not runtime.inbox.blocked
        await runtime.inbox._wake(AsyncMock(return_value=Response()), "inbox_chat", "P1", arguments["msg_ids"])
        assert runtime.inbox.error is None and runtime.inbox.retry_at is None and runtime.inbox.failures == 0


@pytest.mark.asyncio
async def test_missing_reminder_isolated_across_restart_and_recovers_when_restored(tmp_path):
    async with running_runtime(tmp_path) as (runtime, _, _):
        await create_agent(runtime)
        await runtime.run("monitor", AsyncMock(return_value=completed_result(runtime)))
        await runtime.outbox.sync()
        ref = next(iter(runtime.inbox.pending.values()))
        path = tmp_path / ref.file
        original = path.read_bytes()
        path.unlink()
        await runtime.create({"name": "other", "trigger_type": "event", "description": "Other"})
        await runtime.append(
            "other",
            {"id": "other", "event_ts": runtime.now(), "source": "test", "description": "Other"},
        )
        await runtime.run("other", AsyncMock(return_value=completed_result(runtime, "other")))
        await runtime.outbox.sync()
        assert ref.file in runtime.outbox.issues and len(runtime.inbox.pending) == 1
        assert next(iter(runtime.outbox.owners.values())) == "other"
    async with running_runtime(tmp_path) as (runtime, _, _):
        assert ref.file in runtime.outbox.issues and len(runtime.inbox.pending) == 1
        path.write_bytes(original)
        await runtime.outbox.sync()
        assert not runtime.outbox.issues and len(runtime.inbox.pending) == 2


@pytest.mark.parametrize("read", [False, True])
@pytest.mark.asyncio
async def test_corrupt_inbox_views_rebuild_from_canonical_reminders(tmp_path, read):
    async with running_runtime(tmp_path) as (runtime, _, _):
        await create_agent(runtime)
        await runtime.run("monitor", AsyncMock(return_value=completed_result(runtime)))
        await runtime.outbox.sync()
        ref = next(iter(runtime.inbox.pending.values()))
        if read:
            await runtime.inbox.consume([ref.id])
        runtime.files.write(runtime.inbox.path(read), "{corrupt}\n")
        await runtime.outbox.sync()
        assert (ref.id in runtime.inbox.read_refs) is read
        assert (ref.id in runtime.inbox.pending) is not read
        assert (runtime.outbox.records[ref].read_at is not None) is read


@pytest.mark.asyncio
async def test_memory_revisions_preserve_user_edits_and_limit_task_scope(tmp_path):
    async with running_runtime(tmp_path) as (runtime, context, _):
        await create_agent(runtime)
        step = AgentMemoryStep(app_context=context)
        read = (await step(action="read", agent_id="monitor", _agent_id="chat")).answer
        path = tmp_path / read["path"]
        path.write_text("New user requirement\n", encoding="utf-8")
        with pytest.raises(ValueError, match="memory changed"):
            await step(
                action="write",
                agent_id="monitor",
                content="Old text",
                expected_revision=read["revision"],
                _agent_id="chat",
            )
        assert path.read_text() == "New user requirement\n"
        latest = (await step(action="read", _agent_id="monitor")).answer
        await step(
            action="write",
            content=latest["content"] + "Confirmed fact\n",
            expected_revision=latest["revision"],
            _agent_id="monitor",
        )
        assert path.read_text() == "New user requirement\nConfirmed fact\n"
        with pytest.raises(PermissionError, match="only their own"):
            await step(action="read", agent_id="chat", _agent_id="monitor")
        with pytest.raises(ValueError):
            await step(action="write", content="Unconditional overwrite", _agent_id="chat")


def test_memory_profile_cannot_be_replaced_by_a_legacy_turn_sandbox(tmp_path):
    memory = str(tmp_path / "MEMORY.md")
    wrapper = CodexAgentWrapper(cwd=tmp_path)
    arguments = {
        "sandbox": "workspace-write",
        "read_only_paths": [memory],
        "config": {"sandbox_workspace_write": {"network_access": False}},
    }
    options = wrapper._thread_options(arguments)
    profile = options["config"]["permissions"]["reme_agent"]
    assert options["sandbox"] is None and profile["filesystem"][memory] == "read"
    assert profile["filesystem"][str(tmp_path)] == "write" and not profile["network"]["enabled"]
    assert "sandbox" not in wrapper._turn_options(arguments)


@pytest.mark.asyncio
async def test_proactive_timeout_releases_claim_and_preserves_inputs(tmp_path):
    async with running_runtime(tmp_path) as (runtime, context, wrapper):
        await create_agent(runtime)

        async def never_finish(*_args, **_kwargs):
            await asyncio.Event().wait()

        wrapper.reply = AsyncMock(side_effect=never_finish)
        with pytest.raises(TimeoutError):
            await ProactiveAgentStep(app_context=context, timeout_seconds=0.01)(agent_id="monitor")
        snapshot = await runtime.event_stream.snapshot("monitor")
        assert snapshot["pending_count"] == 1 and snapshot["in_flight_count"] == 0
        assert not runtime.workers


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
