"""Shared Codex runtime, isolated conversations, and same-Application MCP tools."""

# pylint: disable=missing-class-docstring,missing-function-docstring,protected-access,redefined-outer-name
# pylint: disable=too-many-lines

import asyncio
import json
from pathlib import Path
import socket
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest
import pytest_asyncio
from fastmcp import Client
from fastmcp.client import StreamableHttpTransport
from fastmcp.exceptions import ToolError
from openai_codex.generated.v2_all import (
    AgentMessageThreadItem,
    AgentMessageDeltaNotification,
    TurnStartedNotification,
    ItemCompletedNotification,
    ThreadTokenUsage,
    ThreadTokenUsageUpdatedNotification,
    TokenUsageBreakdown,
    Turn,
    TurnCompletedNotification,
)
from pydantic import BaseModel

from reme.application import Application
from reme.components.application_context import ApplicationContext
from reme.components.agent_wrapper.codex_agent_wrapper import CodexAgentWrapper
from reme.components.codex_app_server import CodexAppServer
from reme.components.job import BackgroundJob, StreamJob
from reme.components.service.scoped_mcp import ScopedMCP
from reme.config import resolve_app_config
from reme.enumeration import ChunkEnum, ComponentEnum
from reme.schema import Response, TokenUsage
from reme.steps.proactive.pa_chat import PaChatStep, PaNewSessionStep
from reme.steps.proactive.temp_proactive import TempProactiveStep
from reme.utils import global_counter_get_all
from .pa_helpers import bind_pa


class _Job:
    name = "search"
    description = "Search memory"
    enabled = True
    enable_serve = False
    parameters = {
        "type": "object",
        "properties": {"query": {"type": "string"}, "_allowed_paths": {"type": "array"}},
        "required": ["query"],
    }

    def __init__(self):
        self.calls = []

    async def __call__(self, **kwargs):
        self.calls.append(kwargs)
        return Response(answer={"query": kwargs["query"], "calls": len(self.calls)})


class _Turn:
    def __init__(self, thread, inputs):
        self.thread = thread
        self.thread_id = thread.id
        self.inputs = inputs
        self.id = f"turn-{len(thread.inputs)}"

    async def interrupt(self):
        self.thread.interrupts += 1
        self.thread.busy = False

    async def stream(self):
        try:
            yield SimpleNamespace(
                method="turn/started",
                payload=TurnStartedNotification(
                    threadId=self.thread_id,
                    turn=Turn(id=self.id, items=[], status="inProgress", error=None),
                ),
            )
            for event in self.thread.codex.usage_events:
                yield event
            if self.thread.codex.run_hook:
                await self.thread.codex.run_hook(self.thread, self.inputs)
            yield SimpleNamespace(
                method="item/agentMessage/delta",
                payload=AgentMessageDeltaNotification(
                    threadId=self.thread_id,
                    turnId=self.id,
                    itemId="item",
                    delta=str(self.inputs),
                ),
            )
            item = AgentMessageThreadItem(type="agentMessage", id="item", text=str(self.inputs))
            yield SimpleNamespace(
                method="item/completed",
                payload=ItemCompletedNotification(
                    threadId=self.thread.id,
                    turnId=self.id,
                    item=item,
                    completedAtMs=0,
                ),
            )
            self.thread.busy = False
            yield SimpleNamespace(
                method="turn/completed",
                payload=TurnCompletedNotification(
                    threadId=self.thread.id,
                    turn=Turn(
                        id=self.id,
                        items=[item],
                        status="completed",
                        duration_ms=1,
                        error=None,
                    ),
                ),
            )
        finally:
            self.thread.stream_closed = True


class _Thread:
    def __init__(self, codex, thread_id):
        self.codex, self.id = codex, thread_id
        self.inputs = []
        self.options = []
        self._busy = False
        self.interrupts = 0
        self.stream_closed = False

    @property
    def busy(self):
        return self._busy

    @busy.setter
    def busy(self, value):
        self._busy = value
        self.codex.notify_status(self.id, "active" if value else "idle")

    async def turn(self, inputs, **kwargs):
        assert not self.busy, "overlapping turns on one thread"
        self.inputs.append(inputs)
        self.options.append(kwargs)
        self.busy = True
        return _Turn(self, inputs)

    async def read(self, **_kwargs):
        return SimpleNamespace(
            thread=SimpleNamespace(
                status=SimpleNamespace(root=SimpleNamespace(type="active" if self.busy else "idle")),
                path=getattr(self, "path", None),
            ),
        )

    async def compact(self):
        self.busy = True
        if self.codex.compact_hook:
            await self.codex.compact_hook(self)
        else:
            self.busy = False


class _Codex:
    instances = []

    def __init__(self, config):
        self.config = config
        self.threads = {}
        self.loaded = set()
        self.unsubscribed = []
        self.closed = 0
        self.logins = []
        self.opened = []
        self.run_hook = None
        self.usage_events = []
        self.compact_hook = None
        self.notifications = asyncio.Queue()
        self._client = SimpleNamespace(
            request=self.request,
            next_notification=self.next_notification,
            thread_read=self.thread_read,
        )
        self.instances.append(self)

    async def account(self):
        await asyncio.sleep(0)
        return SimpleNamespace(account=object())

    async def login_api_key(self, key):
        self.logins.append(key)

    async def close(self):
        from openai_codex import TransportClosedError

        self.closed += 1
        self.notifications.put_nowait(TransportClosedError("closed"))

    def notify_status(self, thread_id, status):
        self.notifications.put_nowait(
            SimpleNamespace(
                method="thread/status/changed",
                payload=SimpleNamespace(thread_id=thread_id, status=SimpleNamespace(root=SimpleNamespace(type=status))),
            ),
        )

    async def next_notification(self):
        notification = await self.notifications.get()
        if isinstance(notification, Exception):
            raise notification
        return notification

    async def thread_start(self, **options):
        thread = _Thread(self, f"thread-{len(self.threads) + 1}")
        self.threads[thread.id] = thread
        self.loaded.add(thread.id)
        self.opened.append(("start", thread.id, options))
        return thread

    async def thread_resume(self, thread_id, **options):
        self.loaded.add(thread_id)
        self.opened.append(("resume", thread_id, options))
        return self.threads[thread_id]

    async def thread_read(self, thread_id, **kwargs):
        return await self.threads[thread_id].read(**kwargs)

    async def thread_fork(self, thread_id, **options):
        forked = await self.thread_start(**options)
        self.opened[-1] = ("fork", thread_id, options)
        return forked

    async def request(self, method, params, **_kwargs):
        if method == "thread/loaded/list":
            return SimpleNamespace(data=list(self.loaded), next_cursor=None)
        assert method == "thread/unsubscribe"
        self.unsubscribed.append(params["threadId"])
        return SimpleNamespace(status="unsubscribed")


def _wrapper(tmp_path, **kwargs):
    context = ApplicationContext(workspace_dir=str(tmp_path))
    job = _Job()
    context.jobs[job.name] = job
    server = CodexAppServer(
        name="default",
        app_context=context,
        auth_mode="oauth",
        codex_home=kwargs.pop("codex_home", None),
    )
    context.components = {ComponentEnum.CODEX_APP_SERVER: {"default": server}}
    return CodexAgentWrapper(app_context=context, **kwargs), job


@pytest_asyncio.fixture
async def system(tmp_path, monkeypatch):
    _Codex.instances = []
    monkeypatch.setattr("openai_codex.AsyncCodex", _Codex)
    first, job = _wrapper(tmp_path, name="first")
    context = first.app_context
    second = CodexAgentWrapper(name="second", app_context=context)
    server = context.components[ComponentEnum.CODEX_APP_SERVER]["default"]
    monkeypatch.setattr(server, "_mcp_url", AsyncMock(return_value="http://127.0.0.1/mcp"))
    try:
        yield first, second, server, job
    finally:
        await first.close()
        await second.close()
        await server.close()


@pytest.mark.asyncio
async def test_session_listener_observes_native_activity_and_disconnect_without_a_turn(system):
    first, _second, server, _job = system
    session_id = await first.create_session(job_tools=[])
    notifications = []
    unsubscribe = server.watch_session(session_id, notifications.append)
    native = _Codex.instances[0]
    native.notify_status(session_id, "active")
    native.notify_status(session_id, "notLoaded")
    async with asyncio.timeout(1):
        while len(notifications) < 2:
            await asyncio.sleep(0)
    assert [status.status for status in notifications] == ["active", "not_loaded"]
    native.notifications.put_nowait(ConnectionError("upstream disconnected"))
    async with asyncio.timeout(1):
        while notifications[-1] is not None:
            await asyncio.sleep(0)
    assert server._failed
    unsubscribe()
    assert session_id not in server._session_listeners


def _usage_event(last, total, *, turn_id="turn-2", estimate=None):
    def breakdown(tokens, total_tokens=None):
        return TokenUsageBreakdown(
            inputTokens=tokens[0],
            outputTokens=tokens[1],
            totalTokens=sum(tokens) if total_tokens is None else total_tokens,
            cachedInputTokens=0,
            reasoningOutputTokens=0,
        )

    return SimpleNamespace(
        method="thread/tokenUsage/updated",
        payload=ThreadTokenUsageUpdatedNotification(
            threadId="thread-1",
            turnId=turn_id,
            tokenUsage=ThreadTokenUsage(
                last=breakdown(last, estimate),
                total=breakdown(total),
                modelContextWindow=934435,
            ),
        ),
    )


@pytest.mark.asyncio
async def test_rejected_first_turn_keeps_the_empty_session_and_zero_usage_cache(system):
    first, _, server, _ = system
    session_id = await first.create_session(job_tools=[])
    thread = _Codex.instances[0].threads[session_id]
    thread.path = str(server.session_path / "not-materialized.jsonl")
    turn = thread.turn
    thread.turn = AsyncMock(side_effect=RuntimeError("turn rejected before acceptance"))
    with pytest.raises(RuntimeError, match="turn rejected"):
        await first.reply("rejected", resume=session_id, job_tools=[])
    assert server._threads[session_id].pending is thread
    assert server._threads[session_id].token_usage == TokenUsage()
    thread.turn = turn
    assert (await first.reply("retry", resume=session_id, job_tools=[]))["result"] == "retry"


@pytest.mark.asyncio
async def test_wrapper_locates_an_empty_session_before_its_rollout_exists(system, monkeypatch):
    first, _, server, _ = system
    session_id = await first.create_session(job_tools=[])
    client = _Codex.instances[0]
    path = server.session_path / "sessions" / "not-materialized.jsonl"
    read = AsyncMock(return_value=SimpleNamespace(thread=SimpleNamespace(path=str(path))))
    monkeypatch.setattr(client._client, "thread_read", read)

    assert await first.get_session_path(session_id) == path.resolve()
    read.assert_awaited_once_with(session_id, include_turns=False)
    assert not path.exists()
    assert not client.threads[session_id].inputs
    assert [operation for operation, _, _ in client.opened] == ["start"]


@pytest.mark.asyncio
async def test_releasing_a_superseded_empty_session_keeps_other_threads(system):
    first, _, server, _ = system
    previous = await first.create_session(job_tools=[])
    current = await first.create_session(job_tools=[])
    client = _Codex.instances[0]
    await first.release_session(previous)
    assert server._threads[previous].pending is None
    assert server._threads[current].pending is not None
    assert client.unsubscribed == [previous]
    assert not client.threads[previous].inputs and not client.threads[current].inputs


@pytest.mark.asyncio
@pytest.mark.parametrize("language", ["en", "zh"])
async def test_pa_memory_path_reaches_codex_without_snapshotting_contents(system, language, monkeypatch):
    from datetime import datetime

    monkeypatch.setattr(
        "reme.steps.proactive.base_pa_step.BasePaStep._now",
        lambda _self: datetime.fromisoformat("2026-11-08T12:00:00+08:00"),
    )
    agent, _, _, _ = system
    context = agent.app_context
    context.app_config.language = language
    from .pa_helpers import bind_pa

    bind_pa(context, agent)
    context.components["agent_wrapper"] = {"default": agent}
    memory = agent.workspace_path / "daily/personal/MEMORY.md"
    memory.parent.mkdir(parents=True)
    memory.write_text("# Preferences\n- Use Chinese.\n", encoding="utf-8")
    session_id = (await PaNewSessionStep(app_context=context)()).answer
    client = _Codex.instances[0]
    prompt = client.opened[-1][2]["developer_instructions"]
    assert str(memory) in prompt and str(agent.workspace_path) in prompt
    assert "Use Chinese." not in prompt
    assert ("读最新完整记忆" if language == "zh" else "Read the complete latest file") in prompt
    assert ("要求遗忘时立即删除" if language == "zh" else "Remove requested memories immediately") in prompt
    assert ("不转存，也不从旧对话或摘要恢复" if language == "zh" else "restoring them from old history") in prompt
    assert not client.threads[session_id].inputs
    memory.write_text("# Preferences\n- Use English.\n", encoding="utf-8")
    for _ in range(2):
        response = await PaChatStep(app_context=context)(query="hello", stream_queue=asyncio.Queue())
        assert response.success
    assert all(options["developer_instructions"] == prompt for _, _, options in client.opened)
    await PaNewSessionStep(app_context=context)()
    assert client.opened[-1][2]["developer_instructions"] == prompt
    assert memory.read_text(encoding="utf-8") == "# Preferences\n- Use English.\n"


@pytest.mark.asyncio
async def test_empty_session_preserves_thread_options_and_resumes_without_generation(system):
    first, _, _, job = system
    options = {"job_tools": [job.name], "system_prompt": "Session instructions."}
    session_id = await first.create_session(**options)
    client = _Codex.instances[0]
    assert client.opened[0][0] == "start"
    assert client.opened[0][2]["developer_instructions"] == options["system_prompt"]
    assert first.app_server._threads[session_id].binding is not None
    assert client.unsubscribed == []
    assert not client.threads[session_id].inputs
    result = await first.reply("first message", resume=session_id, **options)
    assert result["session_id"] == session_id
    assert [operation for operation, _, _ in client.opened] == ["start"]
    assert client.unsubscribed == [session_id]
    await first.reply("next message", resume=session_id, **options)
    assert [operation for operation, _, _ in client.opened] == ["start", "resume"]


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("baseline,initial_estimate", [((0, 0), False), ((50000, 1000), False), ((50000, 1000), True)])
async def test_turn_usage_counts_all_calls_without_recounting_restored_or_duplicate_usage(
    system,
    stream,
    baseline,
    initial_estimate,
):
    first, _, server, _ = system
    session_id = (await first.reply("initial"))["session_id"]
    client = _Codex.instances[0]
    server.update_token_usage(session_id, TokenUsage(input_tokens=baseline[0], output_tokens=baseline[1]))

    def total(inputs, outputs):
        return baseline[0] + inputs, baseline[1] + outputs

    first_call = _usage_event((10, 2), total(10, 2))
    client.usage_events = [_usage_event((9000, 900), (9000, 900), turn_id="turn-1")]
    if initial_estimate:
        client.usage_events.append(_usage_event((0, 0), baseline, estimate=800000))
    client.usage_events.extend(
        [
            first_call,
            first_call,
            _usage_event((20, 3), total(30, 5)),
            _usage_event((0, 0), total(30, 5), estimate=700000),
            first_call,
            _usage_event((7, 1), total(37, 6)),
        ],
    )
    expected = {"input_tokens": 37, "output_tokens": 6, "total_tokens": 43}
    if stream:
        chunks = [chunk async for chunk in first.reply_stream("multi-call", session_id=session_id)]
        updates = [chunk for chunk in chunks if chunk.chunk_type == ChunkEnum.USAGE]
        assert updates[-1].chunk == expected
        assert [chunk.chunk["total_tokens"] for chunk in updates] == [12, 35, 43]
        assert {chunk.block_id for chunk in updates} == {"usage:turn-2"}
    else:
        response = await first.reply("multi-call", session_id=session_id)
        assert response["result"] == "multi-call"
        assert response["usage"] == expected

    counter = global_counter_get_all(first.app_context.metadata, ["__token_counter", first.name])["children"]
    assert {name: data["value"] for name, data in counter.items()} == expected

    client.usage_events = [_usage_event((2, 1), total(39, 7), turn_id="turn-3")]
    response = await first.reply("next turn", session_id=session_id)
    assert response["usage"] == {"input_tokens": 2, "output_tokens": 1, "total_tokens": 3}
    counter = global_counter_get_all(first.app_context.metadata, ["__token_counter", first.name])["children"]
    assert counter["total_tokens"]["value"] == 46


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("estimate_only", [False, True])
async def test_turn_usage_without_provider_reports_remains_unavailable(system, stream, estimate_only):
    first, _, _, _ = system
    session_id = (await first.reply("initial"))["session_id"]
    if estimate_only:
        first.app_server.update_token_usage(session_id, TokenUsage(input_tokens=50000, output_tokens=1000))
        _Codex.instances[0].usage_events = [_usage_event((0, 0), (50000, 1000), estimate=700000)]
    if stream:
        chunks = [chunk async for chunk in first.reply_stream("no usage", session_id=session_id)]
        assert not any(chunk.chunk_type == ChunkEnum.USAGE for chunk in chunks)
    else:
        assert (await first.reply("no usage", session_id=session_id))["usage"] is None
    assert global_counter_get_all(first.app_context.metadata, ["__token_counter", first.name]) is None


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [False, True])
async def test_failed_turn_records_reported_spending_once(system, stream):
    first, _, _, _ = system
    session_id = (await first.reply("initial"))["session_id"]
    client = _Codex.instances[0]
    first.app_server.update_token_usage(session_id, TokenUsage(input_tokens=5000, output_tokens=500))
    event = _usage_event((10, 2), (5010, 502))
    client.usage_events = [event, event]

    async def fail(*_args):
        raise ValueError("model failed after a completed tool round")

    client.run_hook = fail
    with pytest.raises(ValueError, match="model failed"):
        if stream:
            _ = [chunk async for chunk in first.reply_stream("fails", session_id=session_id)]
        else:
            await first.reply("fails", session_id=session_id)
    counter = global_counter_get_all(first.app_context.metadata, ["__token_counter", first.name])["children"]
    assert counter["input_tokens"]["value"] == 10
    assert counter["output_tokens"]["value"] == 2
    assert counter["total_tokens"]["value"] == 12


@pytest.mark.asyncio
async def test_closing_stream_keeps_reported_usage_and_interrupts_the_turn(system):
    first, _, _, _ = system
    session_id = (await first.reply("initial"))["session_id"]
    client = _Codex.instances[0]
    first.app_server.update_token_usage(session_id, TokenUsage(input_tokens=5000, output_tokens=500))
    client.usage_events = [_usage_event((10, 2), (5010, 502))]
    stream = first.reply_stream("interrupted", session_id=session_id)
    assert (await anext(stream)).chunk_type == ChunkEnum.REPLY_START
    assert (await anext(stream)).chunk_type == ChunkEnum.USAGE
    await stream.aclose()
    assert client.threads[session_id].interrupts == 1
    counter = global_counter_get_all(first.app_context.metadata, ["__token_counter", first.name])["children"]
    assert counter["total_tokens"]["value"] == 12


@pytest.mark.asyncio
async def test_cancelling_reply_keeps_already_reported_usage(system):
    first, _, _, _ = system
    session_id = (await first.reply("initial"))["session_id"]
    client = _Codex.instances[0]
    first.app_server.update_token_usage(session_id, TokenUsage(input_tokens=5000, output_tokens=500))
    client.usage_events = [_usage_event((10, 2), (5010, 502))]
    started = asyncio.Event()

    async def hold(*_args):
        started.set()
        await asyncio.Event().wait()

    client.run_hook = hold
    task = asyncio.create_task(first.reply("cancel", session_id=session_id))
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    counter = global_counter_get_all(first.app_context.metadata, ["__token_counter", first.name])["children"]
    assert counter["total_tokens"]["value"] == 12


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [False, True])
async def test_compaction_replaying_previous_usage_under_current_turn_does_not_recount_it(system, stream):
    first, _, server, _ = system
    session_id = (await first.reply("initial"))["session_id"]
    server.update_token_usage(session_id, TokenUsage(input_tokens=1189426, output_tokens=220))
    _Codex.instances[0].usage_events = [
        _usage_event((610679, 61), (1189426, 220)),
        _usage_event((606779, 471), (1796205, 691)),
        _usage_event((0, 0), (1796205, 691), estimate=23616),
        _usage_event((61232, 23), (1857437, 714)),
    ]
    expected = {"input_tokens": 668011, "output_tokens": 494, "total_tokens": 668505}
    if stream:
        chunks = [chunk async for chunk in first.reply_stream("compaction", session_id=session_id)]
        usages = [chunk.chunk for chunk in chunks if chunk.chunk_type == ChunkEnum.USAGE]
        assert [usage["total_tokens"] for usage in usages] == [607250, 668505]
        assert usages[-1] == expected
    else:
        assert (await first.reply("compaction", session_id=session_id))["usage"] == expected
    counter = global_counter_get_all(first.app_context.metadata, ["__token_counter", first.name])["children"]
    assert {key: value["value"] for key, value in counter.items()} == expected


@pytest.mark.asyncio
@pytest.mark.parametrize("interrupted", [False, True])
async def test_usage_baseline_recovers_after_lost_state_or_missed_stream_usage(system, interrupted):
    first, second, server, _ = system
    session_id = (await first.reply("initial"))["session_id"]
    client = _Codex.instances[0]
    if interrupted:
        stream = first.reply_stream("interrupted", session_id=session_id)
        assert (await anext(stream)).chunk_type == ChunkEnum.REPLY_START
        assert (await anext(stream)).chunk_type == ChunkEnum.CONTENT
    # Native completion can overtake a paused consumer and persist unseen usage.
    path = server.session_path / "rollout.jsonl"
    usage = {"total_token_usage": {"input_tokens": 5000, "output_tokens": 500}}
    record = {"type": "event_msg", "payload": {"type": "token_count", "info": usage}}
    path.write_text(json.dumps(record) + "\n")
    client.threads[session_id].path = str(path)
    if interrupted:
        client.threads[session_id].busy = False
        await stream.aclose()
    else:
        server._threads.pop(session_id)
    turn_id = "turn-3" if interrupted else "turn-2"
    client.usage_events = [
        _usage_event((1000, 100), (5000, 500), turn_id=turn_id),
        _usage_event((10, 2), (5010, 502), turn_id=turn_id),
    ]
    assert (await second.reply("restored", session_id=session_id))["usage"] == {
        "input_tokens": 10,
        "output_tokens": 2,
        "total_tokens": 12,
    }


@pytest.mark.asyncio
async def test_fresh_thread_does_not_require_its_rollout_to_be_materialized(system, monkeypatch):
    first, _, server, _ = system
    original_start = _Codex.thread_start
    pending = server.session_path / "not-materialized.jsonl"

    async def start(client, **options):
        thread = await original_start(client, **options)
        thread.path = str(pending)
        return thread

    monkeypatch.setattr(_Codex, "thread_start", start)
    assert (await first.reply("first turn"))["usage"] is None
    assert not pending.exists()


@pytest.mark.asyncio
async def test_wrappers_share_one_lazy_client_and_keep_independent_sessions(system):
    first, second, server, _ = system
    await first.start()
    assert not _Codex.instances
    results = await asyncio.gather(first.reply("first"), second.reply("second"))
    assert len(_Codex.instances) == 1
    assert results[0]["session_id"] != results[1]["session_id"]
    assert first.app_server is second.app_server is server
    client = _Codex.instances[0]
    await first.close()
    assert client.closed == 0
    assert (await second.reply("still running"))["result"] == "still running"
    await server.close()
    assert client.closed == 1


@pytest.mark.asyncio
async def test_different_threads_run_concurrently_and_same_thread_serializes_across_wrappers(system):
    first, second, server, _ = system
    thread_id = (await first.reply("initial"))["session_id"]
    client = _Codex.instances[0]
    entered, release = asyncio.Event(), asyncio.Event()
    observed = []

    async def run_hook(thread, inputs):
        observed.append((thread.id, inputs))
        if inputs == "hold":
            entered.set()
            await release.wait()

    client.run_hook = run_hook
    async with asyncio.TaskGroup() as group:
        held = group.create_task(first.reply("hold", session_id=thread_id))
        await asyncio.wait_for(entered.wait(), 2)
        queued = group.create_task(second.reply("same", session_id=thread_id))
        try:
            independent = await asyncio.wait_for(second.reply("other"), 2)
            assert independent["session_id"] != thread_id
            assert not queued.done()
            assert (thread_id, "same") not in observed
        finally:
            release.set()
    assert held.result()["session_id"] == queued.result()["session_id"] == thread_id
    assert len(server._threads) == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("approval_mode", [None, "auto_review"])
async def test_resume_fork_and_sdk_options_remain_explicit(system, approval_mode):
    from openai_codex import ApprovalMode, Sandbox
    from openai_codex.types import Personality, ReasoningEffort

    first, _, _, _ = system
    original = await first.reply(
        "one",
        model="custom",
        system_prompt="instructions",
        personality="friendly",
        approval_mode=approval_mode,
    )
    thread_id = original["session_id"]
    resumed = await first.reply("two", resume=thread_id, effort="high")
    forked = await first.reply("three", session_id=thread_id, fork_session=True)
    assert resumed["session_id"] == thread_id
    assert forked["session_id"] != thread_id
    client = _Codex.instances[0]
    options = client.opened[0][2]
    assert options["model"] == "custom"
    assert options["developer_instructions"] == "instructions"
    assert options["base_instructions"] is None
    assert options["approval_mode"] == (ApprovalMode.auto_review if approval_mode else ApprovalMode.deny_all)
    assert options["sandbox"] == Sandbox.full_access
    assert options["personality"] == Personality.friendly
    assert client.threads[thread_id].options[-1]["effort"] == ReasoningEffort.high


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "changes",
    [
        {"tool_context_id": "other"},
        {"job_tools": []},
        {"injected_job_kwargs": {"_allowed_paths": ["other.md"]}},
    ],
)
async def test_loaded_thread_constraints_cannot_be_changed_by_another_wrapper(system, changes):
    first, second, server, _ = system
    binding = {
        "job_tools": ["search"],
        "tool_context_id": "one",
        "injected_job_kwargs": {"_allowed_paths": ["one.md"]},
    }
    original = await first.reply("one", **binding)
    thread_id = original["session_id"]
    token = server._threads[thread_id].token
    with pytest.raises(ValueError, match="cannot change"):
        await second.reply("two", session_id=thread_id, **{**binding, **changes})
    assert token in server.jobs_mcp.scopes
    await second.reply("three", session_id=thread_id, **binding)


@pytest.mark.asyncio
@pytest.mark.parametrize("restore", ["loaded", "unloaded", "restart"])
async def test_fork_inherits_source_instructions_tools_and_turn_settings(system, restore):
    from openai_codex import Sandbox
    from openai_codex.types import ReasoningEffort

    first, second, server, job = system
    original = await first.reply(
        "source work",
        system_prompt="Original instructions.",
        model="original-model",
        effort="high",
        sandbox="read-only",
        personality="friendly",
        job_tools=[job.name],
        injected_job_kwargs={"_allowed_paths": ["original.md"]},
        config={"mcp_servers": {"external": {"url": "https://example.test/mcp"}}},
    )
    source_id = original["session_id"]
    source_token = server._threads[source_id].token
    source_tools = server.jobs_mcp.scopes[source_token]
    settings_path = server._message_path(source_id).with_suffix(".config.jsonl")
    assert source_token not in settings_path.read_text()
    native = _Codex.instances[0]
    if restore != "loaded":
        native.loaded.clear()
    if restore == "restart":
        server._threads.clear()
        server.jobs_mcp.scopes.clear()
    second.kwargs.update(system_prompt="Other instructions.", model="other-model", effort="low", job_tools=[])

    forked = await second.reply("Only summarize.", resume=source_id, fork_session=True)

    assert forked["session_id"] != source_id and native.threads[source_id].inputs == ["source work"]
    assert native.threads[forked["session_id"]].inputs == ["Only summarize."]
    operation, parent, options = native.opened[-1]
    assert operation == "fork" and parent == source_id
    assert options["developer_instructions"] == "Original instructions."
    assert options["model"] == "original-model" and options["sandbox"] == Sandbox.read_only
    assert options["config"]["personality"] == "friendly"
    assert options["config"]["mcp_servers"]["external"]["url"] == "https://example.test/mcp"
    fork_token = server._threads[forked["session_id"]].token
    assert fork_token != source_token
    tools = server.jobs_mcp.scopes[fork_token]
    assert [(tool.name, tool.description, tool.parameters) for tool in tools] == [
        (tool.name, tool.description, tool.parameters) for tool in source_tools
    ]
    await tools[0].fn(query="read")
    assert job.calls[-1]["_allowed_paths"] == ["original.md"]
    assert native.threads[forked["session_id"]].options[-1]["effort"] == ReasoningEffort.high


@pytest.mark.asyncio
@pytest.mark.parametrize("persisted", [False, True])
async def test_fork_of_unstarted_session_preserves_source_options_and_tools(system, monkeypatch, persisted):
    first, second, server, job = system
    source_id = await first.create_session(
        system_prompt="Source instructions.",
        model="source-model",
        job_tools=[job.name],
        injected_job_kwargs={"_allowed_paths": ["source.md"]},
    )
    native = _Codex.instances[0]
    source = native.threads[source_id]
    source.path = str(server.session_path / "sessions" / "source.jsonl")
    if persisted:
        Path(source.path).parent.mkdir(parents=True, exist_ok=True)
        Path(source.path).write_text("", encoding="utf-8")
    fork = AsyncMock(
        wraps=native.thread_fork,
        side_effect=None if persisted else RuntimeError("no rollout found for thread id"),
    )
    monkeypatch.setattr(native, "thread_fork", fork)
    second.kwargs.update(system_prompt="Unrelated instructions.", model="other-model", job_tools=[])

    reply = await second.reply("Prepare context.", resume=source_id, fork_session=True)

    assert reply["status"] == "completed" and reply["session_id"] != source_id
    assert source.inputs == [] and server._threads[source_id].pending is source
    assert source_id not in native.unsubscribed
    assert native.opened[-1][0] == ("fork" if persisted else "start")
    assert native.opened[-1][2]["developer_instructions"] == "Source instructions."
    assert native.opened[-1][2]["model"] == "source-model"
    assert fork.await_count == int(persisted)
    token = server._threads[reply["session_id"]].token
    await server.jobs_mcp.scopes[token][0].fn(query="read")
    assert job.calls[-1]["_allowed_paths"] == ["source.md"]


@pytest.mark.asyncio
async def test_temporary_preparation_uses_an_independent_empty_source_fork(system, monkeypatch):
    first, _, server, job = system
    runtime = bind_pa(first.app_context, first)
    runtime._initialize_files()
    source_id = await first.create_session(system_prompt="Source instructions.", job_tools=[job.name])
    runtime.session_id = source_id
    task = await runtime.temp_tasks.create("Prepare an isolated one-shot task.")
    native = _Codex.instances[0]
    fork = AsyncMock(side_effect=RuntimeError("no rollout found for thread id"))
    monkeypatch.setattr(native, "thread_fork", fork)
    step = TempProactiveStep(app_context=first.app_context)

    async with asyncio.timeout(3):
        prepared = await step._prepare(task)

    assert prepared.status == "scheduled" and prepared.error is None and prepared.retry_at is None
    assert task.query in (runtime.workspace_path / prepared.handoff_path).read_text()
    assert runtime.session_id == source_id and native.threads[source_id].inputs == []
    assert server._threads[source_id].pending is native.threads[source_id]
    fork.assert_not_awaited()


@pytest.mark.asyncio
async def test_fork_waits_for_source_and_cancellation_preserves_its_tools(system):
    first, second, server, _ = system
    source_id = (await first.reply("initial", job_tools=["search"]))["session_id"]
    token = server._threads[source_id].token
    entered, release = asyncio.Event(), asyncio.Event()
    native = _Codex.instances[0]

    async def hold(_thread, _inputs):
        entered.set()
        await release.wait()

    native.run_hook = hold
    running = asyncio.create_task(first.reply("running", resume=source_id, job_tools=["search"]))
    await asyncio.wait_for(entered.wait(), 2)
    forked = asyncio.create_task(second.reply("handoff", resume=source_id, fork_session=True))
    try:
        await asyncio.sleep(0.02)
        assert not any(operation == "fork" for operation, _, _ in native.opened)
        forked.cancel()
        with pytest.raises(asyncio.CancelledError):
            await forked
        assert token in server.jobs_mcp.scopes and len(server.jobs_mcp.scopes) == 1
    finally:
        release.set()
        await running


@pytest.mark.asyncio
@pytest.mark.parametrize("error", ["fork failed", "no rollout found for thread id"])
async def test_failed_fork_revokes_only_its_new_tool_scope(system, monkeypatch, error):
    first, _, server, _ = system
    source_id = (await first.reply("source", job_tools=["search"]))["session_id"]
    token = server._threads[source_id].token
    monkeypatch.setattr(_Codex.instances[0], "thread_fork", AsyncMock(side_effect=RuntimeError(error)))
    with pytest.raises(RuntimeError, match=error):
        await first.reply("handoff", resume=source_id, fork_session=True)
    assert set(server.jobs_mcp.scopes) == {token}


@pytest.mark.asyncio
async def test_application_bound_tool_approval_preserves_native_read_only_and_external_policy(system):
    from openai_codex import Sandbox

    first, _, _, _ = system
    external = {"url": "http://external.test/mcp", "tools": {"write": {"approval_mode": "prompt"}}}
    await first.reply(
        "one",
        job_tools=["search"],
        sandbox="read-only",
        config={"mcp_servers": {"external": external}},
    )
    options = _Codex.instances[0].opened[0][2]
    assert options["sandbox"] == Sandbox.read_only
    servers = options["config"]["mcp_servers"]
    assert servers["reme_jobs"]["enabled_tools"] == ["search"]
    assert servers["reme_jobs"]["tools"] == {"search": {"approval_mode": "approve"}}
    assert servers["external"] == external


@pytest.mark.asyncio
async def test_unloaded_thread_rebinds_tools_and_revokes_old_token(system):
    first, _, server, _ = system
    original = await first.reply("one", job_tools=["search"], tool_context_id="one")
    thread_id = original["session_id"]
    token = server._threads[thread_id].token
    _Codex.instances[0].loaded.clear()
    await first.reply("two", session_id=thread_id, job_tools=["search"], tool_context_id="two")
    assert token not in server.jobs_mcp.scopes
    assert server._threads[thread_id].token != token


@pytest.mark.asyncio
async def test_cancelling_reply_interrupts_only_its_turn_and_keeps_shared_server(system):
    first, second, _, _ = system
    thread_id = (await first.reply("initial"))["session_id"]
    entered = asyncio.Event()

    async def hold(_thread, _inputs):
        entered.set()
        await asyncio.Event().wait()

    client = _Codex.instances[0]
    client.run_hook = hold
    task = asyncio.create_task(first.reply("cancel", session_id=thread_id))
    await asyncio.wait_for(entered.wait(), 2)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    client.run_hook = None
    assert client.threads[thread_id].interrupts == 1
    assert client.closed == 0
    assert (await second.reply("okay"))["result"] == "okay"
    assert not client.threads[thread_id].busy


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["turn", "compact"])
@pytest.mark.parametrize("fail", [False, True])
async def test_cancelled_dispatch_settles_before_resuming_the_thread(system, monkeypatch, operation, fail):
    first, second, server, _ = system
    thread_id = (await first.reply("initial"))["session_id"]
    client = _Codex.instances[0]
    thread = client.threads[thread_id]
    server.update_token_usage(thread_id, TokenUsage(input_tokens=100, output_tokens=10))
    original = getattr(thread, operation)
    entered, release = asyncio.Event(), asyncio.Event()

    async def dispatch(*args, **kwargs):
        entered.set()
        await release.wait()
        if fail:
            raise ValueError("dispatch rejected")
        if operation == "compact":
            path = server.session_path / "rollout.jsonl"
            path.write_text(
                '{"type":"event_msg","payload":{"type":"token_count","info":'
                '{"total_token_usage":{"input_tokens":500,"output_tokens":50}}}}\n',
            )
            thread.path = str(path)
            client.usage_events = [_usage_event((10, 2), (510, 52))]
        return await original(*args, **kwargs)

    monkeypatch.setattr(thread, operation, dispatch)
    request = (
        first.compact_session(thread_id) if operation == "compact" else first.reply("cancel", session_id=thread_id)
    )
    task = asyncio.create_task(request)
    queued = None
    try:
        await asyncio.wait_for(entered.wait(), 2)
        task.cancel("original cancellation")
        await asyncio.sleep(0)
        task.cancel("cancel again")
        await asyncio.sleep(0)
        queued = asyncio.create_task(second.reply("later", session_id=thread_id))
        await asyncio.sleep(0.01)
        assert not queued.done()
        monkeypatch.setattr(thread, operation, original)
        release.set()
        with pytest.raises(asyncio.CancelledError, match="original cancellation"):
            await asyncio.wait_for(task, 2)
        result = await asyncio.wait_for(queued, 2)
        assert result["result"] == "later"
        if operation == "compact" and not fail:
            assert result["usage"] == {"input_tokens": 10, "output_tokens": 2, "total_tokens": 12}
        assert thread.interrupts == int(operation == "turn" and not fail)
        await asyncio.wait_for(server.close(), 2)
    finally:
        release.set()
        task.cancel()
        if queued is not None:
            queued.cancel()
        await asyncio.gather(task, *([queued] if queued is not None else []), return_exceptions=True)


@pytest.mark.asyncio
async def test_stream_close_interrupts_and_routes_chunks_per_session(system):
    first, second, _, _ = system
    stream = first.reply_stream("early")
    start = await anext(stream)
    thread_id = start.session_id
    await stream.aclose()
    thread = _Codex.instances[0].threads[thread_id]
    assert thread.interrupts == 1
    assert thread.stream_closed
    chunks = [chunk async for chunk in second.reply_stream("complete")]
    assert chunks[0].chunk_type == ChunkEnum.REPLY_START
    assert chunks[-1].chunk_type == ChunkEnum.REPLY_END
    assert all(chunk.session_id != thread_id for chunk in chunks)


@pytest.mark.asyncio
async def test_compaction_holds_thread_lock_until_it_finishes(system):
    first, second, _, _ = system
    thread_id = (await first.reply("initial"))["session_id"]
    entered, release = asyncio.Event(), asyncio.Event()

    async def compact_hook(thread):
        entered.set()

        async def finish():
            await release.wait()
            thread.busy = False

        asyncio.create_task(finish())

    _Codex.instances[0].compact_hook = compact_hook
    async with asyncio.TaskGroup() as group:
        group.create_task(first.compact_session(thread_id))
        await asyncio.wait_for(entered.wait(), 2)
        queued = group.create_task(second.reply("later", session_id=thread_id))
        try:
            await asyncio.sleep(0.02)
            assert not queued.done()
        finally:
            release.set()
    assert queued.result()["session_id"] == thread_id


@pytest.mark.asyncio
@pytest.mark.parametrize("cancelled", [False, True])
async def test_compaction_acknowledgement_does_not_release_the_thread_before_it_starts(system, monkeypatch, cancelled):
    first, second, _, _ = system
    thread_id = (await first.reply("initial"))["session_id"]
    client = _Codex.instances[0]
    thread = client.threads[thread_id]
    acknowledged = asyncio.Event()

    async def compact():
        acknowledged.set()

    monkeypatch.setattr(thread, "compact", compact)
    async with asyncio.TaskGroup() as group:
        compacting = group.create_task(first.compact_session(thread_id))
        await asyncio.wait_for(acknowledged.wait(), 2)
        if cancelled:
            compacting.cancel()
        queued = group.create_task(second.reply("later", session_id=thread_id))
        try:
            await asyncio.sleep(0.01)
            # Cancellation may end the caller during cleanup; its barrier must still block the next turn.
            if not cancelled:
                assert not compacting.done()
            assert not queued.done()
            client.notify_status(thread_id, "idle")
            await asyncio.sleep(0)
            if not cancelled:
                assert not compacting.done()
            assert not queued.done()
        finally:
            client.notify_status(thread_id, "active")
            client.notify_status(thread_id, "idle")
    if cancelled:
        assert compacting.cancelled()
    assert queued.result()["session_id"] == thread_id


@pytest.mark.asyncio
async def test_close_waits_for_active_request_without_closing_other_wrappers(system):
    first, second, server, _ = system
    await first.reply("initial")
    entered, release = asyncio.Event(), asyncio.Event()

    async def hold(_thread, _inputs):
        entered.set()
        await release.wait()

    client = _Codex.instances[0]
    client.run_hook = hold
    async with asyncio.TaskGroup() as group:
        group.create_task(first.reply("held"))
        await asyncio.wait_for(entered.wait(), 2)
        closing = group.create_task(first.close())
        try:
            await asyncio.sleep(0)
            assert not closing.done()
            assert client.closed == 0
        finally:
            release.set()
    client.run_hook = None
    await second.reply("still works")
    assert server.is_started


class _StructuredModel(BaseModel):
    ok: bool


@pytest.mark.asyncio
async def test_structured_reply_and_native_inputs(system):
    from openai_codex import LocalImageInput, TextInput

    first, _, _, _ = system
    result = await first.reply('{"ok": true}', output_schema=_StructuredModel)
    assert result["structured_output"] == {"ok": True}
    inputs = [TextInput("describe"), LocalImageInput("image.png")]
    await first.reply(inputs)
    assert _Codex.instances[0].threads["thread-2"].inputs[0] is inputs
    chunks = [chunk async for chunk in first.reply_stream('{"ok": false}', output_schema=_StructuredModel)]
    assert chunks[-1].chunk["structured_output"] == {"ok": False}


@pytest.mark.asyncio
@pytest.mark.parametrize("kwargs", [{"resume": "a", "session_id": "b"}, {"fork_session": True}])
async def test_invalid_session_selection_fails_without_starting_client(system, kwargs):
    first, _, _, _ = system
    with pytest.raises(ValueError):
        await first.reply("invalid", **kwargs)
    assert not _Codex.instances


@pytest.mark.asyncio
@pytest.mark.parametrize("job_class", [BackgroundJob, StreamJob])
async def test_unsupported_job_types_are_rejected(system, job_class):
    first, _, _, _ = system
    first.app_context.jobs["unsupported"] = job_class(name="unsupported")
    with pytest.raises(TypeError, match="non-stream"):
        await first.reply("invalid", job_tools=["unsupported"])


@pytest.mark.parametrize("name", sorted(CodexAgentWrapper._SERVER_OPTIONS))
def test_process_options_must_be_configured_on_server(name):
    with pytest.raises(TypeError, match="codex_app_server"):
        CodexAgentWrapper(**{name: "invalid"})


def test_server_config_authentication_environment_and_home(tmp_path):
    context = ApplicationContext(workspace_dir=str(tmp_path), environment={"CUSTOM": "preserved"})
    server = CodexAppServer(
        name="one",
        app_context=context,
        auth_mode="api_key",
        api_key="key",
        base_url="https://proxy.test/v1",
    )
    config = server._client_config()
    assert config.env["CUSTOM"] == "preserved"
    assert config.env["CODEX_HOME"] == str(tmp_path / ".codex")
    assert 'forced_login_method="api"' in config.config_overrides
    assert 'model_provider="reme"' in config.config_overrides
    assert 'model_providers.reme.base_url="https://proxy.test/v1"' in config.config_overrides
    assert "model_providers.reme.requires_openai_auth=true" in config.config_overrides
    assert "model_providers.reme.supports_websockets=false" in config.config_overrides
    oauth = CodexAppServer(app_context=context, auth_mode="oauth", api_key="ignored", base_url="ignored")
    assert CodexAppServer(codex_home="~/.codex").session_path == Path.home() / ".codex"
    assert CodexAppServer(app_context=context, codex_home="other-home").session_path == tmp_path / "other-home"
    assert oauth._client_config().config_overrides == ('forced_login_method="chatgpt"',)
    with pytest.raises(ValueError, match="requires a non-empty"):
        CodexAppServer(app_context=context, auth_mode="api_key")._client_config()


def test_custom_provider_websocket_support_and_explicit_overrides_are_preserved(tmp_path):
    server = CodexAppServer(
        app_context=ApplicationContext(workspace_dir=str(tmp_path)),
        auth_mode="api_key",
        api_key="key",
        base_url="https://proxy.test/v1",
        supports_websockets=True,
        config_overrides=['model_provider="custom"'],
    )
    overrides = server._client_config().config_overrides
    assert "model_providers.reme.supports_websockets=true" in overrides
    assert overrides.index('model_provider="custom"') > overrides.index('model_provider="reme"')


def test_server_materializes_model_catalog_inside_its_private_home(tmp_path):
    context = ApplicationContext(workspace_dir=str(tmp_path))
    catalog = {"models": [{"slug": "main", "auto_review_model_override": "reviewer"}, {"slug": "reviewer"}]}
    server = CodexAppServer(name="catalog", app_context=context, model_catalog=catalog)
    config = server._client_config()
    path = server.session_path / "reme-model-catalog.json"
    assert json.loads(path.read_text(encoding="utf-8")) == catalog
    assert f"model_catalog_json={json.dumps(str(path))}" in config.config_overrides
    assert path.is_relative_to(tmp_path)
    with pytest.raises(ValueError, match="not both"):
        CodexAppServer(
            app_context=context,
            model_catalog=catalog,
            config_overrides=['model_catalog_json="external.json"'],
        )._client_config()


@pytest.mark.asyncio
async def test_client_initialization_failure_is_closed_and_retryable(system, monkeypatch):
    first, _, server, _ = system
    monkeypatch.setattr(_Codex, "account", AsyncMock(side_effect=RuntimeError("login failed")))
    with pytest.raises(RuntimeError, match="login failed"):
        await first.reply("fails")
    assert _Codex.instances[0].closed == 1
    assert server._codex is None
    assert server._active == 0


def test_default_and_demo_configuration_use_named_shared_servers(monkeypatch):
    monkeypatch.delenv("CODEX_HOME", raising=False)
    monkeypatch.delenv("CODEX_MODEL_NAME", raising=False)
    monkeypatch.setenv("LLM_API_KEY", "demo-api-key")
    monkeypatch.setenv("LLM_BASE_URL", "https://demo.invalid/v1")
    for name in ("default", "demo"):
        config = resolve_app_config(config=name, log_config=False)
        wrappers = config["components"]["agent_wrapper"]
        assert wrappers["codex"]["app_server"] == "default"
        assert "auth_mode" not in wrappers["codex"]
        servers = config["components"]["codex_app_server"]
        assert servers["default"]["auth_mode"] == "api_key"
        assert servers["default"]["codex_home"] == ".codex"
        assert servers["default"]["supports_websockets"] is False
        assert wrappers["codex"]["model"] == "qwen3.8-max"
        assert wrappers["codex"]["approval_mode"] == "deny_all"
        assert wrappers["codex"]["effort"] == "xhigh"
        models = servers["default"]["model_catalog"]["models"]
        assert {model["slug"] for model in models} == {"qwen3.8-max", "qwen3.8-flash"}
        assert all(model["auto_review_model_override"] == "qwen3.8-flash" for model in models)
        assert all(model["default_reasoning_level"] == "xhigh" for model in models)
        assert all(model["multi_agent_reasoning_effort"] == "xhigh" for model in models)
        assert all([level["effort"] for level in model["supported_reasoning_levels"]] == ["xhigh"] for model in models)
        assert set(servers["default"]["config_overrides"]) == {
            'model_reasoning_effort="xhigh"',
            'review_model="qwen3.8-max"',
            'agents.default_subagent_model="qwen3.8-max"',
            'agents.default_subagent_reasoning_effort="xhigh"',
            'memories.extract_model="qwen3.8-flash"',
            'memories.consolidation_model="qwen3.8-max"',
        }
        if name == "demo":
            assert servers["default"]["api_key"] == "demo-api-key"
            assert servers["default"]["base_url"] == "https://demo.invalid/v1"
        else:
            assert servers["oauth"]["codex_home"] == "~/.codex"
            assert wrappers["codex_oauth"]["effort"] == "xhigh"
            assert set(servers["oauth"]["config_overrides"]) == {
                'model_reasoning_effort="xhigh"',
                'agents.default_subagent_reasoning_effort="xhigh"',
            }


@pytest.mark.asyncio
async def test_scoped_mcp_calls_original_jobs_and_isolates_catalog_arguments_and_failures():
    gateway = ScopedMCP()
    job = _Job()
    first = gateway.register([job], {"tool_context_id": "one", "_allowed_paths": ["one.md"]})
    second = gateway.register([job], {"tool_context_id": "two", "_allowed_paths": ["two.md"]})
    empty = gateway.register([], {})
    try:
        url = await gateway.listen()
        async with Client(StreamableHttpTransport(url, headers={"Authorization": f"Bearer {first}"})) as one:
            async with Client(StreamableHttpTransport(url, headers={"Authorization": f"Bearer {second}"})) as two:
                tools = await one.list_tools()
                assert [tool.name for tool in tools] == ["search"]
                assert "_allowed_paths" not in tools[0].input_schema["properties"]
                await one.call_tool("search", {"query": "a"})
                await two.call_tool("search", {"query": "b"})
                await one.call_tool("search", {"query": "c"})
                with pytest.raises(ToolError, match="injected"):
                    await one.call_tool("search", {"query": "bad", "_allowed_paths": ["escape"]})
        assert [call["tool_context_id"] for call in job.calls] == ["one", "two", "one"]
        assert job.calls[1]["_allowed_paths"] == ["two.md"]
        async with Client(StreamableHttpTransport(url, headers={"Authorization": f"Bearer {empty}"})) as client:
            assert not await client.list_tools()
            with pytest.raises(ToolError):
                await client.call_tool("search", {"query": "not allowed"})
        async with httpx.AsyncClient() as client:
            assert (await client.post(url, json={})).status_code == 403
            assert (await client.post(url, headers={"Authorization": "Bearer unknown"}, json={})).status_code == 403
    finally:
        await gateway.close()
    assert gateway._task is None
    assert not gateway.scopes


@pytest.mark.asyncio
async def test_mounted_mcp_transport_lifespan_can_restart():
    gateway = ScopedMCP()
    app = gateway.app
    for _ in range(2):
        async with app.lifespan(app):
            assert gateway.app is app


@pytest.mark.asyncio
async def test_scoped_mcp_can_restart_its_loopback_listener():
    gateway = ScopedMCP()
    job = _Job()
    try:
        for _ in range(2):
            token = gateway.register([job], {})
            url = await gateway.listen()
            async with Client(StreamableHttpTransport(url, headers={"Authorization": f"Bearer {token}"})) as client:
                await client.call_tool("search", {"query": "same application"})
            await gateway.close()
        assert len(job.calls) == 2
    finally:
        await gateway.close()


@pytest.mark.asyncio
async def test_http_service_mounts_scoped_tools_without_another_listener(tmp_path, monkeypatch):
    app = Application(
        workspace_dir=str(tmp_path),
        enable_logo=False,
        log_to_console=False,
        log_to_file=False,
        service={"backend": "http", "web_enabled": False},
        components={
            "codex_app_server": {"default": {"backend": "sdk", "auth_mode": "oauth"}},
            "agent_wrapper": {"one": {"backend": "codex"}, "two": {"backend": "codex"}},
        },
    )
    service = app.context.service
    service.build_service(app)
    server = app.context.components["codex_app_server"]["default"]
    writer = SimpleNamespace(close=lambda: None, wait_closed=AsyncMock())
    monkeypatch.setattr(asyncio, "open_connection", AsyncMock(return_value=(None, writer)))
    job = _Job()
    token = server.jobs_mcp.register([job], {"tool_context_id": "parent"})

    def factory(**kwargs):
        return httpx.AsyncClient(transport=httpx.ASGITransport(app=service.service), **kwargs)

    async with service.service.router.lifespan_context(service.service):
        url = await server._mcp_url()
        async with Client(
            StreamableHttpTransport(
                url,
                headers={"Authorization": f"Bearer {token}"},
                httpx_client_factory=factory,
            ),
        ) as client:
            await client.call_tool("search", {"query": "same application"})
        assert job.calls == [{"query": "same application", "tool_context_id": "parent"}]
        assert server.jobs_mcp._server is None
        assert app.context.components["agent_wrapper"]["one"].app_server is server
        assert app.context.components["agent_wrapper"]["two"].app_server is server
    assert not server.is_started


@pytest.mark.asyncio
@pytest.mark.parametrize("tls_options", [{"ssl_certfile": "server.pem"}, {"ssl_keyfile": "server.key"}])
async def test_tls_http_service_uses_managed_loopback_for_scoped_tools(tmp_path, monkeypatch, tls_options):
    app = Application(
        workspace_dir=str(tmp_path),
        enable_logo=False,
        log_to_console=False,
        log_to_file=False,
        service={"backend": "http", "web_enabled": False, **tls_options},
        components={"codex_app_server": {"default": {"backend": "sdk"}}},
    )
    service = app.context.service
    service.build_service(app)
    server = app.context.components["codex_app_server"]["default"]
    probe = AsyncMock(side_effect=AssertionError("must not probe the TLS listener"))
    monkeypatch.setattr(asyncio, "open_connection", probe)
    job = _Job()
    token = server.jobs_mcp.register([job], {"tool_context_id": "parent"})
    listener_task = None
    async with service.service.router.lifespan_context(service.service):
        url = await server._mcp_url()
        assert url.startswith("http://127.0.0.1:")
        assert await server._mcp_url() == url
        listener_task = server.jobs_mcp._task
        assert listener_task is not None
        async with Client(StreamableHttpTransport(url, headers={"Authorization": f"Bearer {token}"})) as client:
            await client.call_tool("search", {"query": "TLS host"})
        assert job.calls == [{"query": "TLS host", "tool_context_id": "parent"}]
        probe.assert_not_awaited()
    assert listener_task.done()
    assert server.jobs_mcp._task is None
    assert not server.jobs_mcp.scopes


@pytest.mark.asyncio
@pytest.mark.parametrize("option", ["cwd", "app_server"])
@pytest.mark.parametrize("stream", [False, True])
async def test_call_time_wrapper_options_are_rejected_instead_of_silently_ignored(system, option, stream):
    first, _, _, _ = system
    with pytest.raises(TypeError, match="wrapper options"):
        if stream:
            await anext(first.reply_stream("invalid", **{option: "ignored"}))
        else:
            await first.reply("invalid", **{option: "ignored"})
    assert not _Codex.instances


@pytest.mark.asyncio
async def test_mounted_tools_wait_for_the_main_listener_before_returning_a_url(tmp_path):
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    sock.setblocking(False)
    port = sock.getsockname()[1]
    app = Application(
        workspace_dir=str(tmp_path),
        enable_logo=False,
        log_to_console=False,
        log_to_file=False,
        service={"backend": "http", "port": port, "web_enabled": False},
        components={"codex_app_server": {"default": {"backend": "sdk"}}},
    )
    app.context.service.build_service(app)
    server = app.context.components["codex_app_server"]["default"]
    lookup = asyncio.create_task(server._mcp_url())
    listener = None

    async def accept(_reader, writer):
        writer.close()
        await writer.wait_closed()

    try:
        await asyncio.sleep(0.02)
        assert not lookup.done()
        listener = await asyncio.start_server(accept, sock=sock)
        url = await asyncio.wait_for(lookup, 2)
        assert url == f"http://127.0.0.1:{port}{server.jobs_mcp.path}"
        assert server.jobs_mcp._task is None
    finally:
        lookup.cancel()
        await asyncio.gather(lookup, return_exceptions=True)
        if listener is not None:
            listener.close()
            await listener.wait_closed()
        sock.close()


@pytest.mark.asyncio
async def test_replaced_runtime_uses_loopback_when_its_gateway_is_not_mounted(tmp_path, monkeypatch):
    app = Application(
        workspace_dir=str(tmp_path),
        enable_logo=False,
        log_to_console=False,
        log_to_file=False,
        service={"backend": "http", "web_enabled": False},
        components={"codex_app_server": {"default": {"backend": "sdk"}}},
    )
    app.context.service.build_service(app)
    replacement = CodexAppServer(app_context=app.context)
    app.context.service.service.mount("/assets", app=ScopedMCP())
    listen = AsyncMock(return_value="http://127.0.0.1:1234/local")
    monkeypatch.setattr(replacement.jobs_mcp, "listen", listen)
    assert await replacement._mcp_url() == "http://127.0.0.1:1234/local"
    listen.assert_awaited_once()


@pytest.mark.asyncio
async def test_failed_cleanup_preserves_the_original_request_error(system, monkeypatch):
    first, _, server, _ = system
    await first.reply("initial")
    client = _Codex.instances[0]

    async def fail(_thread, _inputs):
        raise ValueError("original failure")

    client.run_hook = fail
    monkeypatch.setattr(server, "wait_idle", AsyncMock(side_effect=RuntimeError("cleanup failure")))
    with pytest.raises(ValueError, match="original failure"):
        await first.reply("failed")


@pytest.mark.asyncio
async def test_cancelled_waiter_keeps_the_running_threads_tool_binding(system):
    first, second, server, _ = system
    binding = {"job_tools": ["search"], "tool_context_id": "one"}
    thread_id = (await first.reply("initial", **binding))["session_id"]
    token = server._threads[thread_id].token
    entered, release = asyncio.Event(), asyncio.Event()

    async def hold(_thread, _inputs):
        entered.set()
        await release.wait()

    client = _Codex.instances[0]
    client.run_hook = hold
    async with asyncio.TaskGroup() as group:
        group.create_task(first.reply("held", session_id=thread_id, **binding))
        await asyncio.wait_for(entered.wait(), 2)
        waiting = group.create_task(second.reply("queued", session_id=thread_id, **binding))
        try:
            await asyncio.sleep(0)
            waiting.cancel()
            with pytest.raises(asyncio.CancelledError):
                await waiting
            assert token in server.jobs_mcp.scopes
        finally:
            release.set()
    client.run_hook = None
    await second.reply("still authorized", session_id=thread_id, **binding)
    assert "queued" not in client.threads[thread_id].inputs


@pytest.mark.asyncio
async def test_disconnected_turn_is_not_replayed_and_next_request_reconnects(system):
    from openai_codex import TransportClosedError

    first, _, server, _ = system
    await first.reply("initial")
    old_client = _Codex.instances[0]

    async def disconnect(_thread, _inputs):
        raise TransportClosedError("disconnected")

    old_client.run_hook = disconnect
    with pytest.raises(TransportClosedError):
        await first.reply("do not replay")
    assert server._failed
    assert old_client.threads["thread-2"].inputs == ["do not replay"]
    assert (await first.reply("new request"))["result"] == "new request"
    assert len(_Codex.instances) == 2
    assert old_client.closed == 1
    assert _Codex.instances[1].threads["thread-1"].inputs == ["new request"]


@pytest.mark.asyncio
async def test_initial_transport_failure_does_not_poison_client_retry(system, monkeypatch):
    from openai_codex import TransportClosedError

    first, _, server, _ = system
    monkeypatch.setattr(
        _Codex,
        "account",
        AsyncMock(
            side_effect=[
                TransportClosedError("initialize failed"),
                SimpleNamespace(account=object()),
            ],
        ),
    )
    with pytest.raises(TransportClosedError):
        await first.reply("failed initialization")
    assert not server._failed
    assert (await first.reply("retry"))["result"] == "retry"
    assert _Codex.instances[0].closed == 1


@pytest.mark.asyncio
async def test_thread_pruning_waits_for_complete_pagination_and_preserves_tokens_on_error(system, monkeypatch):
    first, _, server, _ = system
    one = (await first.reply("one", job_tools=["search"]))["session_id"]
    two = (await first.reply("two", job_tools=["search"]))["session_id"]
    token = server._threads[two].token
    client = _Codex.instances[0]
    pages = [SimpleNamespace(data=[one], next_cursor="next"), RuntimeError("incomplete listing")]
    request = AsyncMock(side_effect=pages)
    monkeypatch.setattr(client._client, "request", request)
    await server._prune_threads(client)
    assert token in server.jobs_mcp.scopes
    request.side_effect = [
        SimpleNamespace(data=[one], next_cursor="next"),
        SimpleNamespace(data=[], next_cursor=None),
    ]
    await server._prune_threads(client)
    assert one in server._threads
    assert two not in server._threads
    assert token not in server.jobs_mcp.scopes


@pytest.mark.asyncio
async def test_stale_unload_snapshot_does_not_revoke_a_resumed_threads_tools(system, monkeypatch):
    first, second, server, _ = system
    binding = {"job_tools": ["search"], "tool_context_id": "one"}
    thread_id = (await first.reply("initial", **binding))["session_id"]
    state = server._threads[thread_id]
    token = state.token
    client = _Codex.instances[0]
    original_request = client.request
    resume_snapshot, stale_snapshot = asyncio.Event(), asyncio.Event()
    resume_query, stale_query = asyncio.Event(), asyncio.Event()
    queries = 0

    async def request(method, params, **kwargs):
        nonlocal queries
        if method == "thread/loaded/list":
            queries += 1
            if queries == 1:
                resume_snapshot.set()
                await resume_query.wait()
                return SimpleNamespace(data=[thread_id], next_cursor=None)
            if queries == 2:
                stale_snapshot.set()
                await stale_query.wait()
                return SimpleNamespace(data=[], next_cursor=None)
        return await original_request(method, params, **kwargs)

    monkeypatch.setattr(client._client, "request", request)
    async with asyncio.TaskGroup() as group:
        resumed = group.create_task(second.reply("resume", session_id=thread_id, **binding))
        await asyncio.wait_for(resume_snapshot.wait(), 2)
        client.loaded.clear()
        pruning = group.create_task(server._prune_threads(client))
        await asyncio.wait_for(stale_snapshot.wait(), 2)
        try:
            resume_query.set()
            await asyncio.wait_for(asyncio.shield(resumed), 2)
        finally:
            resume_query.set()
            stale_query.set()
        await pruning
    assert server._threads.get(thread_id) is state
    assert token in server.jobs_mcp.scopes


@pytest.mark.asyncio
async def test_tool_arguments_are_snapshotted_before_the_caller_mutates_them(system):
    first, _, server, job = system
    allowed = ["one.md"]
    result = await first.reply("one", job_tools=["search"], injected_job_kwargs={"_allowed_paths": allowed})
    allowed[:] = ["other.md"]
    token = server._threads[result["session_id"]].token
    await server.jobs_mcp.scopes[token][0].fn(query="read")
    assert job.calls[0]["_allowed_paths"] == ["one.md"]


def test_event_to_chunks_maps_content_usage_and_completion():
    content_event = SimpleNamespace(
        method="item/agentMessage/delta",
        payload=SimpleNamespace(item_id="item-1", delta="hello"),
    )
    usage = TokenUsageBreakdown(
        cachedInputTokens=1,
        inputTokens=3,
        outputTokens=5,
        reasoningOutputTokens=2,
        totalTokens=8,
    )
    usage_event = SimpleNamespace(
        method="thread/tokenUsage/updated",
        payload=SimpleNamespace(turn_id="turn-1"),
    )
    completed_event = SimpleNamespace(
        method="turn/completed",
        payload=SimpleNamespace(
            turn=SimpleNamespace(id="turn-1", status=SimpleNamespace(value="completed"), duration_ms=10, error=None),
        ),
    )

    content = CodexAgentWrapper._event_to_chunks(content_event, "thread-1")  # pylint: disable=protected-access
    usage_chunks = CodexAgentWrapper._event_to_chunks(  # pylint: disable=protected-access
        usage_event,
        "thread-1",
        usage=TokenUsage.from_provider(usage),
    )
    completed = CodexAgentWrapper._event_to_chunks(completed_event, "thread-1")  # pylint: disable=protected-access

    assert content[0].chunk_type == ChunkEnum.CONTENT
    assert content[0].chunk == "hello"
    assert usage_chunks[0].chunk_type == ChunkEnum.USAGE
    assert usage_chunks[0].input_tokens == 3
    assert usage_chunks[0].output_tokens == 5
    assert completed[0].chunk_type == ChunkEnum.REPLY_END
    assert completed[0].metadata["status"] == "completed"


def test_event_to_chunks_preserves_new_turn_scoped_notifications():
    from openai_codex.types import Notification
    from openai_codex.generated.v2_all import TurnDiffUpdatedNotification

    event = Notification(
        method="turn/diff/updated",
        payload=TurnDiffUpdatedNotification(
            threadId="thread-1",
            turnId="turn-1",
            diff="diff --git a/a b/a",
        ),
    )

    chunk = CodexAgentWrapper._event_to_chunks(event, "thread-1")[0]  # pylint: disable=protected-access

    assert chunk.chunk_type == ChunkEnum.DATA
    assert chunk.chunk == {
        "threadId": "thread-1",
        "turnId": "turn-1",
        "diff": "diff --git a/a b/a",
    }
    assert chunk.metadata == {"codex_method": "turn/diff/updated"}


def test_codex_skills_add_all_without_deleting_existing_content(tmp_path):
    wrapper, _job = _wrapper(tmp_path)
    for name in ("reme_memory", "qwenpaw_memory"):
        source = tmp_path / "skills" / name
        source.mkdir(parents=True)
        (source / "SKILL.md").write_text(f"# {name}", encoding="utf-8")
    existing = tmp_path / ".agents" / "skills" / "user_skill"
    existing.mkdir(parents=True)
    marker = existing / "marker"
    marker.write_text("keep", encoding="utf-8")

    wrapper._ensure_skills("all")  # pylint: disable=protected-access

    assert marker.read_text(encoding="utf-8") == "keep"
    for name in ("reme_memory", "qwenpaw_memory"):
        target = tmp_path / ".agents" / "skills" / name
        assert target.is_symlink()
        assert target.resolve() == (tmp_path / "skills" / name).resolve()


def test_codex_skills_support_single_name_and_are_idempotent(tmp_path):
    wrapper, _job = _wrapper(tmp_path)
    source = tmp_path / "skills" / "one"
    source.mkdir(parents=True)
    (source / "SKILL.md").write_text("# one", encoding="utf-8")

    wrapper._ensure_skills("one")  # pylint: disable=protected-access
    wrapper._ensure_skills(["one"])  # pylint: disable=protected-access

    assert (tmp_path / ".agents" / "skills" / "one").resolve() == source.resolve()


@pytest.mark.parametrize("kind", ["directory", "external_link"])
def test_codex_skills_preserve_conflicts(tmp_path, kind):
    wrapper, _job = _wrapper(tmp_path)
    source = tmp_path / "skills" / "one"
    source.mkdir(parents=True)
    (source / "SKILL.md").write_text("# one", encoding="utf-8")
    target = tmp_path / ".agents" / "skills" / "one"
    target.parent.mkdir(parents=True)
    if kind == "directory":
        target.mkdir()
        (target / "marker").write_text("keep", encoding="utf-8")
    else:
        external = tmp_path / "external"
        external.mkdir()
        target.symlink_to(external, target_is_directory=True)

    with pytest.raises(FileExistsError, match="Codex skill conflict"):
        wrapper._ensure_skills("one")  # pylint: disable=protected-access
    assert target.exists()


@pytest.mark.parametrize("create_dir", [False, True])
def test_codex_skills_reject_missing_or_invalid_skill(tmp_path, create_dir):
    wrapper, _job = _wrapper(tmp_path)
    if create_dir:
        (tmp_path / "skills" / "missing_manifest").mkdir(parents=True)
        name = "missing_manifest"
    else:
        name = "missing"
    with pytest.raises(FileNotFoundError):
        wrapper._ensure_skills(name)  # pylint: disable=protected-access


def test_codex_skills_do_not_modify_codex_home(tmp_path):
    codex_home = tmp_path / "codex-home"
    marker = codex_home / "skills" / "marker"
    marker.parent.mkdir(parents=True)
    marker.write_text("keep", encoding="utf-8")
    wrapper, _job = _wrapper(tmp_path, codex_home=codex_home)
    source = tmp_path / "skills" / "one"
    source.mkdir(parents=True)
    (source / "SKILL.md").write_text("# one", encoding="utf-8")

    wrapper._ensure_skills("one")  # pylint: disable=protected-access

    assert marker.read_text(encoding="utf-8") == "keep"


@pytest.mark.parametrize("review_status", ["approved", "denied"])
def test_event_to_chunks_maps_approval_started_and_completed(review_status):
    action = {"type": "futureApprovalAction", "value": "preserved"}
    review = {"status": review_status, "rationale": "policy"}
    started = SimpleNamespace(
        method="item/autoApprovalReview/started",
        payload=SimpleNamespace(
            action=action,
            review=review,
            review_id="review-1",
            target_item_id="item-1",
            turn_id="turn-1",
        ),
    )
    completed = SimpleNamespace(
        method="item/autoApprovalReview/completed",
        payload=SimpleNamespace(
            action=action,
            review=review,
            review_id="review-1",
            target_item_id="item-1",
            turn_id="turn-1",
            decision_source={"type": "guardian"},
        ),
    )

    started_chunk = CodexAgentWrapper._event_to_chunks(started, "thread-1")[0]  # pylint: disable=protected-access
    completed_chunk = CodexAgentWrapper._event_to_chunks(completed, "thread-1")[0]  # pylint: disable=protected-access

    assert started_chunk.chunk_type == ChunkEnum.APPROVAL
    assert started_chunk.chunk == action
    assert started_chunk.metadata["status"] == "started"
    assert completed_chunk.metadata["status"] == "completed"
    assert completed_chunk.metadata["review"]["status"] == review_status
    assert completed_chunk.metadata["decision_source"] == {"type": "guardian"}


@pytest.mark.asyncio
async def test_reply_activity_reaches_the_selected_proactive_session(system):
    from reme.schema import AgentActivity

    first, _second, server, _job = system
    selected = await first.create_session(job_tools=[])
    observed = []
    unsubscribe = server.watch_session(selected, observed.append)
    try:
        await first.reply("background result", job_tools=[], activity_source="proactive", activity_session_id=selected)
        activities = [event for event in observed if isinstance(event, AgentActivity)]
        assert activities[0].phase == "thinking"
        assert activities[-1].phase == "completed"
        assert all(event.session_id == selected and event.source == "proactive" for event in activities)
        assert len({event.turn_id for event in activities}) == 1
    finally:
        unsubscribe()


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", ["interrupted", "failed"])
async def test_activity_closes_when_the_consumer_stops_or_the_provider_fails(system, outcome):
    from reme.schema import AgentActivity

    first, _second, server, _job = system
    session_id = await first.create_session(job_tools=[])
    observed = []
    unsubscribe = server.watch_session(session_id, observed.append)
    try:
        stream = first.reply_stream("test", session_id=session_id)
        await anext(stream)
        if outcome == "failed":

            async def fail(_thread, _inputs):
                raise ValueError("provider failed")

            _Codex.instances[0].run_hook = fail
            with pytest.raises(ValueError, match="provider failed"):
                await anext(stream)
        else:
            await stream.aclose()
        phases = [event.phase for event in observed if isinstance(event, AgentActivity)]
        assert phases[0] == "thinking"
        assert phases[-1] == outcome
        assert "completed" not in phases
    finally:
        unsubscribe()


@pytest.mark.asyncio
async def test_task_activity_publishes_to_application_observers_without_a_selected_session(system):
    first, _second, server, _job = system
    observed = []
    unsubscribe = server.watch_activity(observed.append)
    try:
        await first.reply(
            "task result",
            job_tools=[],
            activity_source="task",
            activity_run_id="run",
            activity_task_id="research",
        )
        assert observed[0].phase == "thinking" and observed[-1].phase == "completed"
        assert all(item.source == "task" and item.run_id == "run" and item.task_id == "research" for item in observed)
        assert len({item.session_id for item in observed}) == 1
    finally:
        unsubscribe()
