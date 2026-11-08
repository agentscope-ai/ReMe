"""PA subscriptions with real jobs, bounded transport, and mocked session I/O."""

# pylint: disable=protected-access,missing-function-docstring,redefined-outer-name

import asyncio
from contextlib import aclosing
import json
import socket
from unittest.mock import AsyncMock

import httpx
import pytest
import pytest_asyncio
import uvicorn

from reme.application import Application
from reme.components.agent_wrapper import CodexAgentWrapper
from reme.components.codex_app_server import CodexAppServer
from reme.components.job import StreamJob
from reme.components.pa_runtime import PaRuntime
from reme.config import resolve_app_config
from reme.constants import REME_SERVICE_INFO
from reme.enumeration import ChunkEnum
from reme.schema import SessionMessage, SessionSnapshot, SessionStatus, StreamChunk
from reme.steps.proactive import PaChatStep, PaEventsStep


class EventAgent(CodexAgentWrapper):
    """Model/session boundary with durable-looking snapshots and a controllable turn."""

    def __init__(self, server, **kwargs):
        super().__init__(app_server="", **kwargs)
        self.app_server = server
        self.snapshots = {}
        self.read_session = AsyncMock(side_effect=self.read)
        self.create_session = AsyncMock(side_effect=self.create)
        self.release_session = AsyncMock()
        self.entered = asyncio.Event()
        self.release = asyncio.Event()

    async def read(self, session_id):
        return self.snapshots[session_id].model_copy(deep=True)

    async def create(self, **_kwargs):
        session_id = f"session-{len(self.snapshots) + 1}"
        self.snapshots[session_id] = SessionSnapshot(
            session_id=session_id,
            state=SessionStatus(session_id=session_id, status="idle"),
        )
        return session_id

    async def reply_stream(self, inputs, **kwargs):
        self.entered.set()
        await self.release.wait()
        yield StreamChunk(
            chunk_type=ChunkEnum.REPLY_END,
            status="completed",
            chunk={"status": "completed", "result": inputs, "session_id": kwargs["resume"]},
        )


@pytest_asyncio.fixture
async def system(tmp_path):
    config = resolve_app_config(config="proactive")
    app = Application(
        workspace_dir=str(tmp_path),
        enable_logo=False,
        log_to_file=False,
        log_to_console=False,
        jobs={"pa_events": config["jobs"]["pa_events"]},
        service={**config["service"], "mcp_enabled": False, "web_enabled": False},
    )
    server = CodexAppServer(app_context=app.context)
    agent = EventAgent(server, app_context=app.context)
    runtime = PaRuntime(
        app_context=app.context,
        heartbeat_seconds=0.02,
        default_proactive=config["components"]["pa"]["default"]["default_proactive"],
    )
    app.context.components.update(
        codex_app_server={"default": server},
        agent_wrapper={"default": agent},
        pa={"default": runtime},
    )
    async with app:
        yield app, agent, runtime, server


async def next_event(events):
    return await asyncio.wait_for(anext(events), timeout=1)


@pytest.mark.asyncio
async def test_unchanged_status_polling_does_not_starve_initial_snapshot(system):
    _app, agent, runtime, _server = system
    session_id = await runtime.new_session()

    async def read_with_idle_poll(_session_id):
        await asyncio.sleep(0.01)
        runtime.publish_status()
        return agent.snapshots[session_id].model_copy(deep=True)

    agent.read_session = AsyncMock(side_effect=read_with_idle_poll)
    async with aclosing(runtime.events()) as events:
        event = await next_event(events)
    assert event.type == "snapshot" and event.snapshot.session_id == session_id


@pytest.mark.asyncio
async def test_activity_is_replayable_scoped_and_included_in_snapshots(system):
    from reme.schema import AgentActivity

    _app, agent, runtime, server = system
    session_id = await select(agent, runtime)
    activity = AgentActivity(session_id=session_id, turn_id="turn", phase="answering")
    async with aclosing(runtime.events()) as events:
        cursor = (await next_event(events)).cursor
        server.publish_activity(activity)
        event = await next_event(events)
        assert event.activities["chat"] == activity
        server.publish_activity(activity)
        assert (await next_event(events)).type == "heartbeat"
    async with aclosing(runtime.events(cursor)) as events:
        assert (await next_event(events)).activities["chat"] == activity
    background = activity.model_copy(update={"source": "proactive", "turn_id": "background", "phase": "remembering"})
    server.publish_activity(background)
    async with aclosing(runtime.events()) as events:
        snapshot = await next_event(events)
        assert snapshot.activities == {"chat": activity, "proactive": background}
    await select(agent, runtime)
    server.publish_activity(activity)
    async with aclosing(runtime.events()) as events:
        assert (await next_event(events)).activities == {}


async def select(_agent, runtime):
    return await runtime.new_session()


@pytest.mark.asyncio
async def test_parallel_task_activity_survives_chat_switches_and_replays_without_task_history(system):
    from datetime import datetime, timezone
    from reme.schema import AgentActivity
    from reme.schema.pa_task import PaTaskRun

    _app, agent, runtime, server = system
    await select(agent, runtime)
    runs = [
        PaTaskRun(task_id=name, query="private task query", started_at=datetime.now(timezone.utc))
        for name in ("research", "watch")
    ]
    for run in runs:
        runtime.start_task_activity(run)
    first, second = runs
    update = AgentActivity(
        session_id="native-task-session",
        turn_id="native-task-turn",
        source="task",
        run_id=first.run_id,
        task_id=first.task_id,
        phase="remembering",
    )
    async with aclosing(runtime.events()) as events:
        initial = await next_event(events)
        assert set(initial.task_activities) == {first.run_id, second.run_id}
        cursor = initial.cursor
        server.publish_activity(update)
        changed = await next_event(events)
        assert changed.task_activities[first.run_id] == update
        assert changed.activities == {} and changed.message is None
    async with aclosing(runtime.events(cursor)) as events:
        assert (await next_event(events)).task_activities[first.run_id] == update
    server.publish_activity(update.model_copy(update={"phase": "completed"}))
    assert runtime._task_activities[first.run_id].phase == "working", "summary and publication still belong to the run"
    await select(agent, runtime)
    runtime.finish_task_activity(first.run_id)
    server.publish_activity(update)
    async with aclosing(runtime.events()) as events:
        current = await next_event(events)
        assert set(current.task_activities) == {second.run_id}
        assert current.snapshot.messages == []
    server._notify_session(None, None)
    assert runtime._task_activities == {}
    assert server._activity_listeners
    await runtime.close()
    assert not server._activity_listeners


def append(agent, server, session_id, identifier="reminder", *, completed=True):
    message = SessionMessage(
        id=identifier,
        role="assistant",
        source="proactive",
        content=identifier,
        completed=completed,
    )
    agent.snapshots[session_id].messages.append(message)
    server._notify_session(session_id, message)
    return message


@pytest.mark.asyncio
async def test_initial_snapshot_and_heartbeat_are_read_only(system, tmp_path):
    _app, agent, runtime, _server = system
    async with aclosing(runtime.events()) as events:
        initial = await next_event(events)
        assert initial.type == "snapshot" and initial.session_id is None
        assert initial.snapshot is None and initial.status is None
        heartbeat = await next_event(events)
        assert heartbeat.type == "heartbeat" and heartbeat.cursor is None
        assert runtime.cursor == initial.cursor
    agent.create_session.assert_not_awaited()
    agent.read_session.assert_not_awaited()
    assert not (tmp_path / "daily/personal/MEMORY.md").exists()


@pytest.mark.asyncio
async def test_readers_replay_independently_and_upsert_updates_the_same_message(system):
    _app, agent, runtime, server = system
    session_id = await select(agent, runtime)
    async with aclosing(runtime.events()) as first:
        cursor = (await next_event(first)).cursor
        append(agent, server, session_id, completed=False)
        one = await next_event(first)
        assert one.type == "message_upsert" and not one.message.completed
        one.message.content = "client-local edit"
        async with aclosing(runtime.events(cursor)) as second:
            replay = await next_event(second)
            assert replay.message.content == "reminder" and replay.cursor == one.cursor
            server._notify_session(session_id, agent.snapshots[session_id].messages[0])
            assert (await next_event(second)).type == "heartbeat"
            changed = agent.snapshots[session_id].messages[0].model_copy(update={"completed": True})
            server._notify_session(session_id, changed)
            update = await next_event(second)
            assert update.message.id == replay.message.id and update.message.completed
            assert (await next_event(first)).cursor == update.cursor
    assert agent.read_session.await_count == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["malformed", "future", "restart", "overflow", "large_event"])
async def test_unavailable_cursors_reset_to_authoritative_history(system, kind):
    _app, agent, runtime, server = system
    session_id = await select(agent, runtime)
    cursor = runtime.cursor
    if kind == "malformed":
        cursor = "not-a-cursor"
    elif kind == "future":
        cursor = f"{runtime._epoch}:999999"
    elif kind == "restart":
        await runtime.close()
        await runtime.start()
    else:
        runtime.max_events = 1
        if kind == "large_event":
            runtime.max_buffer_bytes = 1
        append(agent, server, session_id, "first")
        append(agent, server, session_id, "second")
    async with aclosing(runtime.events(cursor)) as events:
        assert (await next_event(events)).type == "reset"
        snapshot = await next_event(events)
        assert snapshot.type == "snapshot" and snapshot.cursor == runtime.cursor
        assert snapshot.snapshot == agent.snapshots[session_id]
    assert len(runtime._buffer) <= runtime.max_events
    assert runtime._buffer_bytes <= runtime.max_buffer_bytes


@pytest.mark.asyncio
async def test_snapshot_retries_a_concurrent_session_switch(system, monkeypatch):
    _app, agent, runtime, server = system
    previous = await select(agent, runtime)
    entered, release = asyncio.Event(), asyncio.Event()

    async def held(session_id):
        if session_id == previous:
            entered.set()
            await release.wait()
        return await agent.read(session_id)

    monkeypatch.setattr(agent, "read_session", held)
    async with aclosing(runtime.events()) as events:
        task = asyncio.create_task(next_event(events))
        await entered.wait()
        current = await select(agent, runtime)
        append(agent, server, current, "new-session-message")
        release.set()
        snapshot = await task
        assert snapshot.session_id == current
        assert [item.id for item in snapshot.snapshot.messages] == ["new-session-message"]
        assert previous not in server._session_listeners
        server._notify_session(previous, SessionStatus(session_id=previous, status="active"))
        assert (await next_event(events)).type == "heartbeat"


@pytest.mark.asyncio
async def test_another_readers_snapshot_cannot_hide_a_commit_notification(system):
    _app, agent, runtime, server = system
    session_id = await select(agent, runtime)
    async with aclosing(runtime.events()) as first:
        await next_event(first)
        committed = SessionMessage(id="committed", role="assistant", content="persisted before notification")
        agent.snapshots[session_id].messages.append(committed)
        async with aclosing(runtime.events()) as second:
            assert (await next_event(second)).snapshot.messages == [committed]
        server._notify_session(session_id, committed)
        assert (await next_event(first)).message == committed


@pytest.mark.asyncio
async def test_upstream_gap_resets_existing_readers_and_failed_snapshots_retry(system, monkeypatch):
    _app, agent, runtime, server = system
    session_id = await select(agent, runtime)
    async with aclosing(runtime.events()) as events:
        cursor = (await next_event(events)).cursor
        server._notify_session(session_id, None)
        monkeypatch.setattr(agent, "read_session", AsyncMock(side_effect=ConnectionError("offline")))
        assert (await next_event(events)).reason == "upstream_disconnected"
        failed = await next_event(events)
        assert failed.type == "snapshot" and failed.status == "error" and failed.error == "offline"
        assert failed.cursor != cursor and failed.snapshot is None
        monkeypatch.setattr(agent, "read_session", agent.read)
        recovered = await next_event(events)
        assert recovered.type == "snapshot" and recovered.snapshot.session_id == session_id
        assert recovered.status == "idle"


@pytest.mark.asyncio
async def test_chat_cancellation_publishes_idle_without_ending_other_subscriptions(system):
    app, agent, runtime, _server = system
    async with aclosing(runtime.events()) as events:
        await next_event(events)
        task = asyncio.create_task(
            PaChatStep(app_context=app.context)(query="hold", stream_queue=asyncio.Queue()),
        )
        await agent.entered.wait()
        observed = [await next_event(events) for _ in range(3)]
        assert [event.type for event in observed] == ["status_changed", "session_changed", "status_changed"]
        assert observed[-1].status == "active" and observed[-1].session_id == "session-1"
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert (await next_event(events)).status == "idle"
        assert (await next_event(events)).type == "heartbeat"


@pytest.mark.asyncio
async def test_failed_snapshot_retries_even_when_another_reader_or_owner_has_recovered(system, monkeypatch):
    _app, agent, runtime, server = system
    session_id = await select(agent, runtime)
    monkeypatch.setattr(
        agent,
        "read_session",
        AsyncMock(side_effect=[ConnectionError("offline"), agent.snapshots[session_id]]),
    )
    async with aclosing(runtime.events()) as events:
        failed = await next_event(events)
        assert failed.error == "offline" and failed.snapshot is None
        assert failed.cursor is None
        server._notify_session(session_id, agent.snapshots[session_id])
        recovered = await next_event(events)
        assert recovered.type == "snapshot" and recovered.snapshot.session_id == session_id


@pytest.mark.asyncio
async def test_shutdown_wakes_readers_and_removes_the_session_listener(system):
    _app, agent, runtime, server = system
    session_id = await select(agent, runtime)
    async with aclosing(runtime.events()) as events:
        await next_event(events)
        waiting = asyncio.create_task(anext(events))
        await asyncio.sleep(0)
        await runtime.close()
        with pytest.raises(StopAsyncIteration):
            await asyncio.wait_for(waiting, timeout=1)
    assert session_id not in server._session_listeners


@pytest.mark.asyncio
async def test_http_shutdown_cancels_open_subscription_and_closes_application(system, monkeypatch):
    app, agent, runtime, owner = system
    session_id = await select(agent, runtime)
    monkeypatch.setenv(REME_SERVICE_INFO, "")
    producer_finished = asyncio.Event()
    job = app.context.jobs["pa_events"]
    run = job._run_steps

    async def track(context):
        try:
            return await run(context)
        finally:
            producer_finished.set()

    monkeypatch.setattr(job, "_run_steps", track)
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen(128)
        listener.setblocking(False)
        host, port = listener.getsockname()
        service = app.context.service
        service.host, service.port = host, port
        service.build_service(app)
        service.add_jobs(app)
        config = uvicorn.Config(
            service.service,
            host=host,
            port=port,
            log_config=None,
            ws="none",
            **service.kwargs,
        )
        server = uvicorn.Server(config)
        serving = asyncio.create_task(server.serve(sockets=[listener]))
        try:
            async with asyncio.timeout(5):
                while not server.started:
                    if serving.done():
                        await serving
                        pytest.fail("HTTP server exited before startup")
                    await asyncio.sleep(0.01)
            async with httpx.AsyncClient(timeout=3, trust_env=False) as client:
                async with client.stream("POST", f"http://{host}:{port}/pa_events", json={}) as response:
                    response.raise_for_status()
                    async with aclosing(response.aiter_lines()) as lines:
                        assert "snapshot" in await asyncio.wait_for(anext(lines), timeout=1)
                        server.should_exit = True
                        # Keep the client connected while Uvicorn drains and closes the application.
                        await asyncio.wait_for(asyncio.shield(serving), timeout=8)
                        assert runtime._closed and producer_finished.is_set()
                        assert session_id not in owner._session_listeners
        finally:
            server.should_exit = True
            await asyncio.wait_for(serving, timeout=5)


@pytest.mark.asyncio
async def test_http_backpressure_resets_slow_readers_and_disconnect_cancels_the_job(system, monkeypatch):
    app, agent, runtime, server = system
    session_id = await select(agent, runtime)
    runtime.max_events = 2
    service = app.context.service
    service.build_service(app)
    service.add_jobs(app)
    job = app.context.jobs["pa_events"]
    assert isinstance(job, StreamJob) and job.stream_queue_size == 1
    started, release, disconnect, reset_seen, finished = (asyncio.Event() for _ in range(5))
    queue = None
    run = job._run_steps

    async def track(context):
        nonlocal queue
        queue = context.stream_queue
        try:
            return await run(context)
        finally:
            finished.set()

    monkeypatch.setattr(job, "_run_steps", track)
    first = True
    frames = []

    async def receive():
        nonlocal first
        if first:
            first = False
            return {"type": "http.request", "body": b"{}", "more_body": False}
        await disconnect.wait()
        return {"type": "http.disconnect"}

    async def send(message):
        if message["type"] == "http.response.body" and message.get("body"):
            frames.append(message["body"])
            started.set()
            if b'"type":"reset"' in message["body"]:
                reset_seen.set()
            await release.wait()

    scope = {
        "type": "http",
        "asgi": {"version": "3.0", "spec_version": "2.0"},
        "method": "POST",
        "path": "/pa_events",
        "raw_path": b"/pa_events",
        "query_string": b"",
        "headers": [(b"content-type", b"application/json")],
        "scheme": "http",
        "server": ("test", 80),
        "client": ("test", 1234),
        "root_path": "",
        "http_version": "1.1",
    }
    request = asyncio.create_task(service.service(scope, receive, send))
    try:
        await asyncio.wait_for(started.wait(), timeout=1)
        for index in range(20):
            append(agent, server, session_id, f"message-{index}")
            await asyncio.sleep(0)
        assert queue.maxsize == 1 and queue.qsize() <= 1
        assert len(runtime._buffer) <= 2
        release.set()
        await asyncio.wait_for(reset_seen.wait(), timeout=1)
    finally:
        disconnect.set()
        release.set()
        await asyncio.wait_for(request, timeout=1)
    assert finished.is_set() and not runtime._closed
    assert session_id in server._session_listeners
    payloads = [json.loads(frame.decode().removeprefix("data:").strip()) for frame in frames]
    assert payloads[0]["chunk"]["type"] == "snapshot"


@pytest.mark.asyncio
@pytest.mark.parametrize("parameters", [{"cursor": 1}, {"cursor": "x" * 129}, {"session_id": "other"}])
async def test_event_requests_validate_inputs_without_mutation(system, parameters):
    app, agent, runtime, _server = system
    before = runtime.cursor
    with pytest.raises(ValueError):
        await PaEventsStep(app_context=app.context)(**parameters, stream_queue=asyncio.Queue())
    assert runtime.cursor == before
    agent.read_session.assert_not_awaited()


@pytest.mark.parametrize("value", [-1, True, 1.5])
def test_stream_queue_size_rejects_invalid_capacity(value):
    with pytest.raises(ValueError, match="stream_queue_size"):
        StreamJob(stream_queue_size=value)
