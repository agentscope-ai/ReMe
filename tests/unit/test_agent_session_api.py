"""Session history, proactive injection recovery, activity and service contracts."""

# pylint: disable=protected-access,redefined-outer-name,missing-function-docstring,too-many-locals

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest
import pytest_asyncio
from openai_codex.generated.v2_all import ThreadItem

from reme.application import Application
from reme.components.agent_wrapper import AsAgentWrapper, BaseAgentWrapper, CodexAgentWrapper
from reme.components.agent_wrapper import codex_session_history as history
from reme.components.application_context import ApplicationContext
from reme.components.codex_app_server import CodexAppServer
from reme.config import resolve_app_config
from reme.utils import AsStateHandler


@pytest.mark.asyncio
async def test_agentscope_empty_session_is_persisted_resumable_and_idle(tmp_path):
    wrapper = AsAgentWrapper(app_context=ApplicationContext(workspace_dir=str(tmp_path)), as_llm="")
    session_id = await wrapper.create_session()
    state = await AsStateHandler.for_session(wrapper.session_path, session_id).load()
    assert state.session_id == session_id and not state.context
    assert (await wrapper.get_session_status(session_id)).status == "idle"
    restored = await wrapper._load_state(
        {"resume": session_id},
        state.permission_context.mode,
    )
    assert restored.session_id == session_id and not restored.context


def test_read_native_usage_ignores_estimates_and_an_incomplete_tail_without_writing(tmp_path):
    path = tmp_path / "rollout.jsonl"
    records = [
        {"type": "session_meta", "payload": {}},
        {
            "type": "event_msg",
            "payload": {
                "type": "token_count",
                "info": {
                    "last_token_usage": {"total_tokens": 900000},
                    "total_token_usage": {"input_tokens": 50, "output_tokens": 7, "total_tokens": 57},
                },
            },
        },
        {"type": "event_msg", "payload": {"type": "token_count", "info": None}},
    ]
    original = "\n".join(json.dumps(row) for row in records) + '\n{"type":'
    path.write_text(original)
    assert history.read_token_usage(SimpleNamespace(path=str(path)), tmp_path).model_dump() == {
        "input_tokens": 50,
        "output_tokens": 7,
        "total_tokens": 57,
    }
    assert path.read_text() == original
    assert history.read_token_usage(SimpleNamespace(path=None), tmp_path).total_tokens == 0


def test_read_native_usage_rejects_escaped_paths_and_complete_corruption(tmp_path):
    with pytest.raises(ValueError, match="outside"):
        history.read_token_usage(SimpleNamespace(path=str(tmp_path.parent / "outside.jsonl")), tmp_path)
    path = tmp_path / "bad.jsonl"
    path.write_text("{broken}\n")
    with pytest.raises(json.JSONDecodeError):
        history.read_token_usage(SimpleNamespace(path=str(path)), tmp_path)


def turn(identifier, query, answer):
    return SimpleNamespace(
        id=identifier,
        items=[
            ThreadItem.model_validate(
                {"type": "userMessage", "id": identifier + "-user", "content": [{"type": "text", "text": query}]},
            ),
            ThreadItem.model_validate({"type": "agentMessage", "id": identifier + "-answer", "text": answer}),
        ],
    )


class NativeSession:
    """Mock only the native SDK transport; retain real locking and workspace persistence."""

    def __init__(self, server):
        path = server.session_path / "sessions" / "test.jsonl"
        path.parent.mkdir(parents=True)
        path.write_text("")
        self.thread = SimpleNamespace(
            id="session-1",
            path=str(path),
            turns=[turn("turn-1", "A", "B")],
            status=SimpleNamespace(root=SimpleNamespace(type="idle", active_flags=[])),
        )
        self.threads = {self.thread.id: self.thread}
        self.handle = SimpleNamespace(id=self.thread.id, read=AsyncMock(side_effect=self.read))
        self.client = SimpleNamespace(
            _client=SimpleNamespace(
                thread_read=AsyncMock(side_effect=self.read),
                request=AsyncMock(side_effect=self.request),
            ),
            thread_resume=AsyncMock(return_value=self.handle),
        )
        self.injected = []
        self.inject_hook = None

    async def read(self, session_id=None, **_kwargs):
        return SimpleNamespace(thread=self.threads[session_id] if session_id else self.thread)

    async def request(self, method, params, **_kwargs):
        if method == "thread/loaded/list":
            return SimpleNamespace(data=list(self.threads), next_cursor=None)
        if method == "thread/unsubscribe":
            return SimpleNamespace(status="unsubscribed")
        assert method == "thread/inject_items"
        self.injected.extend(params["items"])
        with open(self.threads[params["threadId"]].path, "a", encoding="utf-8") as stream:
            for item in params["items"]:
                stream.write(json.dumps({"type": "response_item", "payload": item}) + "\n")
        if self.inject_hook:
            await self.inject_hook()
        return SimpleNamespace()


@pytest.mark.asyncio
async def test_session_snapshot_exposes_the_validated_native_file_and_turn(runtime, tmp_path):
    server, native = runtime
    snapshot = await server.read_session("session-1")
    assert snapshot.path == native.thread.path
    assert snapshot.last_turn_id == "turn-1"
    native.thread.path = str(tmp_path.parent / "outside.jsonl")
    with pytest.raises(ValueError, match="outside its configured home"):
        await server.read_session("session-1")


@pytest.mark.asyncio
async def test_session_path_lookup_returns_the_native_file_without_reading_history(runtime):
    server, native = runtime
    assert await server.get_session_path("session-1") == Path(native.thread.path).resolve()
    native.client._client.thread_read.assert_awaited_once_with("session-1", include_turns=False)
    native.client.thread_resume.assert_not_awaited()
    assert not server._message_path("session-1").exists()


@pytest.mark.asyncio
@pytest.mark.parametrize("native_path", [None, ""])
async def test_session_path_lookup_returns_none_when_native_storage_has_no_path(runtime, native_path):
    server, native = runtime
    native.thread.path = native_path
    assert await server.get_session_path("session-1") is None


@pytest.mark.asyncio
@pytest.mark.parametrize("symlink", [False, True])
async def test_session_path_lookup_rejects_paths_outside_the_configured_home(runtime, tmp_path, symlink):
    server, native = runtime
    outside = tmp_path / "outside.jsonl"
    if symlink:
        link = server.session_path / "sessions" / "link.jsonl"
        link.symlink_to(outside)
        native.thread.path = str(link)
    else:
        native.thread.path = str(outside)
    with pytest.raises(ValueError, match="outside its configured home"):
        await server.get_session_path("session-1")


@pytest.mark.asyncio
@pytest.mark.parametrize("session_id", ["", "../outside", "nested/session"])
async def test_session_path_lookup_rejects_invalid_ids_before_calling_codex(runtime, session_id):
    server, native = runtime
    with pytest.raises(ValueError, match="Session/message IDs must contain"):
        await server.get_session_path(session_id)
    native.client._client.thread_read.assert_not_awaited()


@pytest.mark.asyncio
async def test_session_path_lookup_propagates_native_lookup_errors(runtime):
    server, native = runtime
    native.client._client.thread_read.side_effect = RuntimeError("session not found")
    with pytest.raises(RuntimeError, match="session not found"):
        await server.get_session_path("missing-session")


@pytest.mark.asyncio
async def test_context_guard_rejects_stale_reminder_but_recovers_accepted_delivery(runtime):
    from reme.components.codex_app_server import SessionChangedError

    server, native = runtime
    native.thread.turns.append(turn("turn-2", "new question", "new answer"))
    with pytest.raises(SessionChangedError):
        await server.append_session("session-1", "stale", message_id="guard", expected_turn_id="turn-1")
    assert native.injected == []
    accepted = await server.append_session("session-1", "current", message_id="guard", expected_turn_id="turn-2")
    native.thread.turns.append(turn("turn-3", "later", "later answer"))
    assert (
        await server.append_session(
            "session-1",
            "current",
            message_id="guard",
            expected_turn_id="turn-2",
        )
        == accepted
    )
    assert len(native.injected) == 1


@pytest_asyncio.fixture
async def runtime(tmp_path, monkeypatch):
    server = CodexAppServer(
        name="default",
        app_context=ApplicationContext(workspace_dir=str(tmp_path)),
        auth_mode="oauth",
    )
    native = NativeSession(server)
    monkeypatch.setattr(server, "_get_client", AsyncMock(return_value=native.client))
    await server.start()
    try:
        yield server, native
    finally:
        await server.close()


@pytest.mark.asyncio
async def test_empty_session_accepts_proactive_messages_before_its_first_user_turn(runtime):
    server, native = runtime
    native.thread.turns.clear()
    native.client.thread_start = AsyncMock(return_value=native.handle)
    read = native.read

    async def read_empty(**kwargs):
        if kwargs.get("include_turns"):
            raise RuntimeError("empty thread has no persisted turns")
        return await read(**kwargs)

    native.handle.read = AsyncMock(side_effect=read_empty)
    async with server.thread(jobs=[], keep_empty=True) as thread:
        assert thread.id == "session-1"
    message = await server.append_session("session-1", "A proactive first message", message_id="first")
    assert await server.append_session("session-1", message.content, message_id="first") == message
    assert [entry.content for entry in (await server.read_session("session-1")).messages] == [message.content]
    assert len(native.injected) == 1
    native.handle.read.assert_any_call(include_turns=False)


@pytest.mark.asyncio
async def test_multiple_appends_are_ordered_durable_and_idempotent(runtime):
    server, native = runtime
    first = await server.append_session("session-1", "C: proactive", message_id="topic-1")
    assert await server.append_session("session-1", "C: proactive", message_id="topic-1") == first
    native.thread.turns.append(turn("turn-2", "D", "E"))
    await server.append_session("session-1", "F: proactive", message_id="topic-2")
    snapshot = await server.read_session("session-1")
    assert [message.content for message in snapshot.messages] == ["A", "B", "C: proactive", "D", "E", "F: proactive"]
    assert [message.source for message in snapshot.messages] == [
        "chat",
        "chat",
        "proactive",
        "chat",
        "chat",
        "proactive",
    ]
    assert len(native.injected) == 2
    assert all(item["role"] == "assistant" for item in native.injected)
    assert all(record["injected"] for record in history.read_records(server._message_path("session-1")))
    with pytest.raises(ValueError, match="different content"):
        await server.append_session("session-1", "changed", message_id="topic-1")


@pytest.mark.asyncio
async def test_fork_inherits_only_accepted_proactive_records(runtime):
    server, native = runtime
    await server.append_session("session-1", "inherited", message_id="one")
    path = server._message_path("session-1")
    records = history.read_records(path)
    # Include a native-accepted message whose publication was interrupted, and
    # an unaccepted attempt that must stay out of the fork's displayed context.
    records[0]["injected"] = False
    records.append(history.new_record("rejected", "not inherited", native.thread))
    history.write_records(path, records)
    fork_path = server.session_path / "sessions" / "fork.jsonl"
    # Native forks retain injected context but discard the reserved item IDs.
    fork_path.write_text("")
    fork_thread = SimpleNamespace(
        id="fork-1",
        path=str(fork_path),
        turns=[*native.thread.turns, turn("turn-2", "D", "E")],
        status=native.thread.status,
    )
    native.threads[fork_thread.id] = fork_thread
    fork_handle = SimpleNamespace(id="fork-1", read=AsyncMock(return_value=SimpleNamespace(thread=fork_thread)))

    async def fork(_session_id, **_options):
        return fork_handle

    native.client.thread_fork = AsyncMock(side_effect=fork)
    native.client.thread_resume.side_effect = lambda session_id, **_kwargs: (
        fork_handle if session_id == "fork-1" else native.handle
    )
    async with server.thread("session-1", fork=True) as forked:
        assert forked.id == "fork-1"
    snapshot = await server.read_session("fork-1")
    assert [message.content for message in snapshot.messages] == ["A", "B", "inherited", "D", "E"]
    assert history.read_records(path) == records
    assert await server.append_session("fork-1", "inherited", message_id="one") == snapshot.messages[2]
    assert len(native.injected) == 1
    await server.append_session("fork-1", "fork only", message_id="two")
    assert history.read_records(path) == records


@pytest.mark.asyncio
async def test_read_and_status_do_not_wait_behind_generation(runtime):
    server, native = runtime
    async with server.thread("session-1"):
        native.thread.status.root.type = "active"
        native.thread.status.root.active_flags = ["waitingOnUserInput"]
        try:
            async with asyncio.timeout(0.5):
                status = await server.get_session_status("session-1")
                snapshot = await server.read_session("session-1")
                path = await server.get_session_path("session-1")
            assert status.status == snapshot.state.status == "active"
            assert path == Path(native.thread.path).resolve()
            assert status.active_flags == ["waitingOnUserInput"]
            with pytest.raises(TimeoutError):
                await server.append_session("session-1", "later", message_id="busy", timeout=0.02)
            assert native.injected == []
        finally:
            native.thread.status.root.type = "idle"


@pytest.mark.asyncio
async def test_append_waits_until_active_turn_releases_its_lease(runtime):
    server, native = runtime
    async with server.thread("session-1"):
        pending = asyncio.create_task(server.append_session("session-1", "after turn", message_id="waiting"))
        await asyncio.sleep(0)
        assert not pending.done()
        assert not native.injected
    await pending
    assert len(native.injected) == 1


@pytest.mark.asyncio
async def test_recover_publication_failure_without_reinjecting(runtime, monkeypatch):
    server, native = runtime
    notifications = []
    server.watch_session("session-1", notifications.append)
    original = history.write_records

    def fail_commit(path, records):
        if path == server._message_path("session-1") and records[-1]["injected"]:
            raise OSError("publication interrupted")
        original(path, records)

    monkeypatch.setattr(history, "write_records", fail_commit)
    with pytest.raises(OSError, match="publication interrupted"):
        await server.append_session("session-1", "once", message_id="recover")
    assert not notifications
    assert not history.read_records(server._message_path("session-1"))[0]["injected"]
    assert (await server.read_session("session-1")).messages[-1].content == "once"
    monkeypatch.setattr(history, "write_records", original)
    await server.append_session("session-1", "once", message_id="recover")
    assert len(native.injected) == 1
    assert [message.id for message in notifications] == ["reme-proactive-recover"]


@pytest.mark.asyncio
async def test_unaccepted_retry_uses_the_current_turn_anchor(runtime):
    server, native = runtime
    original = native.request

    async def reject(method, params, **kwargs):
        if method == "thread/inject_items":
            raise ValueError("rejected before acceptance")
        return await original(method, params, **kwargs)

    native.client._client.request.side_effect = reject
    with pytest.raises(ValueError, match="before acceptance"):
        await server.append_session("session-1", "later", message_id="retry")
    native.thread.turns.append(turn("turn-2", "D", "E"))
    native.client._client.request.side_effect = original
    await server.append_session("session-1", "later", message_id="retry")
    assert [message.content for message in (await server.read_session("session-1")).messages] == [
        "A",
        "B",
        "D",
        "E",
        "later",
    ]


@pytest.mark.asyncio
async def test_cancelled_append_finishes_publication_before_releasing_lease(runtime):
    server, native = runtime
    notifications = []
    server.watch_session("session-1", notifications.append)
    entered, release = asyncio.Event(), asyncio.Event()

    async def hold():
        entered.set()
        await release.wait()

    native.inject_hook = hold
    pending = asyncio.create_task(server.append_session("session-1", "once", message_id="cancel"))
    await entered.wait()
    pending.cancel()
    await asyncio.sleep(0)
    assert not pending.done()
    assert server._threads["session-1"].lock.locked()
    assert not notifications
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await pending
    assert [message.id for message in notifications] == ["reme-proactive-cancel"]
    await server.append_session("session-1", "once", message_id="cancel")
    assert len(native.injected) == 1


@pytest.mark.asyncio
async def test_message_notifications_follow_committed_records_and_isolate_observers(runtime):
    server, _native = runtime
    notifications = []

    def observe(message):
        records = history.read_records(server._message_path("session-1"))
        assert records[-1]["injected"]
        notifications.append(message)

    def broken(_message):
        raise RuntimeError("observer failure")

    unsubscribe = server.watch_session("session-1", observe)
    server.watch_session("session-1", broken)
    message = await server.append_session("session-1", "published", message_id="published")
    assert notifications == [message]
    unsubscribe()
    unsubscribe()
    await server.append_session("session-1", "later", message_id="later")
    assert notifications == [message]


@pytest.mark.asyncio
@pytest.mark.parametrize("cancelled", [False, True])
async def test_turn_owner_publishes_readable_history_when_the_lease_finishes(runtime, cancelled):
    from reme.schema import SessionSnapshot, SessionStatus

    server, native = runtime
    notifications = []
    server.watch_session("session-1", notifications.append)
    native.handle.turn = AsyncMock(return_value=SimpleNamespace())
    entered = asyncio.Event()

    async def run():
        async with server.thread("session-1") as loaded:
            await server.turn(loaded, "next")
            native.thread.turns.append(turn("turn-2", "next", "answer"))
            server._threads["session-1"].completion.set_result(None)
            entered.set()
            if cancelled:
                await asyncio.Event().wait()

    task = asyncio.create_task(run())
    await asyncio.wait_for(entered.wait(), timeout=1)
    if cancelled:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    else:
        await task
    assert isinstance(notifications[0], SessionStatus) and notifications[0].status == "active"
    assert isinstance(notifications[-1], SessionSnapshot)
    assert [item.content for item in notifications[-1].messages] == ["A", "B", "next", "answer"]


@pytest.mark.asyncio
async def test_failed_final_idle_check_invalidates_subscribers_instead_of_hiding_committed_history(
    runtime,
    monkeypatch,
):
    server, native = runtime
    notifications = []
    server.watch_session("session-1", notifications.append)
    native.handle.turn = AsyncMock(return_value=SimpleNamespace())
    wait_idle = server.wait_idle

    async def fail_final_check(thread):
        if server._threads[thread.id].history_changed:
            raise RuntimeError("status temporarily unavailable")
        await wait_idle(thread)

    monkeypatch.setattr(server, "wait_idle", fail_final_check)
    with pytest.raises(RuntimeError, match="status temporarily unavailable"):
        async with server.thread("session-1") as loaded:
            await server.turn(loaded, "committed")
            native.thread.turns.append(turn("turn-2", "committed", "answer"))
            server._threads["session-1"].completion.set_result(None)
    assert notifications[-1] is None
    assert (await server.read_session("session-1")).messages[-1].content == "answer"


@pytest.mark.asyncio
async def test_session_files_and_native_recovery_are_contained(runtime, tmp_path):
    server, native = runtime
    for identifier in ("../escape", "/absolute", "a/b", ""):
        with pytest.raises(ValueError):
            await server.read_session(identifier)
    native.thread.path = str(tmp_path.parent / "outside.jsonl")
    with pytest.raises(ValueError, match="configured home"):
        await server.append_session("session-1", "no", message_id="contained")
    assert not native.injected


@pytest.mark.asyncio
async def test_non_idle_error_status_is_not_appendable(runtime):
    server, native = runtime
    native.thread.status.root.type = "systemError"
    assert (await server.get_session_status("session-1")).status == "error"
    with pytest.raises(RuntimeError, match="idle"):
        await server.append_session("session-1", "no", message_id="error")
    assert not native.injected


@pytest.mark.asyncio
async def test_unloaded_status_is_read_only_and_append_resumes_before_injecting(runtime):
    server, native = runtime
    native.thread.status.root.type = "notLoaded"
    assert (await server.get_session_status("session-1")).status == "not_loaded"
    assert (await server.read_session("session-1")).state.status == "not_loaded"
    native.client.thread_resume.assert_not_awaited()

    async def resume(*_args, **_kwargs):
        native.thread.status.root.type = "idle"
        return native.handle

    native.client.thread_resume.side_effect = resume
    await server.append_session("session-1", "resumed", message_id="unloaded")
    assert native.injected[-1]["content"][0]["text"] == "resumed"


class UnsupportedAgent(BaseAgentWrapper):
    """Existing wrappers need not implement optional session capabilities."""

    async def reply(self, inputs, **kwargs):
        return {}


@pytest.mark.asyncio
async def test_optional_base_capabilities_fail_explicitly():
    agent = UnsupportedAgent()
    for operation, kwargs in (
        (agent.read_session, {}),
        (agent.get_session_status, {}),
        (agent.append_session, {"content": "text", "message_id": "one"}),
    ):
        with pytest.raises(NotImplementedError, match="UnsupportedAgent"):
            await operation("session-1", **kwargs)


@pytest.mark.asyncio
async def test_http_session_jobs_and_unsupported_error_code(tmp_path, monkeypatch):
    config = resolve_app_config(config="proactive", log_config=False)
    app = Application(
        workspace_dir=str(tmp_path),
        enable_logo=False,
        log_to_file=False,
        log_to_console=False,
        service={"backend": "http", "web_enabled": False, "mcp_enabled": False},
        components={
            "codex_app_server": {"default": {"backend": "sdk", "auth_mode": "oauth"}},
            "agent_wrapper": {"default": {"backend": "codex"}},
        },
        jobs={name: config["jobs"][name] for name in ("session_read", "session_status", "session_append")},
    )
    server = app.context.components["codex_app_server"]["default"]
    native = NativeSession(server)
    monkeypatch.setattr(server, "_get_client", AsyncMock(return_value=native.client))
    service = app.context.service
    service.build_service(app)
    service.add_jobs(app)
    async with app:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=service.service),
            base_url="http://test",
        ) as client:
            for action in ("session_status", "session_read"):
                response = (await client.post("/" + action, json={"session_id": "session-1"})).json()
                assert response["success"]
            response = (
                await client.post(
                    "/session_append",
                    json={"session_id": "session-1", "content": "hello", "message_id": "http", "wait_timeout": 1},
                )
            ).json()
            assert response["success"]
            assert response["answer"]["source"] == "proactive"
            monkeypatch.setattr(CodexAgentWrapper, "read_session", BaseAgentWrapper.read_session)
            response = (await client.post("/session_read", json={"session_id": "session-1"})).json()
            assert not response["success"]
            assert response["metadata"]["code"] == "session_not_supported"
