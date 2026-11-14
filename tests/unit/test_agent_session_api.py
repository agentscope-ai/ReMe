"""Native session history, activity and service contracts."""

# pylint: disable=protected-access,redefined-outer-name,missing-function-docstring,too-many-locals

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
import pytest_asyncio
from openai_codex.generated.v2_all import ThreadItem

from reme.components.agent_wrapper import AsAgentWrapper, BaseAgentWrapper
from reme.components.codex_app_server import _session_history as history
from reme.components.application_context import ApplicationContext
from reme.components.codex_app_server import SdkCodexAppServer
from reme.utils import AsStateHandler


@pytest.mark.asyncio
async def test_agentscope_empty_session_is_persisted_resumable_and_idle(tmp_path):
    wrapper = AsAgentWrapper(app_context=ApplicationContext(workspace_dir=str(tmp_path)), as_llm="")
    session_id = await wrapper.create_session()
    state = await AsStateHandler.for_session(wrapper.session_path, session_id).load()
    assert state.session_id == session_id and not state.context
    assert (await wrapper.get_session_status(session_id)).status == "idle"
    assert await wrapper.session_exists(session_id)
    snapshot = await wrapper.read_session(session_id)
    assert snapshot.session_id == session_id and not snapshot.messages


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

    async def read(self, session_id=None, **_kwargs):
        return SimpleNamespace(thread=self.threads[session_id] if session_id else self.thread)

    async def request(self, method, _params, **_kwargs):
        if method == "thread/loaded/list":
            return SimpleNamespace(data=list(self.threads), next_cursor=None)
        if method == "thread/unsubscribe":
            return SimpleNamespace(status="unsubscribed")
        raise AssertionError(f"Unexpected native request: {method}")


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
    assert not (server.workspace_path / "session" / "proactive").exists()


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


@pytest_asyncio.fixture
async def runtime(tmp_path, monkeypatch):
    server = SdkCodexAppServer(
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
        finally:
            native.thread.status.root.type = "idle"


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


class UnsupportedAgent(BaseAgentWrapper):
    """Existing wrappers need not implement optional session capabilities."""

    async def reply(self, inputs, **kwargs):
        return {}


@pytest.mark.asyncio
async def test_optional_base_capabilities_fail_explicitly():
    agent = UnsupportedAgent()
    for operation, kwargs in (
        (agent.session_exists, {}),
        (agent.read_session, {}),
        (agent.get_session_status, {}),
        (agent.append_session, {"content": "text", "message_id": "one", "role": "user"}),
    ):
        with pytest.raises(NotImplementedError, match="UnsupportedAgent"):
            await operation("session-1", **kwargs)
