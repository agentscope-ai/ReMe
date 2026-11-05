"""Session history, proactive injection recovery, activity and service contracts."""

# pylint: disable=protected-access,redefined-outer-name,missing-function-docstring,too-many-locals

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest
import pytest_asyncio
from openai_codex.generated.v2_all import ThreadItem

from reme.application import Application
from reme.components.agent_wrapper import BaseAgentWrapper, CodexAgentWrapper
from reme.components.agent_wrapper import codex_session_history as history
from reme.components.application_context import ApplicationContext
from reme.components.codex_app_server import CodexAppServer
from reme.config import resolve_app_config


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

    async def read(self, *_args, **_kwargs):
        return SimpleNamespace(thread=self.thread)

    async def request(self, method, params, **_kwargs):
        if method == "thread/loaded/list":
            return SimpleNamespace(data=[self.thread.id], next_cursor=None)
        if method == "thread/unsubscribe":
            return SimpleNamespace(status="unsubscribed")
        assert method == "thread/inject_items"
        self.injected.extend(params["items"])
        with open(self.thread.path, "a", encoding="utf-8") as stream:
            for item in params["items"]:
                stream.write(json.dumps({"type": "response_item", "payload": item}) + "\n")
        if self.inject_hook:
            await self.inject_hook()
        return SimpleNamespace()


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
async def test_read_and_status_do_not_wait_behind_generation(runtime):
    server, native = runtime
    async with server.thread("session-1"):
        native.thread.status.root.type = "active"
        native.thread.status.root.active_flags = ["waitingOnUserInput"]
        try:
            async with asyncio.timeout(0.5):
                status = await server.get_session_status("session-1")
                snapshot = await server.read_session("session-1")
            assert status.status == snapshot.state.status == "active"
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
    original = history.write_records

    def fail_commit(path, records):
        if records[-1]["injected"]:
            raise OSError("publication interrupted")
        original(path, records)

    monkeypatch.setattr(history, "write_records", fail_commit)
    with pytest.raises(OSError, match="publication interrupted"):
        await server.append_session("session-1", "once", message_id="recover")
    assert not history.read_records(server._message_path("session-1"))[0]["injected"]
    assert (await server.read_session("session-1")).messages[-1].content == "once"
    monkeypatch.setattr(history, "write_records", original)
    await server.append_session("session-1", "once", message_id="recover")
    assert len(native.injected) == 1


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
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await pending
    await server.append_session("session-1", "once", message_id="cancel")
    assert len(native.injected) == 1


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
