"""Workspace ownership survives cancellation across the shared SDK process boundary."""

# pylint: disable=protected-access,missing-function-docstring

import asyncio
import json
import threading

import pytest
from openai_codex import AsyncCodex
from openai_codex.errors import InvalidParamsError, TransportClosedError

from reme.components import ApplicationContext
from reme.components.agent_wrapper.codex_agent_wrapper import CodexAgentWrapper
from reme.components.codex_app_server import CodexAppServer
from reme.enumeration import ComponentEnum

from .codex_workspace_test_support import workspace_server


def _config(tmp_path):
    return {"workspace_dir": {"alpha": str(tmp_path / "alpha"), "beta": str(tmp_path / "beta")}}


def _owned_on_disk(tmp_path, alias):
    path = tmp_path / alias / "mem_session" / "codex" / "owned_threads.json"
    return set(json.loads(path.read_text(encoding="utf-8"))["thread_ids"]) if path.is_file() else set()


async def _assert_isolated(tmp_path, context):
    assert context.workspaces["alpha"].codex_thread_state.owned_thread_ids == {"T"}
    assert _owned_on_disk(tmp_path, "alpha") == {"T"}
    for checked_context in (context, ApplicationContext(**_config(tmp_path))):
        # Rejection must happen before a new server initializes any SDK client.
        server = CodexAppServer(app_context=checked_context)
        checked_context.shared_components.setdefault(ComponentEnum.CODEX_APP_SERVER, {})["default"] = server
        beta = CodexAgentWrapper(name="beta", app_context=checked_context)
        try:
            with pytest.raises(ValueError, match="belongs to workspace 'alpha'"):
                await beta.reply("hello", resume="T", workspace="beta")
            assert server._codex is None
        finally:
            await beta.close()
            await server.close()
    assert not _owned_on_disk(tmp_path, "beta")


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel", [False, True])
@pytest.mark.parametrize("operation", ["reply", "compact"])
async def test_failed_compactor_keeps_another_wrappers_success(tmp_path, monkeypatch, cancel, operation):
    context = ApplicationContext(**_config(tmp_path))
    server = workspace_server(context, monkeypatch)
    compactor = CodexAgentWrapper(name="compactor", app_context=context)
    worker = CodexAgentWrapper(name="worker", app_context=context)
    original_get_client = server._get_client
    entered, finish = asyncio.Event(), asyncio.Event()
    calls = 0

    async def get_client():
        nonlocal calls
        calls += 1
        if calls == 1:
            entered.set()
            await finish.wait()
            raise RuntimeError("initialization failed")
        return await original_get_client()

    monkeypatch.setattr(server, "_get_client", get_client)
    task = asyncio.create_task(compactor.compact_session("T", workspace="alpha"))
    try:
        await asyncio.wait_for(entered.wait(), 3)
        if operation == "reply":
            assert (await worker.reply("hello", resume="T", workspace="alpha"))["result"] == "hello"
        else:
            await worker.compact_session("T", workspace="alpha")
        if cancel:
            task.cancel()
        else:
            finish.set()
        with pytest.raises(asyncio.CancelledError if cancel else RuntimeError):
            await asyncio.wait_for(task, 3)
        await _assert_isolated(tmp_path, context)
    finally:
        finish.set()
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        await compactor.close()
        await worker.close()
        await server.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel", [False, True])
async def test_last_predispatch_failure_releases_unused_claim(tmp_path, monkeypatch, cancel):
    context = ApplicationContext(**_config(tmp_path))
    server = workspace_server(context, monkeypatch)
    alpha = CodexAgentWrapper(app_context=context)
    entered = asyncio.Event()

    async def get_client():
        entered.set()
        await asyncio.Event().wait()

    monkeypatch.setattr(server, "_get_client", get_client)
    task = asyncio.create_task(alpha.compact_session("T", workspace="alpha"))
    try:
        await asyncio.wait_for(entered.wait(), 3)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert not _owned_on_disk(tmp_path, "alpha")
        assert not context.workspaces["alpha"].codex_thread_state.owned_thread_ids
        assert not context.workspaces["alpha"].codex_thread_state.provisional_claims
        if not cancel:

            async def reject_client():
                raise RuntimeError("initialization failed")

            monkeypatch.setattr(server, "_get_client", reject_client)
            with pytest.raises(RuntimeError, match="initialization failed"):
                await alpha.compact_session("T", workspace="alpha")
            assert not _owned_on_disk(tmp_path, "alpha")
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        await alpha.close()
        await server.close()


def _resume_response(thread_id, cwd):
    """Valid wire response parsed by the real synchronous SDK client."""
    return {
        "approvalPolicy": "never",
        "approvalsReviewer": "user",
        "cwd": str(cwd),
        "model": "test",
        "modelProvider": "test",
        "sandbox": {"type": "dangerFullAccess"},
        "thread": {
            "cliVersion": "0.163.0",
            "createdAt": 0,
            "cwd": str(cwd),
            "ephemeral": False,
            "id": thread_id,
            "modelProvider": "test",
            "preview": "",
            "sessionId": thread_id,
            "source": "appServer",
            "status": {"type": "idle"},
            "turns": [],
            "updatedAt": 0,
        },
    }


def _native_server(context, sdk, monkeypatch):
    """Use the real SDK clients with controlled RPC and notification transports."""
    from types import SimpleNamespace

    server = CodexAppServer(name="default", app_context=context)
    context.shared_components.setdefault(ComponentEnum.CODEX_APP_SERVER, {})["default"] = server
    sdk._initialized = True
    server._codex = sdk
    notifications = asyncio.Queue()
    monkeypatch.setattr(sdk._client, "next_notification", notifications.get)
    server._notifications = asyncio.create_task(server._watch_notifications(sdk))
    loop = asyncio.get_running_loop()

    def notify(status):
        event = SimpleNamespace(
            method="thread/status/changed",
            payload=SimpleNamespace(thread_id="T", status=SimpleNamespace(root=SimpleNamespace(type=status))),
        )
        loop.call_soon_threadsafe(notifications.put_nowait, event)

    return server, notify


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("operation", "blocked_method"),
    [("compact", "thread/compact/start"), ("compact", "thread/resume"), ("reply", "thread/resume")],
)
@pytest.mark.parametrize("outcome", ["success", "rejected", "transport_closed"])
async def test_cancelled_sdk_worker_preserves_persisted_ownership(
    tmp_path,
    monkeypatch,
    operation,
    blocked_method,
    outcome,
):
    """The real SDK's thread-offloaded RPC stays owned while its waiter is cancelled."""
    context = ApplicationContext(**_config(tmp_path))
    sdk = AsyncCodex()
    server, notify = _native_server(context, sdk, monkeypatch)
    alpha = CodexAgentWrapper(name="alpha", app_context=context)
    entered, release, finished = threading.Event(), threading.Event(), threading.Event()
    ops = []

    def request(method, params=None):
        ops.append(method)
        if method == blocked_method:
            if method == "thread/compact/start":
                notify("active")
            entered.set()
            try:
                if not release.wait(10):
                    raise TimeoutError("transport not released")
                if outcome == "rejected":
                    raise InvalidParamsError(-32602, "rejected")
                if outcome == "transport_closed":
                    raise TransportClosedError("outcome unknown")
                if method == "thread/compact/start":
                    notify("idle")
            finally:
                finished.set()
        if method == "thread/resume":
            return _resume_response(params["threadId"], tmp_path / "alpha")
        if method == "thread/read":
            return {"thread": _resume_response("T", tmp_path / "alpha")["thread"]}
        if method == "thread/compact/start":
            return {}
        if method == "thread/unsubscribe":
            return {"status": "unsubscribed"}
        raise AssertionError(f"Unexpected RPC: {method}")

    monkeypatch.setattr(sdk._client._sync, "_request_raw", request)
    call = (
        alpha.compact_session("T", workspace="alpha")
        if operation == "compact"
        else alpha.reply(
            "hello",
            resume="T",
            workspace="alpha",
        )
    )
    task = asyncio.create_task(call)
    try:
        assert await asyncio.to_thread(entered.wait, 3)
        task.cancel()
        await asyncio.sleep(0)  # Deliver cancellation while the transport remains blocked.
        assert not finished.is_set()
        await _assert_isolated(tmp_path, context)
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, 3)
        assert await asyncio.to_thread(finished.wait, 3)
        await _assert_isolated(tmp_path, context)
        if blocked_method == "thread/compact/start":
            assert "thread/unsubscribe" in ops  # Default cleanup cannot revoke ownership.
    finally:
        release.set()
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        await asyncio.to_thread(finished.wait, 3)
        await alpha.close()
        await server.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("method", ["thread/resume", "thread/compact/start"])
@pytest.mark.parametrize("error_type", [TransportClosedError, TimeoutError, RuntimeError])
async def test_unknown_sdk_failure_retains_ownership(tmp_path, monkeypatch, method, error_type):
    context = ApplicationContext(**_config(tmp_path))
    sdk = AsyncCodex()
    server, _ = _native_server(context, sdk, monkeypatch)
    alpha = CodexAgentWrapper(app_context=context)

    def request(rpc_method, params=None):
        if rpc_method == method:
            raise error_type("outcome unknown")
        if rpc_method == "thread/resume":
            return _resume_response(params["threadId"], tmp_path / "alpha")
        if rpc_method == "thread/read":
            return {"thread": _resume_response("T", tmp_path / "alpha")["thread"]}
        if rpc_method == "thread/unsubscribe":
            return {"status": "unsubscribed"}
        raise AssertionError(f"Unexpected RPC: {rpc_method}")

    monkeypatch.setattr(sdk._client._sync, "_request_raw", request)
    try:
        with pytest.raises(error_type, match="outcome unknown"):
            await alpha.compact_session("T", workspace="alpha")
        await _assert_isolated(tmp_path, context)
    finally:
        await alpha.close()
        await server.close()


@pytest.mark.asyncio
async def test_explicit_resume_rejection_releases_unused_claim(tmp_path, monkeypatch):
    context = ApplicationContext(**_config(tmp_path))
    sdk = AsyncCodex()
    server, _ = _native_server(context, sdk, monkeypatch)
    alpha = CodexAgentWrapper(app_context=context)

    def reject(*_args, **_kwargs):
        raise InvalidParamsError(-32602, "invalid thread")

    monkeypatch.setattr(sdk._client._sync, "_request_raw", reject)
    try:
        with pytest.raises(InvalidParamsError):
            await alpha.compact_session("T", workspace="alpha")
        assert not _owned_on_disk(tmp_path, "alpha")
        assert not context.workspaces["alpha"].codex_thread_state.owned_thread_ids
    finally:
        await alpha.close()
        await server.close()


@pytest.mark.asyncio
async def test_failed_first_reservation_preserves_pending_second_operation(tmp_path, monkeypatch):
    context = ApplicationContext(**_config(tmp_path))
    server = workspace_server(context, monkeypatch)
    wrappers = [CodexAgentWrapper(name=name, app_context=context) for name in ("first", "second")]
    entered = [asyncio.Event(), asyncio.Event()]
    finish = [asyncio.Event(), asyncio.Event()]
    calls = 0

    async def get_client():
        nonlocal calls
        index = calls
        calls += 1
        entered[index].set()
        await finish[index].wait()
        raise RuntimeError("initialization failed")

    monkeypatch.setattr(server, "_get_client", get_client)
    tasks = []
    try:
        for index, wrapper in enumerate(wrappers):
            tasks.append(asyncio.create_task(wrapper.compact_session("T", workspace="alpha")))
            await asyncio.wait_for(entered[index].wait(), 3)
        tasks[0].cancel()
        with pytest.raises(asyncio.CancelledError):
            await tasks[0]
        await _assert_isolated(tmp_path, context)
        finish[1].set()
        with pytest.raises(RuntimeError):
            await tasks[1]
        assert not _owned_on_disk(tmp_path, "alpha")
        assert not context.workspaces["alpha"].codex_thread_state.provisional_claims
    finally:
        for event in finish:
            event.set()
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        for wrapper in wrappers:
            await wrapper.close()
        await server.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["reply", "compact"])
async def test_unknown_workspace_fails_before_sdk_initialization(tmp_path, monkeypatch, operation):
    context = ApplicationContext(**_config(tmp_path))
    server = workspace_server(context, monkeypatch)
    wrapper = CodexAgentWrapper(app_context=context)
    try:
        with pytest.raises(ValueError, match="Unknown workspace 'typo'"):
            if operation == "reply":
                await wrapper.reply("invalid", resume="T", workspace="typo")
            else:
                await wrapper.compact_session("T", workspace="typo")
        assert server._codex is None
        assert not _owned_on_disk(tmp_path, "alpha")
        assert not _owned_on_disk(tmp_path, "beta")
    finally:
        await wrapper.close()
        await server.close()
