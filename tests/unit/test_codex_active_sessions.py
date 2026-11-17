"""Codex snapshot forks and native append acceptance while generation owns its lease."""

# pylint: disable=missing-function-docstring,protected-access,redefined-outer-name

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from reme.components.agent_wrapper import CodexAgentWrapper
from . import test_agent_session_api

runtime = test_agent_session_api.runtime
codex_wrapper = test_agent_session_api.codex_wrapper


@pytest.mark.asyncio
@pytest.mark.parametrize("role", ["user", "assistant"])
async def test_immediate_append_submits_while_generation_runs_and_waits_only_for_persistence(codex_wrapper, role):
    wrapper, native = codex_wrapper
    received = asyncio.Event()
    queued = []
    request = native.client._client.request.side_effect

    async def enqueue(method, params, **kwargs):
        if method == "thread/inject_items":
            queued.append(params)
            received.set()
            return SimpleNamespace()
        return await request(method, params, **kwargs)

    native.client._client.request.side_effect = enqueue
    async with wrapper.app_server.thread("session-1"):
        native.thread.status.root.type = "active"
        pending = asyncio.create_task(
            wrapper.append_session("session-1", "queued", message_id="one", role=role, delivery="immediate"),
        )
        try:
            await asyncio.wait_for(received.wait(), 1)
            assert not pending.done() and not native.injections
            assert native.client.thread_resume.await_count == 1
            assert (await wrapper.read_session("session-1")).state.status == "active"
            await request("thread/inject_items", queued[0])
            result = await asyncio.wait_for(pending, 1)
            assert result.content == "queued" and result.role == role
            assert len(queued) == len(native.injections) == 1
            assert native.thread.status.root.type == "active"
            assert not wrapper.app_server._threads["session-1"].appends
        finally:
            native.thread.status.root.type = "idle"
            pending.cancel()
            await asyncio.gather(pending, return_exceptions=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", ["timeout", "cancelled", "lost_receipt"])
async def test_accepted_but_unpersisted_append_is_not_resubmitted_on_retry(codex_wrapper, outcome):
    wrapper, native = codex_wrapper
    received = asyncio.Event()
    queued = []
    request = native.client._client.request.side_effect

    async def enqueue(method, params, **kwargs):
        if method == "thread/inject_items":
            queued.append(params)
            received.set()
            if outcome == "lost_receipt":
                raise OSError("lost receipt")
            return SimpleNamespace()
        return await request(method, params, **kwargs)

    native.client._client.request.side_effect = enqueue
    pending = asyncio.create_task(
        wrapper.append_session("session-1", "once", message_id="one", delivery="immediate", timeout=0.1),
    )
    await asyncio.wait_for(received.wait(), 1)
    if outcome == "cancelled":
        pending.cancel()
    error = {"timeout": TimeoutError, "cancelled": asyncio.CancelledError, "lost_receipt": OSError}[outcome]
    with pytest.raises(error):
        await pending
    assert "reme-proactive-one" in wrapper.app_server._threads["session-1"].appends
    restarted = CodexAgentWrapper(app_context=wrapper.app_context)
    try:
        with pytest.raises(ValueError, match="different content"):
            await restarted.append_session("session-1", "different", message_id="one", delivery="immediate")
        retry = asyncio.create_task(
            restarted.append_session(
                "session-1",
                "once",
                message_id="one",
                delivery="immediate",
                expected_revision="stale",
            ),
        )
        try:
            await asyncio.sleep(0.02)
            assert not retry.done() and len(queued) == 1
            await request("thread/inject_items", queued[0])
            message = await asyncio.wait_for(retry, 1)
            assert message.content == "once" and len(queued) == 1
        finally:
            retry.cancel()
            await asyncio.gather(retry, return_exceptions=True)
    finally:
        await restarted.close()


@pytest.mark.asyncio
async def test_native_rejection_does_not_leave_an_uncertain_receipt(codex_wrapper):
    wrapper, native = codex_wrapper

    class Rejected(ValueError):
        """A definitive protocol rejection before native acceptance."""

        code = -32600

    native.client._client.request.side_effect = Rejected("invalid request")
    with pytest.raises(Rejected):
        await wrapper.append_session("session-1", "rejected", message_id="one", delivery="immediate")
    assert not wrapper.app_server._threads["session-1"].appends


@pytest.mark.asyncio
async def test_immediate_append_restores_an_unloaded_session(codex_wrapper):
    wrapper, native = codex_wrapper
    native.thread.status.root.type = "notLoaded"

    async def resume(_session_id, **_options):
        native.thread.status.root.type = "idle"
        return native.handle

    native.client.thread_resume.side_effect = resume
    message = await wrapper.append_session("session-1", "restored", message_id="one", delivery="immediate")
    assert message.content == "restored"
    native.client.thread_resume.assert_awaited_once()
    assert len(native.injections) == 1


@pytest.mark.asyncio
async def test_simultaneous_immediate_appends_share_retry_identity(codex_wrapper):
    wrapper, native = codex_wrapper
    messages = await asyncio.gather(
        *(wrapper.append_session("session-1", "once", message_id="one", delivery="immediate") for _ in range(3)),
    )
    assert messages[0] == messages[1] == messages[2]
    assert len(native.injections) == 1


@pytest.mark.asyncio
async def test_pending_receipts_survive_loaded_thread_pruning(codex_wrapper):
    wrapper, native = codex_wrapper
    native.client._client.request.side_effect = AsyncMock(return_value=SimpleNamespace())
    with pytest.raises(TimeoutError):
        await wrapper.append_session("session-1", "queued", message_id="one", delivery="immediate", timeout=0.05)
    await wrapper.app_server._prune_threads(native.client)
    assert "reme-proactive-one" in wrapper.app_server._threads["session-1"].appends
