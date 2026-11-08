"""Workspace selection persistence with actual files and mocked native history/model boundaries."""

# pylint: disable=missing-function-docstring,protected-access,redefined-outer-name

import asyncio
from contextlib import aclosing
import json
from unittest.mock import AsyncMock

import pytest

from reme.components import ApplicationContext, CodexAppServer, PaRuntime
from reme.components.agent_wrapper import CodexAgentWrapper
from reme.schema import SessionMessage, SessionSnapshot, SessionStatus, StreamChunk
from reme.steps.proactive.pa_chat import PaChatStep, PaNewSessionStep, PaSessionStep
from .pa_helpers import PA_CONFIG


@pytest.fixture
def factory(tmp_path):
    snapshots = {}

    def make(workspace=tmp_path, session_dir="conversations", metadata_dir="metadata"):
        context = ApplicationContext(
            workspace_dir=str(workspace),
            session_dir=session_dir,
            metadata_dir=metadata_dir,
            language="zh",
        )
        server = CodexAppServer(app_context=context)
        agent = CodexAgentWrapper(app_context=context, app_server="")
        agent.app_server = server

        async def create(**_kwargs):
            session_id = f"session-{len(snapshots) + 1}"
            snapshots[session_id] = SessionSnapshot(
                session_id=session_id,
                state=SessionStatus(session_id=session_id, status="not_loaded"),
            )
            return session_id

        async def reply(query, *, resume, **_kwargs):
            snapshots[resume].messages.append(SessionMessage(id="answer", role="assistant", content=query))
            yield StreamChunk(
                chunk_type="reply_end",
                event="end",
                status="completed",
                session_id=resume,
                chunk={
                    key: value
                    for key, value in agent._reply_result(session_id=resume, result=query, raw={}).items()
                    if key != "events"
                },
            )

        agent.create_session = AsyncMock(side_effect=create)
        agent.release_session = AsyncMock()
        agent.get_session_status = AsyncMock(side_effect=lambda sid: snapshots[sid].state)
        agent.read_session = AsyncMock(side_effect=lambda sid: snapshots[sid].model_copy(deep=True))
        agent.reply_stream = reply
        runtime = PaRuntime(app_context=context, default_proactive=PA_CONFIG)
        context.components.update(
            agent_wrapper={"default": agent},
            codex_app_server={"default": server},
            pa={"default": runtime},
        )
        return context, agent, runtime

    return make


@pytest.mark.asyncio
async def test_new_runtime_restores_selection_snapshot_and_chat_without_creating_another_session(factory):
    context, agent, runtime = factory()
    async with runtime:
        selected = (await PaNewSessionStep(app_context=context)()).answer
        assert json.loads(runtime.session_ids_path.read_text())[-1]["session_id"] == selected
        assert (await PaChatStep(app_context=context)(query="before restart", stream_queue=asyncio.Queue())).success
        cursor = runtime.cursor
    context, agent, restored = factory()
    async with restored:
        assert (await PaSessionStep(app_context=context)()).answer == {"session_id": selected, "status": "not_loaded"}
        async with aclosing(restored.events(cursor)) as events:
            assert (await anext(events)).type == "reset"
            snapshot = await anext(events)
            assert snapshot.session_id == selected and snapshot.snapshot.messages[0].content == "before restart"
        response = await PaChatStep(app_context=context)(query="after restart", stream_queue=asyncio.Queue())
        assert response.success and response.answer == "after restart"
        agent.create_session.assert_not_awaited()
        assert "reply_kwargs" not in vars(restored) and "pa" not in context.metadata


@pytest.mark.asyncio
async def test_failed_atomic_replace_preserves_selection_file_listener_and_releases_only_new_session(
    factory,
    monkeypatch,
):
    context, agent, runtime = factory()
    async with runtime:
        first = (await PaNewSessionStep(app_context=context)()).answer
        original = runtime.session_ids_path.read_bytes()

        def fail(*_args):
            raise PermissionError("replace denied")

        monkeypatch.setattr("reme.components.pa_runtime.os.replace", fail)
        with pytest.raises(PermissionError, match="replace denied"):
            await PaNewSessionStep(app_context=context)()
        assert runtime.session_id == first
        assert runtime.session_ids_path.read_bytes() == original
        assert first in agent.app_server._session_listeners
        agent.release_session.assert_awaited_once_with("session-2")
        assert not list(runtime.session_ids_path.parent.glob(".pa-*.tmp"))


@pytest.mark.asyncio
async def test_cancelled_rotation_commits_disk_and_memory_before_unlocking_or_publishing(factory, monkeypatch):
    from reme.components import pa_runtime

    context, agent, runtime = factory()
    entered, release = asyncio.Event(), asyncio.Event()
    write = pa_runtime.complete_in_thread

    async def hold(*args):
        entered.set()
        await release.wait()
        return await write(*args)

    async with runtime:
        first = (await PaNewSessionStep(app_context=context)()).answer
        monkeypatch.setattr(pa_runtime, "complete_in_thread", hold)
        cursor = runtime.cursor
        task = asyncio.create_task(PaNewSessionStep(app_context=context)())
        await entered.wait()
        assert runtime.session_id == first
        assert json.loads(runtime.session_ids_path.read_text())[-1]["session_id"] == first
        task.cancel()
        await asyncio.sleep(0)
        assert not task.done() and runtime.session_lock.locked()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert runtime.session_id == "session-2"
        assert json.loads(runtime.session_ids_path.read_text())[-1]["session_id"] == "session-2"
        assert not runtime.session_lock.locked()
        async with aclosing(runtime.events(cursor)) as events:
            changes = [await anext(events), await anext(events)]
            assert any(event.type == "session_changed" and event.session_id == "session-2" for event in changes)
        agent.release_session.assert_awaited_once_with(first)


@pytest.mark.asyncio
async def test_non_regular_selection_is_rejected_without_opening_it(factory):
    _context, agent, runtime = factory()
    runtime.session_ids_path.mkdir(parents=True)
    with pytest.raises(ValueError, match="regular file"):
        await runtime.start()
    agent.create_session.assert_not_awaited()


@pytest.mark.asyncio
async def test_selection_read_permission_error_is_not_treated_as_a_missing_file(factory, monkeypatch):
    from pathlib import Path

    _context, agent, runtime = factory()
    path = runtime.session_ids_path
    path.parent.mkdir(parents=True)
    path.write_text('[{"session_id":"saved-session","last_active_at":"2026-11-08T12:00:00+08:00"}]')
    open_file = Path.open

    def fail(target, *args, **kwargs):
        if target == path:
            raise PermissionError("selection denied")
        return open_file(target, *args, **kwargs)

    monkeypatch.setattr(Path, "open", fail)
    with pytest.raises(PermissionError, match="selection denied"):
        await runtime.start()
    agent.create_session.assert_not_awaited()
    assert not runtime.is_started


@pytest.mark.asyncio
@pytest.mark.parametrize("escape", ["traversal", "symlink"])
async def test_selection_cannot_escape_the_workspace(factory, tmp_path, escape):
    outside = tmp_path / "outside"
    outside.mkdir()
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    if escape == "symlink":
        (workspace / "metadata").symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="workspace"):
        _context, _agent, runtime = factory(
            workspace,
            metadata_dir="../outside" if escape == "traversal" else "metadata",
        )
        await runtime.start()
    assert not list(outside.iterdir())


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "payload",
    ["{", "{}", '[{"session_id":42}]', '[{"session_id":"../outside","last_active_at":"now"}]'],
)
async def test_corrupt_registry_fails_startup_without_replacing_it(factory, payload):
    _context, agent, runtime = factory()
    path = runtime.session_ids_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(payload)
    with pytest.raises(ValueError):
        await runtime.start()
    assert path.read_text() == payload
    assert not runtime.is_started
    agent.create_session.assert_not_awaited()


@pytest.mark.asyncio
async def test_empty_registry_and_missing_native_history_never_create_a_replacement(factory):
    context, agent, runtime = factory()
    async with runtime:
        assert runtime.session_id is None and json.loads(runtime.session_ids_path.read_text()) == []
        agent.read_session.assert_not_awaited()
        agent.create_session.assert_not_awaited()
    runtime._write_json(
        runtime.session_ids_path,
        [{"session_id": "missing-session", "last_active_at": "2026-11-08T12:00:00+08:00"}],
    )
    original = runtime.session_ids_path.read_bytes()
    context, agent, restored = factory()
    async with restored:
        result = await PaSessionStep(app_context=context)()
        assert result.answer == {"session_id": "missing-session", "status": "error"}
        assert "error" in result.metadata
        assert restored.session_ids_path.read_bytes() == original
        agent.create_session.assert_not_awaited()
