"""Shared file locks serialize source edits against committed notification decisions."""

# pylint: disable=missing-function-docstring,protected-access,redefined-outer-name

import asyncio
import gc
import weakref

import pytest

from reme.steps.file_io import get_path_lock
from reme.steps.proactive.chat import ProcessChatNotificationsStep
from reme.steps.proactive.tools import SubmitChatReplyStep
from .proactive_helpers import running_system, emit_notification


@pytest.mark.asyncio
async def test_path_lock_survives_handoff_and_reclaims_unused_entries(tmp_path):
    target = tmp_path / "target.md"
    lock = await get_path_lock(target)
    await lock.acquire()
    entered, release = asyncio.Event(), asyncio.Event()

    async def waiting_writer():
        async with await get_path_lock(target):
            entered.set()
            await release.wait()

    waiter = asyncio.create_task(waiting_writer())
    try:
        await asyncio.sleep(0)
        others = [await get_path_lock(tmp_path / str(index)) for index in range(1024)]
        lock.release()
        # The waiter has been woken, but has not resumed to set locked() yet.
        others.append(await get_path_lock(tmp_path / "overflow"))
        current = await get_path_lock(target)
        assert current is lock
        await asyncio.wait_for(entered.wait(), 2)
        assert current.locked()
    finally:
        release.set()
        await asyncio.wait_for(waiter, 2)
    reference = weakref.ref(current)
    del current, lock
    gc.collect()
    assert reference() is None


@pytest.mark.asyncio
async def test_source_editor_cannot_write_between_native_delivery_and_receipt(tmp_path, monkeypatch):
    async with running_system(tmp_path) as (manager, context, wrapper):
        ref = await emit_notification(manager, context)
        chat = context.components["agent_runtime"]["Remy"]
        entered, release, edited = asyncio.Event(), asyncio.Event(), asyncio.Event()
        append = wrapper.append_session

        async def slow(*args, **kwargs):
            value = await append(*args, **kwargs)
            entered.set()
            await release.wait()
            return value

        monkeypatch.setattr(wrapper, "append_session", slow)

        async def reply(_query, **options):
            return await SubmitChatReplyStep(app_context=context, agent_runtime="Remy")(
                action="send",
                content="Result",
                **options["injected_job_kwargs"],
            )

        monkeypatch.setattr(wrapper, "reply", reply)
        decision = asyncio.create_task(
            ProcessChatNotificationsStep(app_context=context, agent_runtime="Remy")(priority="P1", msg_ids=[ref.id]),
        )
        await asyncio.wait_for(entered.wait(), 1)

        async def edit():
            async with await get_path_lock(tmp_path / ref.record_path):
                assert await chat.inbox.processed(ref.id)
                edited.set()

        writer = asyncio.create_task(edit())
        await asyncio.sleep(0.01)
        assert not edited.is_set()
        release.set()
        await decision
        await writer
        assert edited.is_set()


@pytest.mark.asyncio
async def test_decision_revalidates_after_waiting_for_a_source_editor(tmp_path, monkeypatch):
    async with running_system(tmp_path) as (manager, context, wrapper):
        ref = await emit_notification(manager, context)
        path = tmp_path / ref.record_path
        lock = await get_path_lock(path)
        ready = asyncio.Event()

        async def reply(_query, **options):
            ready.set()
            return await SubmitChatReplyStep(app_context=context, agent_runtime="Remy")(
                action="skip",
                **options["injected_job_kwargs"],
            )

        monkeypatch.setattr(wrapper, "reply", reply)
        async with lock:
            task = asyncio.create_task(
                ProcessChatNotificationsStep(app_context=context, agent_runtime="Remy")(
                    priority="P1",
                    msg_ids=[ref.id],
                ),
            )
            await asyncio.wait_for(ready.wait(), 1)
            path.write_text(path.read_text().replace("Verified result", "Changed"))
        with pytest.raises(RuntimeError, match="source changed"):
            await task
        assert ref.id in await context.components["agent_runtime"]["Remy"].inbox.pending_ids()
