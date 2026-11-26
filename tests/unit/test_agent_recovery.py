"""Durable handoffs recover local and native side effects without rerunning models."""

# pylint: disable=missing-function-docstring,protected-access,redefined-outer-name

from unittest.mock import AsyncMock

import pytest

from reme.steps.proactive.chat import ProcessChatNotificationsStep
from reme.steps.proactive.tools import SubmitChatReplyStep
from .proactive_helpers import running_system, emit_notification


@pytest.mark.asyncio
async def test_committed_input_recovers_missing_sender_outbox_after_restart(tmp_path, monkeypatch):
    async with running_system(tmp_path, runtimes=[{"agent_name": "monitor"}]) as (manager, context, _):
        runtime = await manager.resolve("monitor")
        monkeypatch.setattr(runtime.outbox, "prepare_reference", AsyncMock(side_effect=OSError("disk unavailable")))
        ref = await emit_notification(manager, context)
        assert not await runtime.inbox.pending_ids() and runtime.delivery_error
        assert not await context.components["agent_runtime"]["Remy"].inbox.pending_ids()
    async with running_system(tmp_path) as (manager, context, wrapper):
        runtime = await manager.resolve("monitor")
        await runtime.recover_handoffs()
        assert await context.components["agent_runtime"]["Remy"].inbox.pending_ids() == {ref.id}
        assert not runtime.delivery_error and not wrapper.turns
        await runtime.recover_handoffs()
        assert len(await runtime.outbox.entries()) == 1


@pytest.mark.parametrize("stage", ["native_response", "sender_status", "receiver_ack"])
@pytest.mark.asyncio
async def test_native_side_effect_recovers_before_another_model_decision(tmp_path, monkeypatch, stage):
    async with running_system(tmp_path) as (manager, context, wrapper):
        ref = await emit_notification(manager, context)
        chat = context.components["agent_runtime"]["Remy"]
        if stage == "native_response":
            append = wrapper.append_session

            async def lost(*args, **kwargs):
                await append(*args, **kwargs)
                raise OSError("lost native response")

            monkeypatch.setattr(wrapper, "append_session", lost)
        elif stage == "sender_status":
            save = chat.outbox.save

            async def lost(row):
                if row.status == "sent":
                    raise OSError("status disk write failed")
                return await save(row)

            monkeypatch.setattr(chat.outbox, "save", lost)
        else:
            monkeypatch.setattr(chat.inbox, "ack", AsyncMock(side_effect=OSError("receiver ack failed")))

        async def reply(_query, **options):
            return await SubmitChatReplyStep(app_context=context, agent_runtime="Remy")(
                action="send",
                content="Result",
                **options["injected_job_kwargs"],
            )

        monkeypatch.setattr(wrapper, "reply", reply)
        with pytest.raises(OSError):
            await ProcessChatNotificationsStep(app_context=context, agent_runtime="Remy")(
                priority="P1",
                msg_ids=[ref.id],
            )
        assert ref.id in await chat.inbox.pending_ids()
        messages = {name: list(rows) for name, rows in wrapper.messages.items()}
    async with running_system(tmp_path) as (_, context, wrapper):
        wrapper.messages = messages
        chat = context.components["agent_runtime"]["Remy"]
        await chat.recover_deliveries()
        assert not await chat.inbox.pending_ids()
        assert (await chat.inbox.processed(ref.id)).metadata["decision"] == "send"
        assert (await chat.outbox.entries())[0].status == "acknowledged"
        assert sum(len(rows) for rows in wrapper.messages.values()) == 1 and not wrapper.turns


@pytest.mark.asyncio
async def test_full_receiver_does_not_lose_committed_results_or_block_existing_consumer(tmp_path, monkeypatch):
    async with running_system(tmp_path) as (manager, context, wrapper):
        chat = context.components["agent_runtime"]["Remy"]
        chat.inbox.max_pending_items = 1
        first = await emit_notification(manager, context)
        second = await emit_notification(manager, context, suffix=1)
        runtime = await manager.resolve("monitor")
        assert runtime.delivery_error and not await runtime.inbox.pending_ids()

        async def skip(_query, **options):
            return await SubmitChatReplyStep(app_context=context, agent_runtime="Remy")(
                action="skip",
                **options["injected_job_kwargs"],
            )

        monkeypatch.setattr(wrapper, "reply", skip)
        await ProcessChatNotificationsStep(app_context=context, agent_runtime="Remy")(priority="P1", msg_ids=[first.id])
        await runtime.recover_handoffs()
        assert await chat.inbox.pending_ids() == {second.id} and not runtime.delivery_error
        await ProcessChatNotificationsStep(app_context=context, agent_runtime="Remy")(
            priority="P1",
            msg_ids=[second.id],
        )
        assert not await chat.inbox.pending_ids()


@pytest.mark.parametrize("uncertain", ["read_error", "missing_message"])
@pytest.mark.asyncio
async def test_unresolved_native_delivery_reserves_its_notifications_before_partial_rebatch(
    tmp_path,
    monkeypatch,
    uncertain,
):
    async with running_system(tmp_path) as (manager, context, wrapper):
        first = await emit_notification(manager, context)
        second = await emit_notification(manager, context, suffix=1)
        chat = context.components["agent_runtime"]["Remy"]
        ack = chat.inbox.ack
        monkeypatch.setattr(chat.inbox, "ack", AsyncMock(side_effect=OSError("ack unavailable")))

        async def send(_query, **options):
            return await SubmitChatReplyStep(app_context=context, agent_runtime="Remy")(
                action="send",
                content="Combined result",
                **options["injected_job_kwargs"],
            )

        monkeypatch.setattr(wrapper, "reply", send)
        process = ProcessChatNotificationsStep(app_context=context, agent_runtime="Remy")
        with pytest.raises(OSError, match="ack unavailable"):
            await process(priority="P1", msg_ids=[first.id, second.id])
        monkeypatch.setattr(chat.inbox, "ack", ack)
        session = (await chat.current_session()).session_id
        native = list(wrapper.messages[session])
        read = wrapper.read_session
        probes = 0

        async def uncertain_snapshot(identifier):
            nonlocal probes
            probes += 1
            if probes == 1:
                if uncertain == "read_error":
                    raise OSError("native history temporarily unavailable")
                return (await read(identifier)).model_copy(update={"messages": []})
            return await read(identifier)

        monkeypatch.setattr(wrapper, "read_session", uncertain_snapshot)
        inference = AsyncMock(side_effect=send)
        monkeypatch.setattr(wrapper, "reply", inference)
        with pytest.raises(RuntimeError, match="recovery"):
            await process(priority="P1", msg_ids=[first.id])
        inference.assert_not_awaited()
        assert wrapper.messages[session] == native
        assert await chat.inbox.pending_ids() == {first.id, second.id}
        monkeypatch.setattr(wrapper, "read_session", read)
        await chat.recover_deliveries()
        assert not await chat.inbox.pending_ids()
        assert (await chat.outbox.entries())[0].status == "acknowledged"
