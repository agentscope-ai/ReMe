"""Reminder decision capabilities, source guards and native Chat delivery."""

# pylint: disable=missing-function-docstring,protected-access,redefined-outer-name

import asyncio
from unittest.mock import AsyncMock

import pytest

from reme.schema import SessionMessage
from reme.steps.proactive.chat import ProcessChatNotificationsStep
from reme.steps.proactive.tools import SubmitChatReplyStep
from .proactive_helpers import running_system, emit_notification


@pytest.mark.parametrize("action,content", [("send", "Update"), ("skip", None)])
@pytest.mark.parametrize("language", ["en", "zh"])
@pytest.mark.asyncio
async def test_fork_requires_explicit_decision_and_keeps_formal_chat_binding(
    tmp_path,
    monkeypatch,
    action,
    content,
    language,
):
    async with running_system(tmp_path, language=language) as (manager, context, wrapper):
        ref = await emit_notification(manager, context)
        path = tmp_path / ref.record_path
        before = path.read_bytes()
        chat = context.components["agent_runtime"]["Remy"]
        captured = {}

        async def reply(query, **options):
            captured.update(options)
            assert ref.record_path in query and "Evidence" not in query
            values = {"action": action, **({"content": content} if content else {})}
            tool = SubmitChatReplyStep(app_context=context, agent_runtime="Remy")
            first = await tool(**values, **options["injected_job_kwargs"])
            second = await tool(**values, **options["injected_job_kwargs"])
            assert first.answer == second.answer
            return {"status": "completed"}

        monkeypatch.setattr(wrapper, "reply", reply)
        response = await ProcessChatNotificationsStep(app_context=context, agent_runtime="Remy")(
            priority="P1",
            msg_ids=[ref.id],
        )
        assert response.success and response.answer["sent"] == (action == "send")
        assert (
            not await chat.inbox.pending_ids() and (await chat.inbox.processed(ref.id)).metadata["decision"] == action
        )
        assert path.read_bytes() == before and chat.active_turn is None
        assert captured["job_tools"] == ["submit_chat_reply"] and captured["skills"] == []
        assert captured["fork_session"] and captured["ephemeral"] and captured["sandbox"] == "read-only"
        assert wrapper.created[0]["job_tools"] == ["manage_proactive_runtime", "manage_connector", "manage_notebook"]
        assert len(wrapper.messages[captured["resume"]]) == (1 if action == "send" else 0)
        if action == "send":
            assert (await chat.outbox.entries())[0].status == "acknowledged"
        with pytest.raises(ValueError):
            await SubmitChatReplyStep(app_context=context, agent_runtime="Remy")(
                action="skip",
                **captured["injected_job_kwargs"],
            )


@pytest.mark.parametrize(
    "change",
    ["source_message", "source_priority", "foreground", "native", "disabled", "deleted", "window"],
)
@pytest.mark.parametrize("action", ["send", "skip"])
@pytest.mark.asyncio
async def test_changed_context_prevents_stale_delivery_and_consumption(tmp_path, monkeypatch, change, action):
    async with running_system(tmp_path) as (manager, context, wrapper):
        ref = await emit_notification(manager, context)
        chat = context.components["agent_runtime"]["Remy"]

        async def reply(_query, **options):
            if change.startswith("source"):
                path = tmp_path / ref.record_path
                text = path.read_text()
                path.write_text(
                    (
                        text.replace("Verified result", "Changed")
                        if change == "source_message"
                        else text.replace("P1", "P0")
                    ),
                )
            elif change == "foreground":
                async with chat.chat_turn():
                    pass
            elif change == "native":
                wrapper.messages[options["resume"]].append(
                    SessionMessage(id="user", role="user", content="New instruction"),
                )
            elif change == "disabled":
                await manager.update("monitor", {"enabled": False})
            elif change == "deleted":
                await manager.delete("monitor")
            else:
                now = manager.now()
                hour = (now.hour + 1) % 24
                await manager.update("monitor", {"notify_window": {"start": f"{hour:02}:00", "end": f"{hour:02}:01"}})
            return await SubmitChatReplyStep(app_context=context, agent_runtime="Remy")(
                action=action,
                **({"content": "Stale"} if action == "send" else {}),
                **options["injected_job_kwargs"],
            )

        monkeypatch.setattr(wrapper, "reply", reply)
        with pytest.raises((RuntimeError, ValueError)):
            await ProcessChatNotificationsStep(app_context=context, agent_runtime="Remy")(
                priority="P1",
                msg_ids=[ref.id],
            )
        assert ref.id in await chat.inbox.pending_ids() and not (await chat.inbox.snapshot())["in_flight_count"]
        assert (
            not any(row.role == "assistant" for rows in wrapper.messages.values() for row in rows)
            and chat.active_turn is None
        )


@pytest.mark.parametrize("phase", ["setup", "inference", "no_decision"])
@pytest.mark.asyncio
async def test_failure_or_timeout_releases_claim_and_revokes_capability(tmp_path, monkeypatch, phase):
    async with running_system(tmp_path) as (manager, context, wrapper):
        ref = await emit_notification(manager, context)
        chat = context.components["agent_runtime"]["Remy"]

        async def stall(*_, **__):
            await asyncio.Event().wait()

        if phase == "setup":
            monkeypatch.setattr(wrapper, "create_session", stall)
        elif phase == "inference":
            monkeypatch.setattr(wrapper, "reply", stall)
        else:
            monkeypatch.setattr(wrapper, "reply", AsyncMock(return_value={"status": "completed"}))
        with pytest.raises((TimeoutError, RuntimeError)):
            await ProcessChatNotificationsStep(app_context=context, agent_runtime="Remy", timeout_seconds=0.03)(
                priority="P1",
                msg_ids=[ref.id],
            )
        assert ref.id in await chat.inbox.pending_ids() and not (await chat.inbox.snapshot())["in_flight_count"]
        assert chat.active_turn is None


@pytest.mark.asyncio
async def test_later_arrival_is_not_consumed_by_the_selected_decision(tmp_path, monkeypatch):
    async with running_system(tmp_path) as (manager, context, wrapper):
        ref = await emit_notification(manager, context)
        chat = context.components["agent_runtime"]["Remy"]

        async def reply(_query, **options):
            await emit_notification(manager, context, suffix=1)
            return await SubmitChatReplyStep(app_context=context, agent_runtime="Remy")(
                action="skip",
                **options["injected_job_kwargs"],
            )

        monkeypatch.setattr(wrapper, "reply", reply)
        await ProcessChatNotificationsStep(app_context=context, agent_runtime="Remy")(priority="P1", msg_ids=[ref.id])
        assert len(await chat.inbox.pending_ids()) == 1 and await chat.inbox.processed(ref.id)
