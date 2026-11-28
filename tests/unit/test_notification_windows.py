"""Multiple reminder periods share one contract from configuration through delivery."""

# pylint: disable=missing-function-docstring,redefined-outer-name

from datetime import datetime
from unittest.mock import AsyncMock

import pytest

from reme.components.proactive.runtime.proactive import ProactiveAgentRuntime
from reme.config import resolve_app_config
from reme.schema.proactive_agent import (
    ManageProactiveRuntimeRequest,
    ProactiveRuntimeConfig,
    ProactiveRuntimeSettings,
)
from reme.steps.proactive.chat import (
    ChatNotificationsLoopStep,
    ProcessChatNotificationsStep,
)
from reme.steps.proactive.tools import ManageProactiveRuntimeStep, SubmitChatReplyStep
from .proactive_helpers import running_system, emit_notification

PERIODS = [{"start": "09:00", "end": "12:00"}, {"start": "14:00", "end": "23:00"}]


@pytest.mark.parametrize(
    "time,allowed",
    [
        ("08:59", False),
        ("09:00", True),
        ("11:59", True),
        ("12:00", False),
        ("13:59", False),
        ("14:00", True),
        ("22:59", True),
        ("23:00", False),
        ("00:00", False),
    ],
)
def test_default_period_boundaries_and_lunch_break(time, allowed):
    settings = ProactiveRuntimeSettings()
    assert settings.model_dump()["notify_windows"] == PERIODS
    assert settings.allows_notifications(datetime.fromisoformat(f"2099-01-01T{time}:00+08:00")) is allowed


@pytest.mark.parametrize(
    "periods,time,allowed",
    [
        ([], "03:00", True),
        ([{"start": "22:00", "end": "06:00"}], "23:00", True),
        ([{"start": "22:00", "end": "06:00"}], "05:59", True),
        ([{"start": "22:00", "end": "06:00"}], "06:00", False),
        ([{"start": "22:00", "end": "06:00"}], "12:00", False),
        ([{"start": "08:00", "end": "08:00"}], "03:00", True),
        (PERIODS + [{"start": "22:00", "end": "06:00"}], "01:00", True),
    ],
)
def test_unrestricted_overnight_and_all_day_periods(periods, time, allowed):
    settings = ProactiveRuntimeSettings(notify_windows=periods)
    assert settings.allows_notifications(datetime.fromisoformat(f"2099-01-01T{time}:00+08:00")) is allowed


@pytest.mark.parametrize(
    "values",
    [
        {"notify_window": {"start": "09:00", "end": "18:00"}},
        {"notify_windows": None},
        {"notify_windows": {"start": "09:00", "end": "12:00"}},
        {"notify_windows": [{}]},
        {"notify_windows": [{"start": "9:00", "end": "12:00"}]},
        {"notify_windows": [{"start": "09:00", "end": "24:00"}]},
        {"notify_windows": [{"start": "09:00", "end": "12:00", "extra": True}]},
    ],
)
def test_only_the_new_list_contract_is_accepted(values):
    with pytest.raises(ValueError):
        ProactiveRuntimeSettings.model_validate(values)
    with pytest.raises(ValueError):
        ManageProactiveRuntimeRequest(action="update", agent_name="monitor", **values)


def test_default_lists_are_independent_and_builtin_tool_schema_matches():
    first, second = ProactiveRuntimeSettings(), ProactiveRuntimeSettings()
    first.notify_windows[0].start = "10:00"
    first.notify_windows.pop()
    assert second.model_dump()["notify_windows"] == PERIODS
    assert ProactiveRuntimeConfig(agent_name="monitor", agent_type="event").model_dump()["notify_windows"] == PERIODS
    for preset in ("proactive", "proactive_cc"):
        config = resolve_app_config(config=preset, log_config=False)
        tool = config["jobs"]["manage_proactive_runtime"]["parameters"]["properties"]
        assert tool["notify_windows"]["default"] == PERIODS and "notify_window" not in tool


@pytest.mark.asyncio
async def test_defaults_and_period_updates_persist_without_resetting_omitted_fields(
    tmp_path,
):
    async with running_system(tmp_path) as (manager, context, wrapper):
        preset = ProactiveAgentRuntime(
            name="preset",
            agent_type="event",
            agent_wrapper=wrapper,
            notification_target=context.components["agent_runtime"]["Remy"],
            app_context=context,
        )
        assert preset.initial_config.model_dump()["notify_windows"] == PERIODS
        tool = ManageProactiveRuntimeStep(app_context=context)
        created = (await tool(action="create", agent_name="monitor", agent_type="event")).answer
        assert created["config"]["notify_windows"] == PERIODS
        periods = PERIODS + [{"start": "23:30", "end": "01:00"}]
        await tool(action="update", agent_name="monitor", notify_windows=periods)
        updated = (await tool(action="update", agent_name="monitor", min_trigger_interval_seconds=30)).answer
        assert updated["config"]["notify_windows"] == periods
        assert manager.registry.get("monitor").model_dump()["notify_windows"] == periods
        await tool(action="update", agent_name="monitor", notify_windows=[])
        assert manager.registry.get("monitor").notify_windows == []
        await tool(action="delete", agent_name="monitor")
        restored = (await tool(action="create", agent_name="monitor", agent_type="event")).answer
        assert restored["config"]["notify_windows"] == []


@pytest.mark.parametrize("priority", ["P0", "P1", "P2"])
@pytest.mark.asyncio
async def test_lunch_defers_every_priority_and_afternoon_resumes_processing(tmp_path, monkeypatch, priority):
    async with running_system(tmp_path) as (manager, context, _):
        await manager.create({"agent_type": "event", "agent_name": "monitor"})
        ref = await emit_notification(manager, context, priority=priority)
        chat = context.components["agent_runtime"]["Remy"]
        loop = ChatNotificationsLoopStep(app_context=context, agent_runtime="Remy")
        process = AsyncMock()
        monkeypatch.setattr(loop, "_process", process)
        monkeypatch.setattr(chat, "now", lambda: datetime.fromisoformat("2099-01-01T13:00:00+08:00"))
        await loop.tick()
        assert chat.notification_worker is None and ref.id in await chat.inbox.pending_ids()
        process.assert_not_awaited()
        monkeypatch.setattr(chat, "now", lambda: datetime.fromisoformat("2099-01-01T14:00:00+08:00"))
        await loop.tick()
        await chat.notification_worker
        process.assert_awaited_once_with(priority, [ref.id])


@pytest.mark.asyncio
async def test_crossing_into_lunch_rechecks_periods_before_committing_delivery(tmp_path, monkeypatch):
    async with running_system(tmp_path) as (manager, context, wrapper):
        await manager.create({"agent_type": "event", "agent_name": "monitor"})
        ref = await emit_notification(manager, context)
        chat = context.components["agent_runtime"]["Remy"]
        monkeypatch.setattr(chat, "now", lambda: datetime.fromisoformat("2099-01-01T11:59:00+08:00"))

        async def reply(_query, **options):
            monkeypatch.setattr(chat, "now", lambda: datetime.fromisoformat("2099-01-01T12:00:00+08:00"))
            return await SubmitChatReplyStep(app_context=context, agent_runtime="Remy")(
                action="send",
                content="Update",
                **options["injected_job_kwargs"],
            )

        monkeypatch.setattr(wrapper, "reply", reply)
        with pytest.raises(RuntimeError, match="no longer eligible"):
            await ProcessChatNotificationsStep(app_context=context, agent_runtime="Remy")(
                priority="P1",
                msg_ids=[ref.id],
            )
        assert ref.id in await chat.inbox.pending_ids()
        assert not await chat.outbox.entries() and chat.active_turn is None
