"""Named management actions and small model results."""

# pylint: disable=missing-function-docstring,protected-access

from datetime import datetime
from unittest.mock import AsyncMock

import pytest

from reme.schema.proactive_agent import ManageProactiveRuntimeRequest
from reme.steps.proactive.tools.manage_proactive_runtime import ManageProactiveRuntimeStep
from .proactive_helpers import running_system as running_manager


@pytest.mark.parametrize(
    "arguments",
    [
        {"action": "list", "agent_name": "task"},
        {"action": "list", "agent_name": None},
        {"action": "create", "agent_name": "task", "task_description": "Observe", "agent_type": "proactive"},
        {
            "action": "create",
            "agent_name": "task",
            "task_description": "Observe",
            "agent_type": "schedule",
            "schedule": "0 8 * * *",
        },
        {
            "agent_type": "event",
            "action": "create",
            "agent_name": "task",
            "task_description": "Observe",
            "temporary": True,
        },
        {"agent_type": "event", "action": "create", "agent_name": "task", "task_description": " "},
        {
            "action": "create",
            "config": {"agent_type": "event", "agent_name": "task", "task_description": "Do work"},
        },
        {"action": "status", "agent_id": "task"},
        {"action": "status", "name": "task"},
        {"action": "create", "agent_name": "task", "task_description": "Observe", "agent_type": "event", "cron": None},
        {"action": "create", "agent_name": "task", "task_description": "Observe", "agent_type": "chat"},
        {"agent_type": "event", "action": "create", "agent_name": "task", "instructions": "Observe"},
        {"agent_type": "event", "action": "create", "agent_name": "task", "description": "Observe"},
        {
            "agent_type": "event",
            "action": "create",
            "agent_name": "task",
            "task_description": "Observe",
            "trigger_type": "event",
        },
        {"action": "status"},
        {"action": "status", "agent_name": "task", "enabled": True},
        {"action": "update", "agent_name": "task"},
        {"action": "update", "agent_name": "task", "task_description": "New work"},
        {"action": "update", "agent_name": "task", "temporary": True},
        {"action": "update", "agent_name": "task", "max_batch_events": 10},
        {"action": "update", "agent_name": "task", "min_trigger_interval_seconds": True},
        {"action": "delete", "agent_name": "task", "schedule": None},
        {"action": "query", "agent_name": "task", "query": "Progress?"},
        {"action": "query", "agent_name": "task", "question": " "},
    ],
)
def test_action_arguments_reject_legacy_and_irrelevant_fields(arguments):
    with pytest.raises(ValueError):
        ManageProactiveRuntimeRequest.model_validate(arguments)


@pytest.mark.asyncio
async def test_all_actions_expose_only_their_results_and_keep_task_files(tmp_path, monkeypatch):
    async with running_manager(tmp_path) as (runtime, context, _):
        now = datetime.fromisoformat("2026-11-17T07:00:00+08:00")
        monkeypatch.setattr(
            "reme.components.proactive.runtime.base.datetime",
            type("Clock", (), {"now": staticmethod(lambda tz: now)}),
        )
        tool = ManageProactiveRuntimeStep(app_context=context)
        arguments = {
            "agent_type": "time",
            "action": "create",
            "agent_name": "tech_news",
            "task_description": "每天08:00整理科技新闻。\n保留来源。\n",
            "schedule": "0 8 * * *",
        }
        created = (await tool(**arguments)).answer
        assert created.pop("restored") is False
        assert "run_job" not in created["config"]
        assert created["config"]["schedule"] == "0 8 * * *"
        assert created["next_trigger_at"] == "2026-11-17T08:00:00+08:00"
        task_file = tmp_path / created["task_path"]
        assert task_file.read_text() == arguments["task_description"]
        assert runtime.runtimes["tech_news"].schedule is not None
        assert (await tool(action="list")).answer == {"items": [created], "total": 1}
        assert (await tool(action="status", agent_name="tech_news")).answer == created
        updated = (await tool(action="update", agent_name="tech_news", schedule="0 9 * * *")).answer
        assert (
            updated["config"]["schedule"] == "0 9 * * *" and updated["next_trigger_at"] == "2026-11-17T09:00:00+08:00"
        )
        paused = (await tool(action="update", agent_name="tech_news", enabled=False)).answer
        assert not paused["config"]["enabled"] and paused["config"]["schedule"] == "0 9 * * *"
        assert task_file.read_text() == arguments["task_description"]

        query = AsyncMock(return_value={"status": "completed", "result": "今天的早报已经完成。"})
        monkeypatch.setattr(tool, "query_reply", query)
        assert (
            await tool(action="query", agent_name="tech_news", question="进展如何？")
        ).answer == "今天的早报已经完成。"
        query.assert_awaited_once_with("tech_news", "进展如何？")
        deleted = (await tool(action="delete", agent_name="tech_news")).answer
        assert deleted == {"agent_name": "tech_news", "deleted": True, "files_retained": True}
        assert (await tool(action="list")).answer == {"items": [], "total": 0}
        assert task_file.read_text() == arguments["task_description"]
        restored = (await tool(**{**arguments, "task_description": "Changed request"})).answer
        assert restored["restored"]
        assert task_file.read_text() == arguments["task_description"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "schedule, other_time, agent_type",
    [(None, "0 * * * *", "event"), ("0 * * * *", None, "time")],
)
async def test_trigger_type_is_fixed_after_creation_and_on_restore(tmp_path, schedule, other_time, agent_type):
    async with running_manager(tmp_path) as (runtime, context, _):
        tool = ManageProactiveRuntimeStep(app_context=context)
        name = "experiment_watch"
        task = {
            "agent_type": ("time" if schedule is not None else "event"),
            "agent_name": name,
            "task_description": "跟进实验",
            "schedule": schedule,
        }
        created = (await tool(action="create", **task)).answer
        assert created["config"]["agent_type"] == agent_type
        path = runtime.registry.path(name) / "config.json"
        before = path.read_bytes()
        with pytest.raises(ValueError, match="schedule"):
            await tool(action="update", agent_name=name, schedule=other_time, enabled=False)
        assert path.read_bytes() == before

        changes = {"min_trigger_interval_seconds": 60}
        if schedule is not None:
            changes["schedule"] = "0 8 * * *"
        updated = (await tool(action="update", agent_name=name, **changes)).answer
        assert updated["config"]["schedule"] == changes.get("schedule")
        assert updated["config"]["min_trigger_interval_seconds"] == 60
        await tool(action="update", agent_name=name, notify_window={"start": "09:00", "end": "18:00"})
        assert (await runtime.definition(name)).notify_window.start == "09:00"
        cleared = (await tool(action="update", agent_name=name, notify_window=None)).answer
        assert cleared["config"]["notify_window"] is None

        await tool(action="delete", agent_name=name)
        deleted = path.read_bytes()
        with pytest.raises(ValueError, match="trigger type cannot change"):
            await tool(
                action="create",
                **{**task, "agent_type": "time" if agent_type == "event" else "event", "schedule": other_time},
            )
        assert path.read_bytes() == deleted
        restored = (await tool(action="create", **task)).answer
        assert restored["config"]["schedule"] == schedule


@pytest.mark.asyncio
async def test_query_errors_remain_failed_tool_results(tmp_path, monkeypatch):
    async with running_manager(tmp_path) as (_, context, _):
        tool = ManageProactiveRuntimeStep(app_context=context)
        monkeypatch.setattr(
            tool,
            "query_reply",
            AsyncMock(return_value={"status": "failed", "error": "No completed run"}),
        )
        result = await tool(action="query", agent_name="experiment_watch", question="进展如何？")
        assert not result.success and result.answer == "No completed run"


@pytest.mark.asyncio
async def test_time_task_restore_keeps_omitted_schedule_and_original_requirements(tmp_path):
    async with running_manager(tmp_path) as (_, context, _):
        tool = ManageProactiveRuntimeStep(app_context=context)
        first = (
            await tool(
                action="create",
                agent_name="briefing",
                agent_type="time",
                schedule="0 8 * * *",
                task_description="Keep this briefing requirement.",
            )
        ).answer
        task = tmp_path / first["task_path"]
        await tool(action="delete", agent_name="briefing")
        restored = (await tool(action="create", agent_name="briefing", agent_type="time")).answer
        assert restored["restored"] and restored["config"]["agent_type"] == "time"
        assert restored["config"]["schedule"] == "0 8 * * *"
        assert restored["config"]["skills"] == first["config"]["skills"]
        assert task.read_text() == "Keep this briefing requirement."


@pytest.mark.parametrize(
    "values",
    [
        {"agent_name": "task", "agent_type": "time"},
        {"agent_name": "task", "agent_type": "event", "schedule": "0 8 * * *"},
        {"agent_name": "task", "agent_type": "cron", "schedule": "0 8 * * *"},
        {"agent_name": "task", "schedule": "0 8 * * *"},
    ],
)
def test_explicit_trigger_configuration_rejects_missing_or_conflicting_types(values):
    from reme.schema.proactive_agent import ProactiveRuntimeConfig

    with pytest.raises(ValueError):
        ProactiveRuntimeConfig.model_validate(values)
