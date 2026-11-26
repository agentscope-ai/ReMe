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
            "agent_type": "cron",
            "cron": "0 8 * * *",
        },
        {"action": "create", "agent_name": "task", "task_description": "Observe", "temporary": True},
        {"action": "create", "agent_name": "task", "task_description": " "},
        {
            "action": "create",
            "config": {"agent_name": "task", "task_description": "Do work"},
        },
        {"action": "status", "agent_id": "task"},
        {"action": "status", "name": "task"},
        {"action": "create", "agent_name": "task", "task_description": "Observe", "agent_type": "event"},
        {"action": "create", "agent_name": "task", "task_description": "Observe", "agent_type": "chat"},
        {"action": "create", "agent_name": "task", "instructions": "Observe"},
        {"action": "create", "agent_name": "task", "description": "Observe"},
        {"action": "create", "agent_name": "task", "task_description": "Observe", "trigger_type": "event"},
        {"action": "status"},
        {"action": "status", "agent_name": "task", "enabled": True},
        {"action": "update", "agent_name": "task"},
        {"action": "update", "agent_name": "task", "task_description": "New work"},
        {"action": "update", "agent_name": "task", "temporary": True},
        {"action": "update", "agent_name": "task", "max_batch_events": 10},
        {"action": "update", "agent_name": "task", "min_trigger_interval_seconds": True},
        {"action": "delete", "agent_name": "task", "cron": None},
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
            "action": "create",
            "agent_name": "tech_news",
            "task_description": "每天08:00整理科技新闻。\n保留来源。\n",
            "cron": "0 8 * * *",
        }
        created = (await tool(**arguments)).answer
        assert created.pop("restored") is False
        assert created["config"]["run_job"] == "run_memory_agent"
        assert created["config"]["cron"] == "0 8 * * *"
        assert created["next_trigger_at"] == "2026-11-17T08:00:00+08:00"
        task_file = tmp_path / created["task_path"]
        assert task_file.read_text() == arguments["task_description"]
        assert runtime.runtimes["tech_news"].schedule is not None
        assert (await tool(action="list")).answer == {"items": [created], "total": 1}
        assert (await tool(action="status", agent_name="tech_news")).answer == created
        updated = (await tool(action="update", agent_name="tech_news", cron="0 9 * * *")).answer
        assert updated["config"]["cron"] == "0 9 * * *" and updated["next_trigger_at"] == "2026-11-17T09:00:00+08:00"
        paused = (await tool(action="update", agent_name="tech_news", enabled=False)).answer
        assert not paused["config"]["enabled"] and paused["config"]["cron"] == "0 9 * * *"
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
@pytest.mark.parametrize("cron, other_cron, agent_type", [(None, "0 * * * *", "event"), ("0 * * * *", None, "time")])
async def test_trigger_type_is_fixed_after_creation_and_on_restore(tmp_path, cron, other_cron, agent_type):
    async with running_manager(tmp_path) as (runtime, context, _):
        tool = ManageProactiveRuntimeStep(app_context=context)
        name = "experiment_watch"
        task = {"agent_name": name, "task_description": "跟进实验", "cron": cron}
        created = (await tool(action="create", **task)).answer
        assert ("time" if created["config"]["cron"] else "event") == agent_type
        path = runtime.registry.path(name) / "config.json"
        before = path.read_bytes()
        with pytest.raises(ValueError, match="trigger type cannot change"):
            await tool(action="update", agent_name=name, cron=other_cron, enabled=False)
        assert path.read_bytes() == before

        changes = {"min_trigger_interval_seconds": 60}
        if cron is not None:
            changes["cron"] = "0 8 * * *"
        updated = (await tool(action="update", agent_name=name, **changes)).answer
        assert updated["config"]["cron"] == changes.get("cron")
        assert updated["config"]["min_trigger_interval_seconds"] == 60
        await tool(action="update", agent_name=name, notify_window={"start": "09:00", "end": "18:00"})
        assert (await runtime.definition(name)).notify_window.start == "09:00"
        cleared = (await tool(action="update", agent_name=name, notify_window=None)).answer
        assert cleared["config"]["notify_window"] is None

        await tool(action="delete", agent_name=name)
        deleted = path.read_bytes()
        with pytest.raises(ValueError, match="trigger type cannot change"):
            await tool(action="create", **{**task, "cron": other_cron})
        assert path.read_bytes() == deleted
        restored = (await tool(action="create", **task)).answer
        assert restored["config"]["cron"] == cron


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
