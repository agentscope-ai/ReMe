"""Named management actions, small model results and executable skill examples."""

# pylint: disable=missing-function-docstring,protected-access

from datetime import datetime
import json
from pathlib import Path
import re
from unittest.mock import AsyncMock

import jsonschema
import pytest

from reme.config import resolve_app_config
from reme.schema.proactive_agent import ManageProactiveAgentRequest
from reme.steps.proactive.tools.manage_proactive_agent import ManageProactiveAgentStep
from .test_agent_runtime import running_runtime


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
        {"action": "create", "agent_name": "task"},
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
        ManageProactiveAgentRequest.model_validate(arguments)


@pytest.mark.asyncio
async def test_all_actions_expose_only_their_results_and_keep_task_files(tmp_path, monkeypatch):
    async with running_runtime(tmp_path) as (runtime, context, _):
        now = datetime.fromisoformat("2026-11-17T07:00:00+08:00")
        monkeypatch.setattr(runtime, "now", lambda: now)
        tool = ManageProactiveAgentStep(app_context=context)
        arguments = {
            "action": "create",
            "agent_name": "tech_news",
            "task_description": "每天08:00整理科技新闻。\n保留来源。\n",
            "cron": "0 8 * * *",
        }
        created = (await tool(**arguments)).answer
        assert created == (await runtime.read_agent("tech_news"))["agent"]
        assert created["agent_type"] == "cron" and created["cron"] == "0 8 * * *"
        assert created["next_trigger_at"] == "2026-11-17T08:00:00+08:00"
        task_file = tmp_path / created["task_path"]
        assert task_file.read_text() == arguments["task_description"]
        assert not runtime.cron  # Inspecting the next trigger must not schedule or run work.
        assert (await tool(action="list")).answer == {"items": [created], "total": 1}
        assert (await tool(action="status", agent_name="tech_news")).answer == created
        updated = (await tool(action="update", agent_name="tech_news", cron="0 9 * * *")).answer
        assert updated["cron"] == "0 9 * * *" and updated["next_trigger_at"] == "2026-11-17T09:00:00+08:00"
        paused = (await tool(action="update", agent_name="tech_news", enabled=False)).answer
        assert not paused["enabled"] and paused["cron"] == "0 9 * * *"
        assert task_file.read_text() == arguments["task_description"]

        query = AsyncMock(return_value={"status": "completed", "result": "今天的早报已经完成。"})
        monkeypatch.setattr(tool, "query_reply", query)
        assert (
            await tool(action="query", agent_name="tech_news", question="进展如何？")
        ).answer == "今天的早报已经完成。"
        query.assert_awaited_once_with("tech_news", "进展如何？")
        deleted = (await tool(action="delete", agent_name="tech_news")).answer
        assert deleted == "Deleted task 'tech_news'; history files retained."
        assert (await tool(action="list")).answer == {"items": [], "total": 0}
        assert task_file.read_text() == arguments["task_description"]
        restored = (await tool(**{**arguments, "task_description": "Changed request"})).answer
        assert restored == (await runtime.read_agent("tech_news"))["agent"]
        assert task_file.read_text() == arguments["task_description"]


@pytest.mark.asyncio
async def test_trigger_changes_clear_old_settings_without_ignoring_explicit_conflicts(tmp_path):
    async with running_runtime(tmp_path) as (runtime, context, _):
        tool = ManageProactiveAgentStep(app_context=context)
        await tool(
            action="create",
            agent_name="experiment_watch",
            task_description="跟进实验",
            min_trigger_interval_seconds=30,
        )
        await tool(action="update", agent_name="experiment_watch", cron="0 * * * *", min_trigger_interval_seconds=0)
        assert (await runtime.definition("experiment_watch")).min_trigger_interval_seconds == 0
        await tool(action="update", agent_name="experiment_watch", cron=None)
        assert (await runtime.definition("experiment_watch")).cron is None
        with pytest.raises(ValueError, match="trigger interval"):
            await tool(
                action="update",
                agent_name="experiment_watch",
                cron="0 * * * *",
                min_trigger_interval_seconds=30,
            )
        assert (await runtime.definition("experiment_watch")).agent_type == "proactive"
        await tool(action="update", agent_name="experiment_watch", notify_window={"start": "09:00", "end": "18:00"})
        assert (await runtime.definition("experiment_watch")).notify_window.start == "09:00"
        cleared = (await tool(action="update", agent_name="experiment_watch", notify_window=None)).answer
        assert cleared["notify_window"] is None
        assert (await runtime.definition("experiment_watch")).notify_window is None


@pytest.mark.asyncio
async def test_query_errors_remain_failed_tool_results(tmp_path, monkeypatch):
    async with running_runtime(tmp_path) as (_, context, _):
        tool = ManageProactiveAgentStep(app_context=context)
        monkeypatch.setattr(
            tool,
            "query_reply",
            AsyncMock(return_value={"status": "failed", "error": "No completed run"}),
        )
        result = await tool(action="query", agent_name="experiment_watch", question="进展如何？")
        assert not result.success and result.answer == "No completed run"


def test_skill_action_examples_match_the_live_request_contract():
    path = Path(__file__).resolve().parents[2] / "skills/manage-proactive-agent/SKILL.md"
    schema = resolve_app_config(config="proactive", log_config=False)["jobs"]["manage_proactive_agent"]["parameters"]
    actions = set()
    for block in re.findall(r"```json\n(.*?)\n```", path.read_text(), re.DOTALL):
        value = json.loads(block)
        if isinstance(value, dict) and "action" in value:
            jsonschema.validate(value, schema)
            request = ManageProactiveAgentRequest.model_validate(value)
            actions.add(request.action)
    assert actions == {"list", "create", "status", "update", "query", "delete"}
