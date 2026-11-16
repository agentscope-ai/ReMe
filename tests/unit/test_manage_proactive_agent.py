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
from reme.schema import Response
from reme.schema.proactive_agent import ManageProactiveAgentRequest
from reme.steps.proactive.tools.manage_proactive_agent import ManageProactiveAgentStep
from .test_agent_runtime import running_runtime


@pytest.mark.parametrize(
    "arguments",
    [
        {"action": "list", "name": "task"},
        {"action": "list", "name": None},
        {"action": "create", "name": "task", "trigger_type": "event"},
        {"action": "create", "name": "task", "instructions": " ", "trigger_type": "event"},
        {"action": "create", "config": {"name": "task", "description": "Do work", "trigger_type": "event"}},
        {"action": "status", "agent_id": "task"},
        {"action": "status"},
        {"action": "status", "name": "task", "enabled": True},
        {"action": "update", "name": "task"},
        {"action": "update", "name": "task", "instructions": "New work"},
        {"action": "update", "name": "task", "temporary": True},
        {"action": "update", "name": "task", "max_batch_events": 10},
        {"action": "update", "name": "task", "min_trigger_interval_seconds": True},
        {"action": "delete", "name": "task", "cron": None},
        {"action": "query", "name": "task", "query": "Progress?"},
        {"action": "query", "name": "task", "question": " "},
    ],
)
def test_action_arguments_reject_legacy_and_irrelevant_fields(arguments):
    with pytest.raises(ValueError):
        ManageProactiveAgentRequest.model_validate(arguments)


@pytest.mark.asyncio
async def test_all_actions_expose_only_their_results_and_keep_memory_files(tmp_path, monkeypatch):
    async with running_runtime(tmp_path) as (runtime, context, _):
        now = datetime.fromisoformat("2026-11-17T07:00:00+08:00")
        monkeypatch.setattr(runtime, "now", lambda: now)
        tool = ManageProactiveAgentStep(app_context=context)
        arguments = {
            "action": "create",
            "name": "科技早报",
            "instructions": "每天08:00整理科技新闻。\n保留来源。\n",
            "trigger_type": "cron",
            "cron": "0 8 * * *",
        }
        created = (await tool(**arguments)).answer
        assert created == {
            "name": "科技早报",
            "memory_path": "reme_agents/科技早报/MEMORY.md",
            "next_trigger_at": "2026-11-17T08:00:00+08:00",
        }
        memory = tmp_path / created["memory_path"]
        assert memory.read_text() == arguments["instructions"]
        assert not runtime.cron  # Inspecting the next trigger must not schedule or run work.
        assert (await tool(action="list")).answer == {
            "items": [{"name": "科技早报", "trigger_type": "cron", "enabled": True, "cron": "0 8 * * *"}],
        }
        status = (await tool(action="status", name="科技早报")).answer
        assert status == {
            **created,
            "trigger_type": "cron",
            "enabled": True,
            "temporary": False,
            "cron": "0 8 * * *",
            "running": False,
            "pending_events": 0,
            "in_flight_events": 0,
            "last_completed_at": None,
        }
        updated = (await tool(action="update", name="科技早报", cron="0 9 * * *")).answer
        assert updated == {
            "name": "科技早报",
            "updated": {"cron": "0 9 * * *"},
            "next_trigger_at": "2026-11-17T09:00:00+08:00",
        }
        paused = (await tool(action="update", name="科技早报", enabled=False)).answer
        assert paused == {"name": "科技早报", "updated": {"enabled": False}}
        assert memory.read_text() == arguments["instructions"]

        query = AsyncMock(return_value=Response(answer={"agent_id": "科技早报", "answer": "今天的早报已经完成。"}))
        monkeypatch.setattr(tool, "run_job", query)
        assert (await tool(action="query", name="科技早报", question="进展如何？")).answer == "今天的早报已经完成。"
        query.assert_awaited_once_with(
            "proactive_query",
            agent_id="科技早报",
            query="进展如何？",
            _allowed_paths=None,
        )
        deleted = (await tool(action="delete", name="科技早报")).answer
        assert deleted == "Deleted task '科技早报'; history files retained."
        assert (await tool(action="list")).answer == {"items": []}
        assert memory.read_text() == arguments["instructions"]
        restored = (await tool(**{**arguments, "instructions": "Changed request"})).answer
        assert restored == {**created, "restored": True}
        assert memory.read_text() == arguments["instructions"]


@pytest.mark.asyncio
async def test_trigger_changes_clear_old_settings_without_ignoring_explicit_conflicts(tmp_path):
    async with running_runtime(tmp_path) as (runtime, context, _):
        tool = ManageProactiveAgentStep(app_context=context)
        await tool(
            action="create",
            name="实验监控",
            instructions="跟进实验",
            trigger_type="event",
            min_trigger_interval_seconds=30,
        )
        await tool(action="update", name="实验监控", trigger_type="cron", cron="0 * * * *")
        assert (await runtime.definition("实验监控")).min_trigger_interval_seconds == 0
        await tool(action="update", name="实验监控", trigger_type="event")
        assert (await runtime.definition("实验监控")).cron is None
        with pytest.raises(ValueError, match="trigger interval"):
            await tool(
                action="update",
                name="实验监控",
                trigger_type="cron",
                cron="0 * * * *",
                min_trigger_interval_seconds=30,
            )
        assert (await runtime.definition("实验监控")).trigger_type == "event"
        await tool(action="update", name="实验监控", notify_window={"start": "09:00", "end": "18:00"})
        assert (await runtime.definition("实验监控")).notify_window.start == "09:00"
        cleared = (await tool(action="update", name="实验监控", notify_window=None)).answer
        assert cleared == {"name": "实验监控", "updated": {"notify_window": None}}
        assert (await runtime.definition("实验监控")).notify_window is None


@pytest.mark.asyncio
async def test_query_errors_remain_failed_tool_results(tmp_path, monkeypatch):
    async with running_runtime(tmp_path) as (_, context, _):
        tool = ManageProactiveAgentStep(app_context=context)
        monkeypatch.setattr(tool, "run_job", AsyncMock(return_value=Response(success=False, answer="No completed run")))
        result = await tool(action="query", name="实验监控", question="进展如何？")
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
