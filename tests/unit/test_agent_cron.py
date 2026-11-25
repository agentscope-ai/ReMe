"""Reject unusable schedules before writes and isolate retained, exhausted schedules."""

# pylint: disable=missing-function-docstring,protected-access

import asyncio
from datetime import datetime
import json
from unittest.mock import AsyncMock, Mock
from zoneinfo import ZoneInfo

import pytest

from reme.schema import Event, Response
from .test_proactive_runtime import running_runtime

INVALID_SCHEDULES = ("0 0 31 2 *", "0 0 1 1 * 0 2025")
NOW = datetime(2026, 11, 19, tzinfo=ZoneInfo("Asia/Shanghai"))


@pytest.mark.parametrize(
    "instant,cron,next_trigger",
    [
        ("2026-03-08T01:30:00-05:00", "30 2 * * *", "2026-03-08T03:00:00-04:00"),
        ("2026-03-08T01:45:00-05:00", "30 * * * *", "2026-03-08T03:30:00-04:00"),
        ("2026-11-01T01:45:00-04:00", "30 1 * * *", "2026-11-01T01:30:00-05:00"),
        ("2026-11-01T01:29:00-05:00", "30 1 * * *", "2026-11-01T01:30:00-05:00"),
    ],
)
@pytest.mark.asyncio
async def test_proactive_cron_uses_local_timezone_rules(tmp_path, monkeypatch, instant, cron, next_trigger):
    monkeypatch.setenv("TZ", "America/New_York")
    async with running_runtime(tmp_path) as (runtime, context, _):
        context.app_config.timezone = None
        clock = [datetime.fromisoformat(instant)]
        monkeypatch.setattr(runtime, "now", lambda: clock[0])
        await runtime.create({"agent_name": "timer", "task_description": "Report", "cron": cron})
        assert (await runtime.status("timer"))["next_trigger_at"] == next_trigger

        agent = await runtime.definition("timer")
        await runtime._schedule_cron(agent)
        clock[0] = datetime.fromisoformat(next_trigger)
        await runtime._schedule_cron(agent)
        assert (await runtime.inbox.snapshot("timer"))["pending_count"] == 1


@pytest.mark.parametrize("cron", INVALID_SCHEDULES)
@pytest.mark.asyncio
async def test_unusable_cron_is_rejected_before_creating_files(tmp_path, monkeypatch, cron):
    async with running_runtime(tmp_path) as (runtime, _, _):
        monkeypatch.setattr(runtime, "now", lambda: NOW)
        with pytest.raises(ValueError, match="next date"):
            await runtime.create({"agent_name": "timer", "task_description": "Report", "cron": cron})
        assert not (runtime.agents.directory / "timer").exists()
        assert (await runtime.list_agents())["total"] == 0


@pytest.mark.parametrize("cron", INVALID_SCHEDULES)
@pytest.mark.asyncio
async def test_unusable_cron_update_preserves_existing_files(tmp_path, monkeypatch, cron):
    async with running_runtime(tmp_path) as (runtime, _, _):
        monkeypatch.setattr(runtime, "now", lambda: NOW)
        await runtime.create({"agent_name": "timer", "task_description": "Report", "cron": "* * * * *"})
        directory = runtime.agents.path("timer")
        before = {path: (path.read_bytes(), path.stat().st_mtime_ns) for path in directory.iterdir() if path.is_file()}
        with pytest.raises(ValueError, match="next date"):
            await runtime.update("timer", {"cron": cron})
        assert {path: (path.read_bytes(), path.stat().st_mtime_ns) for path in before} == before
        assert (await runtime.status("timer"))["config"]["cron"] == "* * * * *"


@pytest.mark.parametrize("cron", INVALID_SCHEDULES)
@pytest.mark.asyncio
async def test_retained_unusable_cron_survives_restart_without_blocking_other_tasks(tmp_path, monkeypatch, cron):
    async with running_runtime(tmp_path) as (runtime, _, _):
        monkeypatch.setattr(runtime, "now", lambda: NOW)
        await runtime.create({"agent_name": "aaa_bad", "task_description": "Report", "cron": "* * * * *"})
        path = runtime.agents.path("aaa_bad") / "config.json"
        config = json.loads(path.read_text())
        runtime.files.write_json(path, {**config, "cron": cron})
        saved = path.read_bytes()

    async with running_runtime(tmp_path) as (runtime, _, _):
        monkeypatch.setattr(runtime, "now", lambda: NOW)
        await runtime.create({"agent_name": "healthy", "task_description": "Report"})
        await runtime.append_events("healthy", [Event(event_ts=NOW, source="test", description="Finished")])
        execute = AsyncMock(return_value=(None, "session-test"))

        async def run_job(name, **kwargs):
            assert name == "run_proactive_agent" and kwargs["agent_name"] == "healthy"
            return Response(answer=await runtime.run("healthy", execute))

        await runtime.tick(run_job)
        await asyncio.wait_for(asyncio.gather(*runtime.workers.values()), 2)
        execute.assert_awaited_once()
        assert (await runtime.status("healthy"))["pending_events"] == 0
        status = await runtime.status("aaa_bad")
        assert status["next_trigger_at"] is None and status["error"]
        views = {item["agent_name"]: item for item in (await runtime.list_views())["items"]}
        assert set(views) == {"Remy", "aaa_bad", "healthy"}
        assert views["aaa_bad"]["status"] == "error"
        assert path.read_bytes() == saved

        updated = await runtime.update("aaa_bad", {"min_trigger_interval_seconds": 30})
        assert updated["config"]["min_trigger_interval_seconds"] == 30
        assert updated["next_trigger_at"] is None

        await runtime.update("aaa_bad", {"enabled": False})
        paused = path.read_bytes()
        with pytest.raises(ValueError, match="next date"):
            await runtime.update("aaa_bad", {"enabled": True})
        assert path.read_bytes() == paused
        repaired = await runtime.update("aaa_bad", {"enabled": True, "cron": "* * * * *"})
        assert repaired["next_trigger_at"] is not None and repaired["error"] is None


@pytest.mark.asyncio
async def test_final_cached_cron_trigger_is_delivered_once_after_schedule_expires(tmp_path, monkeypatch):
    async with running_runtime(tmp_path) as (runtime, _, _):
        clock = [datetime(2026, 12, 31, 23, 59, tzinfo=NOW.tzinfo)]
        monkeypatch.setattr(runtime, "now", lambda: clock[0])
        await runtime.create(
            {"agent_name": "timer", "task_description": "Report", "cron": "0 0 1 1 * 0 2027"},
        )
        agent = await runtime.definition("timer")
        await runtime._schedule_cron(agent)
        clock[0] = runtime.execution("timer").schedule[1]
        await runtime._schedule_cron(agent)
        await runtime._schedule_cron(agent)
        assert (await runtime.inbox.snapshot("timer"))["pending_count"] == 1
        assert (await runtime.status("timer"))["next_trigger_at"] is None
        execute = AsyncMock(return_value=(None, "session-test"))

        async def run_job(_name, **_kwargs):
            return Response(answer=await runtime.run("timer", execute))

        await runtime.tick(run_job)
        await asyncio.wait_for(asyncio.gather(*runtime.workers.values()), 2)
        await runtime.tick(run_job)
        execute.assert_awaited_once()
        assert (await runtime.inbox.snapshot("timer"))["pending_count"] == 0
        assert not runtime.workers
        assert runtime.execution("timer").schedule[1] is None


@pytest.mark.asyncio
async def test_schedule_calculation_runs_only_when_needed(tmp_path, monkeypatch):
    async with running_runtime(tmp_path) as (runtime, _, _):
        clock = [NOW]
        monkeypatch.setattr(runtime, "now", lambda: clock[0])
        await runtime.create({"agent_name": "timer", "task_description": "Report", "cron": "* * * * *"})
        agent = await runtime.definition("timer")
        calculate = Mock(wraps=runtime._next_cron_time)
        monkeypatch.setattr(runtime, "_next_cron_time", calculate)

        await runtime._schedule_cron(agent)
        next_trigger = runtime.execution("timer").schedule[1]
        await runtime._schedule_cron(agent)
        await runtime.update("timer", {"min_trigger_interval_seconds": 30})
        assert (await runtime.status("timer"))["next_trigger_at"] == next_trigger.isoformat()
        calculate.assert_called_once()

        clock[0] = next_trigger
        await runtime._schedule_cron(await runtime.definition("timer"))
        assert calculate.call_count == 2
        assert (await runtime.inbox.snapshot("timer"))["pending_count"] == 1
