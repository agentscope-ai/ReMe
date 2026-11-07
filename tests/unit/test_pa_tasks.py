"""Cron-slot deduplication, failure isolation and interrupted execution records."""

import asyncio
import json
from datetime import timedelta
from unittest.mock import AsyncMock

import pytest

from reme.schema import Response


def task(env, identifier="one", query="query"):
    return {"id": identifier, "cron": "*/5 * * * *", "query": query, "enabled": True,
            "created_at": (env.now[0] - timedelta(hours=1)).isoformat()}


@pytest.mark.asyncio
async def test_concurrent_runs_reserve_each_slot_only_once(pa_workspace):
    env = pa_workspace
    env.runtime._write_json(env.runtime.tasks_path, [task(env)])
    runner = AsyncMock(return_value=Response(answer="result"))
    results = await asyncio.gather(env.runtime.run_due_tasks(runner), env.runtime.run_due_tasks(runner))
    runner.assert_awaited_once()
    assert sorted(map(len, results)) == [0, 1]
    rows = [json.loads(line) for line in env.runtime.task_runs_path.read_text().splitlines()]
    assert [row["status"] for row in rows] == ["running", "succeeded"]
    assert rows[0]["run_id"] == rows[1]["run_id"] and rows[0]["scheduled_at"] == rows[1]["scheduled_at"]


@pytest.mark.asyncio
async def test_failures_do_not_prevent_other_tasks(pa_workspace):
    env = pa_workspace
    env.runtime._write_json(env.runtime.tasks_path, [task(env, "one"), task(env, "two")])
    runner = AsyncMock(side_effect=[RuntimeError("task failed"), Response(answer="success")])
    results = await env.runtime.run_due_tasks(runner)
    assert [row["status"] for row in results] == ["failed", "succeeded"]


@pytest.mark.asyncio
async def test_cancelled_task_records_interruption_without_replaying_same_slot(pa_workspace):
    env = pa_workspace
    env.runtime._write_json(env.runtime.tasks_path, [task(env)])
    runner = AsyncMock(side_effect=asyncio.CancelledError())
    with pytest.raises(asyncio.CancelledError):
        await env.runtime.run_due_tasks(runner)
    rows = [json.loads(line) for line in env.runtime.task_runs_path.read_text().splitlines()]
    assert rows[-1]["status"] == "interrupted" and rows[-1]["finished_at"]
    assert await env.runtime.run_due_tasks(runner) == []
    runner.assert_awaited_once()


@pytest.mark.asyncio
async def test_registration_time_and_disabled_tasks_prevent_early_execution(pa_workspace):
    env = pa_workspace
    new = task(env)
    new["created_at"] = env.now[0].isoformat()
    disabled = {**task(env, "two"), "enabled": False}
    env.runtime._write_json(env.runtime.tasks_path, [new, disabled])
    runner = AsyncMock()
    assert await env.runtime.run_due_tasks(runner) == []
    runner.assert_not_awaited()
