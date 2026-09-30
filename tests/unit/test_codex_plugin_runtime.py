"""Daily scheduling, process ownership, and shutdown without real model or network calls."""

# pylint: disable=missing-function-docstring,redefined-outer-name,protected-access

import asyncio
import importlib
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

PLUGIN = Path(__file__).resolve().parents[2] / "integrations/codex/plugins/reme"


@pytest.fixture
def runtime(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(PLUGIN))
    monkeypatch.setenv("CODEX_HOME", str(tmp_path / "codex"))
    return importlib.import_module("reme_runtime"), importlib.import_module("reme_config")


@pytest.mark.parametrize(
    ("cron", "zone", "now", "expected"),
    [
        ("0 23 * * *", "Asia/Shanghai", "2026-09-30T10:00:00+00:00", "2026-09-30T15:00:00+00:00"),
        ("0 23 * * *", "Asia/Shanghai", "2026-09-30T15:00:00+00:00", "2026-10-01T15:00:00+00:00"),
        ("30 2 * * *", "America/New_York", "2026-03-08T06:59:00+00:00", "2026-03-09T06:30:00+00:00"),
        ("30 1 * * *", "America/New_York", "2026-11-01T04:00:00+00:00", "2026-11-01T05:30:00+00:00"),
        ("30 1 * * *", "America/New_York", "2026-11-01T05:45:00+00:00", "2026-11-02T06:30:00+00:00"),
    ],
)
def test_daily_schedule_handles_dst_once_per_local_day(runtime, cron, zone, now, expected):
    module, _ = runtime
    assert module.next_daily_run(cron, zone, datetime.fromisoformat(now)).isoformat() == expected


async def wait_until(predicate):
    async with asyncio.timeout(3):
        while not predicate():
            await asyncio.sleep(0.01)


@pytest.mark.asyncio
async def test_single_scheduler_runs_slot_once_and_fails_over(runtime, monkeypatch):
    module, config = runtime
    config.update_settings({"dreamHint": "retain decisions"})
    due = datetime.now(timezone.utc) + timedelta(milliseconds=50)
    next_times = iter([due, due + timedelta(days=1), due + timedelta(days=1)])
    monkeypatch.setattr(module, "next_daily_run", lambda *_: next(next_times))
    call = AsyncMock(return_value={"answer": "done"})
    monkeypatch.setattr(module, "call_async", call)
    first, second = module.ReMeRuntime(), module.ReMeRuntime()
    await first.start()
    await second.start()
    try:
        await wait_until(lambda: call.call_count == 1)
        assert call.call_args.args[1] == "auto_dream"
        assert call.call_args.args[2]["hint"] == "retain decisions"
        assert call.call_args.args[3] == 3600
        receipt = module.endpoint_dir(config.load_config()) / "dream_receipt.json"
        assert json.loads(receipt.read_text())["scheduled_at"] == due.isoformat()
        await module.execute_dream(config.load_config(), origin="scheduled", scheduled_at=due.isoformat())
        assert call.call_count == 1
        await first.close()
        state = config.data_dir() / "scheduler.json"
        await wait_until(lambda: json.loads(state.read_text()).get("phase") == "running")
        assert call.call_count == 1
    finally:
        await first.close()
        await second.close()
    assert first._task is second._task is None


@pytest.mark.asyncio
async def test_live_settings_reschedule_and_disable_without_starting_a_service(runtime):
    module, config = runtime
    config.update_settings({"autoDreamEnabled": False})
    instance = module.ReMeRuntime()
    await instance.start()
    state = config.data_dir() / "scheduler.json"
    try:
        await wait_until(state.exists)
        assert json.loads(state.read_text())["next_run_at"] is None
        config.update_settings({"autoDreamEnabled": True, "dreamCron": "45 6 * * *", "timezone": "UTC"})
        instance.settings_changed()
        await wait_until(lambda: json.loads(state.read_text()).get("cron") == "45 6 * * *")
        assert json.loads(state.read_text())["next_run_at"].endswith("06:45:00+00:00")
        config.update_settings({"autoDreamEnabled": False})
        instance.settings_changed()
        await wait_until(lambda: json.loads(state.read_text()).get("enabled") is False)
        assert json.loads(state.read_text())["next_run_at"] is None
    finally:
        await instance.close()
    assert json.loads(state.read_text())["phase"] == "stopped"


@pytest.mark.asyncio
async def test_shutdown_cancels_scheduled_call_and_releases_locks(runtime, monkeypatch):
    module, config = runtime
    config.update_settings({"shutdownTimeoutMs": 100})
    started = asyncio.Event()

    async def long_call(*_args):
        started.set()
        await asyncio.Future()

    monkeypatch.setattr(module, "call_async", long_call)
    monkeypatch.setattr(module, "next_daily_run", lambda *_: datetime.now(timezone.utc))
    instance = module.ReMeRuntime()
    await instance.start()
    await asyncio.wait_for(started.wait(), 2)
    await instance.close()
    state = module.endpoint_dir(config.load_config())
    assert json.loads((state / "dream.json").read_text())["status"] == "cancelled"
    with config.lock(state / "dream.lock") as acquired:
        assert acquired
    with config.lock(config.data_dir() / "scheduler.lock") as acquired:
        assert acquired


@pytest.mark.parametrize("cron", ["* * * * *", "60 0 * * *", "0 24 * * *", "0 1 * * 1", "0 1 * * * extra"])
def test_settings_reject_unsupported_schedules_without_overwriting(runtime, cron):
    _, config = runtime
    config.update_settings({"dreamCron": "15 8 * * *"})
    path = config.data_dir() / "config.json"
    before = path.read_bytes()
    with pytest.raises(ValueError):
        config.update_settings({"dreamCron": cron})
    assert path.read_bytes() == before
