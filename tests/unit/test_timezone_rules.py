"""Scheduling and calendar dates share timezone rules even when OS data differs."""

# pylint: disable=protected-access,missing-class-docstring,missing-function-docstring

from datetime import datetime, timedelta, timezone
from importlib.resources import files
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import pytest

from reme.components.file_catalog import LocalFileCatalog
from reme.components.job import cron_job
from reme.components.proactive import ChatAgentRuntime
from reme.core.context import ApplicationContext, RuntimeContext
from reme.steps.evolve import _evolve
from reme.steps.evolve.auto_memory import AutoMemoryStep
from reme.steps.proactive.connector import CodexDialogueStep, ScreenCaptureStep
from reme.utils import timezone_utils


@pytest.mark.parametrize("configured_timezone", [None, "America/Winnipeg"])
@pytest.mark.parametrize("consumer", ["dialogue", "resource", "runtime", "daily", "retention", "message"])
def test_midnight_schedule_and_calendar_dates_share_rules(tmp_path, monkeypatch, configured_timezone, consumer):
    monkeypatch.setenv("TZ", "America/Winnipeg")
    rules = ZoneInfo("America/Winnipeg")
    instant = [datetime(2026, 11, 15, 4, 30, tzinfo=timezone.utc)]
    native_rules = timezone(rules.utcoffset(instant[0]) - timedelta(hours=1))

    class FixedDatetime(datetime):
        @classmethod
        def now(cls, tz=None):
            value = cls.fromtimestamp(instant[0].timestamp(), tz or native_rules)
            return value if tz else value.replace(tzinfo=None)

        def astimezone(self, tz=None):
            return super().astimezone(tz or native_rules)

    monkeypatch.setattr(cron_job, "datetime", SimpleNamespace(datetime=FixedDatetime))
    for module in (
        "reme.components.proactive.runtime.base",
        "reme.components.file_catalog.base_file_catalog",
        "reme.steps.proactive.connector.base",
        "reme.steps.proactive.connector.harness.base",
    ):
        monkeypatch.setattr(f"{module}.datetime", FixedDatetime)
    monkeypatch.setattr(_evolve, "datetime", SimpleNamespace(datetime=FixedDatetime))
    monkeypatch.setattr("reme.steps.evolve.auto_memory.datetime", SimpleNamespace(datetime=FixedDatetime))
    context = ApplicationContext(workspace_dir=str(tmp_path), timezone=configured_timezone)
    job = cron_job.CronJob(cron="0 0 * * *", app_context=context)
    instant[0] += timedelta(seconds=job._next_fire_delay())
    midnight = instant[0].astimezone(rules)
    assert midnight.hour == midnight.minute == midnight.second == 0
    assert instant[0].astimezone(native_rules).date() < midnight.date()

    if consumer == "dialogue":
        day = CodexDialogueStep(connector_job="codex_dialogue_cron", app_context=context)._now().date()
    elif consumer == "resource":
        step = ScreenCaptureStep(connector_job="screen_capture_cron", app_context=context)
        step.context = RuntimeContext()
        target, captured_at = step.resource_target()
        day = captured_at.date()
        assert target.parent.name == midnight.date().isoformat()
    elif consumer == "runtime":
        day = ChatAgentRuntime(name="Remy", app_context=context).now().date()
    elif consumer == "daily":
        day = _evolve.now(configured_timezone).date()
    elif consumer == "retention":
        day = LocalFileCatalog(app_context=context)._today()
    else:
        day = datetime.fromisoformat(AutoMemoryStep._message_day(instant[0].isoformat(), configured_timezone)).date()
    assert day == midnight.date()


@pytest.mark.parametrize("localtime_path", ["etc/localtime", "usr/local/etc/localtime"])
def test_host_localtime_paths_preserve_dst_rules(tmp_path, monkeypatch, localtime_path):
    monkeypatch.delenv("TZ", raising=False)
    paths = tuple(tmp_path / path for path in ("etc/localtime", "usr/local/etc/localtime"))
    localtime = tmp_path / localtime_path
    localtime.parent.mkdir(parents=True)
    localtime.write_bytes(files("tzdata.zoneinfo.America").joinpath("New_York").read_bytes())
    monkeypatch.setattr(timezone_utils, "_LOCALTIME_PATHS", paths)
    monkeypatch.setattr(timezone_utils, "sys", SimpleNamespace(platform="linux"))
    fixed = datetime(2026, 3, 8, 6, 30, tzinfo=timezone.utc)

    class FixedDatetime(datetime):
        @classmethod
        def now(cls, tz=None):
            return cls.fromtimestamp(fixed.timestamp(), tz or timezone(timedelta(hours=-5)))

        def astimezone(self, tz=None):
            return super().astimezone(tz or timezone(timedelta(hours=-5)))

    monkeypatch.setattr(timezone_utils, "datetime", FixedDatetime)
    monkeypatch.setattr(cron_job, "datetime", SimpleNamespace(datetime=FixedDatetime))
    context = ApplicationContext(workspace_dir=str(tmp_path), timezone=None)
    job = cron_job.CronJob(cron="0 3 * * *", app_context=context)
    assert job._next_fire_delay() == 1800
