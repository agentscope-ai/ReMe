"""Cron waits use elapsed seconds across timezone offset changes."""

# pylint: disable=protected-access,missing-function-docstring,missing-class-docstring

import datetime
from importlib.resources import files
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import pytest

from reme.components.job import cron_job as cron_job_module
from reme.components.job.cron_job import CronJob
from reme.utils import timezone_utils


@pytest.mark.parametrize(
    "timezone,instant,expected_delay",
    [
        ("America/New_York", "2026-03-08T06:30:00+00:00", 1800),
        ("America/New_York", "2026-11-01T04:30:00+00:00", 12600),
        ("Europe/London", "2026-03-29T00:30:00+00:00", 5400),
        ("Europe/London", "2026-10-25T00:30:00+00:00", 9000),
        ("Australia/Lord_Howe", "2026-10-03T15:00:00+00:00", 3600),
        ("Australia/Lord_Howe", "2026-04-04T13:30:00+00:00", 10800),
        ("America/New_York", "2026-01-15T06:30:00+00:00", 5400),
        ("Asia/Shanghai", "2026-03-07T17:30:00+00:00", 5400),
        (None, "2026-01-15T06:30:00+00:00", 5400),
        (None, "2026-03-08T06:30:00+00:00", 1800),
        (None, "2026-11-01T04:30:00+00:00", 12600),
    ],
)
def test_cron_delay_uses_elapsed_seconds(monkeypatch, timezone, instant, expected_delay):
    fixed = datetime.datetime.fromisoformat(instant)
    monkeypatch.setenv("TZ", "America/New_York")

    class FixedDatetime(datetime.datetime):
        @classmethod
        def now(cls, tz=None):
            return fixed.astimezone(tz) if tz else fixed.replace(tzinfo=None)

    monkeypatch.setattr(cron_job_module, "datetime", SimpleNamespace(datetime=FixedDatetime))
    job = CronJob(cron="0 3 * * *")
    job.app_context = SimpleNamespace(app_config=SimpleNamespace(timezone=timezone))

    assert job._next_fire_delay() == expected_delay


@pytest.mark.parametrize("with_app_context", [True, False], ids=["local-timezone", "standalone"])
@pytest.mark.parametrize(
    "zone_name,instant,expression,delay",
    [
        ("America/New_York", "2026-11-01T05:59:00+00:00", "* * * * *", 60),
        ("America/New_York", "2026-11-01T06:30:00+00:00", "* * * * *", 60),
        ("Australia/Lord_Howe", "2026-04-04T14:45:00+00:00", "*/20 1 * * *", 1500),
    ],
)
def test_cron_local_time_during_dst_fallback(monkeypatch, with_app_context, zone_name, instant, expression, delay):
    fixed = datetime.datetime.fromisoformat(instant)

    class FixedDatetime(datetime.datetime):
        @classmethod
        def now(cls, tz=None):
            return datetime.datetime.fromtimestamp(fixed.timestamp(), tz)

    monkeypatch.setenv("TZ", zone_name)
    monkeypatch.setattr(cron_job_module, "datetime", SimpleNamespace(datetime=FixedDatetime))
    job = CronJob(cron=expression)
    if with_app_context:
        job.app_context = SimpleNamespace(app_config=SimpleNamespace(timezone=None))

    assert job._next_fire_delay() == delay


def test_date_limited_cron_waits_for_second_fold(monkeypatch):
    fixed = datetime.datetime.fromisoformat("2026-11-01T05:30:00+00:00")

    class FixedDatetime(datetime.datetime):
        @classmethod
        def now(cls, tz=None):
            return fixed.astimezone(tz)

    monkeypatch.setattr(cron_job_module, "datetime", SimpleNamespace(datetime=FixedDatetime))
    job = CronJob(cron="30 1 1 11 * 0 2026")
    job.app_context = SimpleNamespace(app_config=SimpleNamespace(timezone="America/New_York"))
    assert job._next_fire_delay() == 3600


def test_cron_local_time_uses_host_rules_without_an_iana_name(tmp_path, monkeypatch):
    monkeypatch.delenv("TZ", raising=False)
    localtime = tmp_path / "localtime"
    localtime.write_bytes(files("tzdata.zoneinfo.America").joinpath("New_York").read_bytes())
    with localtime.open("rb") as source:
        rules = ZoneInfo.from_file(source, key="local")
    monkeypatch.setattr(timezone_utils, "get_localzone", lambda: rules)
    fixed = datetime.datetime(2026, 11, 25, 8, tzinfo=rules)

    class FixedDatetime(datetime.datetime):
        @classmethod
        def now(cls, tz=None):
            return fixed.astimezone(tz)

    monkeypatch.setattr(cron_job_module, "datetime", SimpleNamespace(datetime=FixedDatetime))
    job = CronJob(cron="0 9 * * *")
    job.app_context = SimpleNamespace(app_config=SimpleNamespace(timezone=None))
    assert job._next_fire_delay() == 3600
