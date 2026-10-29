"""Cron waits use elapsed seconds across timezone offset changes."""

# pylint: disable=protected-access,missing-function-docstring,missing-class-docstring

import datetime
from types import SimpleNamespace

import pytest

from reme.components.job import cron_job as cron_job_module
from reme.components.job.cron_job import CronJob


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
        (None, "2026-01-15T01:30:00+00:00", 5400),
    ],
)
def test_cron_delay_uses_elapsed_seconds(monkeypatch, timezone, instant, expected_delay):
    fixed = datetime.datetime.fromisoformat(instant)

    class FixedDatetime(datetime.datetime):
        @classmethod
        def now(cls, tz=None):
            return fixed.astimezone(tz) if tz else fixed.replace(tzinfo=None)

    monkeypatch.setattr(cron_job_module, "datetime", SimpleNamespace(datetime=FixedDatetime))
    job = CronJob(cron="0 3 * * *")
    job.app_context = SimpleNamespace(app_config=SimpleNamespace(timezone=timezone))

    assert job._next_fire_delay() == expected_delay
