"""Next-fire calculations shared by the Cron and proactive runtimes."""

from datetime import datetime, timedelta
from functools import cache
from importlib.metadata import version


@cache
def _legacy_croniter() -> bool:
    # Hermes pins 6.0, which adjusts ZoneInfo timestamps incorrectly across DST.
    return version("croniter").split(".")[:2] == ["6", "0"]


def _next_legacy_wall_time(expression: str, start: datetime, now: datetime) -> datetime:
    from croniter import croniter

    iterator = croniter(expression, start)
    wall_time = iterator.get_next(datetime)
    after = now.timestamp()
    while True:
        first = wall_time.replace(tzinfo=now.tzinfo, fold=0)
        second = first.replace(fold=1)
        if first.utcoffset() >= second.utcoffset():
            return first if first.timestamp() > after else second
        # The clock skips this wall time. An unrestricted hour can look for the next matching hour.
        if "*" in iterator.expand(expression)[0][1]:
            wall_time = iterator.get_next(datetime)
        else:
            wall_time += timedelta(minutes=1)


def _next_legacy_time(expression: str, now: datetime) -> datetime:
    wall_now = now.replace(tzinfo=None)
    next_fire = _next_legacy_wall_time(expression, wall_now, now)
    offset_change = next_fire.utcoffset() - now.utcoffset()
    if offset_change < timedelta():
        # The clock may repeat an earlier wall time before the first candidate.
        repeated = _next_legacy_wall_time(expression, wall_now + offset_change, now)
        if now.timestamp() < repeated.timestamp() < next_fire.timestamp():
            return repeated
    return next_fire


def next_cron_time(expression: str, now: datetime) -> datetime:
    """Return the next trigger as an aware datetime."""
    from croniter import croniter

    if now.tzinfo is not None and _legacy_croniter():
        return _next_legacy_time(expression, now)
    next_fire = croniter(expression, now).get_next(datetime)
    if next_fire.tzinfo is not None:
        return next_fire
    # croniter drops fold for naive times. Use the system clock to map the wall time to an instant.
    timestamp = next_fire.replace(fold=now.fold).timestamp()
    return datetime.fromtimestamp(timestamp).astimezone()
