"""Next-fire calculations shared by the Cron and proactive runtimes."""

from datetime import datetime, timedelta


def next_cron_time(expression: str, now: datetime) -> datetime:
    """Find the earliest future trigger, including times repeated by a clock rollback."""
    from croniter import croniter

    next_fire = croniter(expression, now).get_next(datetime)
    offset_change = next_fire.utcoffset() - now.utcoffset()
    if offset_change < timedelta():
        # A rollback can repeat an earlier wall time before croniter's first result.
        repeated = croniter(expression, now.replace(tzinfo=None) + offset_change)
        while True:
            candidate = repeated.get_next(datetime).replace(tzinfo=now.tzinfo, fold=1)
            if candidate.timestamp() >= next_fire.timestamp():
                break
            if candidate.timestamp() > now.timestamp():
                return candidate
    return next_fire
