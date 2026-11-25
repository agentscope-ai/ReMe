"""Next-fire calculations shared by the Cron and proactive runtimes."""

from datetime import datetime, timedelta


def next_cron_time(expression: str, now: datetime) -> datetime:
    """Find the earliest future trigger, including times repeated by a clock rollback."""
    from croniter import CroniterBadDateError, croniter

    try:
        next_fire = croniter(expression, now).get_next(datetime)
    except CroniterBadDateError:
        next_fire = None
    # An exhausted wall-clock schedule may still have an occurrence in the second fold.
    next_offset = (next_fire or now.replace(fold=1)).utcoffset()
    offset_change = next_offset - now.utcoffset()
    if offset_change < timedelta():
        # A rollback can repeat an earlier wall time before croniter's first result.
        repeated = croniter(expression, now.replace(tzinfo=None) + offset_change)
        while True:
            try:
                candidate = repeated.get_next(datetime).replace(tzinfo=now.tzinfo, fold=1)
            except CroniterBadDateError:
                break
            if next_fire is not None and candidate.timestamp() >= next_fire.timestamp():
                break
            if candidate.timestamp() > now.timestamp():
                return candidate
    if next_fire is None:
        raise CroniterBadDateError("failed to find next date")
    return next_fire
