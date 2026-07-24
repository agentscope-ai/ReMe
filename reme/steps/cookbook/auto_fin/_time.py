"""Datetime comparison helpers for mixed legacy and timezone-aware Auto Fin data."""

from datetime import datetime


def compare_datetimes(left: datetime, right: datetime) -> int:
    """Compare datetimes, interpreting a lone naive value in the known timezone."""
    if left.utcoffset() is None and right.utcoffset() is not None:
        left = left.replace(tzinfo=right.tzinfo)
    elif left.utcoffset() is not None and right.utcoffset() is None:
        right = right.replace(tzinfo=left.tzinfo)
    return (left > right) - (left < right)
