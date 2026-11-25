"""Resolve the process timezone from an IANA name or the host configuration."""

import os
from zoneinfo import ZoneInfo

from tzlocal import get_localzone


def system_timezone() -> ZoneInfo:
    """Resolve TZ directly so overrides never enter the host timezone cache."""
    name = os.environ.get("TZ")
    if name is not None:
        try:
            return ZoneInfo(name)
        except (LookupError, ValueError) as error:
            raise ValueError(f"TZ must be an IANA timezone name: {name!r}") from error
    try:
        return get_localzone()
    except (LookupError, OSError, ValueError):
        pass
    return ZoneInfo("Asia/Shanghai")
