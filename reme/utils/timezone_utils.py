"""Resolve the process timezone for local clock calculations."""

import os
from datetime import tzinfo
from zoneinfo import ZoneInfo

from tzlocal import get_localzone


def timezone_from_env() -> ZoneInfo | None:
    """Read a named timezone or TZif file from the current TZ environment."""
    name = os.environ.get("TZ", "").removeprefix(":")
    if not name:
        return None
    try:
        if os.path.isabs(name):
            if not os.path.isfile(name):
                return None
            with open(name, "rb") as file:
                return ZoneInfo.from_file(file)
        return ZoneInfo(name)
    except (LookupError, OSError, ValueError):
        return None


def system_timezone() -> tzinfo | None:
    """Use full local rules when available, otherwise let the system clock interpret local times."""
    if "TZ" in os.environ:
        return timezone_from_env()
    try:
        return get_localzone()
    except (LookupError, OSError, ValueError):
        return None
