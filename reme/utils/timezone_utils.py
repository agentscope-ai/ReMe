"""Resolve the process timezone from an IANA name or the host configuration."""

import os
import sys
from datetime import datetime, tzinfo
from pathlib import Path
from zoneinfo import ZoneInfo

from tzlocal import get_localzone, get_localzone_name

_LOCALTIME_PATHS = (Path("/etc/localtime"), Path("/usr/local/etc/localtime"))


def _timezone_from_env() -> ZoneInfo | None:
    """Validate the current process override without caching it as a host setting."""
    name = os.environ.get("TZ")
    if name is not None:
        try:
            return ZoneInfo(name)
        except (LookupError, ValueError) as error:
            raise ValueError(f"TZ must be an IANA timezone name: {name!r}") from error
    return None


def default_timezone_name() -> str:
    """Use a detected IANA name, with the application's fallback when discovery fails."""
    configured = _timezone_from_env()
    if configured is not None:
        return configured.key
    try:
        name = get_localzone_name()
        if name:
            return ZoneInfo(name).key
    except (LookupError, OSError, ValueError):
        pass
    return "Asia/Shanghai"


def system_timezone() -> tzinfo:
    """Resolve process or host rules shared by local-time scheduling and calendar dates."""
    configured = _timezone_from_env()
    if configured is not None:
        return configured
    if sys.platform == "win32":
        try:
            return get_localzone()
        except (LookupError, OSError, ValueError):
            pass
    else:
        for path in _LOCALTIME_PATHS:
            try:
                with path.open("rb") as source:
                    return ZoneInfo.from_file(source)
            except (LookupError, OSError, ValueError):
                continue
    return datetime.now().astimezone().tzinfo


def resolve_timezone(name: str | None) -> tzinfo:
    """Apply an explicit IANA name or the process's local-time rules consistently."""
    return ZoneInfo(name) if name else system_timezone()
