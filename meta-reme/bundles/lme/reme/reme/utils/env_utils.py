"""Load .env files into os.environ (idempotent)."""

import os
from pathlib import Path

_LOADED = False
_LOADED_VALUES: dict[str, str] = {}


def parse_env_file(path: str | Path) -> dict[str, str]:
    """Parse a simple KEY=VALUE env file and return a key/value dict."""
    path = Path(path)
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if not key:
            continue
        values[key] = value.strip().strip("'\"")
    return values


def _load_values(values: dict[str, str], *, override: bool) -> dict[str, str]:
    loaded: dict[str, str] = {}
    for key, value in values.items():
        if override or key not in os.environ:
            os.environ[key] = value
            loaded[key] = value
    return loaded


def load_env(path: str | Path | None = None, *, override: bool = True) -> dict[str, str]:
    """Load .env from given path, or search cwd and up to 5 parents.

    Returns the key/value pairs loaded into ``os.environ``. Repeated calls without
    an explicit path are idempotent and return the values loaded by the first
    successful call.
    """
    global _LOADED
    global _LOADED_VALUES
    if path is None and _LOADED:
        return dict(_LOADED_VALUES)

    if path:
        path = Path(path)
        if path.exists():
            return _load_values(parse_env_file(path), override=override)
        return {}

    for directory in [Path.cwd(), *Path.cwd().parents[:5]]:
        env_path = directory / ".env"
        if env_path.exists():
            _LOADED_VALUES = _load_values(parse_env_file(env_path), override=override)
            _LOADED = True
            return dict(_LOADED_VALUES)
    return {}
