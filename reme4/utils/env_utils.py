"""Load .env files into os.environ (idempotent)."""

import os
from pathlib import Path

_LOADED = False


def _parse(path: Path) -> None:
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ[key.strip()] = value.strip().strip("'\"")


def load_env(path: str | Path | None = None) -> None:
    """Load .env from given path, or search cwd and up to 5 parents."""
    global _LOADED
    if _LOADED:
        return

    if path:
        path = Path(path)
        if path.exists():
            _parse(path)
            _LOADED = True
        return

    for directory in [Path.cwd(), *Path.cwd().parents[:5]]:
        env_path = directory / ".env"
        if env_path.exists():
            _parse(env_path)
            _LOADED = True
            return
