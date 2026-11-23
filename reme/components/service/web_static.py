"""Resolve the optional ReMe workspace frontend build."""

from __future__ import annotations

import os
from pathlib import Path

REME_WEB_STATIC_DIR = "REME_WEB_STATIC_DIR"


def _packaged_studio_dir() -> Path | None:
    """Return the static directory supplied by the optional Studio package."""
    try:
        from reme_studio import static_dir
    except ImportError:
        return None
    return static_dir()


def resolve_web_static_dir(configured_dir: str | None = None) -> Path | None:
    """Return the first directory containing a built workspace ``index.html``."""
    repository = Path(__file__).resolve().parents[3]
    for candidate in (
        configured_dir,
        os.getenv(REME_WEB_STATIC_DIR),
        _packaged_studio_dir(),
        repository / "reme_studio" / "web" / "dist-static",
    ):
        if candidate:
            path = Path(candidate).expanduser().resolve()
            if (path / "index.html").is_file():
                return path
    return None
