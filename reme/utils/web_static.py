"""Resolve the optional ReMe workspace frontend build."""

from __future__ import annotations

import os
from pathlib import Path

REME_WEB_STATIC_DIR = "REME_WEB_STATIC_DIR"


def resolve_web_static_dir(configured_dir: str | None = None) -> Path | None:
    """Return the first directory containing a built workspace ``index.html``."""
    package_dir = Path(__file__).resolve().parent.parent
    repository_dir = package_dir.parent
    cwd = Path.cwd()
    candidates = [
        configured_dir,
        os.getenv(REME_WEB_STATIC_DIR),
        package_dir / "web",
        repository_dir / "website" / "dist-static",
        cwd / "website" / "dist-static",
        cwd / "web_dist",
    ]
    for candidate in candidates:
        if not candidate:
            continue
        path = Path(candidate).expanduser().resolve()
        if path.is_dir() and (path / "index.html").is_file():
            return path
    return None
