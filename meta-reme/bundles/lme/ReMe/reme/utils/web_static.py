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
    package_dir = Path(__file__).resolve().parent.parent
    repository_dir = package_dir.parent
    cwd = Path.cwd()
    explicit_candidates = [
        configured_dir,
        os.getenv(REME_WEB_STATIC_DIR),
    ]
    for candidate in explicit_candidates:
        if not candidate:
            continue
        path = Path(candidate).expanduser().resolve()
        if path.is_dir() and (path / "index.html").is_file():
            return path

    fallback_candidates = [
        _packaged_studio_dir(),
        package_dir / "web",
        repository_dir / "reme_studio" / "dist-static",
        cwd / "reme_studio" / "dist-static",
        cwd / "web_dist",
    ]
    for candidate in fallback_candidates:
        if not candidate:
            continue
        path = Path(candidate).expanduser().resolve()
        if path.is_dir() and (path / "index.html").is_file():
            return path
    return None
