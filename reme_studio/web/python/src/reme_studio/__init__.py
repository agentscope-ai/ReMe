"""Static asset provider for the optional ReMe Studio frontend."""

from pathlib import Path


def static_dir() -> Path:
    """Return the installed Studio static asset directory."""
    return Path(__file__).resolve().parent / "static"


__all__ = ["static_dir"]
