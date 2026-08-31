"""Expose the packaged BEAM application configuration without loading backends."""

from pathlib import Path


def config_path() -> Path:
    """Return the application preset used by the existing benchmark runner."""
    return Path(__file__).parent / "configs" / "beam.yaml"
