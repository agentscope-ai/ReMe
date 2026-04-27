"""Lightweight file watcher for Markdown files only."""

from pathlib import Path

from .base_file_watcher import BaseFileWatcher
from ..component_registry import R

_MD_SUFFIXES = {".md", ".markdown"}


@R.register("light")
class LightFileWatcher(BaseFileWatcher):
    """Watches only Markdown files and delegates parsing to registered parsers."""

    def _watch_filter(self, path: str) -> bool:
        return super()._watch_filter(path) and Path(path).suffix.lower() in _MD_SUFFIXES
