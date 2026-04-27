"""Full file watcher for all supported file types."""

from .base_file_watcher import BaseFileWatcher
from ..component_registry import R


@R.register("full")
class FullFileWatcher(BaseFileWatcher):
    """Watches all supported file types and delegates parsing to registered parsers."""
