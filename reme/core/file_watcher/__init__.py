"""File watcher module for monitoring file system changes.

This module provides file watcher implementations for monitoring file changes
and updating memory stores accordingly.
"""

from .base_file_watcher import BaseFileWatcher
from .delta_file_watcher import DeltaFileWatcher
from .full_file_watcher import FullFileWatcher
from ..registry_factory import R

__all__ = [
    "BaseFileWatcher",
    "DeltaFileWatcher",
    "FullFileWatcher",
]

R.file_watchers.register("full")(FullFileWatcher)
R.file_watchers.register("delta")(DeltaFileWatcher)
