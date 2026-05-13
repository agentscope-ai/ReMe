from .base_file_watcher import BaseFileWatcher
from ..component_registry import R


@R.register("light")
class LightFileWatcher(BaseFileWatcher):
    ...
