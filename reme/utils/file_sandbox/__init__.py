"""Select a local file sandbox for supported operating systems."""

import sys

from .file_sandbox_backend import FileSandboxBackend
from .path_bubble_backend import PathBubbleBackend
from .path_seatbelt_backend import PathSeatbeltBackend


def create_file_sandbox_backend(*, memory_limit_mib: int | None = None, **kwargs) -> FileSandboxBackend:
    """Construct the platform backend; never silently execute without isolation."""
    if sys.platform.startswith("linux"):
        limit = 4096 if memory_limit_mib is None else memory_limit_mib
        return PathBubbleBackend(memory_limit_mib=limit, **kwargs)
    if sys.platform == "darwin":
        limit = 0 if memory_limit_mib is None else memory_limit_mib
        return PathSeatbeltBackend(memory_limit_mib=limit, **kwargs)
    raise RuntimeError(f"No file sandbox backend for {sys.platform}")
