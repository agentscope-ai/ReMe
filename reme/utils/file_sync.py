"""Durable file and directory publication shared by file-native components."""

import os
from pathlib import Path
import sys


def sync_file(fd: int) -> None:
    """Flush file contents through the device write cache where supported."""
    if sys.platform == "darwin":
        import fcntl

        fcntl.fcntl(fd, fcntl.F_FULLFSYNC)
    else:
        os.fsync(fd)


def sync_directory(path: Path) -> None:
    """Persist directory entries after creation, replacement or removal."""
    if os.name != "nt":
        fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
