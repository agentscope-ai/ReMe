"""Remove expired temporary files by mtime, without a file catalog."""

from contextlib import ExitStack
import errno
import os
from pathlib import Path
import stat
import time

from ._resource import resource_root
from ..base_step import BaseStep
from ..file_io._path import _check_path_permission
from ...components import R
from ...utils.async_utils import complete_in_thread
from ...utils.file_utils import get_path_lock


@R.register("cleanup_resources_step")
class CleanupResourcesStep(BaseStep):
    """Scan resource_dir and retain files modified less than ttl_days ago."""

    async def execute(self):
        ttl_days = self.context.get("ttl_days", self.kwargs.get("ttl_days", 7))
        if type(ttl_days) is not int or ttl_days < 1:
            raise ValueError("ttl_days must be a positive integer")
        root = resource_root(self)
        if not _check_path_permission(self.workspace_path, root, self.context.get("_allowed_paths")):
            raise PermissionError("no permission to clean the resource directory")
        async with await get_path_lock(root):
            result = await complete_in_thread(self._cleanup, root, time.time() - ttl_days * 86400)
        self.context.response.answer = result
        self.context.response.success = not result["errors"]
        return self.context.response

    def _cleanup(self, root: Path, cutoff: float) -> dict:
        result = {"deleted_files": [], "deleted_dirs": [], "freed_bytes": 0, "errors": {}}
        if not root.exists():
            return result
        if not hasattr(os, "fwalk") or not hasattr(os, "O_NOFOLLOW"):
            raise RuntimeError("resource cleanup requires descriptor-relative, symlink-safe file operations")
        workspace = self.workspace_path.resolve()
        relative = root.relative_to(workspace)
        flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW

        def scan_error(error: OSError) -> None:
            path = (relative / str(error.filename or ".")).as_posix()
            result["errors"][path] = str(error)

        with ExitStack() as stack:
            root_fd = os.open(workspace, flags)
            stack.callback(os.close, root_fd)
            for part in relative.parts:
                root_fd = os.open(part, flags, dir_fd=root_fd)
                stack.callback(os.close, root_fd)
            for directory, dirs, files, directory_fd in os.fwalk(
                ".",
                topdown=False,
                follow_symlinks=False,
                dir_fd=root_fd,
                onerror=scan_error,
            ):
                for name in files:
                    path = (relative / directory / name).as_posix()
                    try:
                        info = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
                        if not stat.S_ISREG(info.st_mode) or info.st_mtime > cutoff:
                            continue
                        os.unlink(name, dir_fd=directory_fd)
                    except FileNotFoundError:
                        continue
                    except OSError as exc:
                        result["errors"][path] = str(exc)
                    else:
                        result["deleted_files"].append(path)
                        result["freed_bytes"] += info.st_size
                for name in dirs:
                    path = (relative / directory / name).as_posix()
                    try:
                        info = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
                        if not stat.S_ISDIR(info.st_mode):
                            continue
                        os.rmdir(name, dir_fd=directory_fd)
                    except FileNotFoundError:
                        continue
                    except OSError as exc:
                        if exc.errno not in {errno.ENOTEMPTY, errno.EEXIST}:
                            result["errors"][path] = str(exc)
                    else:
                        result["deleted_dirs"].append(path)
        return result
