"""Track the round record until its input receipt is committed."""

from contextlib import asynccontextmanager
from stat import S_ISREG

from ....utils.async_utils import complete_in_thread
from ....utils.files.path_lock import get_path_lock


class RoundRecord:
    """Keep concurrent user edits safe when checking or rolling back a record."""

    def __init__(self, files, path):
        self.files = files
        self.path = path
        self.written = None

    def _version(self):
        try:
            stat = self.files.checked(self.path).stat(follow_symlinks=False)
        except FileNotFoundError:
            return None
        if not S_ISREG(stat.st_mode):
            raise ValueError(f"Round record must be a regular file: {self.path}")
        return stat.st_ino, stat.st_mtime_ns, stat.st_size

    @asynccontextmanager
    async def locked(self):
        """Share the workspace file jobs' write lock."""
        async with await get_path_lock(self.path):
            yield

    async def prepare(self):
        """Ensure the model starts with an unused record path."""
        async with self.locked():
            if await complete_in_thread(self._version) is not None:
                raise FileExistsError(self.path)

    def capture(self):
        """Record the version written by the model before validation."""
        try:
            self.written = self._version()
        except (OSError, ValueError):
            # Validation reports invalid or unreadable destinations.
            self.written = None

    def _matches(self):
        try:
            return self.written is not None and self._version() == self.written
        except (OSError, ValueError):
            return False

    def check_version(self, path):
        """Reject edits made between validation and commit."""
        if path != self.path or not self._matches():
            raise RuntimeError(f"Agent output changed outside this round: {path}")

    async def rollback(self):
        """Remove only the uncommitted version written by this round."""
        async with self.locked():
            await complete_in_thread(self._rollback)

    def _rollback(self):
        if self._matches():
            self.files.remove(self.path)
