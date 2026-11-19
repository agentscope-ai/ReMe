"""Write round outputs in place and keep the previous daily summary recoverable."""

from contextlib import AsyncExitStack, asynccontextmanager
from stat import S_ISREG
from uuid import uuid4

from ....utils.async_utils import complete_in_thread
from ....utils.files.path_lock import get_path_lock


class RoundArtifacts:
    """Back up the summary and track the new round files for cleanup."""

    def __init__(self, files, paths):
        self.files = files
        self.outputs = (paths["record_path"], paths["reminder_path"])
        self.summary = paths["today_summary_path"]
        self.written = None
        self.backup = None
        self.keep_backup = False

    def _version(self, path):
        try:
            stat = self.files.checked(path).stat(follow_symlinks=False)
        except FileNotFoundError:
            return None
        if not S_ISREG(stat.st_mode):
            raise ValueError(f"Round artifact must be a regular file: {path}")
        return stat.st_ino, stat.st_mtime_ns, stat.st_size

    @asynccontextmanager
    async def locked(self):
        """Share the workspace file jobs' write locks."""
        async with AsyncExitStack() as stack:
            for path in sorted((*self.outputs, self.summary)):
                await stack.enter_async_context(await get_path_lock(path))
            yield

    async def prepare(self):
        """Back up the existing summary and return its original content."""
        async with self.locked():
            return await complete_in_thread(self._prepare)

    def _prepare(self):
        for path in self.outputs:
            if self._version(path) is not None:
                raise FileExistsError(path)
        if self._version(self.summary) is None:
            return None
        content = self.files.read(self.summary)
        path = self.summary.with_suffix(".md.bak")
        if path.exists() or path.is_symlink():
            path = path.with_suffix(f".{uuid4().hex}.bak")
        self.files.write(
            path,
            content,
            create_only=True,
            on_write=lambda version: self._remember_backup(path, version),
        )
        if self.backup is None:
            raise FileExistsError(f"Could not back up {self.summary}")
        return content

    def _remember_backup(self, path, version):
        self.backup = path, version[1]

    def _matches(self, path, version):
        try:
            return self._version(path) == version
        except (OSError, ValueError):
            return False

    def capture_outputs(self):
        """Capture the new files before validation and outbox changes."""
        # By convention, the model owns these per-round record and reminder paths during execution.
        self.written = {}
        for path in self.outputs:
            try:
                if (version := self._version(path)) is not None:
                    self.written[path] = version
            except (OSError, ValueError):
                # Validation reports unreadable or invalid destinations.
                continue

    def check_written(self, path):
        """Guard the outbox against changes since validation."""
        if not self.written or path not in self.written or not self._matches(path, self.written[path]):
            raise RuntimeError(f"Agent output changed outside this round: {path}")

    def record_write(self, path, version):
        """Track the version written by the outbox."""
        self.written[path] = version[1]

    async def rollback(self):
        """Restore the summary backup and remove uncommitted round files."""
        # A forced process exit can leave the outputs and backup for manual recovery.
        if self.written is None:
            return
        async with self.locked():
            await complete_in_thread(self._rollback)

    def _rollback(self):
        try:
            self._rollback_summary()
        finally:
            for path, version in self.written.items():
                if self._matches(path, version):
                    self.files.remove(path)

    def _rollback_summary(self):
        # The model owns the summary during execution, so restore the complete pre-run file.
        self.keep_backup = True
        if self.backup is not None:
            path, version = self.backup
            if not self._matches(path, version):
                raise RuntimeError(f"Round backup changed before recovery: {path}")
            self.files.write(self.summary, self.files.read(path))
        elif self._version(self.summary) is not None:
            self.files.remove(self.summary)
        self.keep_backup = False

    def close(self):
        """Remove this round's backup when no recovery is needed."""
        if self.backup is not None and not self.keep_backup:
            path, version = self.backup
            if self._matches(path, version):
                self.files.remove(path)
