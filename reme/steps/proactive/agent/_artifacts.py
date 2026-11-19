"""Stage one round beside its destinations and preserve concurrent source edits."""

from contextlib import AsyncExitStack, asynccontextmanager
from uuid import uuid4

from ....utils.async_utils import complete_in_thread
from ....utils.path_lock import get_path_lock


class RoundArtifacts:
    """Own drafts and conditionally undo only files published by this invocation."""

    keys = ("record_path", "reminder_path", "today_summary_path")

    def __init__(self, files, paths):
        self.files = files
        self.paths = paths
        self.drafts = dict(paths)
        self.before = {}
        self.published = {}
        identifier = uuid4().hex
        for key in self.keys:
            path = paths[key]
            self.before[key] = self.snapshot(path)
            self.drafts[key] = path.with_name(f".round-{identifier}-{path.name}")

    def snapshot(self, path):
        """Retain both content and file identity, including same-content external saves."""
        try:
            content = self.files.read(path)
            stat = path.stat()
            return content, (stat.st_ino, stat.st_mtime_ns, stat.st_size)
        except FileNotFoundError:
            return None, None

    def prepare(self):
        """Seed the summary draft without changing its user-owned destination."""
        before = self.before["today_summary_path"][0]
        if before is not None:
            self.files.write(self.drafts["today_summary_path"], before, create_only=True)

    def check_published(self, path):
        """Called under the destination lock before outbox metadata claims the file."""
        for key, version in self.published.items():
            if self.paths[key] == path:
                if self.snapshot(path) != version:
                    raise RuntimeError(f"Agent output changed outside this round: {path}")
                return
        raise RuntimeError(f"Agent output was not published by this round: {path}")

    @asynccontextmanager
    async def locked(self):
        """Share publication locks with the workspace file jobs."""
        async with AsyncExitStack() as stack:
            for path in sorted(self.paths[key] for key in self.keys):
                await stack.enter_async_context(await get_path_lock(path))
            yield

    async def publish(self):
        """Publish under the same per-path locks as other ReMe writers."""
        async with self.locked():
            await complete_in_thread(self._publish)

    def _publish(self):
        """Validate every destination before replacing any file with complete output."""
        changes = {}
        for key in self.keys:
            content = self.snapshot(self.drafts[key])[0]
            if content is None or content == self.before[key][0]:
                continue
            # Replace private draft names with their published names in saved links.
            for target in self.keys:
                content = content.replace(self.drafts[target].name, self.paths[target].name)
            changes[key] = content
        for key in changes:
            if self.snapshot(self.paths[key]) != self.before[key]:
                raise RuntimeError(f"Agent output changed outside this round: {self.paths[key]}")
        for key, content in changes.items():
            if self.snapshot(self.paths[key]) != self.before[key]:
                raise RuntimeError(f"Agent output changed outside this round: {self.paths[key]}")
            self.files.write(
                self.paths[key],
                content,
                on_write=lambda version, target=key: self.published.__setitem__(target, version),
            )

    async def rollback(self):
        """Coordinate compensation with concurrent workspace file jobs."""
        async with self.locked():
            await complete_in_thread(self._rollback)

    def _rollback(self):
        """Restore our published versions only; external replacements remain authoritative."""
        for key, published in self.published.items():
            path = self.paths[key]
            if self.snapshot(path) != published:
                continue
            before = self.before[key][0]
            if before is None:
                self.files.remove(path)
            else:
                self.files.write(path, before)

    def close(self):
        """Discard only this invocation's private draft names."""
        for key in self.keys:
            path = self.drafts[key]
            if path.exists():
                self.files.remove(path)
