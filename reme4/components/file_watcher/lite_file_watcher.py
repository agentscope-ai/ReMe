"""Polling-based file watcher using watchfiles."""

import asyncio

from watchfiles import Change, awatch

from .base_file_watcher import BaseFileWatcher
from ..component_registry import R
from ...schema import FileChunk, FileNode


@R.register("lite")
class LiteFileWatcher(BaseFileWatcher):
    """Polling-based file watcher using watchfiles awatch."""

    async def _interruptible_sleep(self):
        """Sleep until stop or timeout, whichever comes first."""
        try:
            await asyncio.wait_for(self._stop_event.wait(), timeout=self._retry_interval)
        except asyncio.TimeoutError:
            pass

    async def watch_loop(self):
        if not self.watch_paths:
            self.logger.warning("No watch paths specified")
            return

        while not self._stop_event.is_set():
            valid_paths = [p for p in self.watch_paths if p.exists()]
            if not valid_paths:
                self.logger.warning(f"No valid paths, retrying in {self._retry_interval}s...")
                await self._interruptible_sleep()
                continue

            invalid = set(self.watch_paths) - set(valid_paths)
            if invalid:
                self.logger.warning(f"Skipping invalid paths: {[str(p) for p in invalid]}")

            try:
                self.logger.info(f"Watching: {[str(p) for p in valid_paths]}")
                async for changes in awatch(
                    *valid_paths,
                    watch_filter=self.watch_filter,
                    recursive=self.recursive,
                    force_polling=self.force_polling,
                    debounce=self.debounce,
                    poll_delay_ms=self.poll_delay_ms,
                    stop_event=self._stop_event,
                ):
                    if self._stop_event.is_set():
                        break
                    await self._dispatch_changes(changes)
            except Exception:
                self.logger.exception(f"Watch error, retrying in {self._retry_interval}s...")
                if not self._stop_event.is_set():
                    await self._interruptible_sleep()

    async def _dispatch_changes(self, changes: set[tuple[Change, str]]):
        """Classify raw changes and dispatch to event handlers."""
        buckets: dict[Change, list[str]] = {Change.added: [], Change.modified: [], Change.deleted: []}
        for c, p in changes:
            if c in buckets:
                buckets[c].append(self._get_relative_path(p))
        for change, handler, label in (
            (Change.added, self.on_added, "added"),
            (Change.modified, self.on_modified, "modified"),
            (Change.deleted, self.on_deleted, "deleted"),
        ):
            if buckets[change]:
                self.logger.info(f"Detected {len(buckets[change])} {label} file(s)")
                await handler(buckets[change])

    async def update_store(self, dump: bool = True) -> dict[str, int]:
        if self.file_store is None:
            raise ValueError("file_store is not initialized!")

        existing: dict[str, float] = {
            rel: abs_p.stat().st_mtime for rel, abs_p in (await self.scan_existing_files()).items()
        }
        indexed: dict[str, float] = {n.path: n.st_mtime for n in await self.file_store.get_nodes()}

        to_delete = list(indexed.keys() - existing.keys())
        to_add = list(existing.keys() - indexed.keys())
        to_modify = [p for p in existing.keys() & indexed.keys() if existing[p] != indexed[p]]

        if to_modify:
            self.logger.info(f"Updating {len(to_modify)} modified file(s)")
            await self.on_modified(to_modify)
        if to_delete:
            self.logger.info(f"Removing {len(to_delete)} deleted file(s)")
            await self.on_deleted(to_delete)
        if to_add:
            self.logger.info(f"Indexing {len(to_add)} new file(s)")
            await self.on_added(to_add)

        changed = bool(to_add or to_modify or to_delete)
        if not changed:
            self.logger.info("Store is up to date")
        if dump and changed:
            await self.file_store.dump()
        return {"added": len(to_add), "modified": len(to_modify), "deleted": len(to_delete)}

    async def _parse_and_upsert(self, paths: list[str], action: str):
        """Parse files and upsert into store. Shared by on_added / on_modified."""
        if self.file_parser is None or self.file_store is None:
            raise RuntimeError("file_parser or file_store is not initialized!")

        parsed: list[tuple[FileNode, list[FileChunk]]] = []
        for rel in paths:
            abs_path = self._get_absolute_path(rel)
            if abs_path.is_file():
                self.logger.info(f"{action} file: {rel}")
                parsed.append(await self.file_parser.parse(abs_path))
        if parsed:
            await self.file_store.delete([n.path for n, _ in parsed])
            await self.file_store.upsert(parsed)

    async def on_added(self, path: str | list[str]):
        await self._parse_and_upsert([path] if isinstance(path, str) else path, "Adding")

    async def on_modified(self, path: str | list[str]):
        await self._parse_and_upsert([path] if isinstance(path, str) else path, "Updating")

    async def on_deleted(self, path: str | list[str]):
        if self.file_store is None:
            raise RuntimeError("file_store is not initialized!")
        paths = [path] if isinstance(path, str) else path
        self.logger.info(f"Deleting {len(paths)} file(s)")
        await self.file_store.delete(paths)
