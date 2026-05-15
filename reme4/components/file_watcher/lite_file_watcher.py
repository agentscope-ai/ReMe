import asyncio
from pathlib import Path

from watchfiles import Change, awatch

from .base_file_watcher import BaseFileWatcher
from ..component_registry import R
from ...schema import FileChunk, FileNode
from ...utils import get_logger

logger = get_logger()


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
            logger.warning("No watch paths specified")
            return

        while not self._stop_event.is_set():
            valid_paths = [p for p in self.watch_paths if p.exists()]
            if not valid_paths:
                logger.warning(f"No valid paths, retrying in {self._retry_interval}s...")
                await self._interruptible_sleep()
                continue

            invalid = set(self.watch_paths) - set(valid_paths)
            if invalid:
                logger.warning(f"Skipping invalid paths: {invalid}")

            try:
                logger.info(f"Watching: {valid_paths}")
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
                logger.exception(f"Watch error, retrying in {self._retry_interval}s...")
                if not self._stop_event.is_set():
                    await self._interruptible_sleep()

    async def _dispatch_changes(self, changes: set[tuple[Change, str]]):
        """Classify raw changes and dispatch to event handlers."""
        added = [Path(p) for c, p in changes if c == Change.added]
        modified = [Path(p) for c, p in changes if c == Change.modified]
        deleted = [Path(p) for c, p in changes if c == Change.deleted]
        if added:
            logger.info(f"Detected {len(added)} added file(s)")
            await self.on_added(added)
        if modified:
            logger.info(f"Detected {len(modified)} modified file(s)")
            await self.on_modified(modified)
        if deleted:
            logger.info(f"Detected {len(deleted)} deleted file(s)")
            await self.on_deleted(deleted)

    async def update_store(self):
        if self.file_store is None:
            raise ValueError("file_store is not initialized!")

        # Diff existing files against indexed entries by path and mtime.
        existing = {str(p): p.stat().st_mtime for p in await self.scan_existing_files()}
        indexed = {p: n.st_mtime for p, n in self.file_store.file_nodes.items()}
        existing_keys, indexed_keys = set(existing), set(indexed)

        to_delete = indexed_keys - existing_keys
        to_add = existing_keys - indexed_keys
        to_modify = [p for p in existing_keys & indexed_keys if existing[p] != indexed[p]]

        if to_modify:
            logger.info(f"Updating {len(to_modify)} modified file(s)")
            await self.on_modified([Path(p) for p in to_modify])
        if to_delete:
            logger.info(f"Removing {len(to_delete)} deleted file(s)")
            await self.on_deleted([Path(p) for p in to_delete])
        if to_add:
            logger.info(f"Indexing {len(to_add)} new file(s)")
            await self.on_added([Path(p) for p in to_add])
        if not to_modify and not to_delete and not to_add:
            logger.info("Store is up to date")

    async def _parse_and_upsert(self, paths: list[Path], action: str):
        """Parse files and upsert into store. Shared by on_added / on_modified."""
        if self.file_parser is None or self.file_store is None:
            raise RuntimeError("file_parser or file_store is not initialized!")

        parsed: list[tuple[FileNode, list[FileChunk]]] = []
        for p in paths:
            if p.is_file():
                logger.info(f"{action} file: {p}")
                parsed.append(await self.file_parser.parse(p))
        if parsed:
            file_paths = [str(p) for p in paths if p.is_file()]
            await self.file_store.delete_by_path(file_paths)
            await self.file_store.upsert_file(parsed)

    async def on_added(self, path: Path | list[Path]):
        paths = [path] if isinstance(path, Path) else path
        await self._parse_and_upsert(paths, "Adding")

    async def on_modified(self, path: Path | list[Path]):
        paths = [path] if isinstance(path, Path) else path
        await self._parse_and_upsert(paths, "Updating")

    async def on_deleted(self, path: Path | list[Path]):
        if self.file_store is None:
            raise RuntimeError("file_store is not initialized!")
        paths = [path] if isinstance(path, Path) else path
        logger.info(f"Deleting {len(paths)} file(s)")
        await self.file_store.delete_by_path([str(p) for p in paths])
