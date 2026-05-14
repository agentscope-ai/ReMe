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

    async def _interruptible_sleep(self):
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

            invalid_paths = set(self.watch_paths) - set(valid_paths)
            if invalid_paths:
                logger.warning(f"Skipping invalid paths: {invalid_paths}")

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

            except Exception:
                logger.exception(f"Watch error, retrying in {self._retry_interval}s...")
                if not self._stop_event.is_set():
                    await self._interruptible_sleep()

    async def update_store(self):
        if self.file_store is None:
            raise ValueError("file_store is not initialized!")

        existing_paths: dict[str, float] = {str(p): p.stat().st_mtime for p in await self.scan_existing_files()}
        indexed_paths: dict[str, float] = {p: n.st_mtime for p, n in self.file_store.file_nodes.items()}

        existing_keys = set(existing_paths.keys())
        indexed_keys = set(indexed_paths.keys())

        paths_to_delete = indexed_keys - existing_keys
        paths_to_add = existing_keys - indexed_keys
        paths_to_modify = [p for p in existing_keys & indexed_keys if existing_paths[p] != indexed_paths[p]]

        if paths_to_modify:
            logger.info(f"Updating {len(paths_to_modify)} modified file(s)")
            await self.on_modified([Path(p) for p in paths_to_modify])

        if paths_to_delete:
            logger.info(f"Removing {len(paths_to_delete)} deleted file(s)")
            await self.on_deleted([Path(p) for p in paths_to_delete])

        if paths_to_add:
            logger.info(f"Indexing {len(paths_to_add)} new file(s)")
            await self.on_added([Path(p) for p in paths_to_add])

        if not paths_to_modify and not paths_to_delete and not paths_to_add:
            logger.info("Store is up to date")

    async def on_added(self, path: Path | list[Path]):
        if self.file_parser is None or self.file_store is None:
            raise RuntimeError("file_parser or file_store is not initialized!")

        paths = [path] if isinstance(path, Path) else path
        parsed: list[tuple[FileNode, list[FileChunk]]] = []
        for p in paths:
            if p.is_file():
                logger.info(f"Adding file: {p}")
                parsed.append(await self.file_parser.parse(p))
        if parsed:
            await self.file_store.delete_by_path([str(p) for p in paths if p.is_file()])
            await self.file_store.upsert_file(parsed)

    async def on_modified(self, path: Path | list[Path]):
        if self.file_parser is None or self.file_store is None:
            raise RuntimeError("file_parser or file_store is not initialized!")

        paths = [path] if isinstance(path, Path) else path
        parsed: list[tuple[FileNode, list[FileChunk]]] = []
        for p in paths:
            if p.is_file():
                logger.info(f"Updating file: {p}")
                parsed.append(await self.file_parser.parse(p))
        if parsed:
            await self.file_store.delete_by_path([str(p) for p in paths if p.is_file()])
            await self.file_store.upsert_file(parsed)

    async def on_deleted(self, path: Path | list[Path]):
        if self.file_store is None:
            raise RuntimeError("file_store is not initialized!")

        paths = [path] if isinstance(path, Path) else path
        logger.info(f"Deleting {len(paths)} file(s)")
        await self.file_store.delete_by_path([str(p) for p in paths])
