"""Base file watcher implementation.

This module provides the base class for file watcher implementations
that monitor file system changes and trigger callbacks.
"""

import asyncio
from abc import abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING

from watchfiles import Change, awatch

from ..base_component import BaseComponent
from ..file_store import BaseFileStore
from ...enumeration import ComponentEnum

if TYPE_CHECKING:
    from ..application_context import ApplicationContext


class BaseFileWatcher(BaseComponent):
    """Abstract base class for file watcher implementations.

    This base class provides file monitoring functionality that can be extended
    to implement specific file monitoring requirements.
    """

    def __init__(
            self,
            watch_paths: list[str] | str,
            suffix_filters: list[str] | None = None,
            recursive: bool = False,
            debounce: int = 2000,
            chunk_tokens: int = 400,
            chunk_overlap: int = 80,
            file_store: str = "default",
            rebuild_index_on_start: bool = True,
            poll_delay_ms: int = 2000,
            **kwargs,
    ):
        """Initialize the file watcher.

        Args:
            watch_paths: Paths to watch for changes
            suffix_filters: File suffix filters (e.g., ['.py', '.txt'])
            recursive: Whether to watch directories recursively
            debounce: Debounce time in milliseconds
            chunk_tokens: Token size for chunking
            chunk_overlap: Overlap size for chunks
            file_store: Name of the file store component to use
            rebuild_index_on_start: If True, clear all indexed data on start and rescan existing files.
                           If False, only monitor new changes without initialization.
            poll_delay_ms: Polling delay in milliseconds.
            **kwargs: Additional keyword arguments
        """
        super().__init__(**kwargs)
        self._file_store_name: str = file_store
        self.file_store: BaseFileStore | None = None
        self.watch_paths: list[str] = [watch_paths] if isinstance(watch_paths, str) else watch_paths
        self.suffix_filters: list[str] = suffix_filters or []
        self.recursive: bool = recursive
        self.debounce: int = debounce
        self.chunk_tokens: int = chunk_tokens
        self.chunk_overlap: int = chunk_overlap
        self.rebuild_index_on_start: bool = rebuild_index_on_start
        self.poll_delay_ms: int = poll_delay_ms

        self._stop_event = asyncio.Event()
        self._watch_task: asyncio.Task | None = None

    async def _start(self, app_context: ApplicationContext | None = None):
        """Initialize the file watcher and resolve file_store from app_context."""
        # Resolve file_store from app_context
        if self._file_store_name:
            stores = app_context.components.get(ComponentEnum.FILE_STORE, {})
            if self._file_store_name not in stores:
                raise ValueError(f"File store '{self._file_store_name}' not found.")
            store = stores[self._file_store_name]
            if not isinstance(store, BaseFileStore):
                raise TypeError(f"Expected BaseFileStore, got {type(store).__name__}")
            self.file_store = store

        # Start watching task
        async def _initialize_and_watch():
            if self.rebuild_index_on_start and self.file_store:
                await self.file_store.clear_all()
                self.logger.info("Cleared all indexed data on start")
                await self._scan_existing_files()
            await self._watch_loop()

        self._stop_event.clear()
        self._watch_task = asyncio.create_task(_initialize_and_watch())
        self.logger.info(f"Started watching: {self.watch_paths}")

    async def _close(self):
        """Stop the file watcher and release resources."""
        # Signal stop and cancel watch task
        self._stop_event.set()
        if self._watch_task and not self._watch_task.done():
            self._watch_task.cancel()
            try:
                await self._watch_task
            except asyncio.CancelledError:
                pass

        self._watch_task = None
        self._stop_event.clear()
        self.file_store = None
        self.logger.info("Stopped watching")

    def watch_filter(self, _change: Change, path: str) -> bool:
        """Filter function for file watching."""
        if not self.suffix_filters:
            return True

        for suffix in self.suffix_filters:
            if path.endswith("." + suffix.strip(".")):
                return True

        return False

    async def _scan_existing_files(self):
        """Scan existing files matching watch criteria and trigger on_changes with Change.added."""
        if not self.file_store:
            return

        existing_files: set[tuple[Change, str]] = set()

        for watch_path_str in self.watch_paths:
            watch_path = Path(watch_path_str)

            if not watch_path.exists():
                self.logger.warning(f"Watch path does not exist: {watch_path}")
                continue

            if watch_path.is_file():
                if self.watch_filter(Change.added, str(watch_path)):
                    existing_files.add((Change.added, str(watch_path)))
            elif watch_path.is_dir():
                if self.recursive:
                    for file_path in watch_path.rglob("*"):
                        if file_path.is_file() and self.watch_filter(Change.added, str(file_path)):
                            existing_files.add((Change.added, str(file_path)))
                else:
                    for file_path in watch_path.iterdir():
                        if file_path.is_file() and self.watch_filter(Change.added, str(file_path)):
                            existing_files.add((Change.added, str(file_path)))

        if existing_files:
            self.logger.info(f"[SCAN_ON_START] Found {len(existing_files)} existing files matching watch criteria")
            await self.on_changes(existing_files)
            self.logger.info(f"[SCAN_ON_START] Added {len(existing_files)} files to memory store")
        else:
            self.logger.info("[SCAN_ON_START] No existing files found matching watch criteria")

        files: list[str] = await self.file_store.list_files()
        for file_path in files:
            chunks = await self.file_store.get_file_chunks(file_path)
            self.logger.info(f"Found existing file: {file_path}, {len(chunks)} chunks")

    async def _interruptible_sleep(self, seconds: float):
        """Sleep that can be interrupted by stop_event."""
        try:
            await asyncio.wait_for(self._stop_event.wait(), timeout=seconds)
        except asyncio.TimeoutError:
            pass

    async def _watch_loop(self):
        """Core monitoring loop with auto-restart on failure."""
        if not self.watch_paths:
            self.logger.warning("No watch paths specified")
            return

        while not self._stop_event.is_set():
            valid_paths = [p for p in self.watch_paths if Path(p).exists()]

            if not valid_paths:
                self.logger.warning("No valid watch paths exist, waiting 10 seconds before retry...")
                await self._interruptible_sleep(10)
                continue

            invalid_paths = set(self.watch_paths) - set(valid_paths)
            if invalid_paths:
                self.logger.warning(f"Skipping non-existent paths: {invalid_paths}")

            try:
                self.logger.info(f"Starting watch on valid paths: {valid_paths}")
                async for changes in awatch(
                        *valid_paths,
                        watch_filter=self.watch_filter,
                        recursive=self.recursive,
                        debounce=self.debounce,
                        poll_delay_ms=self.poll_delay_ms,
                        stop_event=self._stop_event,
                ):
                    if self._stop_event.is_set():
                        break

                    await self.on_changes(changes)

            except FileNotFoundError as e:
                self.logger.error(f"Watch path no longer exists: {e}, restarting in 10 seconds...")
                if not self._stop_event.is_set():
                    await self._interruptible_sleep(10)

            except Exception as e:
                self.logger.error(f"Error in watch loop: {e}, restarting in 10 seconds...", exc_info=True)
                if not self._stop_event.is_set():
                    await self._interruptible_sleep(10)

    async def on_changes(self, changes: set[tuple[Change, str]]):
        """Hook method to handle file changes."""
        await self._on_changes(changes)
        self.logger.info(f"[{self.__class__.__name__}] on_changes: {changes}")

    @abstractmethod
    async def _on_changes(self, changes: set[tuple[Change, str]]):
        """Callback method to handle file changes.

        Args:
            changes: Set of (Change, path) tuples representing file changes
        """
