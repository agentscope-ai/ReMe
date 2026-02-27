"""Base file watcher implementation.

This module provides the base class for file watcher implementations
that monitor file system changes and trigger callbacks.
"""

import asyncio
from collections.abc import Coroutine
from pathlib import Path
from typing import Any, Callable

from loguru import logger
from watchfiles import awatch, Change

from ..enumeration import MemorySource
from ..file_store import BaseFileStore


class BaseFileWatcher:
    """
    Minimal file watcher base class

    This base class provides basic file monitoring functionality that can be extended
    to implement specific file monitoring requirements.
    """

    def __init__(
        self,
        watch_paths: list[str] | str,
        suffix_filters: list[str] | None = None,
        recursive: bool = False,
        debounce: int = 500,  # Millisecond debounce
        chunk_tokens: int = 400,
        chunk_overlap: int = 80,
        file_store: BaseFileStore | None = None,
        callback: Callable[[set[tuple[Change, str]]], None | Coroutine[Any, Any, None]] | None = None,
        scan_on_start: bool = False,
        **kwargs,
    ):
        """
        Initialize the file watcher

        Args:
            watch_paths: Paths to watch for changes
            suffix_filters: File suffix filters (e.g., ['.py', '.txt'])
            recursive: Whether to watch directories recursively
            debounce: Debounce time in milliseconds
            chunk_tokens: Token size for chunking
            chunk_overlap: Overlap size for chunks
            file_store: File store instance
            callback: Callback function for changes
            scan_on_start: If True, scan existing files on start and trigger on_changes with Change.added
            **kwargs: Additional keyword arguments
        """
        self.watch_paths: list[str] = [watch_paths] if isinstance(watch_paths, str) else watch_paths
        self.suffix_filters: list[str] = suffix_filters or []
        self.recursive: bool = recursive
        self.debounce: int = debounce
        self.chunk_tokens: int = chunk_tokens
        self.chunk_overlap: int = chunk_overlap
        self.file_store: BaseFileStore = file_store
        self.callback = callback
        self.scan_on_start: bool = scan_on_start
        self.kwargs: dict = kwargs

        self._stop_event = asyncio.Event()
        self._watch_task: asyncio.Task | None = None
        self._running = False

    async def start(self):
        """Start the file watcher"""
        if self._running:
            return

        self._running = True

        # Scan existing files if requested
        if self.scan_on_start:
            await self._scan_existing_files()

        self._watch_task = asyncio.create_task(self._watch_loop())
        logger.info(f"Started watching: {self.watch_paths}")

    async def close(self):
        """Stop the file watcher"""
        if not self._running:
            return

        self._stop_event.set()
        if self._watch_task:
            await self._watch_task
        self._running = False
        logger.info("Stopped watching")

    def watch_filter(self, _change: Change, path: str) -> bool:
        """Filter function for file watching."""
        # If no suffix filters are specified, watch all files
        if not self.suffix_filters:
            return True

        # Check if the file has one of the allowed suffixes
        for suffix in self.suffix_filters:
            if path.endswith("." + suffix.strip(".")):
                return True

        return False

    async def _scan_existing_files(self):
        """Scan existing files matching watch criteria and trigger on_changes with Change.added"""
        existing_files: set[tuple[Change, str]] = set()

        for watch_path_str in self.watch_paths:
            watch_path = Path(watch_path_str)

            if not watch_path.exists():
                logger.warning(f"Watch path does not exist: {watch_path}")
                continue

            if watch_path.is_file():
                # Single file
                if self.watch_filter(Change.added, str(watch_path)):
                    existing_files.add((Change.added, str(watch_path)))
            elif watch_path.is_dir():
                # Directory
                if self.recursive:
                    # Recursive scan
                    for file_path in watch_path.rglob("*"):
                        if file_path.is_file() and self.watch_filter(Change.added, str(file_path)):
                            existing_files.add((Change.added, str(file_path)))
                else:
                    # Non-recursive scan (only immediate children)
                    for file_path in watch_path.iterdir():
                        if file_path.is_file() and self.watch_filter(Change.added, str(file_path)):
                            existing_files.add((Change.added, str(file_path)))

        if existing_files:
            logger.info(f"[SCAN_ON_START] Found {len(existing_files)} existing files matching watch criteria")
            await self.on_changes(existing_files)
            logger.info(f"[SCAN_ON_START] Added {len(existing_files)} files to memory store")
        else:
            logger.info("[SCAN_ON_START] No existing files found matching watch criteria")

        files: list[str] = await self.file_store.list_files(MemorySource.MEMORY)
        for file_path in files:
            chunks = await self.file_store.get_file_chunks(file_path, MemorySource.MEMORY)
            logger.info(f"Found existing file: {file_path}, {len(chunks)} chunks")

    async def _watch_loop(self):
        """Core monitoring loop"""
        if not self.watch_paths:
            logger.warning("No watch paths specified")
            return

        try:
            async for changes in awatch(
                *self.watch_paths,
                watch_filter=self.watch_filter,
                recursive=self.recursive,
                debounce=self.debounce,
                stop_event=self._stop_event,
            ):
                if self._stop_event.is_set():
                    break

                await self.on_changes(changes)
        except FileNotFoundError as e:
            # Watch path was deleted, this is expected during cleanup
            logger.debug(f"Watch path no longer exists: {e}")
        except Exception as e:
            # Log other exceptions but don't crash
            logger.error(f"Error in watch loop: {e}", exc_info=True)

    async def _on_changes(self, changes: set[tuple[Change, str]]):
        """Callback method to handle file changes"""

    async def on_changes(self, changes: set[tuple[Change, str]]):
        """Hook method to handle file changes"""
        if self.callback:
            result = self.callback(changes)
            if asyncio.iscoroutine(result):
                await result
        else:
            await self._on_changes(changes)
        logger.info(f"[{self.__class__.__name__}] on_changes: {changes}")

    def is_running(self) -> bool:
        """Check if the watcher is running"""
        return self._running

    async def add_path(self, path: str):
        """Dynamically add a path to monitor"""
        if path not in self.watch_paths:
            self.watch_paths.append(path)
            if self._running:
                await self.close()
                await self.start()

    async def remove_path(self, path: str):
        """Remove a monitored path"""
        if path in self.watch_paths:
            self.watch_paths.remove(path)
            if self._running:
                await self.close()
                await self.start()
