"""Base file watcher with watchfiles integration."""

import asyncio
from abc import abstractmethod
from pathlib import Path

from watchfiles import Change, awatch

from ..base_component import BaseComponent
from ..file_parser import BaseFileParser
from ..file_store import BaseFileStore
from ...enumeration import ComponentEnum


class BaseFileWatcher(BaseComponent):
    """Abstract base class for file watchers.

    Provides file monitoring with:
        - watchfiles integration for efficient change detection
        - Parser-based file filtering
        - Auto-restart on failure
        - Optional index rebuild on start
    """

    component_type = ComponentEnum.FILE_WATCHER

    def __init__(
        self,
        watch_paths: list[str] | str,
        recursive: bool = False,
        debounce: int = 2000,
        chunk_tokens: int = 400,
        chunk_overlap: int = 80,
        file_store: str = "default",
        default_parser: str | None = None,
        rebuild_index_on_start: bool = True,
        poll_delay_ms: int = 2000,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._file_store_name: str = file_store
        self._default_parser_name: str | None = default_parser
        self.file_store: BaseFileStore | None = None
        self._suffix_to_parser: dict[str, BaseFileParser] = {}
        self._default_parser: BaseFileParser | None = None
        self.watch_paths: list[str] = [watch_paths] if isinstance(watch_paths, str) else watch_paths
        self.recursive: bool = recursive
        self.debounce: int = debounce
        self.chunk_tokens: int = chunk_tokens
        self.chunk_overlap: int = chunk_overlap
        self.rebuild_index_on_start: bool = rebuild_index_on_start
        self.poll_delay_ms: int = poll_delay_ms

        self._stop_event = asyncio.Event()
        self._watch_task: asyncio.Task | None = None

    async def _start(self):
        """Resolve file_store and start watching task."""
        if self._file_store_name:
            assert self.app_context is not None, "app_context must be provided"

            stores = self.app_context.components.get(ComponentEnum.FILE_STORE, {})
            if self._file_store_name not in stores:
                raise ValueError(f"File store '{self._file_store_name}' not found.")
            store = stores[self._file_store_name]
            if not isinstance(store, BaseFileStore):
                raise TypeError(f"Expected BaseFileStore, got {type(store).__name__}")
            self.file_store = store

            parsers = self.app_context.components.get(ComponentEnum.FILE_PARSER, {})
            for parser in parsers.values():
                if isinstance(parser, BaseFileParser):
                    for suffix in parser.suffixes:
                        self._suffix_to_parser[suffix] = parser

            if self._default_parser_name and self._default_parser_name in parsers:
                parser = parsers[self._default_parser_name]
                if isinstance(parser, BaseFileParser):
                    self._default_parser = parser

            if not self._suffix_to_parser and not self._default_parser:
                self.logger.warning("No file parsers registered")

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
        """Stop watching and release resources."""
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
        self._suffix_to_parser.clear()
        self._default_parser = None
        self.logger.info("Stopped watching")

    async def _scan_existing_files(self):
        """Scan existing files and add them as Change.added."""
        if not self.file_store:
            return

        existing_files: set[tuple[Change, str]] = set()

        for watch_path_str in self.watch_paths:
            watch_path = Path(watch_path_str)

            if not watch_path.exists():
                self.logger.warning(f"Watch path does not exist: {watch_path}")
                continue

            if watch_path.is_file():
                existing_files.add((Change.added, str(watch_path)))
            elif watch_path.is_dir():
                iterator = watch_path.rglob("*") if self.recursive else watch_path.iterdir()
                for file_path in iterator:
                    if file_path.is_file():
                        existing_files.add((Change.added, str(file_path)))

        if existing_files:
            self.logger.info(f"[SCAN_ON_START] Found {len(existing_files)} existing files")
            await self.on_changes(existing_files)
            self.logger.info(f"[SCAN_ON_START] Added {len(existing_files)} files to memory store")
        else:
            self.logger.info("[SCAN_ON_START] No existing files found")

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
                self.logger.warning("No valid watch paths exist, waiting 10 seconds...")
                await self._interruptible_sleep(10)
                continue

            invalid_paths = set(self.watch_paths) - set(valid_paths)
            if invalid_paths:
                self.logger.warning(f"Skipping non-existent paths: {invalid_paths}")

            try:
                self.logger.info(f"Starting watch on: {valid_paths}")
                async for changes in awatch(
                    *valid_paths,
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
        """Hook method for handling file changes."""
        await self._on_changes(changes)
        self.logger.info(f"[{self.__class__.__name__}] on_changes: {changes}")

    @abstractmethod
    async def _on_changes(self, changes: set[tuple[Change, str]]):
        """Handle file changes. Override in subclasses."""
