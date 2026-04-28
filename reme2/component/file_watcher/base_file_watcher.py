"""Base file watcher with watchfiles integration."""

import asyncio
from pathlib import Path

from watchfiles import Change, awatch

from ..base_component import BaseComponent
from ..chunk_store import BaseChunkStore
from ..file_parser import BaseFileParser
from ...enumeration import ComponentEnum
from ...schema.file_graph import FileGraph


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
            watch_path: str,
            recursive: bool = False,
            debounce: int = 2000,
            chunk_tokens: int = 400,
            chunk_overlap: int = 80,
            chunk_store: str = "default",
            default_parser: str | None = None,
            rebuild_index_on_start: bool = False,
            poll_delay_ms: int = 2000,
            **kwargs,
    ):
        super().__init__(**kwargs)
        self._chunk_store_name: str = chunk_store
        self._default_parser_name: str | None = default_parser
        self.chunk_store: BaseChunkStore | None = None
        self._suffix_to_parser: dict[str, BaseFileParser] = {}
        self._default_parser: BaseFileParser | None = None
        self.watch_path: str = watch_path
        self.recursive: bool = recursive
        self.debounce: int = debounce
        self.chunk_tokens: int = chunk_tokens
        self.chunk_overlap: int = chunk_overlap
        self.rebuild_index_on_start: bool = rebuild_index_on_start
        self.poll_delay_ms: int = poll_delay_ms

        self.file_graph: FileGraph = FileGraph()
        self._stop_event = asyncio.Event()
        self._watch_task: asyncio.Task | None = None

    _META_DIR = ".reme"

    @property
    def _meta_path(self) -> Path:
        return Path(self.watch_path) / self._META_DIR

    @property
    def _graph_path(self) -> Path:
        return self._meta_path / "file_graph.json"

    async def _start(self):
        """Resolve chunk_store, load or build file_graph, and start watching."""
        if self._chunk_store_name:
            assert self.app_context is not None, "app_context must be provided"

            stores = self.app_context.components.get(ComponentEnum.CHUNK_STORE, {})
            if self._chunk_store_name not in stores:
                raise ValueError(f"Chunk store '{self._chunk_store_name}' not found.")
            store = stores[self._chunk_store_name]
            if not isinstance(store, BaseChunkStore):
                raise TypeError(f"Expected BaseChunkStore, got {type(store).__name__}")
            self.chunk_store = store

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
            await self._load_or_build_graph()
            await self._watch_loop()

        self._stop_event.clear()
        self._watch_task = asyncio.create_task(_initialize_and_watch())
        self.logger.info(f"Started watching: {self.watch_path}")

    async def _load_or_build_graph(self) -> None:
        graph_path = self._graph_path
        if not self.rebuild_index_on_start and graph_path.exists():
            self.file_graph = FileGraph.load(graph_path)
            self.logger.info(f"Loaded file graph from {graph_path} ({len(self.file_graph)} nodes)")
        else:
            if self.rebuild_index_on_start:
                self.logger.info("Rebuild on start enabled, scanning to build")
            else:
                self.logger.info("No persisted file graph found, scanning to build")
            self.file_graph = FileGraph()
            await self._scan_existing_files()
            self.file_graph.save(graph_path)
            self.logger.info(f"Saved file graph to {graph_path} ({len(self.file_graph)} nodes)")

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
        self.chunk_store = None
        self._suffix_to_parser.clear()
        self._default_parser = None
        self.logger.info("Stopped watching")

    async def _scan_existing_files(self) -> None:
        if not self.chunk_store:
            return

        watch_path = Path(self.watch_path)
        if not watch_path.exists():
            self.logger.warning(f"Watch path does not exist: {watch_path}")
            return

        existing_files: set[tuple[Change, str]] = set()
        if watch_path.is_file():
            existing_files.add((Change.added, str(watch_path)))
        elif watch_path.is_dir():
            iterator = watch_path.rglob("*") if self.recursive else watch_path.iterdir()
            for file_path in iterator:
                if file_path.is_file() and self._watch_filter(str(file_path)):
                    existing_files.add((Change.added, str(file_path)))

        if existing_files:
            self.logger.info(f"Scanning {len(existing_files)} existing files")
            await self.on_changes(existing_files)
        else:
            self.logger.info("No existing files found")

    async def _interruptible_sleep(self, seconds: float):
        """Sleep that can be interrupted by stop_event."""
        try:
            await asyncio.wait_for(self._stop_event.wait(), timeout=seconds)
        except asyncio.TimeoutError:
            pass

    async def _watch_loop(self):
        while not self._stop_event.is_set():
            if not Path(self.watch_path).exists():
                self.logger.warning(f"Watch path does not exist: {self.watch_path}, waiting 10 seconds...")
                await self._interruptible_sleep(10)
                continue

            try:
                self.logger.info(f"Starting watch on: {self.watch_path}")
                async for changes in awatch(
                        self.watch_path,
                        watch_filter=lambda _, p: self._watch_filter(p),
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

    def _watch_filter(self, path: str) -> bool:
        return self._meta_path not in Path(path).parents

    def _get_parser(self, path: str) -> BaseFileParser | None:
        suffix = Path(path).suffix.lower()
        return self._suffix_to_parser.get(suffix, self._default_parser)

    async def on_changes(self, changes: set[tuple[Change, str]]) -> None:
        if not self.chunk_store:
            self.logger.warning("File store not initialized, skipping changes")
            return

        for change_type, path in changes:
            try:
                if change_type == Change.added:
                    await self._on_added(path)
                elif change_type == Change.modified:
                    await self._on_modified(path)
                elif change_type == Change.deleted:
                    await self._on_deleted(path)
            except FileNotFoundError:
                self.logger.warning(f"File not found: {path}, skipping")
            except PermissionError:
                self.logger.warning(f"Permission denied: {path}, skipping")
            except Exception as e:
                self.logger.error(f"Error processing {path}: {e}", exc_info=True)

    async def _on_added(self, path: str) -> None:
        parser = self._get_parser(path)
        if not parser:
            self.logger.debug(f"No parser for {path}, skipping")
            return
        file_meta, chunks = await parser.parse(path)
        await self.chunk_store.upsert_chunks(path, chunks)
        self.file_graph.create(file_meta)
        self.logger.info(f"Added {path} ({len(chunks)} chunks)")

    async def _on_modified(self, path: str) -> None:
        parser = self._get_parser(path)
        if not parser:
            self.logger.debug(f"No parser for {path}, skipping")
            return
        file_meta, chunks = await parser.parse(path)
        await self.chunk_store.upsert_chunks(path, chunks)
        self.file_graph.create(file_meta)
        self.logger.info(f"Modified {path} ({len(chunks)} chunks)")

    async def _on_deleted(self, path: str) -> None:
        await self.chunk_store.delete_chunks(path)
        self.file_graph.delete(path)
        self.logger.info(f"Deleted {path}")
