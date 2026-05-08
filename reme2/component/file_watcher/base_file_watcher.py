"""Base file watcher with watchfiles integration.

Per change → single-step pipeline:
  added/modified → existing = file_store.get_chunks(path)
                   parsed   = parser.parse(path, existing_chunks=existing)
                   file_store.upsert_parsed(parsed)
  deleted        → file_store.delete_chunks + delete_file_meta

The parser owns: chunking, edge extraction, and embedding (with hash-diff
cache via the `existing_chunks` argument). The file_store owns: persistence
of meta + edges + chunks (atomic-from-the-caller's-POV via `upsert_parsed`).
The watcher owns: scheduling, cancel-on-modify, retry/timeout, and
startup recovery (re-parsing files whose mtime drifted while offline).
"""

import asyncio
import time
from pathlib import Path

from watchfiles import Change, awatch

from ..base_component import BaseComponent
from ..file_parser import BaseFileParser
from ..file_store import BaseFileStore
from ...enumeration import ComponentEnum


class BaseFileWatcher(BaseComponent):
    """Watches a directory and feeds the file_store via single-step parse+upsert."""

    component_type = ComponentEnum.FILE_WATCHER

    _META_DIR = ".reme"

    def __init__(
        self,
        watch_path: str,
        recursive: bool = False,
        debounce: int = 2000,
        file_store: str = "default",
        default_parser: str | None = None,
        rebuild_index_on_start: bool = False,
        poll_delay_ms: int = 2000,
        parse_max_attempts: int = 3,
        parse_retry_backoff: float = 2.0,
        parse_task_timeout: float = 300.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._file_store_name: str = file_store
        self._default_parser_name: str | None = default_parser
        self.file_store: BaseFileStore | None = None
        self._suffix_to_parser: dict[str, BaseFileParser] = {}
        self._default_parser: BaseFileParser | None = None
        self.watch_path: str = watch_path
        self.recursive: bool = recursive
        self.debounce: int = debounce
        self.rebuild_index_on_start: bool = rebuild_index_on_start
        self.poll_delay_ms: int = poll_delay_ms

        self._stop_event = asyncio.Event()
        self._watch_task: asyncio.Task | None = None

        # Parse pipeline configuration.
        self._parse_max_attempts: int = max(1, int(parse_max_attempts))
        self._parse_retry_backoff: float = float(parse_retry_backoff)
        self._parse_task_timeout: float = float(parse_task_timeout)
        self._tasks: dict[str, asyncio.Task] = {}
        self._mgmt_lock: asyncio.Lock | None = None
        self._run_lock: asyncio.Lock | None = None
        self._last_failure: dict[str, float] = {}

    @property
    def _meta_path(self) -> Path:
        return (Path(self.watch_path) / self._META_DIR).resolve()

    # -- Lifecycle ----------------------------------------------------------

    async def _start(self):
        """Resolve file_store + parsers, sync any disk drift, start watch loop."""
        if self._file_store_name:
            assert self.app_context is not None, "app_context must be provided"

            stores = self.app_context.components.get(ComponentEnum.FILE_STORE, {})
            if self._file_store_name not in stores:
                raise ValueError(f"File store '{self._file_store_name}' not found.")
            store = stores[self._file_store_name]
            if not isinstance(store, BaseFileStore):
                raise TypeError(f"Expected BaseFileStore, got {type(store).__name__}")
            self.file_store = store
            # Vault root drives explicit-path wikilink resolution.
            self.file_store.set_vault_root(self.watch_path)

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
            await self._initial_sync_and_recovery()
            await self._watch_loop()

        self._mgmt_lock = asyncio.Lock()
        self._run_lock = asyncio.Lock()
        self._last_failure.clear()
        self._stop_event.clear()
        self._watch_task = asyncio.create_task(_initialize_and_watch())
        self.logger.info(f"Started watching: {self.watch_path}")

    async def _close(self):
        """Stop watching and release resources."""
        self._stop_event.set()
        if self._watch_task and not self._watch_task.done():
            self._watch_task.cancel()
            try:
                await self._watch_task
            except asyncio.CancelledError:
                pass

        # Cancel all in-flight parse tasks before tearing down state.
        tasks = list(self._tasks.values())
        for t in tasks:
            if not t.done():
                t.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self._tasks.clear()
        self._last_failure.clear()

        self._watch_task = None
        self._stop_event.clear()
        self._mgmt_lock = None
        self._run_lock = None
        self.file_store = None
        self._suffix_to_parser.clear()
        self._default_parser = None
        self.logger.info("Stopped watching")

    # -- Startup sync -------------------------------------------------------

    async def _initial_sync_and_recovery(self) -> None:
        """At startup: file_store has its persisted state loaded already; we
        diff against current disk for any changes that happened while we
        were offline, then re-enqueue them through the parse pipeline."""
        await self._sync_with_disk()

    async def _sync_with_disk(self) -> bool:
        """Diff cached graph (in file_store) vs disk state; reindex changes."""
        if self.file_store is None:
            return False

        watch_path = Path(self.watch_path)
        if not watch_path.exists():
            self.logger.warning(f"Watch path does not exist: {watch_path}")
            return False

        on_disk: dict[str, float] = {}
        if watch_path.is_file():
            on_disk[str(watch_path.resolve())] = watch_path.stat().st_mtime
        elif watch_path.is_dir():
            iterator = watch_path.rglob("*") if self.recursive else watch_path.iterdir()
            for fp in iterator:
                if not fp.is_file():
                    continue
                abs_path = str(fp.resolve())
                if not self._watch_filter(abs_path):
                    continue
                if not self._get_parser(abs_path):
                    continue
                try:
                    on_disk[abs_path] = fp.stat().st_mtime
                except OSError:
                    continue

        cached_paths = set(self.file_store.nodes.keys())
        disk_paths = set(on_disk.keys())

        added = disk_paths - cached_paths
        deleted = cached_paths - disk_paths
        modified: set[str] = set()
        for p in disk_paths & cached_paths:
            cached_meta = self.file_store.get_file_meta(p)
            if cached_meta is None or cached_meta.st_mtime != on_disk[p]:
                modified.add(p)

        unchanged = len(disk_paths & cached_paths) - len(modified)
        if not (added or deleted or modified):
            self.logger.info(f"Index up-to-date ({unchanged} files cached, 0 changes)")
            return False

        self.logger.info(
            f"Incremental sync: +{len(added)} ~{len(modified)} -{len(deleted)} (unchanged {unchanged})",
        )

        changes: set[tuple[Change, str]] = set()
        for p in deleted:
            changes.add((Change.deleted, p))
        for p in added:
            changes.add((Change.added, p))
        for p in modified:
            changes.add((Change.modified, p))
        await self.on_changes(changes)
        return True

    # -- Watch loop ---------------------------------------------------------

    async def _interruptible_sleep(self, seconds: float):
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
        resolved = Path(path).resolve()
        if resolved == self._meta_path:
            return False
        return self._meta_path not in resolved.parents

    def _get_parser(self, path: str) -> BaseFileParser | None:
        suffix = Path(path).suffix.lower()
        return self._suffix_to_parser.get(suffix, self._default_parser)

    # -- Change dispatch ----------------------------------------------------

    async def on_changes(self, changes: set[tuple[Change, str]]) -> None:
        if self.file_store is None:
            self.logger.warning("File store not initialized, skipping changes")
            return

        for change_type, path in changes:
            try:
                if change_type in (Change.added, Change.modified):
                    await self._on_modified(path)
                elif change_type == Change.deleted:
                    await self._on_deleted(path)
            except FileNotFoundError:
                self.logger.warning(f"File not found: {path}, skipping")
            except PermissionError:
                self.logger.warning(f"Permission denied: {path}, skipping")
            except Exception as e:
                self.logger.opt(exception=True).error("Error processing {p}: {err}", p=path, err=str(e))

    async def _on_modified(self, path: str) -> None:
        if not Path(path).is_file():
            return
        parser = self._get_parser(path)
        if parser is None:
            self.logger.debug(f"No parser for {path}, skipping")
            return
        await self._submit_parse_task(path, parser)

    async def _on_deleted(self, path: str) -> None:
        # Cancel any in-flight parse task FIRST so a delayed cancellation
        # can't race the cleanup writes below.
        assert self.file_store is not None, "_on_deleted requires file_store"
        await self._cancel_parse_task(path)
        await self.file_store.delete_chunks(path)
        await self.file_store.delete_file_meta(path)
        self.logger.info(f"Deleted {path}")

    # -- Parse task pipeline ------------------------------------------------

    async def _submit_parse_task(self, path: str, parser: BaseFileParser) -> None:
        """Cancel any prior task + spawn a new one. Atomic under _mgmt_lock."""
        assert self._mgmt_lock is not None, "_submit_parse_task before _start()"
        async with self._mgmt_lock:
            await self._cancel_locked(path)
            self._tasks[path] = asyncio.create_task(
                self._run_parse_task(path, parser),
            )

    async def _cancel_parse_task(self, path: str) -> None:
        """Cancel + await any in-flight parse task for `path`."""
        assert self._mgmt_lock is not None, "_cancel_parse_task before _start()"
        async with self._mgmt_lock:
            await self._cancel_locked(path)

    async def _cancel_locked(self, path: str) -> None:
        """Cancel + await the task for `path`. Caller holds `_mgmt_lock`."""
        task = self._tasks.pop(path, None)
        if task is not None and not task.done():
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass

    async def flush_parse_tasks(self) -> None:
        """Wait until all parse tasks finish (loops because retries spawn new ones)."""
        while self._tasks:
            await asyncio.gather(*list(self._tasks.values()), return_exceptions=True)

    def pending_parse_count(self) -> int:
        return sum(1 for t in self._tasks.values() if not t.done())

    def failed_parse_paths(self) -> dict[str, float]:
        return dict(self._last_failure)

    async def _run_parse_task(self, path: str, parser: BaseFileParser) -> None:
        """One per-path parse task: get prior chunks → parse → upsert_parsed.

        Hash diff happens inside the parser via `existing_chunks`: chunks
        whose hash matches a stored chunk reuse the prior embedding; only
        new-hash blocks hit the embedding API. The whole upsert is one
        call to the file_store.
        """
        try:
            for attempt in range(1, self._parse_max_attempts + 1):
                try:
                    assert self.file_store is not None
                    existing = await self.file_store.get_chunks(path)
                    parsed = await asyncio.wait_for(
                        parser.parse(path, existing_chunks=existing),
                        timeout=self._parse_task_timeout,
                    )

                    assert self._run_lock is not None
                    async with self._run_lock:
                        await asyncio.wait_for(
                            self.file_store.upsert_parsed(parsed),
                            timeout=self._parse_task_timeout,
                        )
                        self._last_failure.pop(path, None)

                    self.logger.info(
                        f"indexed {path}: chunks={len(parsed.chunks)} edges={len(parsed.edges)}",
                    )
                    return
                except asyncio.CancelledError:
                    raise
                except FileNotFoundError:
                    self.logger.debug(f"parse task {path}: file gone, skipping")
                    self._last_failure.pop(path, None)
                    return
                except Exception as e:
                    self._last_failure[path] = time.time()
                    if attempt < self._parse_max_attempts:
                        backoff = self._parse_retry_backoff * attempt
                        self.logger.warning(
                            f"parse task {path} attempt {attempt} failed "
                            f"({type(e).__name__}: {e}); retry in {backoff:.1f}s",
                        )
                        await asyncio.sleep(backoff)
                        continue
                    self.logger.error(
                        f"parse task {path} giving up after {attempt} attempts: "
                        f"{type(e).__name__}: {e}",
                    )
                    return
        finally:
            if self._tasks.get(path) is asyncio.current_task():
                self._tasks.pop(path, None)
