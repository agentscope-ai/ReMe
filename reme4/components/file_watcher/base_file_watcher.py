"""Abstract base for file watchers."""

import asyncio
from abc import abstractmethod
from pathlib import Path

from watchfiles import Change

from ..base_component import BaseComponent
from ..file_parser import BaseFileParser
from ..file_store import BaseFileStore
from ...enumeration import ComponentEnum
from ...utils import get_logger

logger = get_logger()


class BaseFileWatcher(BaseComponent):
    """Abstract base for file watchers. Subclasses implement watch_loop and event handlers."""

    component_type = ComponentEnum.FILE_WATCHER

    def __init__(
        self,
        watch_paths: list[str] | str,
        suffix_filters: list[str] | None = None,
        recursive: bool = True,
        force_polling: bool = True,
        debounce: int = 2000,
        poll_delay_ms: int = 2000,
        file_store: str = "default",
        file_parser: str = "default",
        **kwargs,
    ):
        super().__init__(**kwargs)
        watch_paths = [watch_paths] if isinstance(watch_paths, str) else watch_paths
        base = self.working_path
        self.watch_paths: list[Path] = [base / x for x in watch_paths if (base / x).exists()]
        self.suffix_filters: list[str] = suffix_filters or ["md"]
        self.recursive: bool = recursive
        self.force_polling: bool = force_polling
        self.debounce: int = debounce
        self.poll_delay_ms: int = poll_delay_ms
        self.file_store = self.bind(file_store, BaseFileStore)
        self.file_parser = self.bind(file_parser, BaseFileParser)
        self._stop_event: asyncio.Event = asyncio.Event()
        self._background_task: asyncio.Task | None = None
        self._retry_interval: float = 10

    async def _start(self):
        self._stop_event = asyncio.Event()
        self._background_task = asyncio.create_task(self._background_run())
        logger.info(f"Started watching: {self.watch_paths}")

    async def _background_run(self):
        """Sync store then enter watch loop."""
        await self.update_store()
        await self.watch_loop()

    async def _close(self):
        self._stop_event.set()
        if self._background_task:
            await self._background_task
        logger.info("Stopped watching")

    def watch_filter(self, _change: Change, path: str) -> bool:
        """Return True if the file suffix matches the filter list."""
        if not self.suffix_filters:
            return True
        return any(path.endswith("." + s.strip(".")) for s in self.suffix_filters)

    async def scan_existing_files(self) -> list[Path]:
        """Collect all watchable files under watch_paths."""
        files: list[Path] = []
        for path in self.watch_paths:
            if not path.exists():
                continue
            if path.is_file():
                if self.watch_filter(Change.added, str(path)):
                    files.append(path)
            else:
                items = path.rglob("*") if self.recursive else path.iterdir()
                files.extend(p for p in items if p.is_file() and self.watch_filter(Change.added, str(p)))
        return files

    async def clear_store(self):
        """Remove all entries from the file store."""
        if self.file_store is None:
            raise ValueError("file_store is not initialized!")
        await self.file_store.clear()

    async def reset_store(self):
        """Clear the store and re-index all existing files."""
        if self.file_store is None:
            raise ValueError("file_store is not initialized!")
        await self.file_store.clear()
        await self.on_added(await self.scan_existing_files())

    @abstractmethod
    async def watch_loop(self):
        """Watch for file changes and dispatch events."""

    @abstractmethod
    async def update_store(self):
        """Sync the store with the current state of watch_paths."""

    @abstractmethod
    async def on_added(self, path: Path | list[Path]):
        """Handle file added event."""

    @abstractmethod
    async def on_modified(self, path: Path | list[Path]):
        """Handle file modified event."""

    @abstractmethod
    async def on_deleted(self, path: Path | list[Path]):
        """Handle file deleted event."""
