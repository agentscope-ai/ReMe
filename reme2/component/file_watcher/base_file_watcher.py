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
        self.watch_paths: list[Path] = [self.working_path / x for x in watch_paths if (self.working_path / x).exists()]
        self.suffix_filters: list[str] = suffix_filters or ["md"]
        self.recursive: bool = recursive
        self.force_polling: bool = force_polling
        self.debounce: int = debounce
        self.poll_delay_ms: int = poll_delay_ms
        self.file_store_name: str = file_store
        self.file_parser_name: str = file_parser
        self._stop_event: asyncio.Event = asyncio.Event()
        self._background_task: asyncio.Task | None = None
        self.file_store: BaseFileStore | None = None
        self.file_parser: BaseFileParser | None = None
        self._retry_interval: float = 10

    async def _start(self):
        self.file_store = self.get_component(ComponentEnum.FILE_STORE, self.file_store_name)
        self.file_parser = self.get_component(ComponentEnum.FILE_PARSER, self.file_parser_name)

        async def background_task():
            await self.update_store()
            await self.watch_loop()

        self._background_task = asyncio.create_task(background_task())
        logger.info(f"Started watching: {self.watch_paths}")

    async def _close(self):
        self._stop_event.set()
        if self._background_task:
            await self._background_task
        logger.info("Stopped watching")

    def watch_filter(self, _change: Change, path: str) -> bool:
        if not self.suffix_filters:
            return True
        return any(path.endswith("." + s.strip(".")) for s in self.suffix_filters)

    async def scan_existing_files(self) -> list[Path]:
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
        if self.file_store is None:
            raise ValueError("file_store is not initialized!")
        await self.file_store.clear()

    async def reset_store(self):
        if self.file_store is None:
            raise ValueError("file_store is not initialized!")
        await self.file_store.clear()
        await self.on_added(await self.scan_existing_files())

    @abstractmethod
    async def watch_loop(self):
        """Watch for file changes and update the store."""

    @abstractmethod
    async def update_store(self):
        """Update the store with the latest file changes."""

    @abstractmethod
    async def on_added(self, path: Path | list[Path]):
        """Handle file added event."""

    @abstractmethod
    async def on_modified(self, path: Path | list[Path]):
        """Handle file modified event."""

    @abstractmethod
    async def on_deleted(self, path: Path | list[Path]):
        """Handle file deleted event."""
