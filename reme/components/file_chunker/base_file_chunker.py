"""Abstract base for file chunkers."""

import asyncio
import contextvars
import os
from abc import abstractmethod
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable, TypeVar

from ..base_component import BaseComponent
from ...enumeration import ComponentEnum
from ...schema import FileChunk, FileNode

WorkResult = TypeVar("WorkResult")


async def _finish_work(future: asyncio.Future[WorkResult]) -> WorkResult:
    """Defer cancellation until submitted work finishes, then discard its result."""
    cancellation: asyncio.CancelledError | None = None
    while not future.done():
        try:
            await asyncio.shield(future)
        except asyncio.CancelledError as exc:
            cancellation = exc
        except Exception:
            break
    if cancellation is not None:
        if not future.cancelled():
            future.exception()
        raise cancellation
    return future.result()


class BaseFileChunker(BaseComponent):
    """Abstract base for file chunkers. Subclasses implement `chunk`."""

    component_type = ComponentEnum.FILE_CHUNKER

    def __init__(self, supported_extensions: list[str] | None = None, **kwargs):
        super().__init__(**kwargs)
        self.supported_extensions: list[str] = supported_extensions or []
        self._worker: ThreadPoolExecutor | None = None
        self._worker_lock = asyncio.Lock()
        self._worker_generation = 0
        self._worker_closing = False

    async def _start(self) -> None:
        self._worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix="reme-chunker")
        self._worker_generation += 1
        self._worker_closing = False

    async def _close(self) -> None:
        self._worker_closing = True
        self._worker_generation += 1
        await _finish_work(asyncio.create_task(self._close_worker()))

    async def _close_worker(self) -> None:
        async with self._worker_lock:
            if self._worker is not None:
                # The admission lock is held through completion, so shutdown has no running work.
                self._worker.shutdown(wait=True)
                self._worker = None

    async def _chunk_in_worker(
        self,
        path: str | Path,
        chunk: Callable[[Path, str], tuple[FileNode, list[FileChunk]]],
    ) -> tuple[FileNode, list[FileChunk]]:
        """Admit one parse at a time without accumulating an executor work queue."""
        generation = self._worker_generation
        file_path = Path(path)
        rel_path = self.to_workspace_relative(path)
        async with self._worker_lock:
            if self._worker_closing or generation != self._worker_generation:
                raise RuntimeError("File chunker closed while waiting to parse")
            worker = self._worker
            temporary_worker = worker is None
            if worker is None:
                # Standalone chunk() remains usable without start(), with no persistent threads.
                worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix="reme-chunker")
            try:
                context = contextvars.copy_context()
                future = asyncio.get_running_loop().run_in_executor(
                    worker,
                    context.run,
                    self._chunk_unchanged_file,
                    file_path,
                    rel_path,
                    chunk,
                )
                result, version = await _finish_work(future)
                if self._worker_closing or generation != self._worker_generation:
                    raise RuntimeError("File chunker closed while parsing")
                self._check_file_version(file_path, version, file_path.stat())
                return result
            finally:
                if temporary_worker:
                    worker.shutdown(wait=True)

    @staticmethod
    def _chunk_unchanged_file(
        path: Path,
        rel_path: str,
        chunk: Callable[[Path, str], tuple[FileNode, list[FileChunk]]],
    ) -> tuple[tuple[FileNode, list[FileChunk]], os.stat_result]:
        before = path.stat()
        result = chunk(path, rel_path)
        after = path.stat()
        BaseFileChunker._check_file_version(path, before, after)
        return result, after

    @staticmethod
    def _check_file_version(path: Path, before: os.stat_result, after: os.stat_result) -> None:
        fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
        if any(getattr(before, field) != getattr(after, field) for field in fields):
            raise RuntimeError(f"File changed while chunking: {path}")

    @abstractmethod
    async def chunk(self, path: str | Path) -> tuple[FileNode, list[FileChunk]]:
        """Chunk a file into (node, chunks)."""
