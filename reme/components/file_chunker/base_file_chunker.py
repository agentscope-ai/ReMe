"""Abstract base for file chunkers."""

import asyncio
import contextvars
import os
import threading
from abc import abstractmethod
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable

from ..base_component import BaseComponent
from ...enumeration import ComponentEnum
from ...schema import FileChunk, FileNode
from ...utils.async_utils import wait_for_completion
from ...utils.file_utils import check_file_version


class BaseFileChunker(BaseComponent):
    """Abstract base for file chunkers. Subclasses implement `chunk`."""

    component_type = ComponentEnum.FILE_CHUNKER

    def __init__(self, supported_extensions: list[str] | None = None, **kwargs):
        super().__init__(**kwargs)
        self.supported_extensions: list[str] = supported_extensions or []
        self._worker: ThreadPoolExecutor | None = None
        self._worker_lock = threading.Lock()
        self._worker_state_lock = threading.Lock()
        self._worker_generation = 0
        self._worker_closing = False

    async def _start(self) -> None:
        await self._acquire_worker_lock()
        try:
            with self._worker_state_lock:
                self._worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix="reme-chunker")
                self._worker_generation += 1
                self._worker_closing = False
        finally:
            self._worker_lock.release()

    async def _close(self) -> None:
        with self._worker_state_lock:
            self._worker_closing = True
            self._worker_generation += 1
        await wait_for_completion(asyncio.create_task(self._close_worker()))

    async def _close_worker(self) -> None:
        await self._acquire_worker_lock()
        try:
            with self._worker_state_lock:
                self._worker_closing = True
                self._worker_generation += 1
                worker = self._worker
                self._worker = None
            if worker is not None:
                worker.shutdown(wait=True)
        finally:
            self._worker_lock.release()

    async def _acquire_worker_lock(self) -> None:
        """Acquire cross-loop admission without blocking an event loop thread."""
        # Ownership spans awaited work, so a synchronous lock context would block this loop.
        while not self._worker_lock.acquire(blocking=False):  # pylint: disable=consider-using-with
            await asyncio.sleep(0.01)

    async def _chunk_in_worker(
        self,
        path: str | Path,
        chunk: Callable[[Path, str], tuple[FileNode, list[FileChunk]]],
    ) -> tuple[FileNode, list[FileChunk]]:
        """Admit one parse at a time without accumulating an executor work queue."""
        with self._worker_state_lock:
            generation = self._worker_generation
        file_path = Path(path)
        rel_path = self.to_workspace_relative(path)
        await self._acquire_worker_lock()
        temporary_worker = False
        try:
            with self._worker_state_lock:
                if self._worker_closing or generation != self._worker_generation:
                    raise RuntimeError("File chunker closed while waiting to parse")
                worker = self._worker
                temporary_worker = worker is None
            if temporary_worker:
                # Standalone chunk() remains usable without start(), with no persistent threads.
                with ThreadPoolExecutor(max_workers=1, thread_name_prefix="reme-chunker") as worker:
                    return await self._run_chunk(worker, generation, file_path, rel_path, chunk)
            assert worker is not None
            return await self._run_chunk(worker, generation, file_path, rel_path, chunk)
        finally:
            self._worker_lock.release()

    async def _run_chunk(
        self,
        worker: ThreadPoolExecutor,
        generation: int,
        file_path: Path,
        rel_path: str,
        chunk: Callable[[Path, str], tuple[FileNode, list[FileChunk]]],
    ) -> tuple[FileNode, list[FileChunk]]:
        context = contextvars.copy_context()
        future = asyncio.get_running_loop().run_in_executor(
            worker,
            context.run,
            self._chunk_unchanged_file,
            file_path,
            rel_path,
            chunk,
        )
        result, version = await wait_for_completion(future)
        with self._worker_state_lock:
            if self._worker_closing or generation != self._worker_generation:
                raise RuntimeError("File chunker closed while parsing")
        check_file_version(file_path, version, file_path.stat())
        return result

    @staticmethod
    def _chunk_unchanged_file(
        path: Path,
        rel_path: str,
        chunk: Callable[[Path, str], tuple[FileNode, list[FileChunk]]],
    ) -> tuple[tuple[FileNode, list[FileChunk]], os.stat_result]:
        before = path.stat()
        result = chunk(path, rel_path)
        after = path.stat()
        check_file_version(path, before, after)
        return result, after

    @abstractmethod
    async def chunk(self, path: str | Path) -> tuple[FileNode, list[FileChunk]]:
        """Chunk a file into (node, chunks)."""
