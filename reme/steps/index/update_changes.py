"""Apply bounded file change batches to file_catalog or file_store."""

import asyncio
import math
from abc import abstractmethod
from pathlib import Path
from typing import Any, Callable

import psutil
from watchfiles import Change

from ._change_batch import bucket_changes
from ..base_step import BaseStep
from ...components import R
from ...components.file_chunker import BaseFileChunker
from ...enumeration import ComponentEnum
from ...schema import FileChunk, FileNode

DEFAULT_BATCH_MAX_FILES = 100
DEFAULT_BATCH_AVAILABLE_MEMORY_RATIO = 0.10
DEFAULT_BATCH_MEMORY_TARGET_BYTES = 512 * 1024 * 1024
DEFAULT_BATCH_MEMORY_EXPANSION_FACTOR = 8.0
_FILE_MEMORY_OVERHEAD_BYTES = 32 * 1024
_CHUNK_MEMORY_OVERHEAD_BYTES = 4 * 1024
_ESTIMATED_CHUNK_BYTES = 10_000
_FLOAT16_BYTES = 2


class ChangeApplyStep(BaseStep):
    """Shared added/modified/deleted handling for index update targets."""

    target_name = "target"
    reads_file_content = False

    def __init__(
        self,
        persist: bool | None = None,
        batch_max_files: int = DEFAULT_BATCH_MAX_FILES,
        batch_available_memory_ratio: float = DEFAULT_BATCH_AVAILABLE_MEMORY_RATIO,
        batch_memory_target_bytes: int = DEFAULT_BATCH_MEMORY_TARGET_BYTES,
        batch_memory_expansion_factor: float = DEFAULT_BATCH_MEMORY_EXPANSION_FACTOR,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.persist = persist
        self.batch_max_files = int(batch_max_files)
        self.batch_available_memory_ratio = float(batch_available_memory_ratio)
        self.batch_memory_target_bytes = int(batch_memory_target_bytes)
        self.batch_memory_expansion_factor = float(batch_memory_expansion_factor)
        if self.batch_max_files <= 0:
            raise ValueError("batch_max_files must be greater than zero")
        if not math.isfinite(self.batch_available_memory_ratio) or not 0 < self.batch_available_memory_ratio <= 1:
            raise ValueError("batch_available_memory_ratio must be in the interval (0, 1]")
        if self.batch_memory_target_bytes <= 0:
            raise ValueError("batch_memory_target_bytes must be greater than zero")
        if not math.isfinite(self.batch_memory_expansion_factor) or self.batch_memory_expansion_factor <= 0:
            raise ValueError("batch_memory_expansion_factor must be finite and greater than zero")

    @abstractmethod
    async def build_item(self, path: Path) -> Any:
        """Parse one existing file into the target item shape."""

    @abstractmethod
    async def upsert_items(self, items: list[Any]) -> None:
        """Upsert parsed items into the target."""

    @abstractmethod
    async def delete_paths(self, paths: list[str]) -> None:
        """Delete target-relative paths from the target."""

    @abstractmethod
    async def dump_target(self) -> None:
        """Persist the target."""

    async def execute(self):
        assert self.context is not None
        changes: list[dict] = self.context.get("changes") or []
        persist = bool(self.context.get("persist", True)) if self.persist is None else self.persist
        buckets = bucket_changes(changes, path_exists=lambda p: self._to_abs_path(p).is_file())
        results = await self._apply_existing(buckets)
        results.extend(await self._apply_deleted(buckets[Change.deleted]))
        if persist and results:
            await self.dump_target()
        self.context.response.answer = results
        self.context.response.success = all(r["success"] for r in results) if results else True
        return self.context.response

    async def _apply_existing(self, buckets: dict[Change, list[str]]) -> list[dict]:
        results: list[dict] = []
        for change in (Change.added, Change.modified):
            paths = buckets[change]
            if not paths:
                continue
            self.logger.info(f"Detected {len(paths)} {change.name} file(s)")
            items: list[Any] = []
            ok_paths: list[str] = []
            estimated_bytes = 0
            memory_budget = self._batch_memory_budget()
            for path in paths:
                await asyncio.sleep(0)
                inspected = await self._inspect_path(change, path, results)
                if inspected is None:
                    continue
                abs_path, size_bytes = inspected
                preliminary_bytes = self._try_estimate_memory(
                    change,
                    path,
                    results,
                    self.estimate_source_memory,
                    abs_path,
                    size_bytes,
                )
                if preliminary_bytes is None:
                    continue
                if items and self._batch_is_full(len(items), estimated_bytes, preliminary_bytes, memory_budget):
                    await self._flush_upsert_batch(change, items, ok_paths, estimated_bytes, results)
                    estimated_bytes = 0
                    memory_budget = self._batch_memory_budget()

                item = await self._try_build_item(change, path, abs_path, results)
                if item is None:
                    continue
                item_bytes = self._try_estimate_memory(
                    change,
                    path,
                    results,
                    self.estimate_item_memory,
                    item,
                    abs_path,
                    size_bytes,
                )
                if item_bytes is None:
                    continue
                if items and self._batch_is_full(len(items), estimated_bytes, item_bytes, memory_budget):
                    await self._flush_upsert_batch(change, items, ok_paths, estimated_bytes, results)
                    estimated_bytes = 0
                    memory_budget = self._batch_memory_budget()

                items.append(item)
                ok_paths.append(path)
                estimated_bytes += item_bytes
                if self._batch_is_full(len(items), estimated_bytes, 0, memory_budget):
                    await self._flush_upsert_batch(change, items, ok_paths, estimated_bytes, results)
                    estimated_bytes = 0
                    memory_budget = self._batch_memory_budget()
            if items:
                await self._flush_upsert_batch(change, items, ok_paths, estimated_bytes, results)
        return results

    async def _inspect_path(self, change: Change, path: str, results: list[dict]) -> tuple[Path, int] | None:
        abs_path = self._to_abs_path(path)
        try:
            if not abs_path.is_file():
                results.append(
                    {
                        "change": change.name,
                        "path": path,
                        "success": False,
                        "error": "not a file",
                    },
                )
                return None
            size_bytes = abs_path.stat().st_size
            max_file_bytes = self.max_file_bytes()
            if self.reads_file_content and size_bytes > max_file_bytes:
                await self.delete_paths([self.to_workspace_relative(abs_path)])
                self.logger.warning(
                    f"Skipping oversized file: {path} size_bytes={size_bytes} max_file_bytes={max_file_bytes}",
                )
                results.append(
                    {
                        "change": change.name,
                        "path": path,
                        "success": True,
                        "skipped": True,
                        "reason": "file_too_large",
                        "size_bytes": size_bytes,
                        "max_file_bytes": max_file_bytes,
                    },
                )
                return None
            return abs_path, size_bytes
        except Exception as e:
            self.logger.exception(f"Failed to process {path}")
            results.append({"change": change.name, "path": path, "success": False, "error": str(e)})
            return None

    async def _try_build_item(
        self,
        change: Change,
        path: str,
        abs_path: Path,
        results: list[dict],
    ):
        try:
            self.logger.debug(f"Processing {change.name} file: {path}")
            return await self.build_item(abs_path)
        except Exception as e:
            self.logger.exception(f"Failed to process {path}")
            results.append({"change": change.name, "path": path, "success": False, "error": str(e)})
            return None

    def _try_estimate_memory(
        self,
        change: Change,
        path: str,
        results: list[dict],
        estimate: Callable[..., int],
        *args: Any,
    ) -> int | None:
        """Estimate one item without allowing advisory batching to abort the change set."""
        try:
            return max(0, int(estimate(*args)))
        except Exception as e:
            self.logger.exception(f"Failed to estimate memory for {path}")
            results.append({"change": change.name, "path": path, "success": False, "error": str(e)})
            return None

    def _batch_memory_budget(self) -> int:
        """Return the current estimated-memory budget for a new batch."""
        try:
            available = max(0, int(psutil.virtual_memory().available))
        except (AttributeError, OSError, psutil.Error) as e:
            self.logger.warning(f"Failed to read available memory, using batch memory target: {e}")
            return self.batch_memory_target_bytes
        dynamic_budget = int(available * self.batch_available_memory_ratio)
        return max(1, min(dynamic_budget, self.batch_memory_target_bytes))

    def estimate_source_memory(self, path: Path, size_bytes: int) -> int:
        """Estimate one item before reading it so the current batch can flush first."""
        del path, size_bytes
        return _FILE_MEMORY_OVERHEAD_BYTES

    def estimate_item_memory(self, item: Any, path: Path, size_bytes: int) -> int:
        """Estimate the retained memory for one built target item."""
        del item
        return self.estimate_source_memory(path, size_bytes)

    def _batch_is_full(self, item_count: int, estimated_bytes: int, next_bytes: int, memory_budget: int) -> bool:
        return item_count >= self.batch_max_files or estimated_bytes + next_bytes > memory_budget

    async def _flush_upsert_batch(
        self,
        change: Change,
        items: list[Any],
        ok_paths: list[str],
        estimated_bytes: int,
        results: list[dict],
    ) -> None:
        if not items:
            return
        self.logger.info(
            f"Applying {change.name} batch to {self.target_name}: "
            f"files={len(items)} estimated_bytes={estimated_bytes}",
        )
        results.extend(await self._try_upsert(change, items, ok_paths))
        items.clear()
        ok_paths.clear()

    async def _try_upsert(self, change: Change, items: list[Any], ok_paths: list[str]) -> list[dict]:
        try:
            await self.upsert_items(items)
            return [{"change": change.name, "path": p, "success": True} for p in ok_paths]
        except Exception as e:
            self.logger.exception(f"Failed to upsert {len(items)} {change.name} file(s) into {self.target_name}")
            return [{"change": change.name, "path": p, "success": False, "error": str(e)} for p in ok_paths]

    async def _apply_deleted(self, deleted: list[str]) -> list[dict]:
        if not deleted:
            return []
        self.logger.info(f"Detected {len(deleted)} deleted file(s)")
        results: list[dict] = []
        for start in range(0, len(deleted), self.batch_max_files):
            await asyncio.sleep(0)
            batch = deleted[start : start + self.batch_max_files]
            try:
                await self.delete_paths([self.to_workspace_relative(p) for p in batch])
                results.extend({"change": "deleted", "path": p, "success": True} for p in batch)
            except Exception as e:
                self.logger.exception(f"Failed to delete {len(batch)} file(s) from {self.target_name}")
                results.extend({"change": "deleted", "path": p, "success": False, "error": str(e)} for p in batch)
        return results

    def _to_abs_path(self, path: str | Path) -> Path:
        p = Path(path)
        return p if p.is_absolute() else self.workspace_path / p


@R.register("update_catalog_step")
class UpdateCatalogStep(ChangeApplyStep):
    """Update file_catalog with a batch of file changes."""

    target_name = "file_catalog"

    async def build_item(self, path: Path) -> FileNode:
        stat = path.stat()
        return FileNode(path=self.to_workspace_relative(path), st_mtime=stat.st_mtime)

    async def upsert_items(self, items: list[FileNode]) -> None:
        if self.file_catalog is None:
            raise RuntimeError("file_catalog is not initialized!")
        await self.file_catalog.upsert(items)

    async def delete_paths(self, paths: list[str]) -> None:
        if self.file_catalog is None:
            raise RuntimeError("file_catalog is not initialized!")
        await self.file_catalog.delete(paths)

    async def dump_target(self) -> None:
        if self.file_catalog is not None:
            await self.file_catalog.dump()


@R.register("update_index_step")
class UpdateIndexStep(ChangeApplyStep):
    """Update file_store with a batch of file changes."""

    target_name = "file_store"
    reads_file_content = True

    async def build_item(self, path: Path) -> tuple[FileNode, list[FileChunk]]:
        return await self.chunk_file(path)

    async def upsert_items(self, items: list[tuple[FileNode, list[FileChunk]]]) -> None:
        await self.file_store.upsert(items)

    async def delete_paths(self, paths: list[str]) -> None:
        await self.file_store.delete(paths)

    async def dump_target(self) -> None:
        await self.file_store.dump()

    def estimate_source_memory(self, path: Path, size_bytes: int) -> int:
        chunker = self._resolve_chunker(path)
        chunk_bytes = max(1, int(getattr(chunker, "chunk_byte_size", _ESTIMATED_CHUNK_BYTES)))
        estimated_chunks = max(1, (size_bytes + chunk_bytes - 1) // chunk_bytes)
        return self._estimate_index_memory(size_bytes, estimated_chunks)

    def estimate_item_memory(
        self,
        item: tuple[FileNode, list[FileChunk]],
        path: Path,
        size_bytes: int,
    ) -> int:
        del path
        return self._estimate_index_memory(size_bytes, len(item[1]))

    def _estimate_index_memory(self, size_bytes: int, chunk_count: int) -> int:
        embedding_bytes = 0
        embedding_store = getattr(self.file_store, "embedding_store", None)
        if embedding_store is not None:
            try:
                embedding_bytes = max(0, int(embedding_store.dimensions)) * _FLOAT16_BYTES
            except (AttributeError, TypeError, ValueError):
                embedding_bytes = 0
        expanded_content = int(size_bytes * self.batch_memory_expansion_factor)
        per_chunk = _CHUNK_MEMORY_OVERHEAD_BYTES + embedding_bytes
        return expanded_content + _FILE_MEMORY_OVERHEAD_BYTES + max(0, chunk_count) * per_chunk

    async def chunk_file(self, path: str | Path) -> tuple[FileNode, list[FileChunk]]:
        """Chunk a file into (node, chunks)."""
        if self.app_context is None:
            raise RuntimeError("app_context is not set when resolving file chunker")
        chunker = self._resolve_chunker(Path(path))
        return await chunker.chunk(path)

    def _resolve_chunker(self, path: Path) -> BaseFileChunker:
        """Resolve a file chunker for a given path."""
        chunkers: dict[str, BaseFileChunker] = self.app_context.components[ComponentEnum.FILE_CHUNKER]
        suffix = path.suffix.lstrip(".").lower()
        for candidate in chunkers.values():
            if suffix and suffix in {ext.lower().lstrip(".") for ext in candidate.supported_extensions}:
                return candidate
        if default := chunkers.get("default"):
            return default
        raise RuntimeError(f"No file chunker supports {path} (suffix={suffix!r}) and no default chunker is configured")
