"""Apply file change batches to file_catalog or file_store."""

from abc import abstractmethod
from pathlib import Path
from typing import Any

from watchfiles import Change

from ._change_batch import bucket_changes
from ..base_step import BaseStep
from ...components import R
from ...components.file_chunker import BaseFileChunker
from ...enumeration import ComponentEnum
from ...schema import FileChunk, FileNode


class ChangeApplyStep(BaseStep):
    """Shared added/modified/deleted handling for index update targets."""

    target_name = "target"

    def __init__(self, persist: bool | None = None, **kwargs):
        super().__init__(**kwargs)
        self.persist = persist

    @abstractmethod
    async def build_item(self, path: Path) -> Any:
        """Parse one existing file into the target item shape."""

    @abstractmethod
    def item_path(self, item: Any) -> str:
        """Return the target-relative path for an upsert item."""

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
        for change, action in ((Change.added, "Adding"), (Change.modified, "Updating")):
            paths = buckets[change]
            if not paths:
                continue
            self.logger.info(f"Detected {len(paths)} {change.name} file(s)")
            items, ok_paths = [], []
            for path in paths:
                item = await self._try_build_item(change, action, path, results)
                if item is not None:
                    items.append(item)
                    ok_paths.append(path)
            if items:
                results.extend(await self._try_upsert(change, items, ok_paths))
        return results

    async def _try_build_item(self, change: Change, action: str, path: str, results: list[dict]):
        abs_path = self._to_abs_path(path)
        if not abs_path.is_file():
            results.append({"change": change.name, "path": path, "success": False, "error": "not a file"})
            return None
        self.logger.info(f"{action} file: {path}")
        try:
            return await self.build_item(abs_path)
        except Exception as e:
            self.logger.exception(f"Failed to parse {path}")
            results.append({"change": change.name, "path": path, "success": False, "error": str(e)})
            return None

    async def _try_upsert(self, change: Change, items: list[Any], ok_paths: list[str]) -> list[dict]:
        try:
            await self.delete_paths([self.item_path(item) for item in items])
            await self.upsert_items(items)
            return [{"change": change.name, "path": p, "success": True} for p in ok_paths]
        except Exception as e:
            self.logger.exception(f"Failed to upsert {len(items)} {change.name} file(s) into {self.target_name}")
            return [{"change": change.name, "path": p, "success": False, "error": str(e)} for p in ok_paths]

    async def _apply_deleted(self, deleted: list[str]) -> list[dict]:
        if not deleted:
            return []
        self.logger.info(f"Detected {len(deleted)} deleted file(s)")
        try:
            await self.delete_paths([self.to_workspace_relative(p) for p in deleted])
            return [{"change": "deleted", "path": p, "success": True} for p in deleted]
        except Exception as e:
            self.logger.exception(f"Failed to delete {len(deleted)} file(s) from {self.target_name}")
            return [{"change": "deleted", "path": p, "success": False, "error": str(e)} for p in deleted]

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

    def item_path(self, item: FileNode) -> str:
        return item.path

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

    async def build_item(self, path: Path) -> tuple[FileNode, list[FileChunk]]:
        return await self.chunk_file(path)

    def item_path(self, item: tuple[FileNode, list[FileChunk]]) -> str:
        return item[0].path

    async def upsert_items(self, items: list[tuple[FileNode, list[FileChunk]]]) -> None:
        await self.file_store.upsert(items)

    async def delete_paths(self, paths: list[str]) -> None:
        await self.file_store.delete(paths)

    async def dump_target(self) -> None:
        await self.file_store.dump()

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
