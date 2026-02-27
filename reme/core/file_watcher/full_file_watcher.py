"""Full file watcher for complete file synchronization.

This module provides a file watcher that processes entire files
on any change, ensuring complete synchronization.
"""

import asyncio
from pathlib import Path

from loguru import logger
from watchfiles import Change

from .base_file_watcher import BaseFileWatcher
from ..enumeration import MemorySource
from ..schema import FileMetadata
from ..utils import chunk_markdown, hash_text


class FullFileWatcher(BaseFileWatcher):
    """Full file watcher implementation for full synchronization"""

    def __init__(self, **kwargs):
        """
        Initialize full file watcher"""
        super().__init__(**kwargs)
        self.dirty = False

    @staticmethod
    async def _build_file_metadata(path: str) -> FileMetadata:
        file_path = Path(path)

        def _read_file_sync():
            return file_path.stat(), file_path.read_text(encoding="utf-8")

        stat, content = await asyncio.to_thread(_read_file_sync)
        return FileMetadata(
            hash=hash_text(content),
            mtime_ms=stat.st_mtime * 1000,
            size=stat.st_size,
            path=str(file_path.absolute()),
            content=content,
        )

    async def _on_changes(self, changes: set[tuple[Change, str]]):
        """Handle file changes with full synchronization"""
        self.dirty = True
        for change_type, path in changes:
            if change_type in [Change.added, Change.modified]:
                file_meta = await self._build_file_metadata(path)
                chunks = (
                    chunk_markdown(
                        file_meta.content,
                        file_meta.path,
                        MemorySource.MEMORY,
                        self.chunk_tokens,
                        self.chunk_overlap,
                    )
                    or []
                )
                if chunks:
                    chunks = await self.file_store.get_chunk_embeddings(chunks)
                file_meta.chunk_count = len(chunks)

                await self.file_store.delete_file(file_meta.path, MemorySource.MEMORY)
                logger.info(f"delete_file {file_meta.path}")

                await self.file_store.upsert_file(file_meta, MemorySource.MEMORY, chunks)
                logger.info(f"Upserted {file_meta.chunk_count} chunks for {file_meta.path}")

            elif change_type == Change.deleted:
                await self.file_store.delete_file(path, MemorySource.MEMORY)
                logger.info(f"Deleted {path}")

            else:
                logger.warning(f"Unknown change type: {change_type}")

            logger.info(f"File {change_type} changed: {path}")
        self.dirty = False
