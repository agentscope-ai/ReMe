"""Markdown file watcher for Markdown file synchronization.

This module provides a file watcher that processes Markdown files
on any change, ensuring complete synchronization.
"""

import asyncio
from pathlib import Path

from watchfiles import Change

from .base_file_watcher import BaseFileWatcher
from ...schema import FileMetadata
from ...utils import hash_text, chunk_markdown


class MdFileWatcher(BaseFileWatcher):
    """Markdown file watcher implementation for Markdown file synchronization."""

    def __init__(self, encoding: str = "utf-8", **kwargs):
        """Initialize Markdown file watcher.

        Args:
            encoding: File encoding (default: "utf-8")
            **kwargs: Additional keyword arguments passed to BaseFileWatcher
        """
        super().__init__(**kwargs)
        self.encoding = encoding

    async def _on_changes(self, changes: set[tuple[Change, str]]):
        """Handle file changes with full synchronization."""
        if not self.file_store:
            self.logger.warning("File store not initialized, skipping changes")
            return

        for change_type, path in changes:
            try:
                if change_type in [Change.added, Change.modified]:
                    file_meta = await self._build_file_metadata(path)
                    chunks = (
                            chunk_markdown(
                                file_meta.content,
                                file_meta.path,
                                self.chunk_tokens,
                                self.chunk_overlap,
                            )
                            or []
                    )
                    if chunks:
                        chunks = await self.file_store.get_chunk_embeddings(chunks)
                    file_meta.chunk_count = len(chunks)

                    await self.file_store.delete_file(file_meta.path)
                    self.logger.info(f"delete_file {file_meta.path}")

                    await self.file_store.upsert_file(file_meta, chunks)
                    self.logger.info(f"Upserted {file_meta.chunk_count} chunks for {file_meta.path}")

                elif change_type == Change.deleted:
                    await self.file_store.delete_file(path)
                    self.logger.info(f"Deleted {path}")

                else:
                    self.logger.warning(f"Unknown change type: {change_type}")

                self.logger.info(f"File {change_type} changed: {path}")

            except FileNotFoundError:
                self.logger.warning(f"File not found: {path}, skipping")
            except PermissionError:
                self.logger.warning(f"Permission denied: {path}, skipping")
            except Exception as e:
                self.logger.error(f"Error processing {path}: {e}", exc_info=True)

    async def _build_file_metadata(self, path: str) -> FileMetadata:
        """Build file metadata from path."""
        file_path = Path(path)

        def _read_file_sync():
            return file_path.stat(), file_path.read_text(encoding=self.encoding)

        stat, content = await asyncio.to_thread(_read_file_sync)
        return FileMetadata(
            hash=hash_text(content),
            mtime_ms=stat.st_mtime * 1000,
            size=stat.st_size,
            path=str(file_path.absolute()),
            content=content,
        )
