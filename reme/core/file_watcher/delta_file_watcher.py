"""Delta file watcher for incremental file synchronization.

This module provides a file watcher that detects append-only changes
and only processes newly added content, avoiding redundant operations.
"""

import asyncio
import os

from loguru import logger
from watchfiles import Change

from .base_file_watcher import BaseFileWatcher
from ..enumeration import MemorySource
from ..schema import FileMetadata, MemoryChunk
from ..utils import chunk_markdown, hash_text


class DeltaFileWatcher(BaseFileWatcher):
    """Delta file watcher implementation for incremental synchronization.

    This watcher detects append-only changes (e.g., log files) and only processes
    the newly added content, avoiding redundant embedding requests for unchanged content.

    Strategy:
    - Detect if file is append-only (new lines added at end)
    - Find the safe cutoff point (considering chunk overlap)
    - Only re-chunk and embed content from cutoff to end
    - Delete affected old chunks and insert new chunks
    """

    def __init__(self, overlap_lines: int = 2, **kwargs):
        """
        Initialize delta file watcher.

        Args:
            chunk_tokens: Maximum tokens per chunk
            chunk_overlap: Overlap tokens between chunks
        """
        super().__init__(**kwargs)
        self.overlap_lines = overlap_lines
        self.dirty = False

    @staticmethod
    async def _build_file_metadata(path: str) -> FileMetadata:
        """Build file metadata from filesystem."""

        def _read_file_sync():
            stat_t = os.stat(path)
            with open(path, "r", encoding="utf-8") as f:
                content_t = f.read()
            return stat_t, content_t

        stat, content = await asyncio.to_thread(_read_file_sync)
        return FileMetadata(
            hash=hash_text(content),
            mtime_ms=stat.st_mtime * 1000,
            size=stat.st_size,
            path=path,
            content=content,
        )

    def _find_cutoff_line(
        self,
        old_chunks: list[MemoryChunk],
        old_file_meta: FileMetadata,
        new_file_meta: FileMetadata,
    ) -> int | None:
        """Find the safe cutoff line for incremental update.

        Uses a heuristic approach: if file size increased and hash changed,
        we verify by comparing content. For true append-only files (like logs),
        the old content should be a prefix of new content.

        Args:
            old_chunks: Existing chunks sorted by start_line
            old_file_meta: Previous file metadata
            new_file_meta: Current file metadata (with content)

        Returns:
            Cutoff line number (1-indexed), or None if not append-only
        """
        if not old_chunks:
            return None

        # File shrunk - definitely not append-only
        if new_file_meta.size < old_file_meta.size:
            logger.debug("File shrunk, not append-only")
            return None

        # File didn't grow much - might be a modification
        size_growth = new_file_meta.size - old_file_meta.size
        if size_growth < 10:  # Less than 10 bytes growth
            logger.debug("Minimal size growth, treating as modification")
            return None

        # Verify append-only by checking if old content is prefix
        # We need to read old file content from chunks
        old_chunks_sorted = sorted(old_chunks, key=lambda c: c.start_line)

        # Simple heuristic: check if first few chunks' content matches
        # This avoids reconstructing full old content
        new_lines = new_file_meta.content.split("\n")

        # Sample check: verify first chunk still matches
        first_chunk = old_chunks_sorted[0]
        first_chunk_lines = first_chunk.text.split("\n")
        new_first_lines = new_lines[first_chunk.start_line - 1 : first_chunk.end_line]

        # Compare (allowing for minor whitespace differences at boundaries)
        if len(first_chunk_lines) > 0 and len(new_first_lines) > 0:
            # Check if most of the lines match
            matches = sum(1 for old, new in zip(first_chunk_lines, new_first_lines) if old == new)
            if matches < len(first_chunk_lines) * 0.8:  # Less than 80% match
                logger.debug("First chunk content changed, not append-only")
                return None

        # File appears to be append-only
        # Find the last chunk and set cutoff considering overlap
        last_chunk = max(old_chunks_sorted, key=lambda c: c.end_line)
        cutoff_line = max(1, last_chunk.end_line - self.overlap_lines)

        logger.debug(
            f"Append-only detected: size {old_file_meta.size} -> {new_file_meta.size}, "
            f"cutoff at line {cutoff_line}",
        )

        return cutoff_line

    @staticmethod
    def _extract_content_from_line(content: str, start_line: int) -> str:
        """Extract content starting from a specific line number."""
        lines = content.split("\n")
        if start_line <= 1:
            return content
        if start_line > len(lines):
            return ""
        # start_line is 1-indexed, array is 0-indexed
        return "\n".join(lines[start_line - 1 :])

    async def _on_changes(self, changes: set[tuple[Change, str]]):
        """Handle file changes with incremental synchronization."""
        self.dirty = True

        for change_type, path in changes:
            if change_type == Change.added:
                # New file: process everything
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
                    await self.file_store.upsert_file(file_meta, MemorySource.MEMORY, chunks)
                    logger.info(f"File added: {path} ({len(chunks)} chunks)")
                else:
                    logger.warning(f"No chunks generated for new file {path}")

            elif change_type == Change.modified:
                # Get existing data
                old_chunks = await self.file_store.get_file_chunks(path, MemorySource.MEMORY)
                old_file_meta = await self.file_store.get_file_metadata(path, MemorySource.MEMORY)

                # Read new file
                file_meta = await self._build_file_metadata(path)

                # If no old chunks, fallback to full update
                if not old_chunks or not old_file_meta:
                    logger.debug(f"No existing chunks for {path}, doing full update")
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
                        await self.file_store.delete_file(path, MemorySource.MEMORY)
                        await self.file_store.upsert_file(
                            file_meta,
                            MemorySource.MEMORY,
                            chunks,
                        )
                        logger.info(f"File modified (full): {path} ({len(chunks)} chunks)")
                    continue

                # Check if append-only and find cutoff line
                old_chunks_sorted = sorted(old_chunks, key=lambda c: c.start_line)
                cutoff_line = self._find_cutoff_line(old_chunks_sorted, old_file_meta, file_meta)

                if cutoff_line is None:
                    # Not append-only, do full update
                    logger.debug(f"File {path} has modifications, doing full update")
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
                        await self.file_store.delete_file(path, MemorySource.MEMORY)
                        await self.file_store.upsert_file(file_meta, MemorySource.MEMORY, chunks)
                        logger.info(f"File modified (full): {path} ({len(chunks)} chunks)")
                else:
                    # Append-only: incremental update
                    new_content_part = self._extract_content_from_line(file_meta.content, cutoff_line)

                    new_chunks = (
                        chunk_markdown(
                            new_content_part,
                            file_meta.path,
                            MemorySource.MEMORY,
                            self.chunk_tokens,
                            self.chunk_overlap,
                        )
                        or []
                    )

                    if not new_chunks:
                        logger.debug(f"No new chunks for {path}, skipping")
                        continue

                    for idx, chunk in enumerate(new_chunks):
                        chunk.start_line += cutoff_line - 1
                        chunk.end_line += cutoff_line - 1
                        chunk.id = hash_text(
                            f"{chunk.source}:{chunk.path}:{chunk.start_line}:" f"{chunk.end_line}:{chunk.hash}:{idx}",
                        )

                    new_chunks = await self.file_store.get_chunk_embeddings(new_chunks)

                    chunks_to_delete = [c.id for c in old_chunks_sorted if c.start_line >= cutoff_line]

                    # Apply incremental updates
                    if chunks_to_delete:
                        await self.file_store.delete_file_chunks(path, chunks_to_delete)

                    if new_chunks:
                        await self.file_store.upsert_chunks(new_chunks, MemorySource.MEMORY)

                    # Update file metadata to reflect the changes
                    # Calculate new chunk count: old chunks - deleted + new chunks
                    new_chunk_count = len(old_chunks) - len(chunks_to_delete) + len(new_chunks)
                    file_meta.chunk_count = new_chunk_count
                    await self.file_store.update_file_metadata(file_meta, MemorySource.MEMORY)

                    logger.info(
                        f"File modified (incremental): {path} "
                        f"(cutoff: line {cutoff_line}, "
                        f"+{len(new_chunks)} chunks, -{len(chunks_to_delete)} chunks)",
                    )

            elif change_type == Change.deleted:
                await self.file_store.delete_file(path, MemorySource.MEMORY)
                logger.info(f"File deleted: {path}")

            else:
                logger.warning(f"Unknown change type: {change_type}")

        self.dirty = False
