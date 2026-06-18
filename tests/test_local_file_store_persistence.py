"""Persistence tests for LocalFileStore."""

import hashlib
import time

import pytest

from reme.core.enumeration.memory_source import MemorySource
from reme.core.file_store.local_file_store import LocalFileStore
from reme.core.schema.file_metadata import FileMetadata
from reme.core.schema.memory_chunk import MemoryChunk


def _file_meta(path: str, chunk_count: int) -> FileMetadata:
    content = f"Sample content for {path}"
    return FileMetadata(
        path=path,
        hash=hashlib.md5(content.encode()).hexdigest(),
        mtime_ms=time.time() * 1000,
        size=len(content),
        chunk_count=chunk_count,
    )


def _chunk(path: str) -> MemoryChunk:
    return MemoryChunk(
        id="chunk_persist_1",
        path=path,
        source=MemorySource.MEMORY,
        start_line=1,
        end_line=2,
        text="Persistent memory search content",
        hash=hashlib.md5(b"chunk_persist_1").hexdigest(),
        embedding=[0.0] * 8,
    )


@pytest.mark.asyncio
async def test_local_file_store_persists_upsert_without_close(tmp_path):
    """A fresh instance can load data immediately after upsert."""
    path = "memory/persist.md"
    chunks = [_chunk(path)]
    store = LocalFileStore(
        store_name="memory",
        db_path=tmp_path,
        vector_enabled=False,
        fts_enabled=True,
    )
    await store.start()

    await store.upsert_file(_file_meta(path, len(chunks)), MemorySource.MEMORY, chunks)

    reloaded = LocalFileStore(
        store_name="memory",
        db_path=tmp_path,
        vector_enabled=False,
        fts_enabled=True,
    )
    await reloaded.start()
    try:
        assert await reloaded.list_files(MemorySource.MEMORY) == [path]
        loaded_chunks = await reloaded.get_file_chunks(path, MemorySource.MEMORY)
        assert [chunk.text for chunk in loaded_chunks] == [chunks[0].text]
    finally:
        await reloaded.close()
        await store.close()


@pytest.mark.asyncio
async def test_local_file_store_persists_delete_without_close(tmp_path):
    """Deletes are flushed immediately so stale chunks do not reappear."""
    path = "memory/delete.md"
    chunks = [_chunk(path)]
    store = LocalFileStore(
        store_name="memory",
        db_path=tmp_path,
        vector_enabled=False,
        fts_enabled=True,
    )
    await store.start()

    await store.upsert_file(_file_meta(path, len(chunks)), MemorySource.MEMORY, chunks)
    await store.delete_file(path, MemorySource.MEMORY)

    reloaded = LocalFileStore(
        store_name="memory",
        db_path=tmp_path,
        vector_enabled=False,
        fts_enabled=True,
    )
    await reloaded.start()
    try:
        assert await reloaded.list_files(MemorySource.MEMORY) == []
        assert await reloaded.get_file_chunks(path, MemorySource.MEMORY) == []
    finally:
        await reloaded.close()
        await store.close()
