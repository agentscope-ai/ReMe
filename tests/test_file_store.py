# pylint: disable=too-many-lines
"""Unified test suite for file store implementations.

This module provides comprehensive test coverage for SqliteFileStore, ChromaFileStore,
LocalFileStore and future file store implementations. Tests can be run for specific stores
or all implementations.

Usage:
    python test_file_store.py --sqlite     # Test SqliteFileStore only
    python test_file_store.py --chroma     # Test ChromaFileStore only
    python test_file_store.py --local      # Test LocalFileStore only
    python test_file_store.py --all        # Test all file stores
"""

import argparse
import asyncio
import hashlib
import shutil
import time
from pathlib import Path
from typing import List

from loguru import logger

from reme.core.embedding import OpenAIEmbeddingModel
from reme.core.enumeration.memory_source import MemorySource
from reme.core.file_store.base_file_store import BaseFileStore
from reme.core.file_store.chroma_file_store import ChromaFileStore
from reme.core.file_store.local_file_store import LocalFileStore
from reme.core.file_store.sqlite_file_store import SqliteFileStore
from reme.core.schema.file_metadata import FileMetadata
from reme.core.schema.memory_chunk import MemoryChunk
from reme.core.utils import load_env

# Direct imports to avoid circular dependencies

load_env()


# ==================== Configuration ====================


class TestConfig:
    """Configuration for test execution."""

    # SqliteFileStore settings
    NAME = "test"
    SQLITE_DB_PATH = "./test_file_store_sqlite/memory.db"
    SQLITE_VEC_EXT_PATH = ""  # Empty string to use default vec0/sqlite_vec/vector0
    SQLITE_FTS_ENABLED = True

    # ChromaFileStore settings
    CHROMA_DB_PATH = "./test_file_store_chroma"
    CHROMA_FTS_ENABLED = True

    # LocalFileStore settings
    LOCAL_DB_PATH = "./test_file_store_local"
    LOCAL_FTS_ENABLED = True

    # Embedding model settings
    EMBEDDING_MODEL_NAME = "text-embedding-v4"
    EMBEDDING_DIMENSIONS = 64

    # Test prefix for cleanup
    TEST_PATH_PREFIX = "test_memory_"


# ==================== Sample Data Generator ====================


class SampleDataGenerator:
    """Generator for sample test data."""

    @staticmethod
    def create_sample_chunks(file_path: str, prefix: str = "") -> List[MemoryChunk]:
        """Create sample MemoryChunk instances for testing.

        Args:
            file_path: Path to the file
            prefix: Optional prefix for chunk_id to avoid conflicts

        Returns:
            List[MemoryChunk]: List of sample chunks with diverse content
        """
        id_prefix = f"{prefix}_" if prefix else ""
        base_hash = hashlib.md5(file_path.encode()).hexdigest()[:8]

        return [
            MemoryChunk(
                id=f"{id_prefix}chunk1_{base_hash}",
                path=file_path,
                source=MemorySource.MEMORY,
                start_line=1,
                end_line=5,
                text="Artificial intelligence is a technology that simulates human intelligence.",
                hash=hashlib.md5(b"chunk1").hexdigest(),
                embedding=None,  # Will be populated later
                metadata={"category": "AI", "importance": "high"},
            ),
            MemoryChunk(
                id=f"{id_prefix}chunk2_{base_hash}",
                path=file_path,
                source=MemorySource.MEMORY,
                start_line=6,
                end_line=10,
                text="Machine learning is a subset of artificial intelligence that learns from data.",
                hash=hashlib.md5(b"chunk2").hexdigest(),
                embedding=None,
                metadata={"category": "ML", "importance": "high"},
            ),
            MemoryChunk(
                id=f"{id_prefix}chunk3_{base_hash}",
                path=file_path,
                source=MemorySource.MEMORY,
                start_line=11,
                end_line=15,
                text="Deep learning uses neural networks with multiple layers for complex tasks.",
                hash=hashlib.md5(b"chunk3").hexdigest(),
                embedding=None,
                metadata={"category": "DL", "importance": "medium"},
            ),
        ]

    @staticmethod
    def create_session_chunks(file_path: str, prefix: str = "") -> List[MemoryChunk]:
        """Create sample session chunks for testing.

        Args:
            file_path: Path to the session file
            prefix: Optional prefix for chunk_id to avoid conflicts

        Returns:
            List[MemoryChunk]: List of session chunks
        """
        id_prefix = f"{prefix}_" if prefix else ""
        base_hash = hashlib.md5(file_path.encode()).hexdigest()[:8]

        return [
            MemoryChunk(
                id=f"{id_prefix}session1_{base_hash}",
                path=file_path,
                source=MemorySource.SESSIONS,
                start_line=1,
                end_line=3,
                text="User requested to analyze sales data for Q4 2024.",
                hash=hashlib.md5(b"session1").hexdigest(),
                embedding=None,
                metadata={"session_id": "sess_001", "timestamp": "2024-12-01"},
            ),
            MemoryChunk(
                id=f"{id_prefix}session2_{base_hash}",
                path=file_path,
                source=MemorySource.SESSIONS,
                start_line=4,
                end_line=6,
                text="Analyzed sales data and found a 15% increase in revenue.",
                hash=hashlib.md5(b"session2").hexdigest(),
                embedding=None,
                metadata={"session_id": "sess_001", "timestamp": "2024-12-01"},
            ),
        ]

    @staticmethod
    def create_file_metadata(file_path: str, chunk_count: int = 0) -> FileMetadata:
        """Create sample FileMetadata for testing.

        Args:
            file_path: Path to the file
            chunk_count: Number of chunks (optional)

        Returns:
            FileMetadata: Sample file metadata
        """
        content = f"Sample content for {file_path}"
        return FileMetadata(
            path=file_path,
            hash=hashlib.md5(content.encode()).hexdigest(),
            mtime_ms=time.time() * 1000,
            size=len(content),
            chunk_count=chunk_count,
        )


# ==================== File Store Factory ====================


def get_store_type(store: BaseFileStore) -> str:
    """Get the type identifier of a file store instance.

    Args:
        store: File store instance

    Returns:
        str: Type identifier ("sqlite", "chroma", etc.)
    """
    if isinstance(store, SqliteFileStore):
        return "sqlite"
    elif isinstance(store, ChromaFileStore):
        return "chroma"
    elif isinstance(store, LocalFileStore):
        return "local"
    else:
        raise ValueError(f"Unknown file store type: {type(store)}")


def create_file_store(store_type: str) -> BaseFileStore:
    """Create a file store instance based on type.

    Args:
        store_type: Type of file store ("sqlite", "chroma", etc.)

    Returns:
        BaseFileStore: Initialized file store instance
    """
    config = TestConfig()

    # Initialize embedding model
    embedding_model = OpenAIEmbeddingModel(
        model_name=config.EMBEDDING_MODEL_NAME,
        dimensions=config.EMBEDDING_DIMENSIONS,
    )

    if store_type == "sqlite":
        return SqliteFileStore(
            store_name=config.NAME,
            db_path=config.SQLITE_DB_PATH,
            embedding_model=embedding_model,
            vec_ext_path=config.SQLITE_VEC_EXT_PATH,
            fts_enabled=config.SQLITE_FTS_ENABLED,
        )
    elif store_type == "chroma":
        return ChromaFileStore(
            store_name=config.NAME,
            db_path=config.CHROMA_DB_PATH,
            embedding_model=embedding_model,
            fts_enabled=config.CHROMA_FTS_ENABLED,
        )
    elif store_type == "local":
        return LocalFileStore(
            store_name=config.NAME,
            db_path=config.LOCAL_DB_PATH,
            embedding_model=embedding_model,
            fts_enabled=config.LOCAL_FTS_ENABLED,
        )
    else:
        raise ValueError(f"Unknown store type: {store_type}")


# ==================== Test Functions ====================


async def test_start_store(store: BaseFileStore, _store_name: str):
    """Test store initialization."""
    logger.info("=" * 20 + " START STORE TEST " + "=" * 20)

    await store.start()
    logger.info("✓ Store initialized successfully")

    # Verify tables created (SQLite specific)
    if isinstance(store, SqliteFileStore):
        cursor = store.conn.cursor()
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name",
        )
        tables = [row[0] for row in cursor.fetchall()]
        cursor.close()

        logger.info(f"Created tables: {tables}")
        assert store.files_table_name in tables, f"{store.files_table_name} table should exist"
        assert store.chunks_table_name in tables, f"{store.chunks_table_name} table should exist"
        logger.info("✓ Required tables created")

    # Verify ChromaDB collection created
    if isinstance(store, ChromaFileStore):
        assert store.client is not None, "ChromaDB client should be initialized"
        assert store.chunks_collection is not None, "ChromaDB collection should exist"
        logger.info(f"✓ ChromaDB collection created: {store.collection_name}")

    # Verify LocalFileStore initialized (access internals for test assertions)
    if isinstance(store, LocalFileStore):
        # pylint: disable=protected-access
        assert store._started, "LocalFileStore should be marked as started"
        assert isinstance(store._chunks, dict), "Chunks index should be a dict"
        assert isinstance(store._files, dict), "Files index should be a dict"
        logger.info(f"✓ LocalFileStore ready (chunks file: {store._chunks_file})")


async def test_upsert_file(store: BaseFileStore, _store_name: str) -> tuple[FileMetadata, List[MemoryChunk]]:
    """Test file and chunks insertion."""
    logger.info("=" * 20 + " UPSERT FILE TEST " + "=" * 20)

    # Create sample data
    file_path = "test_memory_file1.txt"
    file_meta = SampleDataGenerator.create_file_metadata(file_path)
    chunks = SampleDataGenerator.create_sample_chunks(file_path, prefix="test")

    # Generate embeddings for chunks
    chunks = await store.get_chunk_embeddings(chunks)
    file_meta.chunk_count = len(chunks)

    # Upsert file
    await store.upsert_file(file_meta, MemorySource.MEMORY, chunks)
    logger.info(f"✓ Upserted file: {file_path} with {len(chunks)} chunks")

    # Verify file exists
    stored_meta = await store.get_file_metadata(file_path, MemorySource.MEMORY)
    assert stored_meta is not None, "File should exist"
    assert stored_meta.hash == file_meta.hash, "File hash should match"
    logger.info(f"✓ Verified file hash: {stored_meta.hash}")

    # Verify chunks
    stored_chunks = await store.get_file_chunks(file_path, MemorySource.MEMORY)
    assert len(stored_chunks) == len(chunks), f"Should have {len(chunks)} chunks"
    logger.info(f"✓ Verified {len(stored_chunks)} chunks stored")

    return file_meta, chunks


async def test_upsert_multiple_sources(store: BaseFileStore, _store_name: str):
    """Test upserting files from different sources."""
    logger.info("=" * 20 + " UPSERT MULTIPLE SOURCES TEST " + "=" * 20)

    # Create memory file
    memory_path = "test_memory_file2.txt"
    memory_meta = SampleDataGenerator.create_file_metadata(memory_path)
    memory_chunks = SampleDataGenerator.create_sample_chunks(memory_path, prefix="mem")
    memory_chunks = await store.get_chunk_embeddings(memory_chunks)
    memory_meta.chunk_count = len(memory_chunks)

    await store.upsert_file(memory_meta, MemorySource.MEMORY, memory_chunks)
    logger.info(f"✓ Upserted MEMORY file: {memory_path}")

    # Create sessions file
    session_path = "test_session_file1.jsonl"
    session_meta = SampleDataGenerator.create_file_metadata(session_path)
    session_chunks = SampleDataGenerator.create_session_chunks(session_path, prefix="sess")
    session_chunks = await store.get_chunk_embeddings(session_chunks)
    session_meta.chunk_count = len(session_chunks)

    await store.upsert_file(session_meta, MemorySource.SESSIONS, session_chunks)
    logger.info(f"✓ Upserted SESSIONS file: {session_path}")

    # List files by source
    memory_files = await store.list_files(MemorySource.MEMORY)
    session_files = await store.list_files(MemorySource.SESSIONS)

    logger.info(f"MEMORY files: {len(memory_files)}")
    logger.info(f"SESSIONS files: {len(session_files)}")

    assert memory_path in memory_files, "Memory file should be listed"
    assert session_path in session_files, "Session file should be listed"
    logger.info("✓ Multiple sources test passed")


async def test_update_file(store: BaseFileStore, _store_name: str):
    """Test updating an existing file."""
    logger.info("=" * 20 + " UPDATE FILE TEST " + "=" * 20)

    file_path = "test_memory_file1.txt"

    # Get original metadata
    original_meta = await store.get_file_metadata(file_path, MemorySource.MEMORY)
    assert original_meta is not None, "Original file should exist"
    original_chunk_count = original_meta.chunk_count
    logger.info(f"Original chunk count: {original_chunk_count}")

    # Update with new chunks
    updated_meta = SampleDataGenerator.create_file_metadata(file_path)
    updated_meta.hash = hashlib.md5(b"updated content").hexdigest()
    updated_chunks = SampleDataGenerator.create_sample_chunks(file_path, prefix="updated")

    # Add one more chunk
    updated_chunks.append(
        MemoryChunk(
            id=f"updated_chunk4_{hashlib.md5(file_path.encode()).hexdigest()[:8]}",
            path=file_path,
            source=MemorySource.MEMORY,
            start_line=16,
            end_line=20,
            text="Natural language processing enables computers to understand human language.",
            hash=hashlib.md5(b"chunk4").hexdigest(),
            embedding=None,
            metadata={"category": "NLP", "importance": "high"},
        ),
    )

    updated_chunks = await store.get_chunk_embeddings(updated_chunks)
    updated_meta.chunk_count = len(updated_chunks)

    # Upsert (update)
    await store.delete_file(file_path, MemorySource.MEMORY)
    await store.upsert_file(updated_meta, MemorySource.MEMORY, updated_chunks)
    logger.info(f"✓ Updated file with {len(updated_chunks)} chunks")

    # Verify update
    new_meta = await store.get_file_metadata(file_path, MemorySource.MEMORY)
    assert new_meta.hash == updated_meta.hash, "Hash should be updated"
    assert new_meta.chunk_count == len(updated_chunks), "Chunk count should be updated"
    logger.info(f"✓ Verified update: new chunk count = {new_meta.chunk_count}")


async def test_get_file_metadata(store: BaseFileStore, _store_name: str):
    """Test retrieving file metadata."""
    logger.info("=" * 20 + " GET FILE METADATA TEST " + "=" * 20)

    file_path = "test_memory_file1.txt"
    meta = await store.get_file_metadata(file_path, MemorySource.MEMORY)

    assert meta is not None, "Metadata should exist"
    assert meta.hash is not None, "Hash should exist"
    assert meta.mtime_ms > 0, "Modification time should be positive"
    assert meta.size > 0, "Size should be positive"
    assert meta.chunk_count is not None and meta.chunk_count > 0, "Should have chunks"

    logger.info(f"File metadata: hash={meta.hash[:8]}..., chunks={meta.chunk_count}, size={meta.size}")
    logger.info("✓ Get file metadata test passed")


async def test_list_files(store: BaseFileStore, _store_name: str):
    """Test listing files by source."""
    logger.info("=" * 20 + " LIST FILES TEST " + "=" * 20)

    memory_files = await store.list_files(MemorySource.MEMORY)
    session_files = await store.list_files(MemorySource.SESSIONS)

    logger.info(f"MEMORY files ({len(memory_files)}):")
    for f in memory_files:
        logger.info(f"  - {f}")

    logger.info(f"SESSIONS files ({len(session_files)}):")
    for f in session_files:
        logger.info(f"  - {f}")

    assert len(memory_files) > 0, "Should have at least one memory file"
    logger.info("✓ List files test passed")


async def test_get_file_chunks(store: BaseFileStore, _store_name: str):
    """Test retrieving chunks for a file."""
    logger.info("=" * 20 + " GET FILE CHUNKS TEST " + "=" * 20)

    file_path = "test_memory_file1.txt"
    chunks = await store.get_file_chunks(file_path, MemorySource.MEMORY)

    assert len(chunks) > 0, "Should have chunks"
    logger.info(f"Retrieved {len(chunks)} chunks")

    for i, chunk in enumerate(chunks, 1):
        logger.info(f"  Chunk {i}: lines {chunk.start_line}-{chunk.end_line}, text={chunk.text[:50]}...")
        assert chunk.id is not None, "Chunk should have ID"
        assert chunk.path == file_path, "Chunk path should match"
        assert chunk.source == MemorySource.MEMORY, "Chunk source should match"
        assert chunk.embedding is not None, "Chunk should have embedding"

    logger.info("✓ Get file chunks test passed")


async def test_vector_search(store: BaseFileStore, _store_name: str):
    """Test vector similarity search."""
    logger.info("=" * 20 + " VECTOR SEARCH TEST " + "=" * 20)

    # Check if vector search is enabled
    if not store.vector_enabled:
        logger.warning("⚠ Vector search not enabled, skipping vector search tests")
        return

    # Search for AI-related content
    query = "What is artificial intelligence and machine learning?"
    results = await store.vector_search(query, limit=5)

    logger.info(f"Vector search for: '{query}'")
    logger.info(f"Found {len(results)} results")

    for i, result in enumerate(results, 1):
        logger.info(f"\n  Result {i}:")
        logger.info(f"    Source: {result.source.value}")
        logger.info(f"    Path: {result.path}")
        logger.info(f"    Lines: {result.start_line}-{result.end_line}")
        logger.info(f"    Score: {result.score:.6f}")
        logger.info(f"    Snippet: {result.snippet}")
        if result.metadata:
            logger.info(f"    Metadata: {result.metadata}")

    assert len(results) > 0, "Should find results"
    assert results[0].score > 0, "Should have positive scores"
    logger.info("\n✓ Vector search test passed")


async def test_vector_search_with_source_filter(store: BaseFileStore, _store_name: str):
    """Test vector search with source filtering."""
    logger.info("=" * 20 + " VECTOR SEARCH WITH SOURCE FILTER TEST " + "=" * 20)

    # Check if vector search is enabled
    if not store.vector_enabled:
        logger.warning("⚠ Vector search not enabled, skipping vector search with source filter test")
        return

    query = "sales data analysis"

    # Search in MEMORY source
    memory_results = await store.vector_search(
        query,
        limit=5,
        sources=[MemorySource.MEMORY],
    )
    logger.info(f"\nMEMORY source results: {len(memory_results)}")
    for i, result in enumerate(memory_results, 1):
        logger.info(f"  {i}. Score: {result.score:.6f} | {result.path}:{result.start_line}-{result.end_line}")
        logger.info(f"     Snippet: {result.snippet}")
        if result.metadata:
            logger.info(f"     Metadata: {result.metadata}")

    # Search in SESSIONS source
    session_results = await store.vector_search(
        query,
        limit=5,
        sources=[MemorySource.SESSIONS],
    )
    logger.info(f"\nSESSIONS source results: {len(session_results)}")
    for i, result in enumerate(session_results, 1):
        logger.info(f"  {i}. Score: {result.score:.6f} | {result.path}:{result.start_line}-{result.end_line}")
        logger.info(f"     Snippet: {result.snippet}")
        if result.metadata:
            logger.info(f"     Metadata: {result.metadata}")

    # Search in all sources
    all_results = await store.vector_search(query, limit=10)
    logger.info(f"\nAll sources results: {len(all_results)}")
    for i, result in enumerate(all_results, 1):
        logger.info(
            f"  {i}. [{result.source.value}] Score: {result.score:.6f} | "
            f"{result.path}:{result.start_line}-{result.end_line}",
        )

    # Verify source filtering
    for r in memory_results:
        assert r.source == MemorySource.MEMORY, "Memory results should only be from MEMORY source"

    for r in session_results:
        assert r.source == MemorySource.SESSIONS, "Session results should only be from SESSIONS source"

    logger.info("\n✓ Vector search with source filter test passed")


async def test_keyword_search(store: BaseFileStore, _store_name: str):
    """Test full-text keyword search."""
    logger.info("=" * 20 + " KEYWORD SEARCH TEST " + "=" * 20)

    # Check if FTS is enabled
    if not store.fts_enabled:
        logger.info("⊘ Skipped: FTS not enabled")
        return

    query = "neural networks"
    results = await store.keyword_search(query, limit=5)

    logger.info(f"Keyword search for: '{query}'")
    logger.info(f"Found {len(results)} results")

    for i, result in enumerate(results, 1):
        logger.info(f"\n  Result {i}:")
        logger.info(f"    Source: {result.source.value}")
        logger.info(f"    Path: {result.path}")
        logger.info(f"    Lines: {result.start_line}-{result.end_line}")
        logger.info(f"    Score: {result.score:.6f}")
        logger.info(f"    Snippet: {result.snippet}")
        if result.metadata:
            logger.info(f"    Metadata: {result.metadata}")

    if len(results) > 0:
        assert results[0].score > 0, "Should have positive scores"
        logger.info("\n✓ Keyword search test passed")
    else:
        logger.info("\n⊘ No results found (may be expected depending on data)")


async def test_keyword_search_with_source_filter(store: BaseFileStore, _store_name: str):
    """Test keyword search with source filtering."""
    logger.info("=" * 20 + " KEYWORD SEARCH WITH SOURCE FILTER TEST " + "=" * 20)

    # Check if FTS is enabled
    if not store.fts_enabled:
        logger.info("⊘ Skipped: FTS not enabled")
        return

    query = "data"

    # Search in different sources
    memory_results = await store.keyword_search(
        query,
        limit=5,
        sources=[MemorySource.MEMORY],
    )
    logger.info(f"\nMEMORY source results: {len(memory_results)}")
    for i, result in enumerate(memory_results, 1):
        logger.info(f"  {i}. Score: {result.score:.6f} | {result.path}:{result.start_line}-{result.end_line}")
        logger.info(f"     Snippet: {result.snippet}")
        if result.metadata:
            logger.info(f"     Metadata: {result.metadata}")

    session_results = await store.keyword_search(
        query,
        limit=5,
        sources=[MemorySource.SESSIONS],
    )
    logger.info(f"\nSESSIONS source results: {len(session_results)}")
    for i, result in enumerate(session_results, 1):
        logger.info(f"  {i}. Score: {result.score:.6f} | {result.path}:{result.start_line}-{result.end_line}")
        logger.info(f"     Snippet: {result.snippet}")
        if result.metadata:
            logger.info(f"     Metadata: {result.metadata}")

    # Verify source filtering
    for r in memory_results:
        assert r.source == MemorySource.MEMORY, "Memory results should only be from MEMORY source"

    for r in session_results:
        assert r.source == MemorySource.SESSIONS, "Session results should only be from SESSIONS source"

    logger.info("\n✓ Keyword search with source filter test passed")


async def test_keyword_search_special_chars(store: BaseFileStore, _store_name: str):
    """Test keyword search with special characters like ?, *, etc."""
    logger.info("=" * 20 + " KEYWORD SEARCH SPECIAL CHARS TEST " + "=" * 20)

    # Check if FTS is enabled
    if not store.fts_enabled:
        logger.info("⊘ Skipped: FTS not enabled")
        return

    # Test various queries with special characters
    test_queries = [
        "What is the status?",
        "How does it work?",
        "Why is this important?",
        "data?",
        "test*",
        "query with ? marks",
    ]

    for query in test_queries:
        logger.info(f"\nTesting query: '{query}'")
        try:
            results = await store.keyword_search(query, limit=3)
            logger.info(f"✓ Query succeeded, found {len(results)} results")
            if results:
                for i, result in enumerate(results[:2], 1):  # Show first 2 results
                    logger.info(
                        f"  {i}. {result.path}:{result.start_line}-{result.end_line} (score: {result.score:.4f})",
                    )
        except Exception as e:
            logger.error(f"✗ Query failed: {e}")
            raise

    logger.info("\n✓ Keyword search with special characters test passed")


async def test_delete_file(store: BaseFileStore, _store_name: str):
    """Test file deletion."""
    logger.info("=" * 20 + " DELETE FILE TEST " + "=" * 20)

    # Create a file to delete
    delete_path = "test_delete_file.txt"
    delete_meta = SampleDataGenerator.create_file_metadata(delete_path)
    delete_chunks = SampleDataGenerator.create_sample_chunks(delete_path, prefix="del")
    delete_chunks = await store.get_chunk_embeddings(delete_chunks)
    delete_meta.chunk_count = len(delete_chunks)

    await store.upsert_file(delete_meta, MemorySource.MEMORY, delete_chunks)
    logger.info(f"✓ Created file: {delete_path}")

    # Verify it exists
    meta_before = await store.get_file_metadata(delete_path, MemorySource.MEMORY)
    assert meta_before is not None, "File should exist before deletion"

    # Delete the file
    await store.delete_file(delete_path, MemorySource.MEMORY)
    logger.info(f"✓ Deleted file: {delete_path}")

    # Verify deletion
    meta_after = await store.get_file_metadata(delete_path, MemorySource.MEMORY)
    assert meta_after is None, "File should not exist after deletion"

    chunks_after = await store.get_file_chunks(delete_path, MemorySource.MEMORY)
    assert len(chunks_after) == 0, "Chunks should be deleted"
    logger.info("✓ Verified deletion")


async def test_batch_upsert(store: BaseFileStore, _store_name: str):
    """Test batch file upsertion."""
    logger.info("=" * 20 + " BATCH UPSERT TEST " + "=" * 20)

    # Create multiple files
    batch_size = 10
    for i in range(batch_size):
        file_path = f"test_batch_file_{i}.txt"
        file_meta = SampleDataGenerator.create_file_metadata(file_path)
        chunks = SampleDataGenerator.create_sample_chunks(file_path, prefix=f"batch{i}")
        chunks = await store.get_chunk_embeddings(chunks)
        file_meta.chunk_count = len(chunks)

        await store.upsert_file(file_meta, MemorySource.MEMORY, chunks)

    logger.info(f"✓ Batch upserted {batch_size} files")

    # Verify all files exist
    memory_files = await store.list_files(MemorySource.MEMORY)
    batch_files = [f for f in memory_files if f.startswith("test_batch_file_")]

    assert len(batch_files) >= batch_size, f"Should have at least {batch_size} batch files"
    logger.info(f"✓ Verified {len(batch_files)} batch files")


async def test_concurrent_searches(store: BaseFileStore, _store_name: str):
    """Test concurrent search operations."""
    logger.info("=" * 20 + " CONCURRENT SEARCHES TEST " + "=" * 20)

    queries = [
        "artificial intelligence",
        "machine learning algorithms",
        "deep learning neural networks",
        "natural language processing",
        "data analysis techniques",
    ]

    # Concurrent vector searches (if available)
    if store.vector_enabled:
        search_tasks = [store.vector_search(q, limit=3) for q in queries]
        results = await asyncio.gather(*search_tasks)

        logger.info(f"✓ Completed {len(results)} concurrent vector searches")
        for i, (query, result) in enumerate(zip(queries, results), 1):
            logger.info(f"  Query {i}: '{query}' -> {len(result)} results")

    # Concurrent keyword searches (if available)
    if store.fts_enabled:
        keyword_tasks = [store.keyword_search(q, limit=3) for q in queries]
        keyword_results = await asyncio.gather(*keyword_tasks)
        logger.info(f"✓ Completed {len(keyword_results)} concurrent keyword searches")

    logger.info("✓ Concurrent searches test passed")


async def test_edge_cases(store: BaseFileStore, _store_name: str):
    """Test edge cases and boundary conditions."""
    logger.info("=" * 20 + " EDGE CASES TEST " + "=" * 20)

    # Test 1: Empty chunk text
    edge_path1 = "test_edge_empty_chunk.txt"
    edge_meta1 = SampleDataGenerator.create_file_metadata(edge_path1)
    edge_chunks1 = [
        MemoryChunk(
            id="edge_empty_chunk",
            path=edge_path1,
            source=MemorySource.MEMORY,
            start_line=1,
            end_line=1,
            text="",
            hash=hashlib.md5(b"").hexdigest(),
            embedding=None,
        ),
    ]

    try:
        edge_chunks1 = await store.get_chunk_embeddings(edge_chunks1)
        await store.upsert_file(edge_meta1, MemorySource.MEMORY, edge_chunks1)
        logger.info("✓ Handled empty chunk text")
    except Exception as e:
        logger.info(f"⊘ Empty chunk not supported: {e}")

    # Test 2: Very long chunk text
    edge_path2 = "test_edge_long_chunk.txt"
    edge_meta2 = SampleDataGenerator.create_file_metadata(edge_path2)
    long_text = "A" * 10000  # 10k characters
    edge_chunks2 = [
        MemoryChunk(
            id="edge_long_chunk",
            path=edge_path2,
            source=MemorySource.MEMORY,
            start_line=1,
            end_line=100,
            text=long_text,
            hash=hashlib.md5(long_text.encode()).hexdigest(),
            embedding=None,
        ),
    ]

    edge_chunks2 = await store.get_chunk_embeddings(edge_chunks2)
    await store.upsert_file(edge_meta2, MemorySource.MEMORY, edge_chunks2)
    retrieved = await store.get_file_chunks(edge_path2, MemorySource.MEMORY)
    assert len(retrieved[0].text) == 10000, "Long text should be preserved"
    logger.info("✓ Handled very long chunk text (10k chars)")

    # Test 3: Special characters in text
    edge_path3 = "test_edge_special_chars.txt"
    edge_meta3 = SampleDataGenerator.create_file_metadata(edge_path3)
    special_text = "Special chars: @#$%^&*()[]{}|\\;:'\",.<>?/~`+=−×÷"
    edge_chunks3 = [
        MemoryChunk(
            id="edge_special_chars",
            path=edge_path3,
            source=MemorySource.MEMORY,
            start_line=1,
            end_line=1,
            text=special_text,
            hash=hashlib.md5(special_text.encode()).hexdigest(),
            embedding=None,
        ),
    ]

    edge_chunks3 = await store.get_chunk_embeddings(edge_chunks3)
    await store.upsert_file(edge_meta3, MemorySource.MEMORY, edge_chunks3)
    retrieved = await store.get_file_chunks(edge_path3, MemorySource.MEMORY)
    assert "@#$%^&*()" in retrieved[0].text, "Special chars should be preserved"
    logger.info("✓ Handled special characters in text")

    # Test 4: Unicode and emoji
    edge_path4 = "test_edge_unicode.txt"
    edge_meta4 = SampleDataGenerator.create_file_metadata(edge_path4)
    unicode_text = "Unicode test: 你好世界 🌍 مرحبا العالم Привет мир"
    edge_chunks4 = [
        MemoryChunk(
            id="edge_unicode",
            path=edge_path4,
            source=MemorySource.MEMORY,
            start_line=1,
            end_line=1,
            text=unicode_text,
            hash=hashlib.md5(unicode_text.encode()).hexdigest(),
            embedding=None,
        ),
    ]

    edge_chunks4 = await store.get_chunk_embeddings(edge_chunks4)
    await store.upsert_file(edge_meta4, MemorySource.MEMORY, edge_chunks4)
    retrieved = await store.get_file_chunks(edge_path4, MemorySource.MEMORY)
    assert "你好世界" in retrieved[0].text, "Unicode should be preserved"
    assert "🌍" in retrieved[0].text, "Emoji should be preserved"
    logger.info("✓ Handled unicode and emoji")

    # Test 5: Search with empty query
    if store.vector_enabled:
        try:
            results = await store.vector_search("", limit=5)
            logger.info(f"✓ Empty query returned {len(results)} results")
        except Exception as e:
            logger.info(f"⊘ Empty query not supported: {e}")

    # Test 6: Very high limit
    if store.vector_enabled:
        results = await store.vector_search("test", limit=1000)
        logger.info(f"✓ High limit search returned {len(results)} results")

    # Test 7: Non-existent file
    non_existent_meta = await store.get_file_metadata("non_existent_file.txt", MemorySource.MEMORY)
    assert non_existent_meta is None, "Non-existent file should return None"
    logger.info("✓ Non-existent file handled gracefully")

    logger.info("✓ Edge cases test passed")


async def test_clear_all(store: BaseFileStore, _store_name: str):
    """Test clearing all data."""
    logger.info("=" * 20 + " CLEAR ALL TEST " + "=" * 20)

    # Verify we have data before clearing
    files_before = await store.list_files(MemorySource.MEMORY)
    logger.info(f"Files before clear: {len(files_before)}")
    assert len(files_before) > 0, "Should have files before clearing"

    # Clear all data
    await store.clear_all()
    logger.info("✓ Cleared all data")

    # Verify all data is gone
    memory_files = await store.list_files(MemorySource.MEMORY)
    session_files = await store.list_files(MemorySource.SESSIONS)

    assert len(memory_files) == 0, "All memory files should be deleted"
    assert len(session_files) == 0, "All session files should be deleted"
    logger.info("✓ Verified all data cleared")

    # Verify we can still insert after clearing
    test_path = "test_after_clear.txt"
    test_meta = SampleDataGenerator.create_file_metadata(test_path)
    test_chunks = SampleDataGenerator.create_sample_chunks(test_path)
    test_chunks = await store.get_chunk_embeddings(test_chunks)
    test_meta.chunk_count = len(test_chunks)

    await store.upsert_file(test_meta, MemorySource.MEMORY, test_chunks)
    logger.info("✓ Can insert data after clearing")

    logger.info("✓ Clear all test passed")


# ==================== Test Runner ====================


async def run_all_tests_for_store(store_type: str, store_name: str):
    """Run all tests for a specific file store type.

    Args:
        store_type: Type of file store ("sqlite", etc.)
        store_name: Display name for the file store
    """
    logger.info(f"\n\n{'#' * 60}")
    logger.info(f"# Running all tests for: {store_name}")
    logger.info(f"{'#' * 60}")

    # Create file store instance
    store = create_file_store(store_type)

    try:
        # ========== Basic Tests ==========
        logger.info(f"\n{'#' * 60}")
        logger.info("# BASIC FUNCTIONALITY TESTS")
        logger.info(f"{'#' * 60}")

        await test_start_store(store, store_name)
        await test_upsert_file(store, store_name)
        await test_upsert_multiple_sources(store, store_name)
        await test_update_file(store, store_name)
        await test_get_file_metadata(store, store_name)
        await test_list_files(store, store_name)
        await test_get_file_chunks(store, store_name)

        # ========== Search Tests ==========
        logger.info(f"\n{'#' * 60}")
        logger.info("# SEARCH FUNCTIONALITY TESTS")
        logger.info(f"{'#' * 60}")

        await test_vector_search(store, store_name)
        await test_vector_search_with_source_filter(store, store_name)
        await test_keyword_search(store, store_name)
        await test_keyword_search_with_source_filter(store, store_name)
        await test_keyword_search_special_chars(store, store_name)

        # ========== Advanced Tests ==========
        logger.info(f"\n{'#' * 60}")
        logger.info("# ADVANCED FUNCTIONALITY TESTS")
        logger.info(f"{'#' * 60}")

        await test_delete_file(store, store_name)
        await test_batch_upsert(store, store_name)
        await test_concurrent_searches(store, store_name)
        await test_edge_cases(store, store_name)

        # ========== Cleanup Test ==========
        logger.info(f"\n{'#' * 60}")
        logger.info("# CLEANUP TESTS")
        logger.info(f"{'#' * 60}")

        await test_clear_all(store, store_name)

        logger.info(f"\n{'=' * 60}")
        logger.info(f"✓ All tests passed for {store_name}!")
        logger.info(f"{'=' * 60}")

    except Exception as e:
        logger.error(f"Test failed: {e}")
        raise
    finally:
        # Cleanup
        await cleanup_store(store, store_type)


async def cleanup_store(store: BaseFileStore, store_type: str):
    """Clean up test resources for a file store.

    Args:
        store: File store instance
        store_type: Type of file store ("sqlite", "chroma", etc.)
    """
    logger.info("=" * 20 + " CLEANUP " + "=" * 20)

    try:
        # Close connections
        await store.close()
        logger.info("✓ Closed store connections")

        # Clean up local directory if SqliteFileStore
        if store_type == "sqlite":
            config = TestConfig()
            db_dir = Path(config.SQLITE_DB_PATH).parent
            if db_dir.exists():
                shutil.rmtree(db_dir)
                logger.info(f"✓ Cleaned up directory: {db_dir}")

        # Clean up local directory if ChromaFileStore
        if store_type == "chroma":
            config = TestConfig()
            db_dir = Path(config.CHROMA_DB_PATH)
            if db_dir.exists():
                shutil.rmtree(db_dir)
                logger.info(f"✓ Cleaned up directory: {db_dir}")
            # Also clean up the metadata file
            metadata_file = db_dir.parent / f"{config.NAME}_file_metadata.json"
            if metadata_file.exists():
                metadata_file.unlink()
                logger.info(f"✓ Cleaned up metadata file: {metadata_file}")

        # Clean up LocalFileStore JSON persistence files
        if store_type == "local":
            config = TestConfig()
            db_dir = Path(config.LOCAL_DB_PATH)
            if db_dir.exists():
                shutil.rmtree(db_dir)
                logger.info(f"✓ Cleaned up directory: {db_dir}")
            for suffix in ("_chunks.jsonl", "_file_metadata.json"):
                json_file = db_dir.parent / f"{config.NAME}{suffix}"
                if json_file.exists():
                    json_file.unlink()
                    logger.info(f"✓ Cleaned up file: {json_file}")

        logger.info("✓ Cleanup completed")
    except Exception as e:
        logger.error(f"Cleanup error: {e}")


# ==================== Main Entry Point ====================


async def main():
    """Main entry point for running tests."""
    parser = argparse.ArgumentParser(
        description="Run file store tests",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python test_file_store.py --sqlite     # Test SqliteFileStore only
  python test_file_store.py --chroma     # Test ChromaFileStore only
  python test_file_store.py --local      # Test LocalFileStore only
  python test_file_store.py --all        # Test all file stores
        """,
    )
    parser.add_argument(
        "--sqlite",
        action="store_true",
        help="Test SqliteFileStore",
    )
    parser.add_argument(
        "--chroma",
        action="store_true",
        help="Test ChromaFileStore",
    )
    parser.add_argument(
        "--local",
        action="store_true",
        help="Test LocalFileStore",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run tests for all available file stores",
    )

    args = parser.parse_args()

    # Determine which file stores to test
    stores_to_test = []

    if args.all:
        stores_to_test = [
            ("sqlite", "SqliteFileStore"),
            ("chroma", "ChromaFileStore"),
            ("local", "LocalFileStore"),
        ]
    else:
        # Build list based on individual flags
        if args.sqlite:
            stores_to_test.append(("sqlite", "SqliteFileStore"))
        if args.chroma:
            stores_to_test.append(("chroma", "ChromaFileStore"))
        if args.local:
            stores_to_test.append(("local", "LocalFileStore"))

        if not stores_to_test:
            # Default to all file stores if no argument provided
            stores_to_test = [
                ("sqlite", "SqliteFileStore"),
                ("chroma", "ChromaFileStore"),
                ("local", "LocalFileStore"),
            ]
            print("No file store specified, defaulting to test all file stores")
            print("Use --sqlite, --chroma, or --local to test specific ones\n")

    # Run tests for each file store
    for store_type, store_name in stores_to_test:
        try:
            await run_all_tests_for_store(store_type, store_name)
        except Exception as e:
            logger.error(f"\n✗ FAILED: {store_name} tests failed with error:")
            logger.error(f"  {type(e).__name__}: {e}")
            raise

    # Final summary
    print(f"\n\n{'#' * 60}")
    print("# TEST SUMMARY")
    print(f"{'#' * 60}")
    print(f"✓ All tests passed for {len(stores_to_test)} file store(s):")
    for _, store_name in stores_to_test:
        print(f"  - {store_name}")
    print(f"{'#' * 60}\n")


if __name__ == "__main__":
    asyncio.run(main())
