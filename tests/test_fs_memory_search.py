"""Tests for ReMeFs memory_search interface.

This module tests the memory_search() method of ReMeFs class which provides
a high-level interface for searching personal information stored in memory files.

The memory_search function should enable:
1. Vector similarity search across memory chunks
2. Keyword/FTS (full-text search) if enabled
3. Hybrid search combining vector and keyword results
4. Source filtering (MEMORY, SESSIONS, etc.)
5. Score-based filtering and result limiting
"""

import asyncio
import hashlib
import shutil
from pathlib import Path

from reme import ReMeFs
from reme.core.enumeration import MemorySource
from reme.core.schema import FileMetadata, MemoryChunk


# ==================== Test Configuration ====================


class TestConfig:
    """Test configuration settings."""

    WORKING_DIR = ".reme_test_search"


# ==================== Sample Data Generator ====================


class SampleDataGenerator:
    """Generator for sample test data."""

    @staticmethod
    def create_personal_info_chunks(test_name: str = "") -> list[MemoryChunk]:
        """Create sample chunks with personal information."""
        base_path = "memory/personal_info.md"
        prefix = f"{test_name}_" if test_name else ""
        return [
            MemoryChunk(
                id=f"{prefix}personal_1",
                path=base_path,
                source=MemorySource.MEMORY,
                start_line=1,
                end_line=3,
                text="My name is Alice Chen. I am a software engineer working at TechCorp.",
                hash=hashlib.md5(b"personal_1").hexdigest(),
                embedding=None,
                metadata={"category": "personal", "type": "basic_info"},
            ),
            MemoryChunk(
                id=f"{prefix}personal_2",
                path=base_path,
                source=MemorySource.MEMORY,
                start_line=4,
                end_line=6,
                text=(
                    "I love Python programming and machine learning. "
                    "My favorite frameworks are PyTorch and scikit-learn."
                ),
                hash=hashlib.md5(b"personal_2").hexdigest(),
                embedding=None,
                metadata={"category": "personal", "type": "interests"},
            ),
            MemoryChunk(
                id=f"{prefix}personal_3",
                path=base_path,
                source=MemorySource.MEMORY,
                start_line=7,
                end_line=9,
                text="In my free time, I enjoy reading science fiction novels and hiking in the mountains.",
                hash=hashlib.md5(b"personal_3").hexdigest(),
                embedding=None,
                metadata={"category": "personal", "type": "hobbies"},
            ),
        ]

    @staticmethod
    def create_technical_chunks(test_name: str = "") -> list[MemoryChunk]:
        """Create sample chunks with technical information."""
        base_path = "memory/technical_notes.md"
        prefix = f"{test_name}_" if test_name else ""
        return [
            MemoryChunk(
                id=f"{prefix}tech_1",
                path=base_path,
                source=MemorySource.MEMORY,
                start_line=1,
                end_line=3,
                text=(
                    "Artificial intelligence is transforming software development "
                    "with automated code generation and testing."
                ),
                hash=hashlib.md5(b"tech_1").hexdigest(),
                embedding=None,
                metadata={"category": "tech", "topic": "AI"},
            ),
            MemoryChunk(
                id=f"{prefix}tech_2",
                path=base_path,
                source=MemorySource.MEMORY,
                start_line=4,
                end_line=6,
                text=(
                    "Machine learning models require careful tuning of hyperparameters "
                    "to achieve optimal performance."
                ),
                hash=hashlib.md5(b"tech_2").hexdigest(),
                embedding=None,
                metadata={"category": "tech", "topic": "ML"},
            ),
            MemoryChunk(
                id=f"{prefix}tech_3",
                path=base_path,
                source=MemorySource.MEMORY,
                start_line=7,
                end_line=9,
                text="Deep learning neural networks excel at image recognition and natural language processing tasks.",
                hash=hashlib.md5(b"tech_3").hexdigest(),
                embedding=None,
                metadata={"category": "tech", "topic": "DL"},
            ),
        ]

    @staticmethod
    def create_session_chunks(test_name: str = "") -> list[MemoryChunk]:
        """Create sample session chunks."""
        base_path = "sessions/2024-01-15.jsonl"
        prefix = f"{test_name}_" if test_name else ""
        return [
            MemoryChunk(
                id=f"{prefix}session_1",
                path=base_path,
                source=MemorySource.SESSIONS,
                start_line=1,
                end_line=2,
                text="User asked about Python best practices for async programming.",
                hash=hashlib.md5(b"session_1").hexdigest(),
                embedding=None,
                metadata={"session_id": "sess_001", "date": "2024-01-15"},
            ),
            MemoryChunk(
                id=f"{prefix}session_2",
                path=base_path,
                source=MemorySource.SESSIONS,
                start_line=3,
                end_line=4,
                text="Discussed asyncio event loop and common pitfalls in concurrent Python code.",
                hash=hashlib.md5(b"session_2").hexdigest(),
                embedding=None,
                metadata={"session_id": "sess_001", "date": "2024-01-15"},
            ),
        ]

    @staticmethod
    def create_file_metadata(path: str, chunk_count: int) -> FileMetadata:
        """Create file metadata."""
        return FileMetadata(
            path=path,
            hash=hashlib.md5(path.encode()).hexdigest(),
            mtime_ms=1704067200000,  # 2024-01-01 00:00:00
            size=1000,
            chunk_count=chunk_count,
        )


# ==================== Helper Functions ====================


def print_search_results(results: list[dict], query: str, title: str = "SEARCH RESULTS"):
    """Pretty print search results."""
    print(f"\n{'=' * 80}")
    print(f"{title}")
    print(f"Query: '{query}'")
    print(f"Found {len(results)} results")
    print(f"{'=' * 80}")

    for i, result in enumerate(results, 1):
        print(f"\n[{i}] Source: {result.get('source', 'N/A')}")
        print(f"    Path: {result.get('path', 'N/A')}")
        print(f"    Lines: {result.get('start_line', 'N/A')}-{result.get('end_line', 'N/A')}")
        print(f"    Score: {result.get('score', 0):.6f}")
        snippet = result.get("snippet", result.get("text", ""))
        if len(snippet) > 100:
            snippet = snippet[:100] + "..."
        print(f"    Snippet: {snippet}")
        if result.get("metadata"):
            print(f"    Metadata: {result.get('metadata')}")

    print(f"{'=' * 80}\n")


# ==================== Test Functions ====================


async def test_memory_search_basic():
    """Test basic memory search functionality.

    Insert sample data and perform a simple search query.
    """
    print("\n" + "=" * 80)
    print("TEST 1: Basic Memory Search")
    print("=" * 80)

    # Initialize ReMeFs with unique store name
    reme_fs = ReMeFs(
        enable_logo=False,
        working_dir=TestConfig.WORKING_DIR,
        default_memory_store_config={
            "backend": "sqlite",
            "store_name": "test_basic",
            "embedding_model": "default",
            "fts_enabled": True,
        },
    )
    await reme_fs.start()

    # Insert personal info chunks
    personal_chunks = SampleDataGenerator.create_personal_info_chunks("test_basic")
    personal_chunks = await reme_fs.default_memory_store.get_chunk_embeddings(personal_chunks)

    file_meta = SampleDataGenerator.create_file_metadata(
        "memory/personal_info.md",
        len(personal_chunks),
    )
    await reme_fs.default_memory_store.upsert_file(
        file_meta,
        MemorySource.MEMORY,
        personal_chunks,
    )
    print(f"✓ Inserted {len(personal_chunks)} personal info chunks")

    # Perform search
    query = "What programming languages does the user like?"
    print(f"\nSearching for: '{query}'")

    result_json = await reme_fs.memory_search(
        query=query,
        max_results=5,
        min_score=0.0,
    )

    # Parse results
    import json

    results = json.loads(result_json)
    print_search_results(results, query, "BASIC SEARCH RESULTS")

    # Verify results
    assert len(results) > 0, "Should find at least one result"
    assert results[0]["score"] > 0, "Top result should have positive score"
    print("✓ Basic memory search test passed")

    await reme_fs.close()


async def test_memory_search_technical_content():
    """Test memory search with technical content.

    Insert technical chunks and search for ML/AI related queries.
    """
    print("\n" + "=" * 80)
    print("TEST 2: Technical Content Search")
    print("=" * 80)

    # Initialize ReMeFs with unique store name
    reme_fs = ReMeFs(
        enable_logo=False,
        working_dir=TestConfig.WORKING_DIR,
        default_memory_store_config={
            "backend": "sqlite",
            "store_name": "test_technical",
            "embedding_model": "default",
            "fts_enabled": True,
        },
    )
    await reme_fs.start()

    # Insert technical chunks
    tech_chunks = SampleDataGenerator.create_technical_chunks("test_technical")
    tech_chunks = await reme_fs.default_memory_store.get_chunk_embeddings(tech_chunks)

    file_meta = SampleDataGenerator.create_file_metadata(
        "memory/technical_notes.md",
        len(tech_chunks),
    )
    await reme_fs.default_memory_store.upsert_file(
        file_meta,
        MemorySource.MEMORY,
        tech_chunks,
    )
    print(f"✓ Inserted {len(tech_chunks)} technical chunks")

    # Test multiple queries
    queries = [
        "artificial intelligence and machine learning",
        "neural networks for image processing",
        "hyperparameter tuning in ML models",
    ]

    for query in queries:
        print(f"\n--- Searching for: '{query}' ---")
        result_json = await reme_fs.memory_search(
            query=query,
            max_results=3,
            min_score=0.0,
        )

        import json

        results = json.loads(result_json)
        print(f"Found {len(results)} results")

        for i, result in enumerate(results, 1):
            print(f"  [{i}] Score: {result['score']:.6f} | {result['path']}")

        assert len(results) > 0, f"Should find results for query: {query}"

    print("\n✓ Technical content search test passed")
    await reme_fs.close()


async def test_memory_search_with_source_filter():
    """Test memory search with source filtering.

    Insert data from different sources and test source-specific searches.
    """
    print("\n" + "=" * 80)
    print("TEST 3: Memory Search with Source Filter")
    print("=" * 80)

    # Initialize ReMeFs with unique store name
    reme_fs = ReMeFs(
        enable_logo=False,
        working_dir=TestConfig.WORKING_DIR,
    )
    await reme_fs.start()

    # Insert MEMORY source data
    personal_chunks = SampleDataGenerator.create_personal_info_chunks("test_source")
    personal_chunks = await reme_fs.default_memory_store.get_chunk_embeddings(personal_chunks)
    personal_meta = SampleDataGenerator.create_file_metadata(
        "memory/personal_info.md",
        len(personal_chunks),
    )
    await reme_fs.default_memory_store.upsert_file(
        personal_meta,
        MemorySource.MEMORY,
        personal_chunks,
    )
    print(f"✓ Inserted {len(personal_chunks)} MEMORY chunks")

    # Insert SESSIONS source data
    session_chunks = SampleDataGenerator.create_session_chunks("test_source")
    session_chunks = await reme_fs.default_memory_store.get_chunk_embeddings(session_chunks)
    session_meta = SampleDataGenerator.create_file_metadata(
        "sessions/2024-01-15.jsonl",
        len(session_chunks),
    )
    await reme_fs.default_memory_store.upsert_file(
        session_meta,
        MemorySource.SESSIONS,
        session_chunks,
    )
    print(f"✓ Inserted {len(session_chunks)} SESSIONS chunks")

    query = "Python programming and async"

    # Search only MEMORY source
    print(f"\n--- Searching MEMORY source for: '{query}' ---")
    # Create a new instance with MEMORY source filter
    reme_fs_memory = ReMeFs(
        enable_logo=False,
        working_dir=TestConfig.WORKING_DIR,
        search_params={"sources": [MemorySource.MEMORY]},
    )
    await reme_fs_memory.start()
    result_json = await reme_fs_memory.memory_search(
        query=query,
        max_results=5,
    )
    await reme_fs_memory.close()

    import json

    memory_results = json.loads(result_json)
    print(f"Found {len(memory_results)} results in MEMORY source")
    for result in memory_results:
        assert result["source"] == MemorySource.MEMORY.value, "Should only return MEMORY source results"

    # Search only SESSIONS source
    print(f"\n--- Searching SESSIONS source for: '{query}' ---")
    # Create a new instance with SESSIONS source filter
    reme_fs_sessions = ReMeFs(
        enable_logo=False,
        working_dir=TestConfig.WORKING_DIR,
        search_params={"sources": [MemorySource.SESSIONS]},
    )
    await reme_fs_sessions.start()
    result_json = await reme_fs_sessions.memory_search(
        query=query,
        max_results=5,
    )
    await reme_fs_sessions.close()
    session_results = json.loads(result_json)
    print(f"Found {len(session_results)} results in SESSIONS source")
    for result in session_results:
        assert result["source"] == MemorySource.SESSIONS.value, "Should only return SESSIONS source results"

    # Search all sources
    print(f"\n--- Searching ALL sources for: '{query}' ---")
    result_json = await reme_fs.memory_search(
        query=query,
        max_results=10,
    )
    all_results = json.loads(result_json)
    print(f"Found {len(all_results)} results across all sources")

    sources_found = {result["source"] for result in all_results}
    print(f"Sources found: {sources_found}")

    print("\n✓ Source filter search test passed")
    await reme_fs.close()


async def test_memory_search_score_filtering():
    """Test memory search with score threshold.

    Test min_score parameter to filter low-relevance results.
    """
    print("\n" + "=" * 80)
    print("TEST 4: Memory Search with Score Filtering")
    print("=" * 80)

    # Initialize ReMeFs with unique store name
    reme_fs = ReMeFs(
        enable_logo=False,
        working_dir=TestConfig.WORKING_DIR,
        default_memory_store_config={
            "backend": "sqlite",
            "store_name": "test_score_filter",
            "embedding_model": "default",
            "fts_enabled": True,
        },
    )
    await reme_fs.start()

    # Insert test data
    chunks = SampleDataGenerator.create_technical_chunks("test_score")
    chunks = await reme_fs.default_memory_store.get_chunk_embeddings(chunks)
    file_meta = SampleDataGenerator.create_file_metadata(
        "memory/technical_notes.md",
        len(chunks),
    )
    await reme_fs.default_memory_store.upsert_file(
        file_meta,
        MemorySource.MEMORY,
        chunks,
    )
    print(f"✓ Inserted {len(chunks)} test chunks")

    query = "machine learning algorithms"

    # Search with different min_score thresholds
    thresholds = [0.0, 0.1, 0.3, 0.5]

    for min_score in thresholds:
        print(f"\n--- Searching with min_score={min_score} ---")
        result_json = await reme_fs.memory_search(
            query=query,
            max_results=10,
            min_score=min_score,
        )

        import json

        results = json.loads(result_json)
        print(f"Found {len(results)} results with min_score >= {min_score}")

        # Verify all results meet threshold
        for result in results:
            assert result["score"] >= min_score, f"Result score {result['score']:.6f} should be >= {min_score}"

        if results:
            print(f"  Top score: {results[0]['score']:.6f}")
            print(f"  Lowest score: {results[-1]['score']:.6f}")

    print("\n✓ Score filtering search test passed")
    await reme_fs.close()


async def test_memory_search_max_results():
    """Test memory search with result limiting.

    Test max_results parameter to limit number of returned results.
    """
    print("\n" + "=" * 80)
    print("TEST 5: Memory Search with Result Limiting")
    print("=" * 80)

    # Initialize ReMeFs with unique store name
    reme_fs = ReMeFs(
        enable_logo=False,
        working_dir=TestConfig.WORKING_DIR,
        default_memory_store_config={
            "backend": "sqlite",
            "store_name": "test_max_results",
            "embedding_model": "default",
            "fts_enabled": True,
        },
    )
    await reme_fs.start()

    # Insert multiple chunks
    personal_chunks = SampleDataGenerator.create_personal_info_chunks("test_max")
    tech_chunks = SampleDataGenerator.create_technical_chunks("test_max")
    all_chunks = personal_chunks + tech_chunks

    all_chunks = await reme_fs.default_memory_store.get_chunk_embeddings(all_chunks)

    # Insert as one file for simplicity
    combined_meta = SampleDataGenerator.create_file_metadata(
        "memory/combined.md",
        len(all_chunks),
    )
    await reme_fs.default_memory_store.upsert_file(
        combined_meta,
        MemorySource.MEMORY,
        all_chunks,
    )

    print(f"✓ Inserted {len(all_chunks)} total chunks")

    query = "programming and technology"

    # Test different max_results values
    result_limits = [1, 2, 3, 5, 20]

    for max_results in result_limits:
        print(f"\n--- Searching with max_results={max_results} ---")
        result_json = await reme_fs.memory_search(
            query=query,
            max_results=max_results,
            min_score=0.0,
        )

        import json

        results = json.loads(result_json)
        print(f"Requested {max_results}, got {len(results)} results")

        assert len(results) <= max_results, f"Should return at most {max_results} results, got {len(results)}"

    print("\n✓ Result limiting search test passed")
    await reme_fs.close()


async def test_memory_search_hybrid_mode():
    """Test memory search with hybrid mode (vector + keyword).

    Test different hybrid configurations and weights.
    """
    print("\n" + "=" * 80)
    print("TEST 6: Memory Search with Hybrid Mode")
    print("=" * 80)

    # Initialize ReMeFs with unique store name
    reme_fs = ReMeFs(
        enable_logo=False,
        working_dir=TestConfig.WORKING_DIR,
        default_memory_store_config={
            "backend": "sqlite",
            "store_name": "test_hybrid",
            "embedding_model": "default",
            "fts_enabled": True,
        },
    )
    await reme_fs.start()

    # Insert test data
    chunks = SampleDataGenerator.create_technical_chunks("test_hybrid")
    chunks = await reme_fs.default_memory_store.get_chunk_embeddings(chunks)
    file_meta = SampleDataGenerator.create_file_metadata(
        "memory/technical_notes.md",
        len(chunks),
    )
    await reme_fs.default_memory_store.upsert_file(
        file_meta,
        MemorySource.MEMORY,
        chunks,
    )
    print(f"✓ Inserted {len(chunks)} test chunks")

    query = "neural networks"

    # Test with hybrid enabled
    print(f"\n--- Hybrid search (enabled) for: '{query}' ---")
    # Create instance with hybrid enabled
    reme_fs_hybrid = ReMeFs(
        enable_logo=False,
        working_dir=TestConfig.WORKING_DIR,
        default_memory_store_config={
            "backend": "sqlite",
            "store_name": "test_hybrid",
            "embedding_model": "default",
            "fts_enabled": True,
        },
        search_params={
            "vector_weight": 0.7,
        },
    )
    await reme_fs_hybrid.start()
    result_json_hybrid = await reme_fs_hybrid.memory_search(
        query=query,
        max_results=5,
    )
    await reme_fs_hybrid.close()

    import json

    hybrid_results = json.loads(result_json_hybrid)
    print(f"Hybrid search found {len(hybrid_results)} results")
    print_search_results(hybrid_results, query, "HYBRID SEARCH RESULTS")

    # Test with hybrid disabled (vector only)
    print(f"\n--- Vector-only search for: '{query}' ---")
    # Create instance with hybrid disabled
    reme_fs_vector = ReMeFs(
        enable_logo=False,
        working_dir=TestConfig.WORKING_DIR,
        default_memory_store_config={
            "backend": "sqlite",
            "store_name": "test_hybrid",
            "embedding_model": "default",
            "fts_enabled": True,
        },
        search_params={},
    )
    await reme_fs_vector.start()
    result_json_vector = await reme_fs_vector.memory_search(
        query=query,
        max_results=5,
    )
    await reme_fs_vector.close()

    vector_results = json.loads(result_json_vector)
    print(f"Vector search found {len(vector_results)} results")
    print_search_results(vector_results, query, "VECTOR-ONLY SEARCH RESULTS")

    # Test different weight configurations
    print("\n--- Testing different hybrid weights ---")
    weight_configs = [
        (0.9, 0.1),  # Mostly vector
        (0.5, 0.5),  # Balanced
        (0.3, 0.7),  # Mostly text
    ]

    for vec_weight, text_weight in weight_configs:
        # Create instance with specific weights
        reme_fs_weights = ReMeFs(
            enable_logo=False,
            working_dir=TestConfig.WORKING_DIR,
            default_memory_store_config={
                "backend": "sqlite",
                "store_name": "test_hybrid",
                "embedding_model": "default",
                "fts_enabled": True,
            },
            search_params={
                "vector_weight": vec_weight,
            },
        )
        await reme_fs_weights.start()
        result_json = await reme_fs_weights.memory_search(
            query=query,
            max_results=5,
        )
        await reme_fs_weights.close()
        results = json.loads(result_json)
        print(f"  Vector:{vec_weight}/Text:{text_weight} -> {len(results)} results")

    print("\n✓ Hybrid mode search test passed")
    await reme_fs.close()


async def cleanup_test_data():
    """Clean up test data directory."""
    print("\n" + "=" * 80)
    print("CLEANUP: Removing test data")
    print("=" * 80)

    test_dir = Path(TestConfig.WORKING_DIR)
    if test_dir.exists():
        shutil.rmtree(test_dir)
        print(f"✓ Removed test directory: {test_dir}")
    else:
        print(f"⊘ Test directory does not exist: {test_dir}")


# ==================== Main Entry Point ====================


async def main():
    """Run all memory search tests."""
    print("\n" + "=" * 80)
    print("ReMeFs Memory Search Interface Tests")
    print("=" * 80)
    print("\nThis test suite validates the memory_search() function:")
    print("  1. Basic semantic search functionality")
    print("  2. Technical content search")
    print("  3. Source filtering (MEMORY, SESSIONS)")
    print("  4. Score threshold filtering")
    print("  5. Result limiting (max_results)")
    print("  6. Hybrid mode (vector + keyword search)")
    print("=" * 80)

    try:
        # Run tests
        await test_memory_search_basic()
        await test_memory_search_technical_content()
        await test_memory_search_with_source_filter()
        await test_memory_search_score_filtering()
        await test_memory_search_max_results()
        await test_memory_search_hybrid_mode()

        print("\n" + "=" * 80)
        print("✓ All memory search tests passed!")
        print("=" * 80)

    finally:
        # Cleanup
        await cleanup_test_data()

    print("\nNote: These tests require:")
    print("  - Valid API keys for embedding model")
    print("  - sqlite-vec extension for vector search")
    print("  - FTS5 enabled for keyword search")


if __name__ == "__main__":
    asyncio.run(main())
