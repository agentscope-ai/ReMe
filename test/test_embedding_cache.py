"""
Async unit tests for embedding cache functionality.

Tests cover:
- Cache hit/miss tracking
- LRU eviction policy
- Cache statistics
- Performance improvements with repeated queries
- Cache clearing

Usage:
    python test_embedding_cache.py
"""

# flake8: noqa: E402
# pylint: disable=C0413

import asyncio
import shutil
import tempfile
from typing import List

from reme.core.utils import load_env

load_env()

from reme.core.embedding import OpenAIEmbeddingModel


def get_test_texts() -> List[str]:
    """Create test texts for embedding cache testing."""
    return [
        "What is machine learning?",
        "How does neural network work?",
        "Explain artificial intelligence",
        "Define deep learning",
        "What is data science?",
    ]


async def test_cache_basic_functionality():
    """Test basic cache hit/miss functionality."""
    print(f"\n{'='*60}")
    print("Test 1: Basic Cache Functionality")
    print(f"{'='*60}")

    temp_dir = tempfile.mkdtemp()
    try:
        model = OpenAIEmbeddingModel(
            model_name="text-embedding-v4",
            dimensions=1024,
            max_cache_size=100,
            max_retries=2,
            raise_exception=True,
            cache_dir=temp_dir,
        )

        test_text = "Hello, this is a test sentence for embedding cache."

        print(f"Input text: {test_text}")
        print(f"Cache directory: {temp_dir}")

        # First call - should be a cache miss
        print("\n1️⃣ First embedding call (cold cache):")
        embedding1 = await model.get_embedding(test_text)
        stats1 = model.get_cache_stats()

        print(f"   Embedding dimension: {len(embedding1)}")
        print(f"   Cache size: {stats1['cache_size']}")
        print(f"   Cache hits: {stats1['cache_hits']}")
        print(f"   Cache misses: {stats1['cache_misses']}")
        print(f"   Hit rate: {stats1['hit_rate']:.2%}")

        assert len(embedding1) == 1024, "Embedding dimension mismatch"
        assert stats1["cache_misses"] == 1, "Should have 1 cache miss"
        assert stats1["cache_hits"] == 0, "Should have 0 cache hits"
        assert stats1["cache_size"] == 1, "Cache should have 1 entry"

        # Second call with same text - should be a cache hit
        print("\n2️⃣ Second embedding call (same text):")
        embedding2 = await model.get_embedding(test_text)
        stats2 = model.get_cache_stats()

        print(f"   Cache hits: {stats2['cache_hits']}")
        print(f"   Cache misses: {stats2['cache_misses']}")
        print(f"   Hit rate: {stats2['hit_rate']:.2%}")

        assert embedding1 == embedding2, "Cached embedding should be identical"
        assert stats2["cache_hits"] == 1, "Should have 1 cache hit"
        assert stats2["cache_misses"] == 1, "Should still have 1 cache miss"
        assert stats2["hit_rate"] == 0.5, "Hit rate should be 50%"

        await model.close()
        print("\n✓ PASSED: Basic cache functionality works correctly")
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


async def test_batch_cache_efficiency():
    """Test cache efficiency with batch embeddings including duplicates."""
    print(f"\n{'='*60}")
    print("Test 2: Batch Cache Efficiency")
    print(f"{'='*60}")

    temp_dir = tempfile.mkdtemp()
    try:
        model = OpenAIEmbeddingModel(
            model_name="text-embedding-v4",
            dimensions=1024,
            max_cache_size=1000,
            max_retries=2,
            raise_exception=True,
            cache_dir=temp_dir,
        )

        texts = get_test_texts()

        # Create a list with duplicates
        texts_with_duplicates = texts + texts[:3]  # 5 unique + 3 duplicates = 8 total

        print(f"Processing {len(texts_with_duplicates)} texts (5 unique + 3 duplicates)")

        # First batch
        print("\n1️⃣ First batch (cold cache):")
        embeddings1 = await model.get_embeddings(texts)
        stats1 = model.get_cache_stats()

        print(f"   Embeddings generated: {len(embeddings1)}")
        print(f"   Cache size: {stats1['cache_size']}")
        print(f"   Cache misses: {stats1['cache_misses']}")
        print(f"   Cache hits: {stats1['cache_hits']}")

        assert len(embeddings1) == len(texts), "Embeddings count mismatch"
        assert stats1["cache_size"] == len(texts), f"Cache should have {len(texts)} entries"
        assert stats1["cache_misses"] == len(texts), "All should be cache misses"

        # Second batch with duplicates
        print("\n2️⃣ Second batch (with duplicates):")
        embeddings2 = await model.get_embeddings(texts_with_duplicates)
        stats2 = model.get_cache_stats()

        print(f"   Embeddings generated: {len(embeddings2)}")
        print(f"   Cache hits: {stats2['cache_hits']}")
        print(f"   Cache misses: {stats2['cache_misses']}")
        print(f"   Hit rate: {stats2['hit_rate']:.2%}")

        assert len(embeddings2) == len(texts_with_duplicates), "Embeddings count mismatch"
        assert stats2["cache_hits"] >= 3, "Should have at least 3 cache hits from duplicates"

        # Verify embeddings are identical for duplicated texts
        for i in range(3):
            assert embeddings2[i] == embeddings2[len(texts) + i], f"Duplicate {i} should have identical embedding"

        await model.close()
        print("\n✓ PASSED: Batch cache efficiently handles duplicates")
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


async def test_cache_lru_eviction():
    """Test LRU cache eviction policy."""
    print(f"\n{'='*60}")
    print("Test 3: LRU Cache Eviction")
    print(f"{'='*60}")

    temp_dir = tempfile.mkdtemp()
    try:
        # Create model with small cache size
        model = OpenAIEmbeddingModel(
            model_name="text-embedding-v4",
            dimensions=1024,
            max_cache_size=3,  # Small cache for testing eviction
            max_retries=2,
            raise_exception=True,
            cache_dir=temp_dir,
        )

        texts = get_test_texts()[:5]  # Use 5 texts, cache size is 3

        print(f"Cache size limit: {model.max_cache_size}")
        print(f"Number of unique texts: {len(texts)}")

        # Fill cache beyond capacity
        print("\n1️⃣ Filling cache with 5 texts (capacity = 3):")
        for i, text in enumerate(texts):
            await model.get_embedding(text)
            stats = model.get_cache_stats()
            print(
                f"   After text {i+1}: cache_size={stats['cache_size']}, "
                f"hits={stats['cache_hits']}, misses={stats['cache_misses']}",
            )

        final_stats = model.get_cache_stats()
        assert final_stats["cache_size"] <= 3, "Cache size should not exceed max_cache_size"
        assert final_stats["cache_misses"] == 5, "Should have 5 cache misses for 5 unique texts"

        # Access the most recent entries - should be cache hits
        print("\n2️⃣ Accessing recent entries (should be cached):")
        recent_texts = texts[-3:]  # Last 3 texts should still be in cache

        for i, text in enumerate(recent_texts):
            await model.get_embedding(text)
            stats = model.get_cache_stats()
            print(f"   Text {len(texts) - 3 + i + 1}: hits={stats['cache_hits']}")

        final_stats = model.get_cache_stats()
        assert final_stats["cache_hits"] == 3, "Should have 3 cache hits for recent entries"

        # Access oldest entries - should be cache misses (evicted)
        print("\n3️⃣ Accessing oldest entries (should be evicted):")
        old_texts = texts[:2]  # First 2 texts should have been evicted

        before_misses = final_stats["cache_misses"]
        for i, text in enumerate(old_texts):
            await model.get_embedding(text)
            stats = model.get_cache_stats()
            print(f"   Text {i + 1}: misses={stats['cache_misses']}")

        final_stats = model.get_cache_stats()
        assert final_stats["cache_misses"] == before_misses + 2, "Should have 2 more cache misses for evicted entries"

        await model.close()
        print("\n✓ PASSED: LRU eviction works correctly")
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


async def test_cache_stats_and_clear():
    """Test cache statistics tracking and clearing."""
    print(f"\n{'='*60}")
    print("Test 4: Cache Statistics and Clearing")
    print(f"{'='*60}")

    temp_dir = tempfile.mkdtemp()
    try:
        model = OpenAIEmbeddingModel(
            model_name="text-embedding-v4",
            dimensions=1024,
            max_cache_size=100,
            max_retries=2,
            raise_exception=True,
            cache_dir=temp_dir,
        )

        texts = get_test_texts()

        # Generate some cache activity
        print("\n1️⃣ Generating cache activity:")
        await model.get_embeddings(texts)
        await model.get_embeddings(texts[:3])  # Repeat first 3

        stats = model.get_cache_stats()
        print(f"   Cache size: {stats['cache_size']}")
        print(f"   Max cache size: {stats['max_cache_size']}")
        print(f"   Cache hits: {stats['cache_hits']}")
        print(f"   Cache misses: {stats['cache_misses']}")
        print(f"   Hit rate: {stats['hit_rate']:.2%}")

        assert stats["cache_size"] > 0, "Cache should not be empty"
        assert stats["cache_hits"] >= 3, "Should have at least 3 cache hits"
        assert "hit_rate" in stats, "Stats should include hit_rate"

        # Clear cache
        print("\n2️⃣ Clearing cache:")
        model.clear_cache()
        stats_after_clear = model.get_cache_stats()

        print(f"   Cache size after clear: {stats_after_clear['cache_size']}")
        print(f"   Hits after clear: {stats_after_clear['cache_hits']}")
        print(f"   Misses after clear: {stats_after_clear['cache_misses']}")
        print(f"   Hit rate after clear: {stats_after_clear['hit_rate']:.2%}")

        assert stats_after_clear["cache_size"] == 0, "Cache should be empty after clear"
        assert stats_after_clear["cache_hits"] == 0, "Hits should be reset"
        assert stats_after_clear["cache_misses"] == 0, "Misses should be reset"
        assert stats_after_clear["hit_rate"] == 0.0, "Hit rate should be 0"

        await model.close()
        print("\n✓ PASSED: Cache statistics and clearing work correctly")
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


async def test_cache_disabled():
    """Test behavior when cache is disabled (max_cache_size=0)."""
    print(f"\n{'='*60}")
    print("Test 5: Cache Disabled")
    print(f"{'='*60}")

    temp_dir = tempfile.mkdtemp()
    try:
        model = OpenAIEmbeddingModel(
            model_name="text-embedding-v4",
            dimensions=1024,
            max_cache_size=0,  # Disable cache
            max_retries=2,
            raise_exception=True,
            cache_dir=temp_dir,
        )

        test_text = "Test text with cache disabled"

        print(f"Cache size limit: {model.max_cache_size} (disabled)")
        print(f"Input text: {test_text}")

        # Call twice with same text
        print("\n1️⃣ First call:")
        embedding1 = await model.get_embedding(test_text)
        stats1 = model.get_cache_stats()
        print(f"   Cache size: {stats1['cache_size']}")
        print(f"   Cache misses: {stats1['cache_misses']}")

        print("\n2️⃣ Second call (same text):")
        embedding2 = await model.get_embedding(test_text)
        stats2 = model.get_cache_stats()
        print(f"   Cache size: {stats2['cache_size']}")
        print(f"   Cache misses: {stats2['cache_misses']}")
        print(f"   Cache hits: {stats2['cache_hits']}")

        assert stats2["cache_size"] == 0, "Cache should remain empty when disabled"
        assert stats2["cache_misses"] == 2, "Both calls should be cache misses"
        assert stats2["cache_hits"] == 0, "Should have no cache hits when disabled"
        assert embedding1 == embedding2, "Embeddings should still be consistent"

        await model.close()
        print("\n✓ PASSED: Cache correctly disabled when max_cache_size=0")
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


async def test_cache_performance_demo():
    """Demonstrate cache performance improvements."""
    print(f"\n{'='*60}")
    print("Test 6: Cache Performance Demo")
    print(f"{'='*60}")

    temp_dir = tempfile.mkdtemp()
    try:
        model = OpenAIEmbeddingModel(
            model_name="text-embedding-v4",
            dimensions=1024,
            max_cache_size=1000,
            max_retries=2,
            raise_exception=True,
            cache_dir=temp_dir,
        )

        texts = get_test_texts()

        # Create a realistic workload with many repeated queries
        workload = texts * 3  # 15 queries total, 5 unique

        print(f"\nProcessing {len(workload)} queries ({len(texts)} unique texts)")
        print("This simulates a realistic scenario with repeated queries\n")

        # Process all queries
        for i, text in enumerate(workload, 1):
            await model.get_embedding(text)
            if i % 5 == 0:  # Report every 5 queries
                stats = model.get_cache_stats()
                print(
                    f"After {i:2d} queries: hits={stats['cache_hits']:2d}, "
                    f"misses={stats['cache_misses']:2d}, "
                    f"hit_rate={stats['hit_rate']:5.1%}",
                )

        final_stats = model.get_cache_stats()
        total_requests = final_stats["cache_hits"] + final_stats["cache_misses"]

        print(f"\n{'─'*60}")
        print("📊 Final Statistics:")
        print(f"{'─'*60}")
        print(f"  Total queries: {total_requests}")
        print(f"  Unique texts: {len(texts)}")
        print(f"  Cache hits: {final_stats['cache_hits']}")
        print(f"  Cache misses: {final_stats['cache_misses']}")
        print(f"  Hit rate: {final_stats['hit_rate']:.1%}")
        print(f"  Cache size: {final_stats['cache_size']}/{final_stats['max_cache_size']}")
        print(f"{'─'*60}")
        print(
            f"💰 API calls saved: {final_stats['cache_hits']} out of {total_requests} "
            f"({final_stats['cache_hits']/total_requests*100:.1f}%)",
        )
        print(f"{'─'*60}")

        assert final_stats["cache_hits"] == 10, "Should have 10 cache hits (2 repeats × 5 texts)"
        assert final_stats["cache_misses"] == 5, "Should have 5 cache misses (5 unique texts)"
        assert final_stats["hit_rate"] > 0.6, "Hit rate should be > 60%"

        await model.close()
        print("\n✓ PASSED: Cache provides significant performance improvement")
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


async def main():
    """Run all cache tests."""
    print("\n" + "#" * 60)
    print("# EMBEDDING CACHE TESTS")
    print("#" * 60)

    try:
        await test_cache_basic_functionality()
        await test_batch_cache_efficiency()
        await test_cache_lru_eviction()
        await test_cache_stats_and_clear()
        await test_cache_disabled()
        await test_cache_performance_demo()

        print("\n" + "=" * 60)
        print("✅ ALL CACHE TESTS PASSED")
        print("=" * 60)
        print("\nKey takeaways:")
        print("  • Cache correctly tracks hits/misses")
        print("  • LRU eviction works as expected")
        print("  • Duplicate queries are efficiently cached")
        print("  • Cache can be disabled or cleared")
        print("  • Significant performance improvement with realistic workloads")
        print("=" * 60 + "\n")

    except Exception as e:
        print(f"\n✗ TEST FAILED: {type(e).__name__}: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())
