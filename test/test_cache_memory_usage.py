"""
Calculate and measure the memory usage of embedding cache.

This script estimates and measures the actual memory footprint
of the embedding cache at different sizes.
"""

import sys
from collections import OrderedDict


def calculate_theoretical_memory():
    """Calculate theoretical memory usage for embedding cache."""
    print("=" * 70)
    print("Theoretical Memory Usage Calculation")
    print("=" * 70)

    # Constants
    dimensions = 1024  # Default embedding dimensions
    bytes_per_float = 8  # Python float is 64-bit
    cache_sizes = [1000, 5000, 10000, 50000, 100000]

    print("\nAssumptions:")
    print(f"  • Embedding dimensions: {dimensions}")
    print(f"  • Bytes per float: {bytes_per_float}")
    print("  • Hash key (SHA256 hex): 64 characters ≈ 128 bytes (UTF-8)")
    print("  • Python list overhead: ~56 bytes")
    print("  • Python string overhead: ~50 bytes")
    print("  • OrderedDict per-entry overhead: ~230 bytes")

    # Memory per single cache entry
    embedding_size = dimensions * bytes_per_float  # Vector data
    list_overhead = 56  # Python list object overhead
    hash_key_size = 64 + 50  # String chars + string object overhead
    ordereddict_entry_overhead = 230  # Dict entry + ordering overhead

    entry_size = embedding_size + list_overhead + hash_key_size + ordereddict_entry_overhead

    print(f"\n{'─' * 70}")
    print("Memory per cache entry:")
    print(f"  • Embedding vector: {embedding_size:,} bytes ({embedding_size/1024:.1f} KB)")
    print(f"  • List overhead: {list_overhead} bytes")
    print(f"  • Hash key: {hash_key_size} bytes")
    print(f"  • OrderedDict overhead: {ordereddict_entry_overhead} bytes")
    print(f"  • Total per entry: {entry_size:,} bytes ({entry_size/1024:.2f} KB)")

    print(f"\n{'─' * 70}")
    print("Memory usage at different cache sizes:")
    print(f"{'─' * 70}")
    print(f"{'Cache Size':>12} | {'Memory (MB)':>12} | {'Memory (GB)':>12}")
    print(f"{'─' * 70}")

    for size in cache_sizes:
        total_bytes = size * entry_size
        mb = total_bytes / (1024 * 1024)
        gb = total_bytes / (1024 * 1024 * 1024)
        print(f"{size:>12,} | {mb:>12.2f} | {gb:>12.4f}")

    print(f"{'─' * 70}")

    # Highlight default size
    default_size = 10000
    default_memory_mb = (default_size * entry_size) / (1024 * 1024)

    print(f"\n✨ Default cache size (max_cache_size={default_size:,}):")
    print(f"   Estimated memory: ~{default_memory_mb:.1f} MB")
    print("\n💡 Recommendation:")
    print("   • For memory-constrained environments: 1,000-5,000 (~8-42 MB)")
    print("   • For balanced performance: 10,000 (~84 MB) [DEFAULT]")
    print("   • For high-throughput applications: 50,000-100,000 (~420-840 MB)")
    print("   • To disable cache: max_cache_size=0")


def measure_actual_memory():
    """Measure actual memory usage with real data."""
    print(f"\n\n{'=' * 70}")
    print("Actual Memory Measurement")
    print(f"{'=' * 70}")

    import random

    # Create a sample cache
    cache = OrderedDict()
    dimensions = 1024

    test_sizes = [100, 1000, 5000, 10000]

    print("\nCreating embedding cache with random data...")
    print(f"{'─' * 70}")
    print(f"{'Cache Size':>12} | {'Memory (MB)':>12} | {'Per Entry (KB)':>15}")
    print(f"{'─' * 70}")

    for size in test_sizes:
        # Clear cache
        cache.clear()

        # Fill with sample data
        for i in range(size):
            hash_key = f"hash_{i:064x}"  # 64-char hex string
            embedding = [random.random() for _ in range(dimensions)]
            cache[hash_key] = embedding

        # Measure memory (approximate)
        # Calculate size of all embeddings
        total_bytes = 0
        for key, value in cache.items():
            total_bytes += sys.getsizeof(key)  # Key size
            total_bytes += sys.getsizeof(value)  # List object
            total_bytes += len(value) * sys.getsizeof(float())  # Float elements

        # Add OrderedDict overhead
        total_bytes += sys.getsizeof(cache)

        mb = total_bytes / (1024 * 1024)
        per_entry_kb = total_bytes / size / 1024

        print(f"{size:>12,} | {mb:>12.2f} | {per_entry_kb:>15.2f}")

    print(f"{'─' * 70}")

    # Measure default size
    cache.clear()
    default_size = 10000
    print(f"\n📊 Measuring default size ({default_size:,} entries)...")

    for i in range(default_size):
        hash_key = f"hash_{i:064x}"
        embedding = [random.random() for _ in range(dimensions)]
        cache[hash_key] = embedding

    total_bytes = sys.getsizeof(cache)
    for key, value in cache.items():
        total_bytes += sys.getsizeof(key)
        total_bytes += sys.getsizeof(value)
        total_bytes += len(value) * sys.getsizeof(float())

    mb = total_bytes / (1024 * 1024)

    print(f"   Actual memory usage: {mb:.2f} MB")
    print(f"   Per entry: {total_bytes / default_size / 1024:.2f} KB")


def print_usage_guidelines():
    """Print guidelines for choosing cache size."""
    print(f"\n\n{'=' * 70}")
    print("Cache Size Selection Guidelines")
    print(f"{'=' * 70}")

    print("\n📋 How to choose the right cache size:\n")

    print("1️⃣  Estimate your query patterns:")
    print("   • How many unique texts will you embed?")
    print("   • What's the repetition rate?")
    print("   • Example: 1,000 unique texts with 70% repetition → cache_size=1,000\n")

    print("2️⃣  Consider available memory:")
    print("   • ~84 MB per 10,000 entries (1024-dim embeddings)")
    print("   • ~420 MB per 50,000 entries")
    print("   • Scale linearly: ~8.4 MB per 1,000 entries\n")

    print("3️⃣  Monitor cache size:\n")

    print("4️⃣  Configuration examples:\n")

    examples = [
        ("Embedded device / IoT", "max_cache_size=100", "~840 KB"),
        ("Development / Testing", "max_cache_size=1000", "~8.4 MB"),
        ("Production (default)", "max_cache_size=10000", "~84 MB"),
        ("High-volume service", "max_cache_size=50000", "~420 MB"),
        ("Disable cache", "max_cache_size=0", "0 MB"),
    ]

    print(f"{'Use Case':<25} | {'Configuration':<22} | {'Memory':<10}")
    print(f"{'─' * 25}-+-{'─' * 22}-+-{'─' * 10}")
    for use_case, config, memory in examples:
        print(f"{use_case:<25} | {config:<22} | {memory:<10}")

    print("\n💡 Code example:\n")
    print("   # Default configuration (recommended)")
    print("   model = OpenAIEmbeddingModel(")
    print("       model_name='text-embedding-v4',")
    print("       dimensions=1024,")
    print("       max_cache_size=10000,  # ~84 MB")
    print("   )")
    print()
    print("   # Memory-constrained configuration")
    print("   model = OpenAIEmbeddingModel(")
    print("       model_name='text-embedding-v4',")
    print("       dimensions=1024,")
    print("       max_cache_size=1000,  # ~8.4 MB")
    print("   )")
    print()
    print("   # Monitor cache")
    print("   print(f'Cache entries: {len(model._embedding_cache)}')")


if __name__ == "__main__":
    calculate_theoretical_memory()
    measure_actual_memory()
    print_usage_guidelines()

    print(f"\n{'=' * 70}")
    print("✅ Memory analysis complete")
    print(f"{'=' * 70}\n")
