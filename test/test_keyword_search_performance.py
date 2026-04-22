"""Performance test for LocalFileStore keyword_search.

Tests keyword_search efficiency with:
- Query length: 20 characters
- Chunk count: 1000 chunks
"""

import asyncio
import hashlib
import random
import shutil
import time
from pathlib import Path

from reme.core.enumeration.memory_source import MemorySource
from reme.core.file_store.local_file_store import LocalFileStore
from reme.core.schema.memory_chunk import MemoryChunk


def generate_random_text(length: int = 200) -> str:
    """Generate random text content."""
    words = [
        "python",
        "function",
        "class",
        "memory",
        "search",
        "algorithm",
        "database",
        "vector",
        "embedding",
        "chunk",
        "file",
        "store",
        "query",
        "result",
        "performance",
        "test",
        "data",
        "index",
        "keyword",
        "text",
        "content",
        "process",
        "system",
        "module",
        "import",
        "return",
        "value",
        "parameter",
        "method",
        "object",
        "instance",
        "variable",
        "constant",
        "string",
        "integer",
        "float",
        "list",
        "dictionary",
        "tuple",
        "set",
        "array",
        "matrix",
    ]
    text_words = []
    current_length = 0
    while current_length < length:
        word = random.choice(words)
        text_words.append(word)
        current_length += len(word) + 1  # +1 for space
    return " ".join(text_words)[:length]


def create_test_chunks(count: int = 1000, text_length: int = 10000) -> list[MemoryChunk]:
    """Create test chunks for performance testing."""
    chunks = []
    for i in range(count):
        text = generate_random_text(text_length)
        chunk = MemoryChunk(
            id=f"perf_test_chunk_{i}",
            path=f"/test/file_{i % 100}.py",
            source=MemorySource.MEMORY,
            start_line=i * 10 + 1,
            end_line=(i + 1) * 10,
            text=text,
            hash=hashlib.md5(text.encode()).hexdigest(),
            embedding=None,
            metadata={"index": i},
        )
        chunks.append(chunk)
    return chunks


async def run_performance_test():
    """Run keyword_search performance test."""
    # pylint: disable=protected-access
    # Setup
    test_db_path = Path("./test_keyword_perf")
    test_db_path.mkdir(exist_ok=True)

    store = LocalFileStore(
        db_path=test_db_path,
        store_name="perf_test",
        vector_enabled=False,  # Disable vector search for this test
        fts_enabled=True,
    )
    await store.start()

    # Create test data
    print("Creating 1000 test chunks...")
    chunks = create_test_chunks(1000)

    # Manually add chunks to store (bypass embedding)
    for chunk in chunks:
        store._chunks[chunk.id] = chunk

    print(f"Loaded {len(store._chunks)} chunks into memory")

    # Create a 20-character query
    query = "python function data"  # 20 characters including spaces
    print(f"Query: '{query}' (length: {len(query)})")

    # Warmup
    await store.keyword_search(query, limit=10)

    # Performance test - multiple runs
    num_runs = 100
    times = []

    print(f"\nRunning {num_runs} iterations...")

    for _ in range(num_runs):
        start = time.perf_counter()
        _ = await store.keyword_search(query, limit=10)
        elapsed = time.perf_counter() - start
        times.append(elapsed)

    # Statistics
    avg_time = sum(times) / len(times)
    min_time = min(times)
    max_time = max(times)

    print("\n" + "=" * 50)
    print("Performance Results (keyword_search)")
    print("=" * 50)
    print(f"Query length: {len(query)} characters")
    print(f"Chunk count: {len(store._chunks)}")
    print(f"Iterations: {num_runs}")
    print("-" * 50)
    print(f"Average time: {avg_time * 1000:.4f} ms")
    print(f"Min time:     {min_time * 1000:.4f} ms")
    print(f"Max time:     {max_time * 1000:.4f} ms")
    print(f"Total time:   {sum(times) * 1000:.2f} ms")
    print("=" * 50)

    # Cleanup
    await store.close()

    # Remove test directory
    shutil.rmtree(test_db_path, ignore_errors=True)


if __name__ == "__main__":
    asyncio.run(run_performance_test())
