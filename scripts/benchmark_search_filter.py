"""Compare prepared filtering with the unchanged dict matcher on real searches.

Run: python -m scripts.benchmark_search_filter --repeats 7
Setup is excluded; deterministic chunks and fake query embeddings need no files/network.
"""

# pylint: disable=protected-access

import argparse
import asyncio
import json
import platform
import statistics
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import numpy as np

from reme.components.file_store import LocalFileStore
from reme.components.keyword_index import BM25Index
from reme.components.tokenizer import RegexTokenizer
from reme.schema import FileChunk


async def measure(operation, repeats):
    """Warm up and report per-call latency in milliseconds."""
    for _ in range(2):
        await operation()
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        await operation()
        samples.append((time.perf_counter() - start) * 1000)
    return {"median_ms": statistics.median(samples), "p95_ms": float(np.percentile(samples, 95))}


async def benchmark(size, repeats):
    """Measure predicate traversal, keyword_search and vector_search separately."""
    store = LocalFileStore(embedding_store="")
    index = BM25Index()
    index.tokenizer = RegexTokenizer(filter_stopwords=False)
    store.keyword_index = index
    store.embedding_store = SimpleNamespace(dimensions=2, is_healthy=True)
    store._get_query_embedding = AsyncMock(return_value=np.array([1.0, 0.0]))
    for i in range(size):
        store.file_chunks[str(i)] = FileChunk(
            id=str(i),
            path=f"daily/{i % 2000}.md",
            text="alpha beta",
            embedding=np.array([1.0, 0.0]),
        )
    await index.add_docs({key: chunk.text for key, chunk in store.file_chunks.items()})
    results = []
    for path_count in (1, 100, 1000):
        config = {"paths": [f"daily/{i}.md" for i in range(path_count)]}

        async def predicate(config=config):
            matches = store._prepare_search_filter(config)
            return [chunk.id for chunk in store.file_chunks.values() if matches(chunk)]

        async def keyword(config=config):
            return [(c.id, c.scores) for c in await store.keyword_search("alpha", 10, config)]

        async def vector(config=config):
            return [(c.id, c.scores) for c in await store.vector_search("alpha", 10, config)]

        def legacy(search_filter):
            return lambda chunk: store._matches_search_filter(chunk, search_filter)

        for name, operation in (("predicate", predicate), ("keyword_search", keyword), ("vector_search", vector)):
            expected = await operation()
            prepared = await measure(operation, repeats)
            with patch.object(store, "_prepare_search_filter", side_effect=legacy):
                assert await operation() == expected
                baseline = await measure(operation, repeats)
            results.append(
                {
                    "chunks": size,
                    "paths": path_count,
                    "operation": name,
                    "baseline": baseline,
                    "prepared": prepared,
                },
            )
    return results


async def main():
    """Print reproducible environment and results as JSON."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", nargs="+", type=int, default=[20000, 100000])
    parser.add_argument("--repeats", type=int, default=7)
    args = parser.parse_args()
    rows = []
    for size in args.sizes:
        rows.extend(await benchmark(size, args.repeats))
    print(
        json.dumps(
            {
                "python": platform.python_version(),
                "platform": platform.platform(),
                "numpy": np.__version__,
                "repeats": args.repeats,
                "results": rows,
            },
            indent=2,
        ),
    )


if __name__ == "__main__":
    asyncio.run(main())
