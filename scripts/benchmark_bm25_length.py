"""Compare live-length lookup and real BM25 queries with the previous scan.

Run from the repository: python -m scripts.benchmark_bm25_length --repeats 15
Setup is excluded; no files, models or network are used. JSON goes to stdout.
"""

# pylint: disable=protected-access

import argparse
import asyncio
import json
import platform
import statistics
import time
from unittest.mock import patch

import numpy as np

from reme.components.keyword_index import BM25Index
from reme.components.tokenizer import RegexTokenizer


def scanning_length(index):
    """Original length calculation, including lazy-deleted slots."""
    return int(index._doc_lens[~index._deleted].sum())


async def measure(operation, repeats):
    """Warm up and report milliseconds per operation."""
    for _ in range(3):
        await operation()
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        await operation()
        samples.append((time.perf_counter() - start) * 1000)
    return {"median_ms": statistics.median(samples), "p95_ms": float(np.percentile(samples, 95))}


async def benchmark(size, repeats):
    """Exercise a rare-term query globally and within 100 selected documents."""
    index = BM25Index()
    index.tokenizer = RegexTokenizer(filter_stopwords=False)
    for start in range(0, size, 10000):
        docs = {}
        for i in range(start, min(start + 10000, size)):
            docs[str(i)] = "alpha beta rare" if i % 1000 == 0 else "alpha beta"
        await index.add_docs(docs)
    selected = [str(i) for i in range(min(size, 100))]

    async def length():
        return index.total_len

    async def query():
        return await index.retrieve("rare", 10)

    async def filtered():
        return await index.retrieve_filtered("rare", 10, selected)

    results = []
    for deleted_fraction in (0.0, 0.5):
        if deleted_fraction:
            await index.delete_docs([str(i) for i in range(1, size, 2)])
        for name, operation in (("length", length), ("retrieve", query), ("retrieve_filtered", filtered)):
            expected = await operation()
            cached = await measure(operation, repeats)
            with patch.object(BM25Index, "total_len", property(scanning_length)):
                assert await operation() == expected
                baseline = await measure(operation, repeats)
            results.append(
                {
                    "slots": size,
                    "deleted_fraction": deleted_fraction,
                    "operation": name,
                    "baseline": baseline,
                    "incremental": cached,
                },
            )
    return results


async def main():
    """Print reproducible environment and benchmark results."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", nargs="+", type=int, default=[10000, 100000, 1000000])
    parser.add_argument("--repeats", type=int, default=15)
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
