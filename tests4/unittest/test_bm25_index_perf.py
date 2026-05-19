"""BM25Index performance tests for add_docs and retrieve."""

import asyncio
import os
import random
import tempfile
import time

from reme4.components.keyword_index import BM25Index
from reme4.components.tokenizer import RegexTokenizer

# A small vocab of realistic-looking words for generating random text
_VOCAB = [
    "algorithm",
    "data",
    "machine",
    "learning",
    "model",
    "network",
    "neural",
    "training",
    "optimization",
    "gradient",
    "loss",
    "function",
    "parameter",
    "weight",
    "bias",
    "layer",
    "activation",
    "relu",
    "sigmoid",
    "softmax",
    "backpropagation",
    "forward",
    "pass",
    "batch",
    "epoch",
    "iteration",
    "convergence",
    "divergence",
    "regularization",
    "dropout",
    "attention",
    "transformer",
    "encoder",
    "decoder",
    "embedding",
    "token",
    "vector",
    "matrix",
    "tensor",
    "computation",
    "graph",
    "node",
    "edge",
    "vertex",
    "path",
    "search",
    "retrieval",
    "index",
    "query",
    "document",
    "corpus",
    "term",
    "frequency",
    "inverse",
    "score",
    "rank",
    "relevance",
    "precision",
    "recall",
    "f1",
    "metric",
    "evaluation",
    "benchmark",
    "dataset",
    "sample",
    "feature",
    "label",
    "class",
    "predict",
    "classification",
    "regression",
    "clustering",
    "dimension",
    "reduction",
    "pca",
    "tsne",
    "visualization",
    "matplotlib",
    "plot",
    "chart",
    "histogram",
    "scatter",
    "line",
    "bar",
    "database",
    "sql",
    "query",
    "table",
    "row",
    "column",
    "index",
    "primary",
    "foreign",
    "key",
    "constraint",
    "schema",
    "migration",
    "version",
    "control",
    "git",
    "commit",
    "branch",
    "merge",
    "conflict",
    "resolution",
    "review",
    "approve",
    "reject",
    "pull",
    "request",
    "issue",
    "bug",
    "fix",
    "feature",
    "enhancement",
    "refactor",
    "test",
    "deploy",
    "production",
    "staging",
    "development",
    "environment",
    "configuration",
    "setting",
    "variable",
    "constant",
    "global",
    "local",
    "scope",
    "closure",
    "callback",
    "promise",
    "async",
    "await",
    "synchronous",
    "asynchronous",
    "concurrent",
    "parallel",
    "thread",
    "process",
    "memory",
    "cache",
    "buffer",
    "queue",
    "stack",
    "heap",
    "pool",
]


class temp_chdir:
    """Context manager to temporarily chdir into a path and restore on exit."""

    def __init__(self, path):
        self.path = path
        self.old = None

    def __enter__(self):
        self.old = os.getcwd()
        os.chdir(self.path)
        return self

    def __exit__(self, *exc):
        os.chdir(self.old)


def _gen_random_text(n_tokens: int) -> str:
    """Generate random text with approximately n_tokens words."""
    words = random.choices(_VOCAB, k=n_tokens)
    return " ".join(words)


def _gen_random_query(n_words: int) -> str:
    """Generate a random query with n_words words."""
    words = random.choices(_VOCAB, k=n_words)
    return " ".join(words)


async def _make_index() -> BM25Index:
    """Create and start a BM25Index using cwd as working dir, with non-filtering tokenizer."""
    index = BM25Index()
    tokenizer = RegexTokenizer(filter_stopwords=False)
    index.tokenizer = tokenizer
    index._owned.append(tokenizer)  # pylint: disable=protected-access
    await index.start()
    return index


async def _setup_index_for_retrieve(n_docs: int = 100, doc_tokens: int = 1000) -> BM25Index:
    """Build an index with n_docs medium-sized docs in cwd."""
    index = await _make_index()
    docs = {f"doc_{i}": _gen_random_text(doc_tokens) for i in range(n_docs)}
    await index.add_docs(docs)
    return index


def test_add_docs_small():
    """Add 100 small docs (~100 tokens each)."""

    async def run():
        docs = {f"doc_{i}": _gen_random_text(100) for i in range(100)}
        with tempfile.TemporaryDirectory() as tmp, temp_chdir(tmp):
            index = await _make_index()
            t0 = time.perf_counter()
            await index.add_docs(docs)
            elapsed = time.perf_counter() - t0
            print(f"  add_docs (100 docs x ~100 tokens): {elapsed:.4f}s")
            await index.close()

    asyncio.run(run())


def test_add_docs_medium():
    """Add 100 medium docs (~1000 tokens each)."""

    async def run():
        docs = {f"doc_{i}": _gen_random_text(1000) for i in range(100)}
        with tempfile.TemporaryDirectory() as tmp, temp_chdir(tmp):
            index = await _make_index()
            t0 = time.perf_counter()
            await index.add_docs(docs)
            elapsed = time.perf_counter() - t0
            print(f"  add_docs (100 docs x ~1000 tokens): {elapsed:.4f}s")
            await index.close()

    asyncio.run(run())


def test_add_docs_large():
    """Add 100 large docs (~10000 tokens each)."""

    async def run():
        docs = {f"doc_{i}": _gen_random_text(10000) for i in range(100)}
        with tempfile.TemporaryDirectory() as tmp, temp_chdir(tmp):
            index = await _make_index()
            t0 = time.perf_counter()
            await index.add_docs(docs)
            elapsed = time.perf_counter() - t0
            print(f"  add_docs (100 docs x ~10000 tokens): {elapsed:.4f}s")
            await index.close()

    asyncio.run(run())


def test_retrieve_short_query():
    """Retrieve with 1-word query."""

    async def run():
        with tempfile.TemporaryDirectory() as tmp, temp_chdir(tmp):
            index = await _setup_index_for_retrieve()
            query = _gen_random_query(1)
            t0 = time.perf_counter()
            await index.retrieve(query, limit=10)
            elapsed = time.perf_counter() - t0
            print(f"  retrieve (1-word query, 100 docs): {elapsed:.6f}s")
            await index.close()

    asyncio.run(run())


def test_retrieve_medium_query():
    """Retrieve with 5-word query."""

    async def run():
        with tempfile.TemporaryDirectory() as tmp, temp_chdir(tmp):
            index = await _setup_index_for_retrieve()
            query = _gen_random_query(5)
            t0 = time.perf_counter()
            await index.retrieve(query, limit=10)
            elapsed = time.perf_counter() - t0
            print(f"  retrieve (5-word query, 100 docs): {elapsed:.6f}s")
            await index.close()

    asyncio.run(run())


def test_retrieve_long_query():
    """Retrieve with 20-word query."""

    async def run():
        with tempfile.TemporaryDirectory() as tmp, temp_chdir(tmp):
            index = await _setup_index_for_retrieve()
            query = _gen_random_query(20)
            t0 = time.perf_counter()
            await index.retrieve(query, limit=10)
            elapsed = time.perf_counter() - t0
            print(f"  retrieve (20-word query, 100 docs): {elapsed:.6f}s")
            await index.close()

    asyncio.run(run())


def test_retrieve_very_long_query():
    """Retrieve with 100-word query."""

    async def run():
        with tempfile.TemporaryDirectory() as tmp, temp_chdir(tmp):
            index = await _setup_index_for_retrieve()
            query = _gen_random_query(100)
            t0 = time.perf_counter()
            await index.retrieve(query, limit=10)
            elapsed = time.perf_counter() - t0
            print(f"  retrieve (100-word query, 100 docs): {elapsed:.6f}s")
            await index.close()

    asyncio.run(run())


if __name__ == "__main__":
    random.seed(42)
    print("=== BM25Index Performance Tests ===\n")

    print("[add_docs]")
    test_add_docs_small()
    test_add_docs_medium()
    test_add_docs_large()

    print("\n[retrieve]")
    test_retrieve_short_query()
    test_retrieve_medium_query()
    test_retrieve_long_query()
    test_retrieve_very_long_query()

    print("\nDone.")
