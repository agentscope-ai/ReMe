"""BM25Index performance tests for add_docs and retrieve."""

import asyncio
import random
import time
import tempfile
from pathlib import Path

from reme2.component.keyword_index.bm25_index import BM25Index
from reme2.component.tokenizer import RegexTokenizer

# A small vocab of realistic-looking words for generating random text
_VOCAB = [
    "algorithm", "data", "machine", "learning", "model", "network", "neural", "training",
    "optimization", "gradient", "loss", "function", "parameter", "weight", "bias", "layer",
    "activation", "relu", "sigmoid", "softmax", "backpropagation", "forward", "pass",
    "batch", "epoch", "iteration", "convergence", "divergence", "regularization", "dropout",
    "attention", "transformer", "encoder", "decoder", "embedding", "token", "vector",
    "matrix", "tensor", "computation", "graph", "node", "edge", "vertex", "path",
    "search", "retrieval", "index", "query", "document", "corpus", "term", "frequency",
    "inverse", "score", "rank", "relevance", "precision", "recall", "f1", "metric",
    "evaluation", "benchmark", "dataset", "sample", "feature", "label", "class", "predict",
    "classification", "regression", "clustering", "dimension", "reduction", "pca", "tsne",
    "visualization", "matplotlib", "plot", "chart", "histogram", "scatter", "line", "bar",
    "database", "sql", "query", "table", "row", "column", "index", "primary", "foreign",
    "key", "constraint", "schema", "migration", "version", "control", "git", "commit",
    "branch", "merge", "conflict", "resolution", "review", "approve", "reject", "pull",
    "request", "issue", "bug", "fix", "feature", "enhancement", "refactor", "test",
    "deploy", "production", "staging", "development", "environment", "configuration",
    "setting", "variable", "constant", "global", "local", "scope", "closure", "callback",
    "promise", "async", "await", "synchronous", "asynchronous", "concurrent", "parallel",
    "thread", "process", "memory", "cache", "buffer", "queue", "stack", "heap", "pool",
]


def _gen_random_text(n_tokens: int) -> str:
    """Generate random text with approximately n_tokens words."""
    words = random.choices(_VOCAB, k=n_tokens)
    return " ".join(words)


def _gen_random_query(n_words: int) -> str:
    """Generate a random query with n_words words."""
    words = random.choices(_VOCAB, k=n_words)
    return " ".join(words)


async def _make_index(tmp_dir: Path) -> BM25Index:
    """Create and start a BM25Index instance, bypassing app_context."""
    index = BM25Index.__new__(BM25Index)
    # Manually run BaseComponent.__init__ fields
    index.name = "perf_test"
    index.backend = ""
    index.app_context = None
    index.kwargs = {}
    index.logger = __import__("logging").getLogger("perf_test")
    index._is_started = False
    index._lock = asyncio.Lock()
    # BaseKeywordIndex fields
    index.tokenizer_name = "default"
    index.index_path = tmp_dir / "keyword_index"
    index.index_path.mkdir(parents=True, exist_ok=True)
    # BM25Index fields
    index.k1 = 1.5
    index.b = 0.75
    index.vocab = {}
    index.inverted_index = {}
    index.doc_meta = {}
    index.total_len = 0
    index._idf_cache = {}
    # Init tokenizer
    index.tokenizer = RegexTokenizer(filter_stopwords=False)
    await index.tokenizer.start()
    index._is_started = True
    return index


async def test_add_docs_small():
    """Add 100 small docs (~100 tokens each)."""
    docs = {f"doc_{i}": _gen_random_text(100) for i in range(100)}
    with tempfile.TemporaryDirectory() as tmp:
        index = await _make_index(Path(tmp))
        t0 = time.perf_counter()
        await index.add_docs(docs)
        elapsed = time.perf_counter() - t0
        print(f"  add_docs (100 docs x ~100 tokens): {elapsed:.4f}s")
        await index.close()


async def test_add_docs_medium():
    """Add 100 medium docs (~1000 tokens each)."""
    docs = {f"doc_{i}": _gen_random_text(1000) for i in range(100)}
    with tempfile.TemporaryDirectory() as tmp:
        index = await _make_index(Path(tmp))
        t0 = time.perf_counter()
        await index.add_docs(docs)
        elapsed = time.perf_counter() - t0
        print(f"  add_docs (100 docs x ~1000 tokens): {elapsed:.4f}s")
        await index.close()


async def test_add_docs_large():
    """Add 100 large docs (~10000 tokens each)."""
    docs = {f"doc_{i}": _gen_random_text(10000) for i in range(100)}
    with tempfile.TemporaryDirectory() as tmp:
        index = await _make_index(Path(tmp))
        t0 = time.perf_counter()
        await index.add_docs(docs)
        elapsed = time.perf_counter() - t0
        print(f"  add_docs (100 docs x ~10000 tokens): {elapsed:.4f}s")
        await index.close()


async def _setup_index_for_retrieve(n_docs: int = 100, doc_tokens: int = 1000) -> tuple[BM25Index, str]:
    """Build an index with n_docs medium-sized docs, return (index, tmp_dir)."""
    tmp = tempfile.mkdtemp()
    index = await _make_index(Path(tmp))
    docs = {f"doc_{i}": _gen_random_text(doc_tokens) for i in range(n_docs)}
    await index.add_docs(docs)
    return index, tmp


async def test_retrieve_short_query():
    """Retrieve with 1-word query."""
    index, tmp = await _setup_index_for_retrieve()
    query = _gen_random_query(1)
    t0 = time.perf_counter()
    await index.retrieve(query, limit=10)
    elapsed = time.perf_counter() - t0
    print(f"  retrieve (1-word query, 100 docs): {elapsed:.6f}s")
    await index.close()


async def test_retrieve_medium_query():
    """Retrieve with 5-word query."""
    index, tmp = await _setup_index_for_retrieve()
    query = _gen_random_query(5)
    t0 = time.perf_counter()
    await index.retrieve(query, limit=10)
    elapsed = time.perf_counter() - t0
    print(f"  retrieve (5-word query, 100 docs): {elapsed:.6f}s")
    await index.close()


async def test_retrieve_long_query():
    """Retrieve with 20-word query."""
    index, tmp = await _setup_index_for_retrieve()
    query = _gen_random_query(20)
    t0 = time.perf_counter()
    await index.retrieve(query, limit=10)
    elapsed = time.perf_counter() - t0
    print(f"  retrieve (20-word query, 100 docs): {elapsed:.6f}s")
    await index.close()


async def test_retrieve_very_long_query():
    """Retrieve with 100-word query."""
    index, tmp = await _setup_index_for_retrieve()
    query = _gen_random_query(100)
    t0 = time.perf_counter()
    await index.retrieve(query, limit=10)
    elapsed = time.perf_counter() - t0
    print(f"  retrieve (100-word query, 100 docs): {elapsed:.6f}s")
    await index.close()


async def main():
    random.seed(42)
    print("=== BM25Index Performance Tests ===\n")

    print("[add_docs]")
    await test_add_docs_small()
    await test_add_docs_medium()
    await test_add_docs_large()

    print("\n[retrieve]")
    await test_retrieve_short_query()
    await test_retrieve_medium_query()
    await test_retrieve_long_query()
    await test_retrieve_very_long_query()

    print("\nDone.")


if __name__ == "__main__":
    asyncio.run(main())
