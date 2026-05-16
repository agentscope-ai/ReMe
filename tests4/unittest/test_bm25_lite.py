"""Tests for BM25Index search engine."""

# pylint: disable=protected-access

import asyncio
import os
import tempfile
import warnings

from reme4.components.keyword_index import BM25Index
from reme4.components.tokenizer import RegexTokenizer

# Filter jieba/pkg_resources deprecation warnings
warnings.filterwarnings("ignore", category=DeprecationWarning, module="jieba")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="pkg_resources")


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


async def create_bm25(k1: float = 1.5, b: float = 0.75) -> BM25Index:
    """Create and start a BM25Index in cwd with a non-filtering tokenizer.

    The non-filtering tokenizer keeps short test texts (e.g. "hello world") visible,
    since several common test words ("hello", "我", "的") are in the default stopwords.
    """
    bm25 = BM25Index(k1=k1, b=b)
    # Replace the unresolved Dependency placeholder with a real tokenizer instance.
    tokenizer = RegexTokenizer(filter_stopwords=False)
    bm25.tokenizer = tokenizer
    bm25._owned.append(tokenizer)
    await bm25.start()
    return bm25


def test_basic_init():
    """Test BM25Index initialization."""

    async def run():
        with tempfile.TemporaryDirectory() as tmpdir, temp_chdir(tmpdir):
            bm25 = BM25Index()
            assert bm25.k1 == 1.5
            assert bm25.b == 0.75
            assert bm25.vocab == {}
            assert bm25.inverted_index == {}
            assert bm25.doc_meta == {}
            assert bm25.n_docs == 0
            assert bm25.avg_len == 0.0
            print("✓ test_basic_init passed")

    asyncio.run(run())


def test_start_with_tokenizer():
    """Test BM25Index starts and initializes tokenizer."""

    async def run():
        with tempfile.TemporaryDirectory() as tmpdir, temp_chdir(tmpdir):
            bm25 = await create_bm25()
            assert bm25.tokenizer is not None
            assert bm25.is_started

            await bm25.close()
            assert not bm25.is_started
            print("✓ test_start_with_tokenizer passed")

    asyncio.run(run())


def test_add_single_doc():
    """Test adding a single document."""

    async def run():
        with tempfile.TemporaryDirectory() as tmpdir, temp_chdir(tmpdir):
            bm25 = await create_bm25()

            await bm25.add_docs({"doc1": "hello world"})

            assert bm25.n_docs == 1
            assert bm25.total_len > 0
            assert "doc1" in bm25.doc_meta

            await bm25.close()
            print("✓ test_add_single_doc passed")

    asyncio.run(run())


def test_add_multiple_docs():
    """Test adding multiple documents."""

    async def run():
        with tempfile.TemporaryDirectory() as tmpdir, temp_chdir(tmpdir):
            bm25 = await create_bm25()

            docs = {
                "doc1": "hello world",
                "doc2": "hello python",
                "doc3": "world python",
            }
            await bm25.add_docs(docs)

            assert bm25.n_docs == 3
            assert len(bm25.vocab) > 0
            assert len(bm25.inverted_index) > 0

            await bm25.close()
            print("✓ test_add_multiple_docs passed")

    asyncio.run(run())


def test_retrieve_basic():
    """Test basic retrieval functionality."""

    async def run():
        with tempfile.TemporaryDirectory() as tmpdir, temp_chdir(tmpdir):
            bm25 = await create_bm25()

            docs = {
                "doc1": "python programming language",
                "doc2": "java programming language",
                "doc3": "python data analysis",
            }
            await bm25.add_docs(docs)

            results = await bm25.retrieve("python", limit=3)
            assert len(results) <= 3
            assert "doc1" in results or "doc3" in results

            await bm25.close()
            print("✓ test_retrieve_basic passed")

    asyncio.run(run())


def test_retrieve_with_limit():
    """Test retrieval with result limit."""

    async def run():
        with tempfile.TemporaryDirectory() as tmpdir, temp_chdir(tmpdir):
            bm25 = await create_bm25()

            docs = {f"doc{i}": f"python programming {i}" for i in range(10)}
            await bm25.add_docs(docs)

            results = await bm25.retrieve("python", limit=3)
            assert len(results) == 3

            results = await bm25.retrieve("python", limit=5)
            assert len(results) == 5

            await bm25.close()
            print("✓ test_retrieve_with_limit passed")

    asyncio.run(run())


def test_retrieve_empty_query():
    """Test retrieval with empty or unknown query."""

    async def run():
        with tempfile.TemporaryDirectory() as tmpdir, temp_chdir(tmpdir):
            bm25 = await create_bm25()

            docs = {"doc1": "hello world"}
            await bm25.add_docs(docs)

            results = await bm25.retrieve("", limit=3)
            assert results == {}

            results = await bm25.retrieve("unknownxyz", limit=3)
            assert results == {}

            await bm25.close()
            print("✓ test_retrieve_empty_query passed")

    asyncio.run(run())


def test_retrieve_empty_index():
    """Test retrieval from empty index."""

    async def run():
        with tempfile.TemporaryDirectory() as tmpdir, temp_chdir(tmpdir):
            bm25 = await create_bm25()

            results = await bm25.retrieve("python", limit=3)
            assert results == {}

            await bm25.close()
            print("✓ test_retrieve_empty_index passed")

    asyncio.run(run())


def test_update_doc():
    """Test updating an existing document."""

    async def run():
        with tempfile.TemporaryDirectory() as tmpdir, temp_chdir(tmpdir):
            bm25 = await create_bm25()

            await bm25.add_docs({"doc1": "hello world python"})
            old_len = bm25.total_len

            await bm25.add_docs({"doc1": "java"})
            assert bm25.n_docs == 1
            assert bm25.total_len != old_len

            results = await bm25.retrieve("java", limit=1)
            assert "doc1" in results

            await bm25.close()
            print("✓ test_update_doc passed")

    asyncio.run(run())


def test_remove_doc():
    """Test removing a document."""

    async def run():
        with tempfile.TemporaryDirectory() as tmpdir, temp_chdir(tmpdir):
            bm25 = await create_bm25()

            docs = {
                "doc1": "hello world",
                "doc2": "hello python",
            }
            await bm25.add_docs(docs)
            assert bm25.n_docs == 2

            bm25._remove_doc("doc1")
            assert bm25.n_docs == 1
            assert "doc1" not in bm25.doc_meta

            results = await bm25.retrieve("hello", limit=2)
            assert "doc1" not in results
            assert "doc2" in results

            await bm25.close()
            print("✓ test_remove_doc passed")

    asyncio.run(run())


def test_remove_nonexistent_doc():
    """Test removing a nonexistent document (should be no-op)."""

    async def run():
        with tempfile.TemporaryDirectory() as tmpdir, temp_chdir(tmpdir):
            bm25 = await create_bm25()

            await bm25.add_docs({"doc1": "hello world"})
            bm25._remove_doc("nonexistent")
            assert bm25.n_docs == 1

            await bm25.close()
            print("✓ test_remove_nonexistent_doc passed")

    asyncio.run(run())


def test_clear():
    """Test clearing the index."""

    async def run():
        with tempfile.TemporaryDirectory() as tmpdir, temp_chdir(tmpdir):
            bm25 = await create_bm25()

            await bm25.add_docs(
                {
                    "doc1": "hello world",
                    "doc2": "hello python",
                },
            )
            assert bm25.n_docs == 2

            await bm25.clear()
            assert bm25.n_docs == 0
            assert bm25.vocab == {}
            assert bm25.inverted_index == {}
            assert bm25.doc_meta == {}
            assert bm25.total_len == 0
            assert bm25._idf_cache == {}

            await bm25.close()
            print("✓ test_clear passed")

    asyncio.run(run())


def test_optimize_index():
    """Test optimize_index functionality to compact vocab."""

    async def run():
        with tempfile.TemporaryDirectory() as tmpdir, temp_chdir(tmpdir):
            bm25 = await create_bm25()

            await bm25.add_docs({"doc1": "hello world"})
            bm25._remove_doc("doc1")

            assert bm25.n_docs == 0
            assert len(bm25.vocab) > 0

            await bm25.optimize_index()
            assert bm25.vocab == {}
            assert bm25.inverted_index == {}

            await bm25.close()
            print("✓ test_optimize_index passed")

    asyncio.run(run())


def test_optimize_index_with_docs():
    """Test optimize_index with remaining documents."""

    async def run():
        with tempfile.TemporaryDirectory() as tmpdir, temp_chdir(tmpdir):
            bm25 = await create_bm25()

            await bm25.add_docs(
                {
                    "doc1": "hello world",
                    "doc2": "hello python",
                },
            )

            old_vocab = bm25.vocab.copy()
            bm25._remove_doc("doc1")

            await bm25.optimize_index()

            assert bm25.n_docs == 1
            assert "doc2" in bm25.doc_meta
            assert len(bm25.vocab) < len(old_vocab)

            results = await bm25.retrieve("hello", limit=1)
            assert "doc2" in results

            await bm25.close()
            print("✓ test_optimize_index_with_docs passed")

    asyncio.run(run())


def test_persistence():
    """Test dump and load persistence."""

    async def run():
        with tempfile.TemporaryDirectory() as tmpdir, temp_chdir(tmpdir):
            bm25 = await create_bm25()
            docs = {
                "doc1": "hello world",
                "doc2": "hello python",
                "doc3": "programming language",
            }
            await bm25.add_docs(docs)

            old_vocab = bm25.vocab.copy()
            old_doc_meta = {k: dict(v) for k, v in bm25.doc_meta.items()}

            await bm25.dump()
            await bm25.close()

            bm25_new = await create_bm25()

            assert bm25_new.vocab == old_vocab
            assert bm25_new.n_docs == 3
            for doc_id in old_doc_meta:
                assert doc_id in bm25_new.doc_meta

            results = await bm25_new.retrieve("hello", limit=2)
            assert "doc1" in results or "doc2" in results

            await bm25_new.close()
            print("✓ test_persistence passed")

    asyncio.run(run())


def test_custom_params():
    """Test custom k1 and b parameters."""

    async def run():
        with tempfile.TemporaryDirectory() as tmpdir, temp_chdir(tmpdir):
            bm25 = await create_bm25(k1=2.0, b=0.5)

            assert bm25.k1 == 2.0
            assert bm25.b == 0.5

            await bm25.add_docs({"doc1": "test document"})
            results = await bm25.retrieve("test", limit=1)
            assert "doc1" in results

            await bm25.close()
            print("✓ test_custom_params passed")

    asyncio.run(run())


def test_chinese_text():
    """Test with Chinese text."""

    async def run():
        with tempfile.TemporaryDirectory() as tmpdir, temp_chdir(tmpdir):
            bm25 = await create_bm25()

            docs = {
                "doc1": "我爱北京天安门",
                "doc2": "北京是中国的首都",
                "doc3": "上海的天气很好",
            }
            await bm25.add_docs(docs)

            results = await bm25.retrieve("北", limit=2)
            assert len(results) <= 2
            assert "doc1" in results or "doc2" in results

            await bm25.close()
            print("✓ test_chinese_text passed")

    asyncio.run(run())


def test_mixed_chinese_english():
    """Test with mixed Chinese and English text."""

    async def run():
        with tempfile.TemporaryDirectory() as tmpdir, temp_chdir(tmpdir):
            bm25 = await create_bm25()

            docs = {
                "doc1": "Python 是一种编程语言",
                "doc2": "Java 编程语言",
                "doc3": "Python 数据分析",
            }
            await bm25.add_docs(docs)

            results = await bm25.retrieve("Python", limit=3)
            assert len(results) > 0

            results = await bm25.retrieve("编", limit=2)
            assert len(results) > 0

            await bm25.close()
            print("✓ test_mixed_chinese_english passed")

    asyncio.run(run())


def test_idf_cache():
    """Test IDF cache functionality."""

    async def run():
        with tempfile.TemporaryDirectory() as tmpdir, temp_chdir(tmpdir):
            bm25 = await create_bm25()

            await bm25.add_docs(
                {
                    "doc1": "hello world",
                    "doc2": "hello python",
                },
            )

            token = "hello"
            if token in bm25.vocab:
                tid = bm25.vocab[token]
                idf1 = bm25._get_idf(tid)
                assert tid in bm25._idf_cache
                idf2 = bm25._get_idf(tid)
                assert idf1 == idf2

            await bm25.close()
            print("✓ test_idf_cache passed")

    asyncio.run(run())


def test_avg_len():
    """Test average document length calculation."""

    async def run():
        with tempfile.TemporaryDirectory() as tmpdir, temp_chdir(tmpdir):
            bm25 = await create_bm25()

            assert bm25.avg_len == 0.0

            await bm25.add_docs({"doc1": "hello world python"})
            assert bm25.avg_len > 0

            await bm25.add_docs({"doc2": "test"})
            new_avg = bm25.avg_len
            assert new_avg > 0

            await bm25.close()
            print("✓ test_avg_len passed")

    asyncio.run(run())


def test_score_ordering():
    """Test that results are ordered by score descending."""

    async def run():
        with tempfile.TemporaryDirectory() as tmpdir, temp_chdir(tmpdir):
            bm25 = await create_bm25()

            docs = {
                "doc1": "python python python",
                "doc2": "python python",
                "doc3": "python",
            }
            await bm25.add_docs(docs)

            results = await bm25.retrieve("python", limit=3)
            scores = list(results.values())

            for i in range(len(scores) - 1):
                assert scores[i] >= scores[i + 1]

            await bm25.close()
            print("✓ test_score_ordering passed")

    asyncio.run(run())


def test_empty_doc():
    """Test adding empty document."""

    async def run():
        with tempfile.TemporaryDirectory() as tmpdir, temp_chdir(tmpdir):
            bm25 = await create_bm25()

            await bm25.add_docs({"doc1": ""})
            assert bm25.n_docs == 0

            await bm25.add_docs({"doc2": "   "})
            assert bm25.n_docs == 0

            await bm25.close()
            print("✓ test_empty_doc passed")

    asyncio.run(run())


if __name__ == "__main__":
    print("\n=== BM25Index Tests ===")
    test_basic_init()
    test_start_with_tokenizer()
    test_add_single_doc()
    test_add_multiple_docs()
    test_retrieve_basic()
    test_retrieve_with_limit()
    test_retrieve_empty_query()
    test_retrieve_empty_index()
    test_update_doc()
    test_remove_doc()
    test_remove_nonexistent_doc()
    test_clear()
    test_optimize_index()
    test_optimize_index_with_docs()
    test_persistence()
    test_custom_params()
    test_chinese_text()
    test_mixed_chinese_english()
    test_idf_cache()
    test_avg_len()
    test_score_ordering()
    test_empty_doc()
    print("\n所有测试通过!")
