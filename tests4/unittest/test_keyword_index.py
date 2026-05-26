"""Tests for BaseKeywordIndex implementations (currently: BM25Index).

Covers full lifecycle, CRUD, retrieval, persistence, optimize and — as the focus
of this file — Chinese / English / mixed-language behaviour driven by the
default RegexTokenizer (Chinese split per char, English words lowercased,
single-char ASCII words dropped).
"""

# pylint: disable=protected-access

import asyncio
import os
import tempfile
import warnings

from reme4.components.keyword_index import BM25Index
from reme4.components.tokenizer import RegexTokenizer

warnings.filterwarnings("ignore", category=DeprecationWarning, module="jieba")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="pkg_resources")


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #


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


async def create_bm25(
    k1: float = 1.5,
    b: float = 0.75,
    filter_stopwords: bool = False,
) -> BM25Index:
    """Create and start a BM25Index in cwd with a non-filtering RegexTokenizer.

    Stopword filtering is off so short test words ("hello", "我", "的") survive.
    """
    bm25 = BM25Index(k1=k1, b=b)
    tokenizer = RegexTokenizer(filter_stopwords=filter_stopwords)
    bm25.tokenizer = tokenizer
    bm25._owned.append(tokenizer)
    await bm25.start()
    return bm25


def run(coro):
    """Tiny shorthand to avoid repeating asyncio.run wrappers."""
    return asyncio.run(coro)


# --------------------------------------------------------------------------- #
# Initialisation & lifecycle                                                  #
# --------------------------------------------------------------------------- #


def test_basic_init():
    """Default constructor produces empty BM25 state."""

    async def go():
        with tempfile.TemporaryDirectory() as tmp, temp_chdir(tmp):
            bm25 = BM25Index()
            assert bm25.k1 == 1.5
            assert bm25.b == 0.75
            assert bm25.index_version == "v1"
            assert bm25.vocab == {}
            assert not bm25.inverted_index
            assert bm25.doc_meta == {}
            assert bm25.n_docs == 0
            assert bm25.total_len == 0
            assert bm25.avg_len == 0.0

    run(go())


def test_custom_params():
    """k1, b and index_version are honoured."""

    async def go():
        with tempfile.TemporaryDirectory() as tmp, temp_chdir(tmp):
            bm25 = BM25Index(k1=2.0, b=0.5, index_version="v2")
            assert bm25.k1 == 2.0
            assert bm25.b == 0.5
            assert bm25.index_version == "v2"

    run(go())


def test_index_file_raises_when_tokenizer_is_none():
    """index_file must raise when tokenizer is explicitly None."""

    async def go():
        with tempfile.TemporaryDirectory() as tmp, temp_chdir(tmp):
            bm25 = BM25Index()
            bm25.tokenizer = None
            try:
                _ = bm25.index_file
            except RuntimeError:
                return
            raise AssertionError("expected RuntimeError when tokenizer is None")

    run(go())


def test_start_close_lifecycle():
    """start/close toggles is_started and runs underlying tokenizer."""

    async def go():
        with tempfile.TemporaryDirectory() as tmp, temp_chdir(tmp):
            bm25 = await create_bm25()
            assert bm25.is_started
            assert bm25.tokenizer is not None
            await bm25.close()
            assert not bm25.is_started

    run(go())


def test_index_file_path_includes_tokenizer_and_version():
    """index_file path embeds tokenizer name + index_version."""

    async def go():
        with tempfile.TemporaryDirectory() as tmp, temp_chdir(tmp):
            bm25 = await create_bm25()
            path = str(bm25.index_file)
            assert "bm25_regex_v1.pkl" in path
            await bm25.close()

    run(go())


# --------------------------------------------------------------------------- #
# add_docs / delete_docs / update                                             #
# --------------------------------------------------------------------------- #


def test_add_empty_dict_noop():
    """Adding an empty dict must not touch state."""

    async def go():
        with tempfile.TemporaryDirectory() as tmp, temp_chdir(tmp):
            bm25 = await create_bm25()
            await bm25.add_docs({})
            assert bm25.n_docs == 0
            assert bm25.vocab == {}
            await bm25.close()

    run(go())


def test_add_single_doc():
    """A single doc populates length, vocab and metadata."""

    async def go():
        with tempfile.TemporaryDirectory() as tmp, temp_chdir(tmp):
            bm25 = await create_bm25()
            await bm25.add_docs({"d1": "hello world"})
            assert bm25.n_docs == 1
            assert bm25.total_len == 2  # 'hello', 'world'
            assert bm25.avg_len == 2.0
            assert set(bm25.vocab) == {"hello", "world"}
            assert "d1" in bm25.doc_meta
            assert bm25.doc_meta["d1"]["len"] == 2
            await bm25.close()

    run(go())


def test_add_multiple_docs_and_inverted_index():
    """Inverted index lists postings for every term across multiple docs."""

    async def go():
        with tempfile.TemporaryDirectory() as tmp, temp_chdir(tmp):
            bm25 = await create_bm25()
            await bm25.add_docs(
                {
                    "d1": "hello world",
                    "d2": "hello python",
                    "d3": "world python",
                },
            )
            assert bm25.n_docs == 3
            inv = bm25.inverted_index
            tid_hello = bm25.vocab["hello"]
            tid_world = bm25.vocab["world"]
            tid_python = bm25.vocab["python"]
            assert set(inv[tid_hello]) == {"d1", "d2"}
            assert set(inv[tid_world]) == {"d1", "d3"}
            assert set(inv[tid_python]) == {"d2", "d3"}
            await bm25.close()

    run(go())


def test_add_doc_empty_or_whitespace_is_skipped():
    """Empty / whitespace-only content yields no tokens and is silently dropped."""

    async def go():
        with tempfile.TemporaryDirectory() as tmp, temp_chdir(tmp):
            bm25 = await create_bm25()
            await bm25.add_docs({"d1": "", "d2": "   ", "d3": "\n\t"})
            assert bm25.n_docs == 0
            await bm25.close()

    run(go())


def test_update_existing_doc_swaps_content():
    """Re-adding same doc_id replaces tokens; old terms no longer match it."""

    async def go():
        with tempfile.TemporaryDirectory() as tmp, temp_chdir(tmp):
            bm25 = await create_bm25()
            await bm25.add_docs({"d1": "hello world python"})
            old_len = bm25.total_len
            await bm25.add_docs({"d1": "java"})
            assert bm25.n_docs == 1
            assert bm25.total_len != old_len
            assert bm25.doc_meta["d1"]["len"] == 1

            # Old term must no longer return d1.
            assert "d1" not in await bm25.retrieve("hello", limit=5)
            # New term does.
            assert "d1" in await bm25.retrieve("java", limit=5)
            await bm25.close()

    run(go())


def test_delete_single_doc():
    """delete_docs removes a single doc from retrieval and meta."""

    async def go():
        with tempfile.TemporaryDirectory() as tmp, temp_chdir(tmp):
            bm25 = await create_bm25()
            await bm25.add_docs({"d1": "hello world", "d2": "hello python"})
            assert bm25.n_docs == 2

            await bm25.delete_docs(["d1"])
            assert bm25.n_docs == 1
            assert "d1" not in bm25.doc_meta
            assert "d2" in bm25.doc_meta

            results = await bm25.retrieve("hello", limit=2)
            assert "d1" not in results
            assert "d2" in results
            await bm25.close()

    run(go())


def test_delete_multiple_docs():
    """delete_docs handles a batch list."""

    async def go():
        with tempfile.TemporaryDirectory() as tmp, temp_chdir(tmp):
            bm25 = await create_bm25()
            await bm25.add_docs({f"d{i}": "hello world" for i in range(5)})
            assert bm25.n_docs == 5

            await bm25.delete_docs(["d0", "d2", "d4"])
            assert bm25.n_docs == 2
            assert set(bm25.doc_meta) == {"d1", "d3"}
            await bm25.close()

    run(go())


def test_delete_nonexistent_is_noop():
    """Deleting unknown doc_ids must not raise."""

    async def go():
        with tempfile.TemporaryDirectory() as tmp, temp_chdir(tmp):
            bm25 = await create_bm25()
            await bm25.add_docs({"d1": "hello"})
            await bm25.delete_docs(["nope", "still_nope"])
            assert bm25.n_docs == 1
            await bm25.close()

    run(go())


def test_re_add_after_delete():
    """Adding a doc_id back after deletion yields a fresh idx and is retrievable."""

    async def go():
        with tempfile.TemporaryDirectory() as tmp, temp_chdir(tmp):
            bm25 = await create_bm25()
            await bm25.add_docs({"d1": "hello"})
            await bm25.delete_docs(["d1"])
            assert bm25.n_docs == 0
            await bm25.add_docs({"d1": "world"})
            assert bm25.n_docs == 1
            assert "d1" in await bm25.retrieve("world", limit=1)
            await bm25.close()

    run(go())


# --------------------------------------------------------------------------- #
# Retrieval                                                                   #
# --------------------------------------------------------------------------- #


def test_retrieve_empty_index():
    """Retrieving from an empty index returns {}."""

    async def go():
        with tempfile.TemporaryDirectory() as tmp, temp_chdir(tmp):
            bm25 = await create_bm25()
            assert await bm25.retrieve("python", limit=3) == {}
            await bm25.close()

    run(go())


def test_retrieve_empty_or_unknown_query():
    """Empty queries and out-of-vocab queries both return {}."""

    async def go():
        with tempfile.TemporaryDirectory() as tmp, temp_chdir(tmp):
            bm25 = await create_bm25()
            await bm25.add_docs({"d1": "hello world"})
            assert await bm25.retrieve("", limit=3) == {}
            assert await bm25.retrieve("zzzunknownxyz", limit=3) == {}
            await bm25.close()

    run(go())


def test_retrieve_limit_caps_results():
    """retrieve honours `limit`."""

    async def go():
        with tempfile.TemporaryDirectory() as tmp, temp_chdir(tmp):
            bm25 = await create_bm25()
            await bm25.add_docs({f"d{i}": f"python lang {i}" for i in range(10)})
            assert len(await bm25.retrieve("python", limit=3)) == 3
            assert len(await bm25.retrieve("python", limit=5)) == 5
            # limit greater than matches: bounded by positive matches.
            assert len(await bm25.retrieve("python", limit=99)) == 10
            await bm25.close()

    run(go())


def test_retrieve_score_ordering_by_tf():
    """A doc with higher term frequency for the query token outranks others."""

    async def go():
        with tempfile.TemporaryDirectory() as tmp, temp_chdir(tmp):
            bm25 = await create_bm25()
            await bm25.add_docs(
                {
                    "high": "python python python",
                    "mid": "python python other",
                    "low": "python alpha beta",
                },
            )
            results = await bm25.retrieve("python", limit=3)
            assert list(results.keys()) == ["high", "mid", "low"]
            scores = list(results.values())
            assert scores[0] >= scores[1] >= scores[2]
            await bm25.close()

    run(go())


def test_retrieve_idf_favours_rare_terms():
    """In a query of {common, rare}, the doc containing the rare term wins."""

    async def go():
        with tempfile.TemporaryDirectory() as tmp, temp_chdir(tmp):
            bm25 = await create_bm25()
            # 'common' appears everywhere → low IDF.
            # 'rare' appears in only one doc → high IDF.
            docs = {f"d{i}": "common filler text" for i in range(10)}
            docs["target"] = "common rare term"
            await bm25.add_docs(docs)

            results = await bm25.retrieve("common rare", limit=3)
            assert next(iter(results)) == "target"
            await bm25.close()

    run(go())


def test_retrieve_length_normalization():
    """With b=0.75 (default), a much longer doc with same tf scores lower."""

    async def go():
        with tempfile.TemporaryDirectory() as tmp, temp_chdir(tmp):
            bm25 = await create_bm25()
            await bm25.add_docs(
                {
                    "short": "python",
                    "long": "python " + " ".join(f"w{i}" for i in range(50)),
                },
            )
            results = await bm25.retrieve("python", limit=2)
            assert results["short"] > results["long"]
            await bm25.close()

    run(go())


def test_retrieve_duplicate_query_tokens_dont_double_count():
    """Repeating the same query token should not boost its contribution."""

    async def go():
        with tempfile.TemporaryDirectory() as tmp, temp_chdir(tmp):
            bm25 = await create_bm25()
            await bm25.add_docs({"d1": "python rocks"})
            once = await bm25.retrieve("python", limit=1)
            many = await bm25.retrieve("python python python", limit=1)
            assert once["d1"] == many["d1"]
            await bm25.close()

    run(go())


# --------------------------------------------------------------------------- #
# Chinese / English / mixed-language behaviour (focus)                        #
# --------------------------------------------------------------------------- #


def test_chinese_only_corpus():
    """Pure Chinese corpus indexes per-character and retrieves correctly."""

    async def go():
        with tempfile.TemporaryDirectory() as tmp, temp_chdir(tmp):
            bm25 = await create_bm25()
            await bm25.add_docs(
                {
                    "d1": "我爱北京天安门",
                    "d2": "北京是中国的首都",
                    "d3": "上海的天气很好",
                },
            )
            # Regex tokenizer splits Chinese per character.
            assert "北" in bm25.vocab
            assert "京" in bm25.vocab

            # Query "北京" → two tokens, both d1 and d2 match; d3 does not.
            results = await bm25.retrieve("北京", limit=3)
            assert set(results) == {"d1", "d2"}
            await bm25.close()

    run(go())


def test_english_only_corpus_is_lowercased():
    """English tokens are lowercased so case-insensitive retrieval works."""

    async def go():
        with tempfile.TemporaryDirectory() as tmp, temp_chdir(tmp):
            bm25 = await create_bm25()
            await bm25.add_docs(
                {
                    "d1": "Python Programming Language",
                    "d2": "Java Programming Language",
                },
            )
            assert "python" in bm25.vocab
            assert "Python" not in bm25.vocab

            r_upper = await bm25.retrieve("PYTHON", limit=2)
            r_lower = await bm25.retrieve("python", limit=2)
            assert r_upper == r_lower
            assert "d1" in r_upper
            await bm25.close()

    run(go())


def test_single_char_english_dropped():
    """RegexTokenizer's \\w\\w+ pattern drops single-letter ASCII tokens like 'I'."""

    async def go():
        with tempfile.TemporaryDirectory() as tmp, temp_chdir(tmp):
            bm25 = await create_bm25()
            await bm25.add_docs({"d1": "I love Beijing"})
            # 'I' must not appear; 'love' and 'beijing' must.
            assert "i" not in bm25.vocab
            assert "love" in bm25.vocab
            assert "beijing" in bm25.vocab
            # Querying with just "I" returns nothing.
            assert await bm25.retrieve("I", limit=1) == {}
            await bm25.close()

    run(go())


def test_mixed_doc_chinese_query_matches():
    """A Chinese query hits docs containing those Chinese chars even when mixed."""

    async def go():
        with tempfile.TemporaryDirectory() as tmp, temp_chdir(tmp):
            bm25 = await create_bm25()
            await bm25.add_docs(
                {
                    "d1": "Python 是一种编程语言",
                    "d2": "Java 编程语言",
                    "d3": "Python 数据分析",
                },
            )
            results = await bm25.retrieve("编程", limit=3)
            assert set(results) >= {"d1", "d2"}
            assert "d3" not in results
            await bm25.close()

    run(go())


def test_mixed_doc_english_query_matches():
    """An English query hits the right mixed-language docs."""

    async def go():
        with tempfile.TemporaryDirectory() as tmp, temp_chdir(tmp):
            bm25 = await create_bm25()
            await bm25.add_docs(
                {
                    "d1": "Python 是一种编程语言",
                    "d2": "Java 编程语言",
                    "d3": "Python 数据分析",
                },
            )
            results = await bm25.retrieve("python", limit=3)
            assert set(results) == {"d1", "d3"}
            assert "d2" not in results
            await bm25.close()

    run(go())


def test_mixed_query_combines_chinese_and_english_signal():
    """A query mixing English and Chinese aggregates IDF×tf contributions."""

    async def go():
        with tempfile.TemporaryDirectory() as tmp, temp_chdir(tmp):
            bm25 = await create_bm25()
            await bm25.add_docs(
                {
                    "py_cn": "Python 编程",  # matches both 'python' and '编','程'
                    "py_only": "Python tutorial",  # matches only 'python'
                    "cn_only": "编程入门",  # matches only '编','程'
                    # Avoid Chinese chars that the query splits into ('编','程') — '教程' would
                    # leak '程' into 'other' and pollute IDF, so use unrelated chars only.
                    "other": "Java 教学",
                },
            )
            results = await bm25.retrieve("Python 编程", limit=4)
            # py_cn should rank highest because it matches both branches.
            assert next(iter(results)) == "py_cn"
            # 'other' should not appear.
            assert "other" not in results
            # Both unimodal matches should still appear.
            assert "py_only" in results and "cn_only" in results
            await bm25.close()

    run(go())


def test_mixed_doc_more_matches_outrank_fewer():
    """Doc covering more query tokens (Chinese+English) outranks partial matches."""

    async def go():
        with tempfile.TemporaryDirectory() as tmp, temp_chdir(tmp):
            bm25 = await create_bm25()
            await bm25.add_docs(
                {
                    "full": "machine learning 机器 学习",
                    "en_only": "machine learning algorithm",
                    "cn_only": "机器 学习 算法",
                },
            )
            results = await bm25.retrieve("machine 机器", limit=3)
            # full has both English and Chinese hits → highest score.
            assert next(iter(results)) == "full"
            await bm25.close()

    run(go())


def test_unicode_word_with_digits_preserved():
    """Alphanumeric tokens like 'iphone15' stay whole; trailing Chinese still split."""

    async def go():
        with tempfile.TemporaryDirectory() as tmp, temp_chdir(tmp):
            bm25 = await create_bm25()
            await bm25.add_docs(
                {
                    "d1": "iPhone15 Pro 售价 9999 元",
                    "d2": "Android 旗舰 999 元",
                },
            )
            assert "iphone15" in bm25.vocab
            assert "9999" in bm25.vocab
            assert "元" in bm25.vocab

            r1 = await bm25.retrieve("iphone15", limit=2)
            assert list(r1) == ["d1"]
            r2 = await bm25.retrieve("元", limit=2)
            assert set(r2) == {"d1", "d2"}
            await bm25.close()

    run(go())


def test_chinese_punctuation_ignored():
    """CJK punctuation should not produce tokens."""

    async def go():
        with tempfile.TemporaryDirectory() as tmp, temp_chdir(tmp):
            bm25 = await create_bm25()
            await bm25.add_docs({"d1": "你好，世界！这是 Python。"})
            for sym in ["，", "！", "。"]:
                assert sym not in bm25.vocab
            assert "你" in bm25.vocab
            assert "python" in bm25.vocab
            await bm25.close()

    run(go())


def test_mixed_persistence_roundtrip():
    """A mixed-language index round-trips through dump/load with identical retrieval."""

    async def go():
        with tempfile.TemporaryDirectory() as tmp, temp_chdir(tmp):
            bm25 = await create_bm25()
            await bm25.add_docs(
                {
                    "d1": "Python 编程语言",
                    "d2": "Java 编程",
                    "d3": "数据分析 with Python",
                },
            )
            before = await bm25.retrieve("Python 编程", limit=3)
            await bm25.close()  # close triggers dump

            bm25_2 = await create_bm25()  # start triggers load
            assert bm25_2.n_docs == 3
            after = await bm25_2.retrieve("Python 编程", limit=3)
            assert before == after
            await bm25_2.close()

    run(go())


# --------------------------------------------------------------------------- #
# Persistence                                                                 #
# --------------------------------------------------------------------------- #


def test_dump_load_roundtrip_preserves_state():
    """dump → fresh instance → load reconstructs vocab, postings and params."""

    async def go():
        with tempfile.TemporaryDirectory() as tmp, temp_chdir(tmp):
            bm25 = await create_bm25(k1=2.0, b=0.4)
            await bm25.add_docs(
                {
                    "d1": "hello world",
                    "d2": "hello python",
                    "d3": "programming language",
                },
            )
            old_vocab = dict(bm25.vocab)
            old_meta = {k: dict(v) for k, v in bm25.doc_meta.items()}
            await bm25.dump()
            await bm25.close()

            bm25_2 = await create_bm25()  # default k1/b — load must overwrite
            assert bm25_2.vocab == old_vocab
            assert bm25_2.n_docs == 3
            assert set(bm25_2.doc_meta) == set(old_meta)
            assert bm25_2.k1 == 2.0
            assert bm25_2.b == 0.4
            await bm25_2.close()

    run(go())


def test_load_missing_file_keeps_empty_state():
    """Calling load() with no file on disk is a no-op."""

    async def go():
        with tempfile.TemporaryDirectory() as tmp, temp_chdir(tmp):
            bm25 = await create_bm25()
            # No add_docs, nothing persisted.
            assert not bm25.index_file.exists()
            await bm25.load()
            assert bm25.n_docs == 0
            await bm25.close()

    run(go())


def test_load_corrupt_file_resets_index():
    """A corrupt pickle on disk is reported, deleted, and the index is cleared."""

    async def go():
        with tempfile.TemporaryDirectory() as tmp, temp_chdir(tmp):
            bm25 = await create_bm25()
            await bm25.add_docs({"d1": "hello"})
            await bm25.dump()

            # Corrupt the file.
            bm25.index_file.write_bytes(b"not a pickle")
            await bm25.load()
            assert bm25.n_docs == 0
            assert bm25.vocab == {}
            assert not bm25.index_file.exists()
            await bm25.close()

    run(go())


# --------------------------------------------------------------------------- #
# clear / optimize / reset_index                                              #
# --------------------------------------------------------------------------- #


def test_clear_wipes_everything():
    """clear() empties state and removes the index file."""

    async def go():
        with tempfile.TemporaryDirectory() as tmp, temp_chdir(tmp):
            bm25 = await create_bm25()
            await bm25.add_docs({"d1": "hello", "d2": "world"})
            await bm25.dump()
            assert bm25.index_file.exists()

            await bm25.clear()
            assert bm25.n_docs == 0
            assert bm25.vocab == {}
            assert bm25.inverted_index == {}
            assert bm25.doc_meta == {}
            assert bm25.total_len == 0
            assert bm25._idf_cache == {}
            assert not bm25.index_file.exists()
            await bm25.close()

    run(go())


def test_optimize_drops_deleted_only_terms():
    """After deleting, optimize_index drops vocab entries that no live doc uses."""

    async def go():
        with tempfile.TemporaryDirectory() as tmp, temp_chdir(tmp):
            bm25 = await create_bm25()
            await bm25.add_docs(
                {
                    "d1": "alpha unique_to_d1",
                    "d2": "alpha beta",
                },
            )
            assert "unique_to_d1" in bm25.vocab

            await bm25.delete_docs(["d1"])
            await bm25.optimize_index()

            assert bm25.n_docs == 1
            assert "d2" in bm25.doc_meta
            # Term that only existed in d1 is gone.
            assert "unique_to_d1" not in bm25.vocab
            # Shared/own terms of d2 survive.
            assert "alpha" in bm25.vocab and "beta" in bm25.vocab
            # Retrieval still works correctly.
            assert "d2" in await bm25.retrieve("alpha", limit=1)
            await bm25.close()

    run(go())


def test_optimize_when_all_deleted_clears_index():
    """optimize_index on a fully-deleted state collapses to empty index."""

    async def go():
        with tempfile.TemporaryDirectory() as tmp, temp_chdir(tmp):
            bm25 = await create_bm25()
            await bm25.add_docs({"d1": "hello world"})
            await bm25.delete_docs(["d1"])
            await bm25.optimize_index()
            assert bm25.n_docs == 0
            assert bm25.vocab == {}
            assert bm25.inverted_index == {}
            await bm25.close()

    run(go())


def test_optimize_noop_when_no_deletions():
    """With nothing deleted, optimize_index leaves state intact."""

    async def go():
        with tempfile.TemporaryDirectory() as tmp, temp_chdir(tmp):
            bm25 = await create_bm25()
            await bm25.add_docs({"d1": "hello world"})
            vocab_before = dict(bm25.vocab)
            await bm25.optimize_index()
            assert bm25.vocab == vocab_before
            assert bm25.n_docs == 1
            await bm25.close()

    run(go())


def test_reset_index_replaces_all_docs():
    """reset_index (inherited from base) wipes and re-adds in one call."""

    async def go():
        with tempfile.TemporaryDirectory() as tmp, temp_chdir(tmp):
            bm25 = await create_bm25()
            await bm25.add_docs({"d1": "old content"})
            await bm25.reset_index({"d2": "new content"})
            assert bm25.n_docs == 1
            assert "d2" in bm25.doc_meta
            assert "d1" not in bm25.doc_meta
            await bm25.close()

    run(go())


# --------------------------------------------------------------------------- #
# Internal invariants                                                         #
# --------------------------------------------------------------------------- #


def test_idf_cache_populates_and_invalidates():
    """_get_idf caches results; add/delete clear the cache."""

    async def go():
        with tempfile.TemporaryDirectory() as tmp, temp_chdir(tmp):
            bm25 = await create_bm25()
            await bm25.add_docs({"d1": "hello world", "d2": "hello python"})
            tid_hello = bm25.vocab["hello"]
            idf1 = bm25._get_idf(tid_hello)
            assert tid_hello in bm25._idf_cache
            assert bm25._get_idf(tid_hello) == idf1

            # Mutating the index must invalidate the cache.
            await bm25.add_docs({"d3": "hello there"})
            assert bm25._idf_cache == {}

            await bm25.delete_docs(["d1"])
            # delete_docs also clears cache; populate again then trigger via add.
            _ = bm25._get_idf(bm25.vocab["hello"])
            assert bm25._idf_cache  # non-empty now
            await bm25.add_docs({"d4": "x y z"})
            assert bm25._idf_cache == {}
            await bm25.close()

    run(go())


def test_avg_len_tracks_live_docs_only():
    """avg_len excludes deleted docs.

    Note: RegexTokenizer's `\\w\\w+` pattern drops 1-letter words, so we
    use multi-letter tokens to keep length math predictable.
    """

    async def go():
        with tempfile.TemporaryDirectory() as tmp, temp_chdir(tmp):
            bm25 = await create_bm25()
            assert bm25.avg_len == 0.0
            await bm25.add_docs({"d1": "alpha beta gamma delta"})  # 4 tokens
            await bm25.add_docs({"d2": "alpha beta"})  # 2 tokens
            assert bm25.avg_len == 3.0

            await bm25.delete_docs(["d1"])
            assert bm25.avg_len == 2.0
            await bm25.close()

    run(go())


def test_deleted_docs_excluded_from_scoring():
    """A deleted doc must score 0 and never appear in retrieve()."""

    async def go():
        with tempfile.TemporaryDirectory() as tmp, temp_chdir(tmp):
            bm25 = await create_bm25()
            await bm25.add_docs({"d1": "python", "d2": "python", "d3": "python"})
            await bm25.delete_docs(["d2"])

            results = await bm25.retrieve("python", limit=10)
            assert set(results) == {"d1", "d3"}
            assert all(s > 0 for s in results.values())
            await bm25.close()

    run(go())


def test_inverted_index_hides_deleted_postings():
    """inverted_index view skips postings whose doc is deleted."""

    async def go():
        with tempfile.TemporaryDirectory() as tmp, temp_chdir(tmp):
            bm25 = await create_bm25()
            await bm25.add_docs({"d1": "alpha", "d2": "alpha beta"})
            tid_alpha = bm25.vocab["alpha"]
            await bm25.delete_docs(["d1"])

            inv = bm25.inverted_index
            # 'alpha' posting now contains only the live doc.
            assert tid_alpha in inv
            assert set(inv[tid_alpha]) == {"d2"}
            await bm25.close()

    run(go())


# --------------------------------------------------------------------------- #
# Manual runner                                                               #
# --------------------------------------------------------------------------- #


if __name__ == "__main__":
    import inspect
    import sys

    mod = sys.modules[__name__]
    tests = [(name, obj) for name, obj in inspect.getmembers(mod, inspect.isfunction) if name.startswith("test_")]
    print(f"\n=== BaseKeywordIndex / BM25Index Tests ({len(tests)}) ===\n")
    failed = []
    for name, fn in tests:
        try:
            fn()
            print(f"✓ {name}")
        except Exception as exc:  # noqa: BLE001
            print(f"✗ {name}: {exc!r}")
            failed.append(name)
    print()
    if failed:
        print(f"FAILED: {len(failed)} / {len(tests)}")
        for n in failed:
            print(f"  - {n}")
        sys.exit(1)
    print(f"所有 {len(tests)} 项测试通过!")
