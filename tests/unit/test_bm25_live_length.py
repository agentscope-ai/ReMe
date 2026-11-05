"""Derived BM25 length stays consistent across mutations and old snapshots."""

# pylint: disable=protected-access,missing-function-docstring

import random
from unittest.mock import patch

import numpy as np
import pytest

from reme.components.keyword_index import BM25Index
from reme.components.tokenizer import RegexTokenizer


class ScanningIndex(BM25Index):
    """Original implementation used as a scoring oracle."""

    @property
    def total_len(self):
        return int(self._doc_lens[~self._deleted].sum())


def assert_length(index):
    assert index.total_len == int(index._doc_lens[~index._deleted].sum())


def make_index(cls=BM25Index):
    index = cls()
    index.tokenizer = RegexTokenizer(filter_stopwords=False)
    return index


@pytest.mark.asyncio
async def test_mutations_and_scores_match_scanning_oracle(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    index, reference = make_index(), make_index(ScanningIndex)
    rng = random.Random(417)
    for iteration in range(80):
        doc_id = f"doc-{rng.randrange(12)}"
        if iteration % 4 == 0:
            for target in (index, reference):
                await target.delete_docs([doc_id, doc_id, "absent"])
        else:
            text = " ".join(rng.choices(["alpha", "beta", "gamma"], k=rng.randrange(8)))
            for target in (index, reference):
                await target.add_docs({doc_id: text})
        if iteration % 9 == 0:
            for target in (index, reference):
                await target.optimize_index()
        assert_length(index)
        assert index.total_len == reference.total_len
        assert index.avg_len == reference.avg_len
        assert await index.retrieve("alpha beta", 6) == await reference.retrieve("alpha beta", 6)
        selected = [f"doc-{i}" for i in range(5)]
        assert await index.retrieve_filtered("gamma beta", 3, selected) == await reference.retrieve_filtered(
            "gamma beta",
            3,
            selected,
        )
    await index.clear()
    assert index.total_len == 0
    await index.add_docs({"new": "alpha beta"})
    await index.delete_docs(["new"])
    await index.optimize_index()
    assert index.total_len == 0


@pytest.mark.asyncio
async def test_snapshot_roundtrip_rebuilds_length_without_new_fields(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    index = make_index()
    await index.add_docs({"old": "alpha beta", "live": "gamma beta alpha"})
    await index.delete_docs(["old"])
    snapshot = index._snapshot()
    assert "total_len" not in snapshot and "_total_len" not in snapshot
    restored = make_index()
    await restored.add_docs({"unrelated": "word " * 100})
    restored._restore(snapshot)
    assert restored.total_len == 3
    assert await restored.retrieve("beta") == await index.retrieve("beta")
    await index.dump()
    loaded = make_index()
    await loaded.load()
    assert loaded.total_len == 3
    await loaded.add_docs({"live": "alpha"})
    assert loaded.total_len == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "batch",
    [
        {"existing": "unavailable"},
        {"new": "fresh memory", "failing": "unavailable"},
        {"new": "fresh memory", "existing": "unavailable"},
        {"existing": "replacement memory", "failing": "unavailable"},
    ],
)
async def test_failed_tokenization_preserves_documents_and_allows_retry(batch):
    """A tokenizer error must not retire old documents or leave incomplete new slots."""
    index = make_index()
    tokenizer = index.tokenizer
    await index.add_docs({"existing": "original memory", "untouched": "stable memory"})
    before = await index.retrieve("original stable fresh replacement", limit=10)
    tokenize = tokenizer.tokenize

    def fail_on_unavailable(texts, **kwargs):
        if "unavailable" in texts:
            raise RuntimeError("tokenizer unavailable")
        return tokenize(texts, **kwargs)

    with patch.object(tokenizer, "tokenize", side_effect=fail_on_unavailable):
        with pytest.raises(RuntimeError, match="tokenizer unavailable"):
            await index.add_docs(batch)

    assert_length(index)
    assert index.total_len == 4
    assert set(index.document_ids) == {"existing", "untouched"}
    assert await index.retrieve("original stable fresh replacement", limit=10) == before

    retry = {doc_id: "recovered memory" if text == "unavailable" else text for doc_id, text in batch.items()}
    await index.add_docs(retry)
    assert set(index.document_ids) == {"existing", "untouched"} | set(batch)
    assert set(await index.retrieve("fresh replacement recovered", limit=10)) == set(batch)
    assert set(await index.retrieve("stable")) == {"untouched"}


@pytest.mark.asyncio
async def test_postings_failure_still_counts_appended_arrays(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    index = make_index()
    with patch.object(index, "_extend_postings", side_effect=RuntimeError("posting failed")):
        with pytest.raises(RuntimeError, match="posting failed"):
            await index.add_docs({"new": "alpha beta"})
    assert_length(index)
    assert index.total_len == 2


@pytest.mark.asyncio
async def test_query_never_scans_lengths_with_live_mask():
    class NoBooleanScan(np.ndarray):
        """Integer posting lookups are allowed, corpus-wide mask scans are not."""

        def __getitem__(self, key):
            if isinstance(key, np.ndarray) and key.dtype == bool:
                raise AssertionError("query scanned all document lengths")
            return super().__getitem__(key)

    index = make_index()
    await index.add_docs({"one": "alpha beta", "two": "beta beta"})
    index._doc_lens = index._doc_lens.view(NoBooleanScan)
    assert index.total_len == 4
    assert await index.retrieve("beta")
    assert await index.retrieve_filtered("beta", 1, ["two"])
