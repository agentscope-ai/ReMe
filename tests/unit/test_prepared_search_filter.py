"""Prepared filters preserve the dict matcher and all search entrypoints."""

# pylint: disable=protected-access,missing-function-docstring

import copy
import random
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import numpy as np
import pytest

from reme.components.file_store import FaissLocalFileStore, LocalFileStore, ZvecLocalFileStore
from reme.components.keyword_index import BM25Index
from reme.components.tokenizer import RegexTokenizer
from reme.schema import FileChunk


def test_differential_filter_combinations():
    rng = random.Random(714)
    paths = ["daily/2026-01-01/n.md", "daily/2026-02-01.md", "digest/topic.md", "daily/invalid.md"]
    chunks = [
        FileChunk(id=str(i), path=rng.choice(paths), metadata={"group": rng.choice([None, "a", "b"]), "n": i % 3})
        for i in range(80)
    ]
    path_filters = [{}, {"paths": []}, {"path": paths[0]}, {"path": paths[0], "paths": paths[1:]}]
    prefix_filters = [{}, {"prefixes": []}, {"prefix": "daily/"}, {"path_prefix": "digest", "prefixes": ["daily/"]}]
    date_filters = [
        {},
        {"strict_date_filter": True},
        {"start_date": "2026-01-15"},
        {
            "end_date": "2026-01-31",
            "strict_date_filter": True,
        },
    ]
    metadata_filters = [
        {},
        {"group": ["a", "b"]},
        {"metadata": {"group": "a"}, "group": "b"},
        {
            "metadata": {"n": (1, 2), "absent": None},
        },
    ]
    for _ in range(250):
        config = {
            **rng.choice(path_filters),
            **rng.choice(prefix_filters),
            **rng.choice(date_filters),
            **rng.choice(metadata_filters),
        }
        original = copy.deepcopy(config)
        prepared = LocalFileStore._prepare_search_filter(config)
        assert [prepared(chunk) for chunk in chunks] == [
            LocalFileStore._matches_search_filter(chunk, config) for chunk in chunks
        ]
        assert config == original
    assert LocalFileStore._prepare_search_filter(None)(chunks[0])


def test_preparation_is_query_local_and_preserves_subclass_hooks():
    candidate = FileChunk(path="one")
    config = {"paths": ["one"]}
    prepared = LocalFileStore._prepare_search_filter(config)
    config["paths"][:] = ["two"]
    assert prepared(candidate)
    assert not LocalFileStore._prepare_search_filter(config)(candidate)

    class CustomStore(LocalFileStore):
        """Existing subclass hook still governs the new hot path."""

        @classmethod
        def _matches_search_filter(cls, chunk, search_filter):
            return chunk.path == "special"

    assert not CustomStore._prepare_search_filter({})(candidate)
    assert CustomStore._prepare_search_filter({})(FileChunk(path="special"))


@pytest.mark.parametrize(
    "config",
    [
        {"paths": [], "metadata": {"bad": [[1]]}},
        {"paths": ["one"], "metadata": {"bad": [[1]]}},
        {"prefix": "else", "metadata": 123},
        {"paths": [[1]]},
    ],
)
def test_malformed_conditions_keep_short_circuit_behavior(config):
    candidate = FileChunk(path="one")
    prepared = LocalFileStore._prepare_search_filter(config)
    try:
        expected = LocalFileStore._matches_search_filter(candidate, config)
    except (TypeError, ValueError) as error:
        with pytest.raises(type(error)):
            prepared(candidate)
    else:
        assert prepared(candidate) == expected


def make_store(cls=LocalFileStore):
    # Native ANN boundaries are faked below; execute the real backend search methods.
    store = object.__new__(cls)
    LocalFileStore.__init__(store, embedding_store="")
    store.file_chunks = {
        str(i): FileChunk(id=str(i), path=f"daily/{i}.md", text="alpha beta", embedding=np.array([1.0, 0.0]))
        for i in range(30)
    }
    store.embedding_store = SimpleNamespace(dimensions=2, is_healthy=True)
    store._get_query_embedding = AsyncMock(return_value=np.array([1.0, 0.0]))
    return store


@pytest.mark.asyncio
@pytest.mark.parametrize("fallback", [False, True])
async def test_keyword_prepares_once_and_preserves_fallback(fallback):
    store = make_store()
    index = BM25Index()
    index.tokenizer = RegexTokenizer(filter_stopwords=False)
    await index.add_docs({key: chunk.text for key, chunk in store.file_chunks.items()})
    store.keyword_index = index
    if fallback:
        index.retrieve_filtered = AsyncMock(side_effect=NotImplementedError)
    config = {"paths": ["daily/28.md", "daily/29.md"]}
    with patch.object(store, "_prepare_search_filter", wraps=store._prepare_search_filter) as prepare:
        result = await store.keyword_search("alpha", 2, config)
        assert prepare.call_count == 1
    with patch.object(
        store,
        "_prepare_search_filter",
        side_effect=lambda filt: lambda c: store._matches_search_filter(c, filt),
    ):
        original = await store.keyword_search("alpha", 2, config)
    assert [(c.id, c.scores) for c in result] == [(c.id, c.scores) for c in original]
    assert {c.path for c in result} == set(config["paths"])


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", [LocalFileStore, FaissLocalFileStore, ZvecLocalFileStore])
async def test_vector_prepares_once_including_progressive_recall(backend):
    store = make_store(backend)
    searches = []
    if backend is FaissLocalFileStore:

        def search(_query, count):
            searches.append(count)
            return np.ones((1, count)), np.arange(count).reshape(1, -1)

        store._faiss_index = SimpleNamespace(ntotal=30, search=search, hnsw=SimpleNamespace(efSearch=0))
        store._prepare = Mock(side_effect=lambda value: value.reshape(1, -1))
        store.hnsw_m = 1
        store._id_map = list(store.file_chunks)
        store._tombstones = set()
    elif backend is ZvecLocalFileStore:

        def query(_collection, _vector, count):
            searches.append(count)
            return [(str(i), 1.0) for i in range(count)]

        store._collection = object()
        store._indexed_ids = set(store.file_chunks)
        store._query_collection = Mock(side_effect=query)
    config = {"paths": ["daily/29.md"]}
    with patch.object(store, "_prepare_search_filter", wraps=store._prepare_search_filter) as prepare:
        result = await store.vector_search("alpha", 1, config)
        assert prepare.call_count == 1
    if backend is not LocalFileStore:
        assert searches == [3, 6, 12, 24, 30]
    with patch.object(
        store,
        "_prepare_search_filter",
        side_effect=lambda filt: lambda c: store._matches_search_filter(c, filt),
    ):
        original = await store.vector_search("alpha", 1, config)
    assert [(c.id, c.scores) for c in result] == [(c.id, c.scores) for c in original]
    assert [c.id for c in result] == ["29"]
