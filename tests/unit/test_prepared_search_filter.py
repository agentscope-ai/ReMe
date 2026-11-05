"""Prepared filters preserve the dict matcher and all search entrypoints."""

# pylint: disable=protected-access,missing-function-docstring

import copy
import random
from itertools import product
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import numpy as np
import pytest

from reme.components.file_store import FaissLocalFileStore, LocalFileStore, ZvecLocalFileStore
from reme.components.keyword_index import BM25Index
from reme.components.runtime_context import RuntimeContext
from reme.components.tokenizer import RegexTokenizer
from reme.schema import FileChunk
from reme.steps.index import SearchStep


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
    store = make_store()
    for path, prefix, date, metadata in product(path_filters, prefix_filters, date_filters, metadata_filters):
        config = {
            **path,
            **prefix,
            **date,
            **metadata,
        }
        original = copy.deepcopy(config)
        prepared = store._prepare_search_filter(config)
        assert [prepared(chunk) for chunk in chunks] == [
            LocalFileStore._matches_search_filter(chunk, config) for chunk in chunks
        ]
        assert config == original
    assert store._prepare_search_filter(None)(chunks[0])


def test_preparation_is_query_local_and_preserves_subclass_hooks():
    store = make_store()
    candidate = FileChunk(path="one")
    config = {"paths": ["one"]}
    prepared = store._prepare_search_filter(config)
    config["paths"][:] = ["two"]
    assert prepared(candidate)
    assert not store._prepare_search_filter(config)(candidate)

    class CustomStore(LocalFileStore):
        """Existing subclass hook still governs the new hot path."""

        @classmethod
        def _matches_search_filter(cls, chunk, search_filter):
            return chunk.path == "special"

    custom_store = make_store(CustomStore)
    assert not custom_store._prepare_search_filter({})(candidate)
    assert custom_store._prepare_search_filter({})(FileChunk(path="special"))


@pytest.mark.asyncio
@pytest.mark.parametrize("hook", ["_matches_search_filter", "_value_matches"])
@pytest.mark.parametrize("config", [None, {}, {"group": "selected"}])
async def test_static_subclass_hooks_preserve_search_results(hook, config):
    class CustomStore(LocalFileStore):
        """Third-party stores can implement either matching hook as a static method."""

    if hook == "_matches_search_filter":
        CustomStore._matches_search_filter = staticmethod(lambda chunk, _config: chunk.id == "29")
    else:
        CustomStore._value_matches = staticmethod(lambda actual, expected: actual == expected.upper())

    store = make_store(CustomStore)
    store.file_chunks["29"].metadata["group"] = "SELECTED"
    index = BM25Index()
    index.tokenizer = RegexTokenizer(filter_stopwords=False)
    await index.add_docs({key: chunk.text for key, chunk in store.file_chunks.items()})
    store.keyword_index = index
    expected_ids = {chunk.id for chunk in store.file_chunks.values() if store._matches_search_filter(chunk, config)}
    assert expected_ids == ({"29"} if hook == "_matches_search_filter" or config else set(store.file_chunks))
    for search in (store.keyword_search, store.vector_search):
        result = await search("alpha", 30, config)
        with patch.object(
            store,
            "_prepare_search_filter",
            side_effect=lambda filt: lambda chunk: store._matches_search_filter(chunk, filt),
        ):
            legacy = await search("alpha", 30, config)
        assert {chunk.id for chunk in result} == expected_ids
        assert [(chunk.id, chunk.scores) for chunk in result] == [(chunk.id, chunk.scores) for chunk in legacy]


@pytest.mark.asyncio
@pytest.mark.parametrize("override", ["instance_method", "instance_attribute"])
@pytest.mark.parametrize("config", [None, {}, {"group": "selected"}])
async def test_bound_instance_matchers_preserve_search_results(override, config):
    class CustomStore(LocalFileStore):
        """A matcher may depend on state belonging to this store instance."""

        def _matches_search_filter(self, chunk, search_filter):
            return chunk.id == self.kwargs["allowed_id"]

    store = make_store(CustomStore if override == "instance_method" else LocalFileStore)
    store.kwargs["allowed_id"] = "29"
    if override == "instance_attribute":
        store._matches_search_filter = lambda chunk, _config: chunk.id == store.kwargs["allowed_id"]
    index = BM25Index()
    index.tokenizer = RegexTokenizer(filter_stopwords=False)
    await index.add_docs({key: chunk.text for key, chunk in store.file_chunks.items()})
    store.keyword_index = index
    for search in (store.keyword_search, store.vector_search):
        result = await search("alpha", 30, config)
        with patch.object(
            store,
            "_prepare_search_filter",
            side_effect=lambda filt: lambda chunk: store._matches_search_filter(chunk, filt),
        ):
            legacy = await search("alpha", 30, config)
        assert [chunk.id for chunk in result] == ["29"]
        assert [(chunk.id, chunk.scores) for chunk in result] == [(chunk.id, chunk.scores) for chunk in legacy]


@pytest.mark.parametrize(
    ("search_filter", "expected"),
    [
        ({"tags": ["python"]}, True),
        ({"tags": "python"}, True),
        ({"tags": ["python", "java"]}, True),
        ({"tags": ["java"]}, False),
        ({"tags": "java"}, False),
        ({"metadata": {"tags": ["notes"]}}, True),
        ({"metadata": {"tags": ["java"]}}, False),
        ({"tags": ["python"], "name": "note"}, True),
        ({"tags": ["python"], "name": "other"}, False),
        ({"absent": "value"}, False),
    ],
)
def test_list_valued_metadata_matches_search_filters(search_filter, expected):
    """Frontmatter lists are matched by content instead of raising or missing."""
    store = make_store()
    chunk = FileChunk(path="daily/2026-01-01/n.md", metadata={"tags": ["python", "notes"], "name": "note"})
    assert store._prepare_search_filter(search_filter)(chunk) is expected
    assert LocalFileStore._matches_search_filter(chunk, search_filter) is expected


@pytest.mark.asyncio
async def test_search_entrypoints_filter_list_valued_metadata():
    """Vector and keyword search keep list-valued metadata filterable."""
    store = make_store()
    store.file_chunks["7"].metadata["tags"] = ["python", "notes"]
    index = BM25Index()
    index.tokenizer = RegexTokenizer(filter_stopwords=False)
    await index.add_docs({key: chunk.text for key, chunk in store.file_chunks.items()})
    store.keyword_index = index
    for search in (store.vector_search, store.keyword_search):
        assert [chunk.id for chunk in await search("alpha", 5, {"tags": ["python"]})] == ["7"]
        assert [chunk.id for chunk in await search("alpha", 5, {"tags": "python"})] == ["7"]
        assert await search("alpha", 5, {"tags": ["java"]}) == []


@pytest.mark.asyncio
async def test_search_step_returns_list_valued_metadata_matches():
    """The search Step returns the list-valued metadata match instead of failing."""
    store = make_store()
    store.file_chunks["7"].metadata["tags"] = ["python", "notes"]
    index = BM25Index()
    index.tokenizer = RegexTokenizer(filter_stopwords=False)
    await index.add_docs({key: chunk.text for key, chunk in store.file_chunks.items()})
    store.keyword_index = index
    step = SearchStep(file_store=store, expand_links=False)

    resp = await step(RuntimeContext(query="alpha", limit=5, search_filter={"tags": ["python"]}))

    assert resp.success is True
    assert [item["id"] for item in resp.metadata["results"]] == ["7"]


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
    store = make_store()
    candidate = FileChunk(path="one")
    prepared = store._prepare_search_filter(config)
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
