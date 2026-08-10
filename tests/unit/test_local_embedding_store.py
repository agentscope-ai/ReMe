"""Regression tests for LocalEmbeddingStore dimension and vector space handling."""

# pylint: disable=protected-access

import asyncio
from types import SimpleNamespace

import numpy as np

from reme.components.as_embedding import OpenAIAsEmbedding
from reme.components.embedding_store.base_embedding_store import BaseEmbeddingStore
from reme.components.embedding_store.local_embedding_store import LocalEmbeddingStore
from reme.schema import EmbNode


class FakeAsEmbedding:
    """Fake AgentScope embedding component."""

    dimensions = 2
    vector_space_id = "fakespace000"

    async def __call__(self, texts: list[str], **_kwargs):
        return [[1.0] if text == "bad" else [1.0, 0.0] for text in texts]


class BadHealthAsEmbedding:
    """Fake provider whose health probe returns the wrong dimension."""

    dimensions = 2
    vector_space_id = "fakespace000"

    async def __call__(self, _texts: list[str], **_kwargs):
        return [[1.0]]


class FakeProviderModel:
    """Stand-in for a constructed AgentScope embedding model object."""

    def __init__(self, model: str, dimensions: int = 2, base_url: str = ""):
        self.model = model
        self.dimensions = dimensions
        self.credential = SimpleNamespace(base_url=base_url)


class InsufficientQuotaError(Exception):
    """OpenAI-compatible quota error used without importing the provider SDK."""

    body = {"error": {"code": "insufficient_quota"}}


class QuotaThenSuccessAsEmbedding:
    """Fail once for quota, then return a valid embedding."""

    dimensions = 2

    def __init__(self):
        self.calls = 0

    async def __call__(self, texts: list[str], **_kwargs):
        self.calls += 1
        if self.calls == 1:
            raise InsufficientQuotaError("quota exhausted")
        return [[1.0, 0.0] for _ in texts]


class BadNodeEmbeddingStore(BaseEmbeddingStore):
    """Embedding store that returns wrong-dimensional vectors."""

    dimensions = 2

    async def health_check(self, timeout: float = 2.0) -> bool:
        return True

    async def get_embeddings(self, input_text: list[str], **_kwargs):
        return [np.array([1.0], dtype=np.float16) for _ in input_text]


def run(coro):
    """Run an async test body."""
    return asyncio.run(coro)


def test_truncate_uses_cjk_aware_integer_budget():
    """Truncation should preserve ASCII behavior and budget non-ASCII text."""
    store = BadNodeEmbeddingStore(name="t_base_embedding_truncate", max_input_length=10)

    assert store._truncate("abcdefghijk") == "abcdefghij"
    assert store._truncate("中文中文中文中文") == "中文中文中文"
    assert store._truncate("éabcdefghij") == "éabcdefgh"

    store.max_input_length = -1
    assert store._truncate("text") == ""
    assert store._truncate("中文") == ""


def test_compute_batch_rejects_embeddings_with_wrong_dimension():
    """Provider results with wrong dimensions are not padded, truncated, or cached."""

    async def go():
        store = LocalEmbeddingStore(name="t_local_embedding_dim")
        store.as_embedding = FakeAsEmbedding()

        results = await store._compute_batch(
            [
                (0, "ok", "ok-cache-key"),
                (1, "bad", "bad-cache-key"),
            ],
        )

        assert len(results) == 1
        assert results[0][0] == 0
        assert results[0][2].tolist() == [1.0, 0.0]
        assert isinstance(results[0][2], np.ndarray)

    run(go())


def test_base_get_node_embeddings_rejects_wrong_dimension():
    """Base node assignment should not accept wrong-dimensional vectors."""

    async def go():
        store = BadNodeEmbeddingStore(name="t_base_embedding_dim")
        node = EmbNode(text="bad")

        await store.get_node_embeddings([node])

        assert node.embedding is None

    run(go())


def test_health_check_rejects_wrong_dimension():
    """Health check should fail when the provider returns the wrong vector length."""

    async def go():
        store = LocalEmbeddingStore(name="t_local_embedding_health_dim")
        store.as_embedding = BadHealthAsEmbedding()

        assert await store.health_check() is False
        assert store.is_healthy is False

    run(go())


def test_insufficient_quota_waits_sixty_seconds_before_retry(monkeypatch):
    """Quota exhaustion uses the dedicated delay before ReMe retries."""

    async def go():
        sleeps = []

        async def fake_sleep(delay):
            sleeps.append(delay)

        store = LocalEmbeddingStore(
            name="t_local_embedding_quota",
            max_retries=2,
            quota_retry_delay=60.0,
        )
        embedding = QuotaThenSuccessAsEmbedding()
        store.as_embedding = embedding
        monkeypatch.setattr(asyncio, "sleep", fake_sleep)

        result = await store._call_with_retry(["text"])

        assert result == [[1.0, 0.0]]
        assert embedding.calls == 2
        assert sleeps == [60.0]

    run(go())


def test_insufficient_quota_does_not_retry_without_opt_in(monkeypatch):
    """The default store behavior remains unchanged for embedded consumers."""

    async def go():
        sleeps = []

        async def fake_sleep(delay):
            sleeps.append(delay)

        store = LocalEmbeddingStore(name="t_local_embedding_default_quota", max_retries=2)
        embedding = QuotaThenSuccessAsEmbedding()
        store.as_embedding = embedding
        monkeypatch.setattr(asyncio, "sleep", fake_sleep)

        result = await store._call_with_retry(["text"])

        assert result is None
        assert embedding.calls == 1
        assert not sleeps

    run(go())


def test_vector_space_id_separates_models_of_equal_dimension():
    """Two models of the same width must not claim the same vector space."""
    common = {"backend": "openai", "dimensions": 1024, "credential": {"base_url": "https://example.com/v1"}}
    v3 = OpenAIAsEmbedding(name="t_space_v3", model="text-embedding-v3", **common)
    v4 = OpenAIAsEmbedding(name="t_space_v4", model="text-embedding-v4", **common)

    assert v3.dimensions == v4.dimensions
    assert v3.vector_space_id != v4.vector_space_id


def test_vector_space_id_separates_endpoints_of_one_model_name():
    """The same model name served by two endpoints is two vector spaces."""
    common = {"backend": "openai", "model": "text-embedding-v4", "dimensions": 1024}
    official = OpenAIAsEmbedding(name="t_space_a", credential={"base_url": "https://example.com/v1"}, **common)
    self_hosted = OpenAIAsEmbedding(name="t_space_b", credential={"base_url": "http://127.0.0.1:8000/v1"}, **common)

    assert official.vector_space_id != self_hosted.vector_space_id


def test_vector_space_id_ignores_trailing_slash_and_api_key():
    """Cosmetic and secret credential changes must not invalidate stored vectors."""
    common = {"backend": "openai", "model": "text-embedding-v4", "dimensions": 1024}
    first = OpenAIAsEmbedding(
        name="t_space_c",
        credential={"base_url": "https://example.com/v1", "api_key": "key-one"},
        **common,
    )
    second = OpenAIAsEmbedding(
        name="t_space_d",
        credential={"base_url": "https://example.com/v1/", "api_key": "key-two"},
        **common,
    )

    assert first.vector_space_id == second.vector_space_id


def test_vector_space_id_follows_a_model_swapped_in_after_start():
    """A provider replaced at runtime must win over the original kwargs."""
    embedding = OpenAIAsEmbedding(name="t_space_swap", backend="openai", model="v3", dimensions=2)
    before = embedding.vector_space_id

    # Mirrors Application.update_component("as_embedding", "default", model=<new provider>).
    embedding.model = FakeProviderModel("v4")

    assert embedding.vector_space_id != before


def test_vector_space_id_is_stable_across_lazy_provider_construction():
    """Constructing the configured provider must not look like a model switch."""
    embedding = OpenAIAsEmbedding(
        name="t_space_lazy",
        backend="openai",
        model="v3",
        dimensions=2,
        credential={"base_url": "https://example.com/v1"},
    )
    before = embedding.vector_space_id

    embedding.model = FakeProviderModel("v3", base_url="https://example.com/v1")

    assert embedding.vector_space_id == before


def test_cache_is_saved_and_restored_per_vector_space(monkeypatch, tmp_path):
    """Switching models persists the old cache and restores it when switched back."""

    async def go():
        monkeypatch.setattr(
            LocalEmbeddingStore,
            "component_metadata_path",
            property(lambda _self: tmp_path),
        )
        embedding = OpenAIAsEmbedding(name="t_space_store", backend="openai", model="v3", dimensions=2)
        store = LocalEmbeddingStore(name="t_local_space")
        store.as_embedding = embedding
        await store.load()

        key = store._cache_key("hello")
        store._cache_put(key, np.array([1.0, 0.0], dtype=np.float16))
        v3_path = store.cache_path

        embedding.model = FakeProviderModel("v4")
        await store._sync_cache_space()

        assert store._cache_key("hello") == key
        assert store.cache_path != v3_path
        assert store._cache_get(store._cache_key("hello")) is None
        assert v3_path.exists()

        embedding.model = FakeProviderModel("v3")
        await store._sync_cache_space()

        np.testing.assert_array_equal(store._cache_get(key), np.array([1.0, 0.0], dtype=np.float16))

    run(go())


def test_start_ignores_cache_file_without_vector_space_tag(monkeypatch, tmp_path):
    """An unattributable legacy cache is ignored without deleting derived data."""

    async def go():
        monkeypatch.setattr(
            LocalEmbeddingStore,
            "component_metadata_path",
            property(lambda _self: tmp_path),
        )
        store = LocalEmbeddingStore(name="t_local_untagged")
        store.as_embedding = FakeAsEmbedding()
        untagged = tmp_path / f"{store.name}_{store.cache_version}.npz"
        untagged.write_bytes(b"vectors from an unknown model")

        await store._start()

        assert untagged.exists()
        assert not store._cache

    run(go())
