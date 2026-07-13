"""Regression tests for LocalEmbeddingStore dimension handling."""

# pylint: disable=protected-access

import asyncio

import numpy as np

from reme.components.embedding_store.base_embedding_store import BaseEmbeddingStore
from reme.components.embedding_store.local_embedding_store import LocalEmbeddingStore
from reme.schema import EmbNode


class FakeAsEmbedding:
    """Fake AgentScope embedding component."""

    dimensions = 2

    async def __call__(self, texts: list[str], **_kwargs):
        return [[1.0] if text == "bad" else [1.0, 0.0] for text in texts]


class BadHealthAsEmbedding:
    """Fake provider whose health probe returns the wrong dimension."""

    dimensions = 2

    async def __call__(self, _texts: list[str], **_kwargs):
        return [[1.0]]


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


def test_compute_batch_rejects_embeddings_with_wrong_dimension():
    """Provider results with wrong dimensions are not padded, truncated, or cached."""

    async def go():
        store = LocalEmbeddingStore(name="t_local_embedding_dim")
        store.as_embedding = FakeAsEmbedding()
        store._key_suffix = f"|{store.dimensions}".encode()

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
