"""Integration tests: drive embedding store through Application wiring.

Requires EMBEDDING_API_KEY (and optionally EMBEDDING_BACKEND / EMBEDDING_BASE_URL /
EMBEDDING_MODEL_NAME) in the environment or a .env file at the repo root.
Hits the real embedding API.
"""

import asyncio
import os
import tempfile

import numpy as np

from reme4 import Application
from reme4.config import resolve_app_config
from reme4.enumeration import ComponentEnum
from reme4.schema import EmbNode
from reme4.utils import cosine_similarity, load_env

load_env()


class _temp_chdir:
    """chdir to path for the duration of the block; restore on exit."""

    def __init__(self, path):
        self.path = path
        self._old = None

    def __enter__(self):
        self._old = os.getcwd()
        os.chdir(self.path)
        return self

    def __exit__(self, *exc):
        os.chdir(self._old)


async def _make_app() -> Application:
    """Build and start an Application from the default config."""
    cfg = resolve_app_config(log_to_console=False, log_to_file=False, enable_logo=False)
    app = Application(**cfg)
    await app.start()
    return app


def test_embedding_health_check():
    """health_check() returns True with a working API key."""

    async def run():
        with tempfile.TemporaryDirectory() as tmp, _temp_chdir(tmp):
            app = await _make_app()
            try:
                store = app.context.components[ComponentEnum.EMBEDDING_STORE]["default"]
                result = await store.health_check(timeout=10.0)
                assert result is True, f"health_check returned {result}"
                assert store.is_healthy is True
                print("✓ test_embedding_health_check passed")
            finally:
                await app.close()

    asyncio.run(run())


def test_embedding_single_text():
    """Single text produces a valid embedding vector."""

    async def run():
        with tempfile.TemporaryDirectory() as tmp, _temp_chdir(tmp):
            app = await _make_app()
            try:
                store = app.context.components[ComponentEnum.EMBEDDING_STORE]["default"]
                emb = await store.get_embedding("Hello, world!")
                assert emb is not None, "get_embedding returned None"
                assert emb.shape == (store.dimensions,), f"shape {emb.shape} != ({store.dimensions},)"
                assert emb.dtype == np.float16, f"dtype {emb.dtype} != float16"
                assert np.linalg.norm(emb) > 0, "embedding is a zero vector"
                print(f"\n  [single] len={len(emb)}, first5={emb[:5].tolist()}")
                print("✓ test_embedding_single_text passed")
            finally:
                await app.close()

    asyncio.run(run())


def test_embedding_multiple_texts():
    """Batch embedding returns correct count and shapes."""

    async def run():
        with tempfile.TemporaryDirectory() as tmp, _temp_chdir(tmp):
            app = await _make_app()
            try:
                store = app.context.components[ComponentEnum.EMBEDDING_STORE]["default"]
                texts = ["cat", "dog", "house"]
                results = await store.get_embeddings(texts)
                assert len(results) == 3, f"expected 3 results, got {len(results)}"
                for i, emb in enumerate(results):
                    assert emb is not None, f"result[{i}] is None"
                    assert emb.shape == (store.dimensions,), f"result[{i}] shape mismatch"
                    assert np.linalg.norm(emb) > 0, f"result[{i}] is a zero vector"
                    print(f"\n  [{texts[i]}] len={len(emb)}, first5={emb[:5].tolist()}")
                print("✓ test_embedding_multiple_texts passed")
            finally:
                await app.close()

    asyncio.run(run())


def test_embedding_cache_hit():
    """Same text returns cached result on second call."""

    async def run():
        with tempfile.TemporaryDirectory() as tmp, _temp_chdir(tmp):
            app = await _make_app()
            try:
                store = app.context.components[ComponentEnum.EMBEDDING_STORE]["default"]
                text = "test caching behavior"
                emb1 = await store.get_embedding(text)
                cache_size_after_first = len(store._cache)  # pylint: disable=protected-access

                emb2 = await store.get_embedding(text)
                cache_size_after_second = len(store._cache)  # pylint: disable=protected-access

                assert emb1 is not None and emb2 is not None
                assert cache_size_after_second == cache_size_after_first, "cache grew on second call"
                assert np.array_equal(emb1, emb2), "cached embedding differs from original"
                print(f"\n  [cache] len={len(emb1)}, first5={emb1[:5].tolist()}")
                print("✓ test_embedding_cache_hit passed")
            finally:
                await app.close()

    asyncio.run(run())


def test_embedding_similarity():
    """Semantically similar texts have higher cosine similarity."""

    async def run():
        with tempfile.TemporaryDirectory() as tmp, _temp_chdir(tmp):
            app = await _make_app()
            try:
                store = app.context.components[ComponentEnum.EMBEDDING_STORE]["default"]
                text_a = "The cat sat on the mat"
                text_b = "A kitten rested on the rug"
                text_c = "Quantum computing uses qubits for parallel computation"

                results = await store.get_embeddings([text_a, text_b, text_c])
                emb_a, emb_b, emb_c = results

                sim_ab = cosine_similarity(emb_a.tolist(), emb_b.tolist())
                sim_ac = cosine_similarity(emb_a.tolist(), emb_c.tolist())

                print(f"\n  sim(cat/kitten) = {sim_ab:.4f}")
                print(f"  sim(cat/quantum) = {sim_ac:.4f}")

                assert sim_ab > 0.4, f"similar texts sim={sim_ab:.4f}, expected > 0.4"
                assert sim_ab > sim_ac, f"similar pair ({sim_ab:.4f}) not > dissimilar ({sim_ac:.4f})"
                print("✓ test_embedding_similarity passed")
            finally:
                await app.close()

    asyncio.run(run())


def test_embedding_node_embeddings():
    """get_node_embeddings fills embedding field on EmbNode objects."""

    async def run():
        with tempfile.TemporaryDirectory() as tmp, _temp_chdir(tmp):
            app = await _make_app()
            try:
                store = app.context.components[ComponentEnum.EMBEDDING_STORE]["default"]
                nodes = [
                    EmbNode(text="first node text"),
                    EmbNode(text="second node text"),
                ]
                result = await store.get_node_embeddings(nodes)
                assert result is nodes, "get_node_embeddings should return the same list"
                for i, node in enumerate(nodes):
                    assert node.embedding is not None, f"node[{i}].embedding is None"
                    assert node.embedding.shape == (store.dimensions,), f"node[{i}] shape mismatch"
                    print(f"\n  [node{i}] len={len(node.embedding)}, first5={node.embedding[:5].tolist()}")
                print("✓ test_embedding_node_embeddings passed")
            finally:
                await app.close()

    asyncio.run(run())


if __name__ == "__main__":
    print("=== Embedding integration tests ===")
    test_embedding_health_check()
    test_embedding_single_text()
    test_embedding_multiple_texts()
    test_embedding_cache_hit()
    test_embedding_similarity()
    test_embedding_node_embeddings()
    print("\nAll embedding integration tests passed!")
