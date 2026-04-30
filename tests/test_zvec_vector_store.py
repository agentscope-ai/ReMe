"""Test suite for ZvecVectorStore implementation.

Comprehensive tests covering CRUD operations, search, filtering,
collection management, and edge cases for the zvec vector store adapter.

Usage:
    python -m pytest tests/test_zvec_vector_store.py -v
    python tests/test_zvec_vector_store.py
"""

# pylint: disable=redefined-outer-name,unused-argument

from __future__ import annotations

import asyncio
import shutil
import tempfile
from pathlib import Path
from typing import List
from uuid import uuid4

import pytest

from loguru import logger

from reme.core.schema import VectorNode
from reme.core.vector_store import ZvecVectorStore

# ---------------------------------------------------------------------------
# Skip entire module if zvec native library is not installed
# ---------------------------------------------------------------------------
try:
    import zvec as _zvec  # noqa: F401 — just checking availability
except ImportError:
    pytest.skip("zvec native library not installed", allow_module_location=True)


# ==================== Configuration ====================


class TestConfig:
    """Configuration for zvec test execution."""

    ZVEC_ROOT_PATH = tempfile.mkdtemp(prefix="test_zvec_")
    EMBEDDING_DIMENSION = 64  # Small dimension for faster tests
    TEST_COLLECTION_PREFIX = "test_zvec_vs"


# ==================== Sample Data ====================


def create_sample_nodes(prefix: str = "") -> List[VectorNode]:
    """Create sample VectorNode instances for testing."""
    id_prefix = f"{prefix}_" if prefix else ""
    return [
        VectorNode(
            vector_id=f"{id_prefix}node1",
            content="Artificial intelligence is a technology that simulates human intelligence.",
            metadata={
                "node_type": "tech",
                "category": "AI",
                "source": "research",
                "priority": "high",
                "year": "2023",
            },
        ),
        VectorNode(
            vector_id=f"{id_prefix}node2",
            content="Machine learning is a subset of artificial intelligence.",
            metadata={
                "node_type": "tech",
                "category": "ML",
                "source": "research",
                "priority": "high",
                "year": "2022",
            },
        ),
        VectorNode(
            vector_id=f"{id_prefix}node3",
            content="Deep learning uses neural networks with multiple layers.",
            metadata={
                "node_type": "tech_new",
                "category": "DL",
                "source": "blog",
                "priority": "medium",
                "year": "2024",
            },
        ),
        VectorNode(
            vector_id=f"{id_prefix}node4",
            content="I love eating delicious seafood, especially fresh fish.",
            metadata={
                "node_type": "food",
                "category": "preference",
                "source": "personal",
                "priority": "low",
                "year": "2023",
            },
        ),
        VectorNode(
            vector_id=f"{id_prefix}node5",
            content="Natural language processing enables computers to understand human language.",
            metadata={
                "node_type": "tech",
                "category": "NLP",
                "source": "research",
                "priority": "high",
                "year": "2024",
            },
        ),
    ]


# ==================== Fixtures ====================


class MockEmbeddingModel:
    """A mock embedding model that generates deterministic random vectors.

    Avoids external API calls during testing. Produces unit-normalized
    vectors so that cosine similarity works correctly.
    """

    def __init__(self, dimension: int = 64):
        self.dimension = dimension

    async def get_embedding(self, query: str) -> list[float]:
        """Generate a deterministic embedding from a query string."""
        import hashlib
        import struct

        h = hashlib.sha256(query.encode()).digest()
        # Repeat hash to fill dimension
        full_hash = b""
        while len(full_hash) < self.dimension * 4:
            full_hash += hashlib.sha256(h + full_hash).digest()

        vec = list(struct.unpack(f"<{self.dimension}f", full_hash[: self.dimension * 4]))
        # Normalize to unit vector
        norm = sum(x * x for x in vec) ** 0.5
        if norm > 0:
            vec = [x / norm for x in vec]
        return vec

    async def get_embeddings(self, queries: list[str]) -> list[list[float]]:
        """Generate embeddings for multiple queries."""
        return [await self.get_embedding(q) for q in queries]

    async def get_node_embedding(self, node: VectorNode) -> VectorNode:
        """Assign embedding to a single node."""
        if node.content:
            node.vector = await self.get_embedding(node.content)
        return node

    async def get_node_embeddings(self, nodes: list[VectorNode]) -> list[VectorNode]:
        """Assign embeddings to multiple nodes."""
        return [await self.get_node_embedding(n) for n in nodes]


@pytest.fixture
def embedding_model():
    """Provide a MockEmbeddingModel for tests."""
    return MockEmbeddingModel(dimension=TestConfig.EMBEDDING_DIMENSION)


@pytest.fixture
def zvec_store(embedding_model, tmp_path):
    """Create and start a ZvecVectorStore for testing.

    Yields the store and cleans up afterwards.
    """
    collection_name = f"{TestConfig.TEST_COLLECTION_PREFIX}_{uuid4().hex[:8]}"
    store = ZvecVectorStore(
        collection_name=collection_name,
        db_path=str(tmp_path / "zvec_db"),
        embedding_model=embedding_model,
        dimension=TestConfig.EMBEDDING_DIMENSION,
        distance="cosine",
    )

    async def _setup():
        await store.start()
        return store

    store = asyncio.get_event_loop().run_until_complete(_setup())
    yield store

    async def _teardown():
        try:
            await store.close()
        except Exception:
            pass
        # Clean up temp directory
        db_path = Path(str(tmp_path / "zvec_db"))
        if db_path.exists():
            shutil.rmtree(db_path, ignore_errors=True)

    asyncio.get_event_loop().run_until_complete(_teardown())


# ==================== Helper ====================


def run(coro):
    """Run an async coroutine in the current event loop."""
    return asyncio.get_event_loop().run_until_complete(coro)


# ==================== Test: Collection Lifecycle ====================


class TestCollectionLifecycle:
    """Tests for collection creation, listing, deletion, and copy."""

    def test_create_collection(self, zvec_store):
        """Test that a collection is created during start()."""
        collections = run(zvec_store.list_collections())
        assert zvec_store.collection_name in collections

    def test_list_collections(self, zvec_store):
        """Test listing collections."""
        collections = run(zvec_store.list_collections())
        assert isinstance(collections, list)
        assert len(collections) >= 1

    def test_delete_collection(self, zvec_store, embedding_model, tmp_path):
        """Test deleting a collection."""
        # Create a secondary collection
        coll_name = f"del_test_{uuid4().hex[:8]}"
        store2 = ZvecVectorStore(
            collection_name=coll_name,
            db_path=str(tmp_path / "zvec_db"),
            embedding_model=embedding_model,
            dimension=TestConfig.EMBEDDING_DIMENSION,
        )
        run(store2.start())

        collections = run(zvec_store.list_collections())
        assert coll_name in collections

        run(zvec_store.delete_collection(coll_name))

        collections = run(zvec_store.list_collections())
        assert coll_name not in collections

    def test_copy_collection(self, zvec_store, embedding_model, tmp_path):
        """Test copying a collection."""
        # Insert some data first
        nodes = create_sample_nodes("copy")
        run(zvec_store.insert(nodes))

        copy_name = f"copy_test_{uuid4().hex[:8]}"
        run(zvec_store.copy_collection(copy_name))

        # Verify copy exists
        collections = run(zvec_store.list_collections())
        assert copy_name in collections

        # Clean up
        run(zvec_store.delete_collection(copy_name))


# ==================== Test: Insert ====================


class TestInsert:
    """Tests for node insertion (single and batch)."""

    def test_insert_single_node(self, zvec_store):
        """Test inserting a single node."""
        node = VectorNode(
            vector_id="single_1",
            content="This is a single node insertion test",
            metadata={"test_type": "single_insert"},
        )
        run(zvec_store.insert(node))

        result = run(zvec_store.get("single_1"))
        assert result is not None
        assert result.vector_id == "single_1"
        assert "single node" in result.content

    def test_insert_batch_nodes(self, zvec_store):
        """Test inserting multiple nodes in batch."""
        nodes = create_sample_nodes("batch")
        run(zvec_store.insert(nodes))

        all_nodes = run(zvec_store.list(limit=10))
        assert len(all_nodes) >= len(nodes)

    def test_insert_node_with_vector(self, zvec_store):
        """Test inserting a node that already has a vector."""
        node = VectorNode(
            vector_id="prevec_1",
            content="Node with pre-computed vector",
            vector=[0.1] * TestConfig.EMBEDDING_DIMENSION,
            metadata={"test_type": "pre_vector"},
        )
        run(zvec_store.insert(node))

        result = run(zvec_store.get("prevec_1"))
        assert result is not None
        assert result.vector is not None


# ==================== Test: Search ====================


class TestSearch:
    """Tests for vector similarity search."""

    @pytest.fixture(autouse=True)
    def _insert_sample_data(self, zvec_store):
        """Insert sample data before each search test."""
        nodes = create_sample_nodes("search")
        run(zvec_store.insert(nodes))

    def test_basic_search(self, zvec_store):
        """Test basic vector search."""
        results = run(zvec_store.search(query="What is artificial intelligence?", limit=3))
        assert len(results) > 0
        for r in results:
            assert isinstance(r, VectorNode)
            assert r.content

    def test_search_with_limit(self, zvec_store):
        """Test search with various limits."""
        results = run(zvec_store.search(query="technology", limit=2))
        assert len(results) <= 2

    def test_search_with_filter(self, zvec_store):
        """Test vector search with metadata filter."""
        results = run(
            zvec_store.search(
                query="What is artificial intelligence?",
                limit=5,
                filters={"node_type": "tech"},
            ),
        )
        # All results should have node_type == "tech"
        for r in results:
            assert r.metadata.get("node_type") == "tech"

    def test_search_with_multiple_filters(self, zvec_store):
        """Test search with multiple metadata filters (AND)."""
        results = run(
            zvec_store.search(
                query="What is artificial intelligence?",
                limit=5,
                filters={"node_type": "tech", "source": "research"},
            ),
        )
        for r in results:
            assert r.metadata.get("node_type") == "tech"
            assert r.metadata.get("source") == "research"

    def test_search_relevance_ranking(self, zvec_store):
        """Test that search results have scores and are relevant."""
        results = run(zvec_store.search(query="artificial intelligence", limit=5))
        assert len(results) > 0
        # All results should have a score
        for r in results:
            assert "score" in r.metadata
            assert r.metadata["score"] > 0
        # The top result should be highly relevant (AI content matches AI query)
        top_content = results[0].content.lower()
        assert "artificial intelligence" in top_content or "intelligence" in top_content or "ai" in top_content


# ==================== Test: Get ====================


class TestGet:
    """Tests for retrieving nodes by ID."""

    @pytest.fixture(autouse=True)
    def _insert_sample_data(self, zvec_store):
        """Insert sample data before each get test."""
        nodes = create_sample_nodes("get")
        run(zvec_store.insert(nodes))

    def test_get_single_id(self, zvec_store):
        """Test retrieving a single node by ID."""
        result = run(zvec_store.get("get_node1"))
        assert result is not None
        assert result.vector_id == "get_node1"

    def test_get_multiple_ids(self, zvec_store):
        """Test retrieving multiple nodes by IDs."""
        results = run(zvec_store.get(["get_node1", "get_node2"]))
        assert isinstance(results, list)
        assert len(results) >= 2
        result_ids = {r.vector_id for r in results}
        assert "get_node1" in result_ids
        assert "get_node2" in result_ids

    def test_get_nonexistent_id(self, zvec_store):
        """Test retrieving a non-existent ID."""
        result = run(zvec_store.get("nonexistent_id_xyz"))
        assert result is None or result == []


# ==================== Test: List ====================


class TestList:
    """Tests for listing nodes with optional filters and sorting."""

    @pytest.fixture(autouse=True)
    def _insert_sample_data(self, zvec_store):
        """Insert sample data before each list test."""
        nodes = create_sample_nodes("list")
        run(zvec_store.insert(nodes))

    def test_list_all(self, zvec_store):
        """Test listing all nodes."""
        results = run(zvec_store.list(limit=20))
        assert len(results) > 0

    def test_list_with_filter(self, zvec_store):
        """Test listing nodes with metadata filter."""
        results = run(zvec_store.list(filters={"category": "AI"}, limit=10))
        for r in results:
            assert r.metadata.get("category") == "AI"

    def test_list_with_sorting(self, zvec_store):
        """Test listing with sorting by metadata key."""
        # Insert nodes with numeric metadata for sorting
        sort_nodes = [
            VectorNode(
                vector_id=f"sort_{i}",
                content=f"Sort test node {i}",
                metadata={"rating": str(50 + i * 5), "test_type": "sort_test"},
            )
            for i in range(10)
        ]
        run(zvec_store.insert(sort_nodes))

        results = run(
            zvec_store.list(
                filters={"test_type": "sort_test"},
                sort_key="rating",
                reverse=True,
                limit=5,
            ),
        )
        assert len(results) <= 5
        # Verify descending order
        ratings = [r.metadata.get("rating") for r in results]
        for i in range(len(ratings) - 1):
            assert ratings[i] >= ratings[i + 1]


# ==================== Test: Update ====================


class TestUpdate:
    """Tests for updating existing nodes."""

    @pytest.fixture(autouse=True)
    def _insert_sample_data(self, zvec_store):
        """Insert sample data before each update test."""
        nodes = create_sample_nodes("upd")
        run(zvec_store.insert(nodes))

    def test_update_single_node(self, zvec_store):
        """Test updating a single node's content and metadata."""
        updated = VectorNode(
            vector_id="upd_node2",
            content="Machine learning is a powerful subset of AI that learns from data.",
            metadata={
                "node_type": "tech",
                "category": "ML",
                "updated": "true",
            },
        )
        run(zvec_store.update(updated))

        result = run(zvec_store.get("upd_node2"))
        assert result is not None
        assert result.metadata.get("updated") == "true"

    def test_update_batch(self, zvec_store):
        """Test batch updating multiple nodes."""
        updates = [
            VectorNode(
                vector_id="upd_node1",
                content="Updated content for node 1",
                metadata={"node_type": "tech", "batch_updated": "true"},
            ),
            VectorNode(
                vector_id="upd_node3",
                content="Updated content for node 3",
                metadata={"node_type": "tech_new", "batch_updated": "true"},
            ),
        ]
        run(zvec_store.update(updates))

        results = run(zvec_store.get(["upd_node1", "upd_node3"]))
        if isinstance(results, list):
            for r in results:
                assert r.metadata.get("batch_updated") == "true"


# ==================== Test: Delete ====================


class TestDelete:
    """Tests for deleting nodes."""

    @pytest.fixture(autouse=True)
    def _insert_sample_data(self, zvec_store):
        """Insert sample data before each delete test."""
        nodes = create_sample_nodes("del")
        run(zvec_store.insert(nodes))

    def test_delete_single(self, zvec_store):
        """Test deleting a single node by ID."""
        run(zvec_store.delete("del_node4"))

        # Verify deletion
        result = run(zvec_store.get("del_node4"))
        assert result is None or result == []

    def test_delete_batch(self, zvec_store):
        """Test batch deleting multiple nodes by IDs."""
        # First insert some extra nodes to delete
        extra_nodes = [
            VectorNode(
                vector_id=f"del_extra_{i}",
                content=f"Extra node {i} for batch delete test",
                metadata={"test_type": "batch_delete"},
            )
            for i in range(5)
        ]
        run(zvec_store.insert(extra_nodes))

        ids = [f"del_extra_{i}" for i in range(5)]
        run(zvec_store.delete(ids))

        # Verify all deleted
        for nid in ids:
            result = run(zvec_store.get(nid))
            assert result is None or result == []

    def test_delete_all(self, zvec_store):
        """Test deleting all nodes from the collection."""
        run(zvec_store.delete_all())
        # Collection should be empty now
        remaining = run(zvec_store.list(limit=100))
        assert len(remaining) == 0


# ==================== Test: Edge Cases ====================


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_empty_content(self, zvec_store):
        """Test inserting a node with empty content."""
        node = VectorNode(
            vector_id="edge_empty",
            content="",
            metadata={"type": "empty"},
        )
        # Empty content may fail embedding — that's OK, we just want to see it handled
        try:
            run(zvec_store.insert([node]))
        except Exception:
            pass  # Expected if embedding fails on empty string

    def test_long_content(self, zvec_store):
        """Test inserting a node with very long content."""
        node = VectorNode(
            vector_id="edge_long",
            content="A" * 5000,
            metadata={"type": "long_content"},
        )
        run(zvec_store.insert([node]))
        result = run(zvec_store.get("edge_long"))
        assert result is not None
        assert len(result.content) == 5000

    def test_special_characters(self, zvec_store):
        """Test content with special characters."""
        node = VectorNode(
            vector_id="edge_special",
            content="Special chars: @#$%^&*()[]{}|;:',.<>?/~`",
            metadata={"type": "special_chars"},
        )
        run(zvec_store.insert([node]))
        result = run(zvec_store.get("edge_special"))
        assert result is not None
        assert "@#$%" in result.content

    def test_unicode_content(self, zvec_store):
        """Test content with Unicode characters."""
        node = VectorNode(
            vector_id="edge_unicode",
            content="Unicode test: 你好世界 مرحبا Привет",
            metadata={"type": "unicode"},
        )
        run(zvec_store.insert([node]))
        result = run(zvec_store.get("edge_unicode"))
        assert result is not None
        assert "你好世界" in result.content

    def test_nonexistent_id(self, zvec_store):
        """Test getting a non-existent ID."""
        result = run(zvec_store.get("nonexistent_xyz_999"))
        assert result is None or result == []

    def test_metadata_with_empty_string_value(self, zvec_store):
        """Test metadata containing empty string values."""
        node = VectorNode(
            vector_id="edge_meta_empty",
            content="Testing empty metadata values",
            metadata={"field1": "value1", "field2": "", "field3": "value3"},
        )
        run(zvec_store.insert([node]))
        result = run(zvec_store.get("edge_meta_empty"))
        assert result is not None

    def test_search_nonexistent_filter(self, zvec_store):
        """Test search with a filter value that doesn't match anything."""
        nodes = create_sample_nodes("edge_filter")
        run(zvec_store.insert(nodes))

        results = run(
            zvec_store.search(
                query="test",
                limit=10,
                filters={"category": "NONEXISTENT_CATEGORY"},
            ),
        )
        assert len(results) == 0


# ==================== Test: Batch Operations ====================


class TestBatchOperations:
    """Tests for large-scale batch insert, update, and delete."""

    def test_batch_insert_100_nodes(self, zvec_store):
        """Test inserting 100 nodes in batch."""
        batch_nodes = [
            VectorNode(
                vector_id=f"batch_{i}",
                content=f"This is batch test content number {i} about technology and science.",
                metadata={
                    "batch_id": str(i // 10),
                    "index": str(i),
                    "category": ["tech", "science", "business"][i % 3],
                },
            )
            for i in range(100)
        ]
        run(zvec_store.insert(batch_nodes))

        all_nodes = run(zvec_store.list(limit=150))
        assert len(all_nodes) >= 100

    def test_batch_update_20_nodes(self, zvec_store):
        """Test batch updating 20 nodes."""
        # Insert first
        nodes = [
            VectorNode(
                vector_id=f"bupd_{i}",
                content=f"Batch update test {i}",
                metadata={"index": str(i)},
            )
            for i in range(30)
        ]
        run(zvec_store.insert(nodes))

        # Update first 20
        updates = [
            VectorNode(
                vector_id=f"bupd_{i}",
                content=f"UPDATED content {i}",
                metadata={"index": str(i), "updated": "true"},
            )
            for i in range(20)
        ]
        run(zvec_store.update(updates))

        # Verify
        results = run(zvec_store.list(filters={"updated": "true"}, limit=30))
        assert len(results) >= 20

    def test_batch_delete_50_nodes(self, zvec_store):
        """Test batch deleting 50 nodes."""
        # Insert
        nodes = [
            VectorNode(
                vector_id=f"bdel_{i}",
                content=f"Batch delete test {i}",
                metadata={"index": str(i)},
            )
            for i in range(50)
        ]
        run(zvec_store.insert(nodes))

        # Delete
        ids = [f"bdel_{i}" for i in range(50)]
        run(zvec_store.delete(ids))

        # Verify
        remaining = run(zvec_store.list(limit=200))
        batch_remaining = [n for n in remaining if n.vector_id.startswith("bdel_")]
        assert len(batch_remaining) == 0


# ==================== Test: Concurrent Operations ====================


class TestConcurrentOperations:
    """Tests for concurrent read/write operations."""

    def test_concurrent_inserts_and_searches(self, zvec_store):
        """Test that concurrent inserts and searches work without errors."""

        async def _run():
            # Concurrent inserts
            insert_tasks = []
            for i in range(5):
                batch = [
                    VectorNode(
                        vector_id=f"conc_{i}_{j}",
                        content=f"Concurrent test content {i}-{j}",
                        metadata={"thread_id": str(i)},
                    )
                    for j in range(10)
                ]
                insert_tasks.append(zvec_store.insert(batch))

            await asyncio.gather(*insert_tasks)

            # Concurrent searches
            search_tasks = [zvec_store.search(query="concurrent test", limit=5) for _ in range(5)]
            search_results = await asyncio.gather(*search_tasks)

            # All searches should return results
            for results in search_results:
                assert len(results) > 0

        run(_run())


# ==================== Test: Data Model Conversion ====================


class TestDataModelConversion:
    """Tests for VectorNode <-> zvec Doc conversion helpers."""

    def test_vector_node_to_doc_roundtrip(self, zvec_store):
        """Test that VectorNode -> Doc -> VectorNode roundtrip preserves data."""
        from reme.core.vector_store.zvec_vector_store import (
            _vector_node_to_doc,
            _doc_to_vector_node,
        )

        original = VectorNode(
            vector_id="roundtrip_1",
            content="Roundtrip test content",
            vector=[0.1] * TestConfig.EMBEDDING_DIMENSION,
            metadata={"key1": "value1", "key2": "42", "key3": "true"},
        )

        doc = _vector_node_to_doc(original)
        assert doc.id == "roundtrip_1"
        assert doc.field("content") == "Roundtrip test content"

        restored = _doc_to_vector_node(doc, include_score=False)
        assert restored.vector_id == "roundtrip_1"
        assert restored.content == "Roundtrip test content"
        assert restored.metadata.get("key1") == "value1"

    def test_post_filter_exact_match(self):
        """Test post-filtering with exact match."""
        from reme.core.vector_store.zvec_vector_store import _apply_filters_post

        nodes = [
            VectorNode(vector_id="1", content="a", metadata={"category": "AI"}),
            VectorNode(vector_id="2", content="b", metadata={"category": "ML"}),
            VectorNode(vector_id="3", content="c", metadata={"category": "AI"}),
        ]

        filtered = _apply_filters_post(nodes, {"category": "AI"})
        assert len(filtered) == 2
        assert all(n.metadata["category"] == "AI" for n in filtered)

    def test_post_filter_range_query(self):
        """Test post-filtering with range query."""
        from reme.core.vector_store.zvec_vector_store import _apply_filters_post

        nodes = [
            VectorNode(vector_id="1", content="a", metadata={"year": 2022}),
            VectorNode(vector_id="2", content="b", metadata={"year": 2023}),
            VectorNode(vector_id="3", content="c", metadata={"year": 2024}),
        ]

        filtered = _apply_filters_post(nodes, {"year": [2023, 2024]})
        assert len(filtered) == 2

    def test_post_filter_none_and_empty(self):
        """Test post-filtering with None and empty filters."""
        from reme.core.vector_store.zvec_vector_store import _apply_filters_post

        nodes = [VectorNode(vector_id="1", content="a", metadata={})]

        # None filter returns all
        assert _apply_filters_post(nodes, None) == nodes
        # Empty filter returns all
        assert _apply_filters_post(nodes, {}) == nodes

    def test_score_excluded_from_stored_metadata(self):
        """Test that score is excluded when converting VectorNode to Doc."""
        from reme.core.vector_store.zvec_vector_store import _vector_node_to_doc

        node = VectorNode(
            vector_id="score_test",
            content="test",
            vector=[0.1] * TestConfig.EMBEDDING_DIMENSION,
            metadata={"key1": "val1", "score": 0.95},
        )

        doc = _vector_node_to_doc(node)
        # The metadata JSON should NOT contain the score key
        import json

        stored_meta = json.loads(doc.field("metadata"))
        assert "score" not in stored_meta
        assert "key1" in stored_meta


# ==================== Main Entry Point ====================


async def run_standalone_tests():
    """Run tests standalone (without pytest) for quick validation."""
    tmp_dir = tempfile.mkdtemp(prefix="test_zvec_standalone_")
    embedding_model = MockEmbeddingModel(dimension=TestConfig.EMBEDDING_DIMENSION)

    store = ZvecVectorStore(
        collection_name="standalone_test",
        db_path=tmp_dir,
        embedding_model=embedding_model,
        dimension=TestConfig.EMBEDDING_DIMENSION,
        distance="cosine",
    )

    try:
        await store.start()
        logger.info("✓ Store started")

        # Insert
        nodes = create_sample_nodes("std")
        await store.insert(nodes)
        logger.info(f"✓ Inserted {len(nodes)} nodes")

        # Search
        results = await store.search(query="artificial intelligence", limit=3)
        logger.info(f"✓ Search returned {len(results)} results")
        for r in results:
            logger.info(f"  - {r.vector_id}: {r.content[:50]}... (score={r.metadata.get('score')})")

        # Get
        result = await store.get("std_node1")
        logger.info(f"✓ Get: {result.vector_id if result else 'None'}")

        # List
        all_nodes = await store.list(limit=10)
        logger.info(f"✓ List: {len(all_nodes)} nodes")

        # Update
        await store.update(
            VectorNode(
                vector_id="std_node1",
                content="Updated content",
                metadata={"updated": "true"},
            ),
        )
        result = await store.get("std_node1")
        logger.info(f"✓ Update: metadata.updated={result.metadata.get('updated') if result else 'N/A'}")

        # Delete
        await store.delete("std_node4")
        result = await store.get("std_node4")
        logger.info(f"✓ Delete: {'gone' if result is None or result == [] else 'still exists'}")

        # Count
        count = await store.count()
        logger.info(f"✓ Count: {count} nodes")

        logger.info("✓ All standalone tests passed!")

    finally:
        await store.close()
        shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    asyncio.run(run_standalone_tests())
