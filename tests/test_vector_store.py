# pylint: disable=too-many-lines
"""Unified test suite for vector store implementations.

This module provides comprehensive test coverage for LocalVectorStore, ESVectorStore,
PGVectorStore, QdrantVectorStore, and ChromaVectorStore implementations. Tests can be
run for specific vector stores or all implementations.

Usage:
    python test_vector_store.py --local      # Test LocalVectorStore only
    python test_vector_store.py --es         # Test ESVectorStore only
    python test_vector_store.py --pgvector   # Test PGVectorStore only
    python test_vector_store.py --qdrant     # Test QdrantVectorStore only
    python test_vector_store.py --chroma     # Test ChromaVectorStore only
    python test_vector_store.py --all        # Test all vector stores

"""

import argparse
import asyncio
import shutil
from pathlib import Path
from typing import List

from loguru import logger

from reme_ai.core.embedding import OpenAIEmbeddingModel
from reme_ai.core.schema import VectorNode
from reme_ai.core.vector_store import (
    BaseVectorStore,
    ChromaVectorStore,
    LocalVectorStore,
    ESVectorStore,
    PGVectorStore,
    QdrantVectorStore,
)


# ==================== Configuration ====================


class TestConfig:
    """Configuration for test execution."""

    # LocalVectorStore settings
    LOCAL_ROOT_PATH = "./test_vector_store_local"

    # ESVectorStore settings
    ES_HOSTS = "http://11.160.132.46:8200"
    ES_BASIC_AUTH = None  # Set to ("username", "password") if authentication is required

    # QdrantVectorStore settings
    QDRANT_PATH = None  # "./test_vector_store_qdrant"  # For local mode
    QDRANT_HOST = None  # Set to host address for remote mode (e.g., "localhost")
    QDRANT_PORT = None  # Set to port for remote mode (e.g., 6333)
    QDRANT_URL = "http://11.160.132.46:6333"  # Alternative to host/port (e.g., http://localhost:6333)
    QDRANT_API_KEY = None  # Set for Qdrant Cloud authentication

    # PGVectorStore settings
    PG_DSN = "postgresql://localhost/postgres"  # PostgreSQL connection string
    PG_MIN_SIZE = 1  # Minimum connections in pool
    PG_MAX_SIZE = 5  # Maximum connections in pool
    PG_USE_HNSW = True  # Use HNSW index for faster search
    PG_USE_DISKANN = False  # Use DiskANN index (requires vectorscale extension)

    # ChromaVectorStore settings
    CHROMA_PATH = "./test_vector_store_chroma"  # For local persistent mode
    CHROMA_HOST = None  # Set to host address for remote mode (e.g., "localhost")
    CHROMA_PORT = None  # Set to port for remote mode (e.g., 8000)
    CHROMA_API_KEY = None  # Set for ChromaDB Cloud authentication
    CHROMA_TENANT = None  # Set for ChromaDB Cloud tenant
    CHROMA_DATABASE = None  # Set for ChromaDB Cloud database

    # Embedding model settings
    EMBEDDING_MODEL_NAME = "text-embedding-v4"
    EMBEDDING_DIMENSIONS = 64

    # Test collection naming
    TEST_COLLECTION_PREFIX = "test_vector_store"


# ==================== Sample Data Generator ====================


class SampleDataGenerator:
    """Generator for sample test data."""

    @staticmethod
    def create_sample_nodes(prefix: str = "") -> List[VectorNode]:
        """Create sample VectorNode instances for testing.

        Args:
            prefix: Optional prefix for vector_id to avoid conflicts

        Returns:
            List[VectorNode]: List of sample nodes with diverse metadata
        """
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
                    "department": "engineering",
                    "language": "english",
                    "status": "published",
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
                    "department": "engineering",
                    "language": "english",
                    "status": "published",
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
                    "department": "marketing",
                    "language": "chinese",
                    "status": "draft",
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
                    "department": "lifestyle",
                    "language": "english",
                    "status": "published",
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
                    "department": "engineering",
                    "language": "english",
                    "status": "review",
                },
            ),
        ]


# ==================== Vector Store Factory ====================


def get_store_type(store: BaseVectorStore) -> str:
    """Get the type identifier of a vector store instance.

    Args:
        store: Vector store instance

    Returns:
        str: Type identifier ("local", "es", "pgvector", "qdrant", or "chroma")
    """
    if isinstance(store, LocalVectorStore):
        return "local"
    elif isinstance(store, QdrantVectorStore):
        return "qdrant"
    elif isinstance(store, ESVectorStore):
        return "es"
    elif isinstance(store, PGVectorStore):
        return "pgvector"
    elif isinstance(store, ChromaVectorStore):
        return "chroma"
    else:
        raise ValueError(f"Unknown vector store type: {type(store)}")


def create_vector_store(store_type: str, collection_name: str) -> BaseVectorStore:
    """Create a vector store instance based on type.

    Args:
        store_type: Type of vector store ("local", "es", or "qdrant")
        collection_name: Name of the collection

    Returns:
        BaseVectorStore: Initialized vector store instance
    """
    config = TestConfig()

    # Initialize embedding model
    embedding_model = OpenAIEmbeddingModel(
        model_name=config.EMBEDDING_MODEL_NAME,
        dimensions=config.EMBEDDING_DIMENSIONS,
    )

    if store_type == "local":
        return LocalVectorStore(
            collection_name=collection_name,
            embedding_model=embedding_model,
            root_path=config.LOCAL_ROOT_PATH,
        )
    elif store_type == "es":
        return ESVectorStore(
            collection_name=collection_name,
            embedding_model=embedding_model,
            hosts=config.ES_HOSTS,
            basic_auth=config.ES_BASIC_AUTH,
        )
    elif store_type == "qdrant":
        return QdrantVectorStore(
            collection_name=collection_name,
            embedding_model=embedding_model,
            path=config.QDRANT_PATH,
            host=config.QDRANT_HOST,
            port=config.QDRANT_PORT,
            url=config.QDRANT_URL,
            api_key=config.QDRANT_API_KEY,
            distance="cosine",
            on_disk=False,
        )
    elif store_type == "pgvector":
        return PGVectorStore(
            collection_name=collection_name,
            embedding_model=embedding_model,
            dsn=config.PG_DSN,
            min_size=config.PG_MIN_SIZE,
            max_size=config.PG_MAX_SIZE,
            use_hnsw=config.PG_USE_HNSW,
            use_diskann=config.PG_USE_DISKANN,
        )
    elif store_type == "chroma":
        return ChromaVectorStore(
            collection_name=collection_name,
            embedding_model=embedding_model,
            path=config.CHROMA_PATH,
            host=config.CHROMA_HOST,
            port=config.CHROMA_PORT,
            api_key=config.CHROMA_API_KEY,
            tenant=config.CHROMA_TENANT,
            database=config.CHROMA_DATABASE,
        )
    else:
        raise ValueError(f"Unknown store type: {store_type}")


# ==================== Test Functions ====================


async def test_create_collection(store: BaseVectorStore, _store_name: str):
    """Test collection creation."""
    logger.info("=" * 20 + " CREATE COLLECTION TEST " + "=" * 20)

    # Clean up if exists
    collections = await store.list_collections()
    if store.collection_name in collections:
        await store.delete_collection(store.collection_name)
        logger.info(f"Cleaned up existing collection: {store.collection_name}")

    # Create collection
    await store.create_collection(store.collection_name)

    # Verify creation
    collections = await store.list_collections()
    assert store.collection_name in collections, "Collection should exist after creation"
    logger.info(f"✓ Created collection: {store.collection_name}")


async def test_insert(store: BaseVectorStore, _store_name: str) -> List[VectorNode]:
    """Test node insertion (single and batch)."""
    logger.info("=" * 20 + " INSERT TEST " + "=" * 20)

    # Test single node insertion
    single_node = VectorNode(
        vector_id="test_single_insert",
        content="This is a single node insertion test",
        metadata={"test_type": "single_insert"},
    )
    await store.insert(single_node)
    logger.info("✓ Inserted single node")

    # Test batch insertion
    sample_nodes = SampleDataGenerator.create_sample_nodes("test")
    await store.insert(sample_nodes)
    logger.info(f"✓ Batch inserted {len(sample_nodes)} nodes")

    # Verify total insertions
    all_nodes = await store.list(limit=20)
    assert len(all_nodes) >= len(sample_nodes) + 1, "Should have at least sample nodes + single node"
    logger.info(f"✓ Total nodes in collection: {len(all_nodes)}")

    return sample_nodes


async def test_search(store: BaseVectorStore, _store_name: str):
    """Test basic vector search."""
    logger.info("=" * 20 + " SEARCH TEST " + "=" * 20)

    results = await store.search(
        query="What is artificial intelligence?",
        limit=3,
    )

    logger.info(f"Search returned {len(results)} results")
    for i, r in enumerate(results, 1):
        score = r.metadata.get("_score", "N/A")
        logger.info(f"  Result {i}: {r.content[:60]}... (score: {score})")

    assert len(results) > 0, "Search should return results"
    logger.info("✓ Basic search test passed")


async def test_search_with_single_filter(store: BaseVectorStore, _store_name: str):
    """Test vector search with single metadata filter."""
    logger.info("=" * 20 + " SINGLE FILTER SEARCH TEST " + "=" * 20)

    # Test single value filter
    filters = {"node_type": "tech"}
    results = await store.search(
        query="What is artificial intelligence?",
        limit=5,
        filters=filters,
    )

    logger.info(f"Filtered search (node_type=tech) returned {len(results)} results")
    for i, r in enumerate(results, 1):
        node_type = r.metadata.get("node_type")
        logger.info(f"  Result {i}: type={node_type}, content={r.content[:50]}...")
        assert node_type == "tech", "Result should have node_type='tech'"

    logger.info("✓ Single filter search test passed")


async def test_search_with_list_filter(store: BaseVectorStore, _store_name: str):
    """Test vector search with list filter (IN operation)."""
    logger.info("=" * 20 + " LIST FILTER SEARCH TEST " + "=" * 20)

    # Test list filter (IN operation)
    filters = {"node_type": ["tech", "tech_new"]}
    results = await store.search(
        query="What is artificial intelligence?",
        limit=5,
        filters=filters,
    )

    logger.info(f"Filtered search (node_type IN [tech, tech_new]) returned {len(results)} results")
    for i, r in enumerate(results, 1):
        node_type = r.metadata.get("node_type")
        logger.info(f"  Result {i}: type={node_type}, content={r.content[:50]}...")
        assert node_type in ["tech", "tech_new"], "Result should have node_type in [tech, tech_new]"

    logger.info("✓ List filter search test passed")


async def test_search_with_multiple_filters(store: BaseVectorStore, _store_name: str):
    """Test vector search with multiple metadata filters (AND operation)."""
    logger.info("=" * 20 + " MULTIPLE FILTERS SEARCH TEST " + "=" * 20)

    # Test multiple filters (AND operation)
    filters = {
        "node_type": ["tech", "tech_new"],
        "source": "research",
    }
    results = await store.search(
        query="What is artificial intelligence?",
        limit=5,
        filters=filters,
    )

    logger.info(
        f"Multi-filter search (node_type IN [tech, tech_new] AND source=research) " f"returned {len(results)} results",
    )
    for i, r in enumerate(results, 1):
        node_type = r.metadata.get("node_type")
        source = r.metadata.get("source")
        logger.info(f"  Result {i}: type={node_type}, source={source}, content={r.content[:40]}...")
        assert node_type in ["tech", "tech_new"], "Result should have node_type in [tech, tech_new]"
        assert source == "research", "Result should have source='research'"

    logger.info("✓ Multiple filters search test passed")


async def test_get_by_id(store: BaseVectorStore, _store_name: str):
    """Test retrieving nodes by vector_id (single and batch)."""
    logger.info("=" * 20 + " GET BY ID TEST " + "=" * 20)

    # Test single ID retrieval
    target_id = "test_node1"
    result = await store.get(target_id)

    assert isinstance(result, VectorNode), "Should return a VectorNode for single ID"
    assert result.vector_id == target_id, f"Result should have vector_id={target_id}"
    logger.info(f"✓ Retrieved single node: {result.vector_id}")

    # Test batch retrieval (small batch)
    target_ids = ["test_node1", "test_node2"]
    results = await store.get(target_ids)

    assert isinstance(results, list), "Should return a list for multiple IDs"
    assert len(results) == 2, f"Should return 2 results, got {len(results)}"
    result_ids = {r.vector_id for r in results}
    assert result_ids == set(target_ids), f"Result IDs should match {target_ids}"
    logger.info(f"✓ Batch retrieved {len(results)} nodes")

    # Test larger batch retrieval
    large_batch_ids = ["test_node1", "test_node2", "test_node3", "test_node5"]
    large_results = await store.get(large_batch_ids)
    assert isinstance(large_results, list), "Should return a list for batch IDs"
    assert len(large_results) >= 3, "Should return at least 3 results"
    logger.info(f"✓ Large batch retrieved {len(large_results)} nodes")


async def test_list_all(store: BaseVectorStore, _store_name: str):
    """Test listing all nodes in collection."""
    logger.info("=" * 20 + " LIST ALL TEST " + "=" * 20)

    results = await store.list(limit=10)

    logger.info(f"Collection contains {len(results)} nodes")
    for i, node in enumerate(results, 1):
        logger.info(f"  Node {i}: id={node.vector_id}, content={node.content[:50]}...")

    assert len(results) > 0, "Collection should contain nodes"
    logger.info("✓ List all nodes test passed")


async def test_list_with_filters(store: BaseVectorStore, _store_name: str):
    """Test listing nodes with metadata filters."""
    logger.info("=" * 20 + " LIST WITH FILTERS TEST " + "=" * 20)

    filters = {"category": "AI"}
    results = await store.list(filters=filters, limit=10)

    logger.info(f"Filtered list (category=AI) returned {len(results)} nodes")
    for i, node in enumerate(results, 1):
        category = node.metadata.get("category")
        logger.info(f"  Node {i}: category={category}, id={node.vector_id}")
        assert category == "AI", "All nodes should have category=AI"

    logger.info("✓ List with filters test passed")


async def test_update(store: BaseVectorStore, _store_name: str):
    """Test updating existing nodes (single and batch)."""
    logger.info("=" * 20 + " UPDATE TEST " + "=" * 20)

    # Test single node update
    updated_node = VectorNode(
        vector_id="test_node2",
        content="Machine learning is a powerful subset of AI that learns from data.",
        metadata={
            "node_type": "tech",
            "category": "ML",
            "updated": "true",
            "update_timestamp": "2024-12-26",
        },
    )

    await store.update(updated_node)

    # Verify single update
    result = await store.get("test_node2")
    assert "updated" in result.metadata, "Updated metadata should be present"
    logger.info(f"✓ Updated single node: {result.vector_id}")
    logger.info(f"  New content: {result.content[:60]}...")

    # Test batch update (update multiple nodes at once)
    batch_update_nodes = [
        VectorNode(
            vector_id="test_node1",
            content="Artificial intelligence is evolving rapidly with new breakthroughs.",
            metadata={
                "node_type": "tech",
                "category": "AI",
                "batch_updated": "true",
                "update_timestamp": "2024-12-31",
            },
        ),
        VectorNode(
            vector_id="test_node3",
            content="Deep learning revolutionizes neural network architectures.",
            metadata={
                "node_type": "tech_new",
                "category": "DL",
                "batch_updated": "true",
                "update_timestamp": "2024-12-31",
            },
        ),
    ]

    await store.update(batch_update_nodes)
    logger.info(f"✓ Batch updated {len(batch_update_nodes)} nodes")

    # Verify batch updates
    results = await store.get(["test_node1", "test_node3"])
    for r in results:
        assert r.metadata.get("batch_updated") == "true", f"Node {r.vector_id} should have batch_updated metadata"
    logger.info(f"✓ Verified batch update for {len(results)} nodes")


async def test_delete(store: BaseVectorStore, _store_name: str):
    """Test deleting nodes (single and batch)."""
    logger.info("=" * 20 + " DELETE TEST " + "=" * 20)

    # Test single node deletion
    node_to_delete = "test_node4"
    await store.delete(node_to_delete)

    # Verify single deletion - try to get the deleted node
    try:
        result = await store.get(node_to_delete)
        # If result is empty list or None, deletion was successful
        if isinstance(result, list):
            assert len(result) == 0, "Deleted node should not be retrievable"
        else:
            assert result is None, "Deleted node should not be retrievable"
    except Exception:
        pass  # Expected if node doesn't exist

    logger.info(f"✓ Deleted single node: {node_to_delete}")

    # Test batch deletion - first insert some nodes to delete
    batch_delete_nodes = [
        VectorNode(
            vector_id=f"delete_test_{i}",
            content=f"Node to be deleted {i}",
            metadata={"test_type": "delete_batch"},
        )
        for i in range(5)
    ]
    await store.insert(batch_delete_nodes)
    logger.info(f"✓ Inserted {len(batch_delete_nodes)} nodes for batch delete test")

    # Batch delete
    delete_ids = [f"delete_test_{i}" for i in range(5)]
    await store.delete(delete_ids)
    logger.info(f"✓ Batch deleted {len(delete_ids)} nodes")

    # Verify batch deletion
    try:
        results = await store.get(delete_ids)
        if isinstance(results, list):
            assert len(results) == 0, "All deleted nodes should not be retrievable"
    except Exception:
        pass  # Expected if nodes don't exist
    logger.info("✓ Verified batch deletion")


async def test_copy_collection(store: BaseVectorStore, store_name: str):
    """Test copying a collection."""
    logger.info("=" * 20 + " COPY COLLECTION TEST " + "=" * 20)

    config = TestConfig()
    copy_collection_name = f"{config.TEST_COLLECTION_PREFIX}_{store_name}_copy"

    # Elasticsearch and PostgreSQL require lowercase table/index names
    store_type = get_store_type(store)
    if store_type in ("es", "pgvector"):
        copy_collection_name = copy_collection_name.lower()

    # Clean up if exists
    collections = await store.list_collections()
    if copy_collection_name in collections:
        await store.delete_collection(copy_collection_name)

    # Copy collection
    await store.copy_collection(copy_collection_name)

    # Verify copy
    collections = await store.list_collections()
    assert copy_collection_name in collections, "Copied collection should exist"
    logger.info(f"✓ Copied collection to: {copy_collection_name}")

    # Verify content in copied collection
    copied_store = create_vector_store(store_type, copy_collection_name)
    copied_nodes = await copied_store.list()
    logger.info(f"✓ Copied collection has {len(copied_nodes)} nodes")
    await copied_store.close()

    # Clean up copied collection
    await store.delete_collection(copy_collection_name)
    logger.info("✓ Cleaned up copied collection")


async def test_list_collections(store: BaseVectorStore, _store_name: str):
    """Test listing all collections."""
    logger.info("=" * 20 + " LIST COLLECTIONS TEST " + "=" * 20)

    collections = await store.list_collections()

    logger.info(f"Found {len(collections)} collections")
    config = TestConfig()
    test_collections = [c for c in collections if c.startswith(config.TEST_COLLECTION_PREFIX)]
    logger.info(f"  Test collections: {test_collections}")

    assert store.collection_name in collections, "Main test collection should be listed"
    logger.info("✓ List collections test passed")


async def test_delete_collection(store: BaseVectorStore, _store_name: str):
    """Test deleting a collection."""
    logger.info("=" * 20 + " DELETE COLLECTION TEST " + "=" * 20)

    await store.delete_collection(store.collection_name)

    # Verify deletion
    collections = await store.list_collections()
    assert store.collection_name not in collections, "Collection should not exist after deletion"
    logger.info(f"✓ Deleted collection: {store.collection_name}")


async def test_cosine_similarity(store_name: str):
    """Test manual cosine similarity calculation (LocalVectorStore only)."""
    if store_name != "LocalVectorStore":
        logger.info("=" * 20 + " COSINE SIMILARITY TEST (SKIPPED) " + "=" * 20)
        logger.info("⊘ Skipped: Only applicable to LocalVectorStore")
        return

    logger.info("=" * 20 + " COSINE SIMILARITY TEST " + "=" * 20)

    vec1 = [1.0, 0.0, 0.0]
    vec2 = [0.0, 1.0, 0.0]
    vec3 = [1.0, 0.0, 0.0]

    # Test perpendicular vectors (similarity = 0)
    sim1 = LocalVectorStore._cosine_similarity(vec1, vec2)  # pylint: disable=protected-access
    logger.info(f"Similarity between perpendicular vectors: {sim1:.4f}")
    assert abs(sim1) < 0.0001, "Perpendicular vectors should have similarity close to 0"

    # Test identical vectors (similarity = 1)
    sim2 = LocalVectorStore._cosine_similarity(vec1, vec3)  # pylint: disable=protected-access
    logger.info(f"Similarity between identical vectors: {sim2:.4f}")
    assert abs(sim2 - 1.0) < 0.0001, "Identical vectors should have similarity close to 1"

    # Test with real-world like vectors
    vec4 = [0.5, 0.5, 0.5]
    vec5 = [0.6, 0.4, 0.5]
    sim3 = LocalVectorStore._cosine_similarity(vec4, vec5)  # pylint: disable=protected-access
    logger.info(f"Similarity between similar vectors: {sim3:.4f}")
    assert sim3 > 0.9, "Similar vectors should have high similarity"

    logger.info("✓ Cosine similarity tests passed")


async def test_batch_operations(store: BaseVectorStore, _store_name: str):
    """Test large-scale batch insert, update, and delete operations.

    This test validates the efficiency and correctness of batch operations
    by processing 100 nodes at once, which is more realistic for production use.
    """
    logger.info("=" * 20 + " BATCH OPERATIONS TEST " + "=" * 20)

    # Create a large batch of nodes (100 nodes)
    batch_nodes = []
    for i in range(100):
        batch_nodes.append(
            VectorNode(
                vector_id=f"batch_node_{i}",
                content=f"This is batch test content number {i} about various topics in technology and science.",
                metadata={
                    "batch_id": str(i // 10),  # Group into batches of 10
                    "index": str(i),
                    "category": ["tech", "science", "business"][i % 3],
                    "priority": ["high", "medium", "low"][i % 3],
                },
            ),
        )

    # Batch insert
    await store.insert(batch_nodes)
    logger.info(f"✓ Inserted {len(batch_nodes)} nodes in batch")

    # Verify batch insert
    results = await store.list(limit=150)
    assert len(results) >= 100, f"Should have at least 100 nodes, got {len(results)}"
    logger.info(f"✓ Verified batch insert: {len(results)} total nodes")

    # Batch update (update first 20 nodes)
    update_nodes = []
    for i in range(20):
        update_nodes.append(
            VectorNode(
                vector_id=f"batch_node_{i}",
                content=f"UPDATED: This is updated batch content {i}",
                metadata={
                    "batch_id": str(i // 10),
                    "index": str(i),
                    "updated": "true",
                    "update_timestamp": "2024-12-31",
                },
            ),
        )

    await store.update(update_nodes)
    logger.info(f"✓ Updated {len(update_nodes)} nodes in batch")

    # Verify updates
    updated_results = await store.list(filters={"updated": "true"}, limit=50)
    assert len(updated_results) >= 20, "Should have at least 20 updated nodes"
    logger.info(f"✓ Verified batch update: {len(updated_results)} updated nodes")

    # Batch delete (delete nodes with batch_id >= 5)
    delete_ids = [f"batch_node_{i}" for i in range(50, 100)]
    await store.delete(delete_ids)
    logger.info(f"✓ Deleted {len(delete_ids)} nodes in batch")

    # Verify deletions
    remaining = await store.list(limit=150)
    batch_nodes_remaining = [n for n in remaining if n.vector_id.startswith("batch_node_")]
    assert len(batch_nodes_remaining) <= 50, "Should have at most 50 batch nodes remaining"
    logger.info(f"✓ Verified batch delete: {len(batch_nodes_remaining)} nodes remaining")


async def test_complex_metadata_queries(store: BaseVectorStore, _store_name: str):
    """Test complex metadata filtering with nested conditions."""
    logger.info("=" * 20 + " COMPLEX METADATA QUERIES TEST " + "=" * 20)

    # Insert nodes with rich metadata
    complex_nodes = [
        VectorNode(
            vector_id="complex_1",
            content="Advanced neural networks for computer vision applications",
            metadata={
                "domain": "AI",
                "subdomain": "computer_vision",
                "year": "2024",
                "citations": "150",
                "impact_factor": "high",
                "tags": "neural_networks,vision,deep_learning",
            },
        ),
        VectorNode(
            vector_id="complex_2",
            content="Natural language processing with transformer models",
            metadata={
                "domain": "AI",
                "subdomain": "nlp",
                "year": "2023",
                "citations": "200",
                "impact_factor": "high",
                "tags": "transformers,nlp,language_models",
            },
        ),
        VectorNode(
            vector_id="complex_3",
            content="Reinforcement learning for robotics control",
            metadata={
                "domain": "AI",
                "subdomain": "robotics",
                "year": "2024",
                "citations": "80",
                "impact_factor": "medium",
                "tags": "reinforcement_learning,robotics,control",
            },
        ),
        VectorNode(
            vector_id="complex_4",
            content="Quantum computing algorithms and applications",
            metadata={
                "domain": "quantum",
                "subdomain": "algorithms",
                "year": "2024",
                "citations": "50",
                "impact_factor": "medium",
                "tags": "quantum,algorithms,computing",
            },
        ),
    ]

    await store.insert(complex_nodes)
    logger.info(f"✓ Inserted {len(complex_nodes)} nodes with complex metadata")

    # Test 1: Multiple field filters with list values
    filters_1 = {
        "domain": "AI",
        "year": ["2023", "2024"],
        "impact_factor": "high",
    }
    results_1 = await store.search(
        query="artificial intelligence research",
        limit=10,
        filters=filters_1,
    )
    logger.info(f"Test 1 - AI + high impact + recent years: {len(results_1)} results")
    for r in results_1:
        assert r.metadata.get("domain") == "AI"
        assert r.metadata.get("impact_factor") == "high"
        assert r.metadata.get("year") in ["2023", "2024"]

    # Test 2: List filter with multiple subdomains
    filters_2 = {
        "subdomain": ["nlp", "computer_vision"],
    }
    results_2 = await store.search(
        query="deep learning applications",
        limit=10,
        filters=filters_2,
    )
    logger.info(f"Test 2 - NLP or Computer Vision: {len(results_2)} results")
    for r in results_2:
        assert r.metadata.get("subdomain") in ["nlp", "computer_vision"]

    # Test 3: Year-based filtering
    filters_3 = {
        "year": "2024",
    }
    results_3 = await store.list(filters=filters_3, limit=10)
    logger.info(f"Test 3 - Year 2024 only: {len(results_3)} results")
    for r in results_3:
        assert r.metadata.get("year") == "2024"

    logger.info("✓ Complex metadata queries test passed")


async def test_edge_cases(store: BaseVectorStore, _store_name: str):
    """Test edge cases and boundary conditions."""
    logger.info("=" * 20 + " EDGE CASES TEST " + "=" * 20)

    # Test 1: Empty content
    edge_node_1 = VectorNode(
        vector_id="edge_empty_content",
        content="",
        metadata={"type": "empty"},
    )
    try:
        await store.insert([edge_node_1])
        logger.info("✓ Handled empty content node")
    except Exception as e:
        logger.info(f"⊘ Empty content not supported: {e}")

    # Test 2: Very long content
    edge_node_2 = VectorNode(
        vector_id="edge_long_content",
        content="A" * 10000,  # 10k characters
        metadata={"type": "long_content"},
    )
    await store.insert([edge_node_2])
    result = await store.get("edge_long_content")
    assert len(result.content) == 10000
    logger.info("✓ Handled very long content (10k chars)")

    # Test 3: Special characters in content
    edge_node_3 = VectorNode(
        vector_id="edge_special_chars",
        content="Special chars: @#$%^&*()[]{}|\\;:'\",.<>?/~`+=−×÷",
        metadata={"type": "special_chars"},
    )
    await store.insert([edge_node_3])
    result = await store.get("edge_special_chars")
    assert "@#$%^&*()" in result.content
    logger.info("✓ Handled special characters in content")

    # Test 4: Unicode and emoji content
    edge_node_4 = VectorNode(
        vector_id="edge_unicode",
        content="Unicode test: 你好世界 🌍 مرحبا العالم Привет мир",
        metadata={"type": "unicode", "language": "multi"},
    )
    await store.insert([edge_node_4])
    result = await store.get("edge_unicode")
    assert "你好世界" in result.content
    assert "🌍" in result.content
    logger.info("✓ Handled unicode and emoji content")

    # Test 5: Search with empty query
    try:
        results = await store.search(query="", limit=5)
        logger.info(f"✓ Empty query returned {len(results)} results")
    except Exception as e:
        logger.info(f"⊘ Empty query not supported: {e}")

    # Test 6: Search with very high limit
    results = await store.search(query="test", limit=1000)
    logger.info(f"✓ High limit search returned {len(results)} results")

    # Test 7: Get non-existent ID
    result = await store.get("non_existent_id_12345")
    if isinstance(result, list):
        assert len(result) == 0, "Non-existent ID should return empty list"
    else:
        assert result is None, "Non-existent ID should return None"
    logger.info("✓ Handled non-existent ID gracefully")

    # Test 8: Metadata with empty string values
    edge_node_5 = VectorNode(
        vector_id="edge_empty_metadata",
        content="Testing empty string values in metadata",
        metadata={"field1": "value1", "field2": "", "field3": "value3"},
    )
    await store.insert([edge_node_5])
    logger.info("✓ Handled empty string values in metadata")

    logger.info("✓ Edge cases test passed")


async def test_concurrent_operations(store: BaseVectorStore, _store_name: str):
    """Test concurrent read/write operations."""
    logger.info("=" * 20 + " CONCURRENT OPERATIONS TEST " + "=" * 20)

    # Prepare concurrent insert nodes
    concurrent_nodes = [
        VectorNode(
            vector_id=f"concurrent_{i}",
            content=f"Concurrent test content {i}",
            metadata={"thread_id": str(i % 5), "index": str(i)},
        )
        for i in range(50)
    ]

    # Test concurrent inserts
    insert_tasks = []
    for i in range(0, 50, 10):
        batch = concurrent_nodes[i : i + 10]
        insert_tasks.append(store.insert(batch))

    await asyncio.gather(*insert_tasks)
    logger.info("✓ Completed concurrent inserts")

    # Test concurrent searches
    search_tasks = [store.search(query=f"concurrent test {i}", limit=5) for i in range(10)]
    search_results = await asyncio.gather(*search_tasks)
    logger.info(f"✓ Completed {len(search_results)} concurrent searches")

    # Test concurrent reads
    get_tasks = [store.get(f"concurrent_{i}") for i in range(0, 50, 5)]
    get_results = await asyncio.gather(*get_tasks)
    logger.info(f"✓ Completed {len(get_results)} concurrent reads")

    # Test batch updates (using batch update instead of concurrent individual updates)
    update_nodes = [
        VectorNode(
            vector_id=f"concurrent_{i}",
            content=f"UPDATED concurrent content {i}",
            metadata={"thread_id": str(i % 5), "updated": "true"},
        )
        for i in range(0, 20, 2)
    ]
    await store.update(update_nodes)
    logger.info(f"✓ Completed batch update of {len(update_nodes)} nodes")

    logger.info("✓ Concurrent operations test passed")


async def test_search_relevance_ranking(store: BaseVectorStore, _store_name: str):
    """Test search result relevance and ranking."""
    logger.info("=" * 20 + " SEARCH RELEVANCE RANKING TEST " + "=" * 20)

    # Insert nodes with varying relevance
    relevance_nodes = [
        VectorNode(
            vector_id="relevance_exact",
            content="Machine learning is a subset of artificial intelligence focused on learning from data.",
            metadata={"relevance": "exact"},
        ),
        VectorNode(
            vector_id="relevance_high",
            content="Artificial intelligence and machine learning are transforming technology.",
            metadata={"relevance": "high"},
        ),
        VectorNode(
            vector_id="relevance_medium",
            content="Deep learning uses neural networks for pattern recognition.",
            metadata={"relevance": "medium"},
        ),
        VectorNode(
            vector_id="relevance_low",
            content="Software engineering best practices for code quality.",
            metadata={"relevance": "low"},
        ),
        VectorNode(
            vector_id="relevance_none",
            content="Cooking recipes for delicious Italian pasta dishes.",
            metadata={"relevance": "none"},
        ),
    ]

    await store.insert(relevance_nodes)
    logger.info(f"✓ Inserted {len(relevance_nodes)} nodes with varying relevance")

    # Search with specific query
    query = "What is machine learning and artificial intelligence?"
    results = await store.search(query=query, limit=5)

    logger.info(f"Search results for: '{query}'")
    for i, result in enumerate(results, 1):
        score = result.metadata.get("_score", "N/A")
        relevance = result.metadata.get("relevance", "unknown")
        logger.info(f"  {i}. [{relevance}] score={score}: {result.content[:60]}...")

    # Verify that more relevant results appear first
    if len(results) >= 2:
        # The exact match should be in top results
        top_ids = [r.vector_id for r in results[:3]]
        assert (
            "relevance_exact" in top_ids or "relevance_high" in top_ids
        ), "Most relevant results should appear in top 3"
        logger.info("✓ Relevance ranking verified")

    # Test with different query
    query2 = "neural networks deep learning"
    results2 = await store.search(query=query2, limit=5)
    logger.info(f"\nSearch results for: '{query2}'")
    for i, result in enumerate(results2, 1):
        score = result.metadata.get("_score", "N/A")
        logger.info(f"  {i}. score={score}: {result.content[:60]}...")

    logger.info("✓ Search relevance ranking test passed")


async def test_metadata_statistics(store: BaseVectorStore, _store_name: str):
    """Test metadata aggregation and statistics."""
    logger.info("=" * 20 + " METADATA STATISTICS TEST " + "=" * 20)

    # Get all nodes and analyze metadata
    all_nodes = await store.list(limit=500)
    logger.info(f"Total nodes in collection: {len(all_nodes)}")

    # Count by category
    category_counts = {}
    for node in all_nodes:
        category = node.metadata.get("category", "unknown")
        category_counts[category] = category_counts.get(category, 0) + 1

    logger.info("Category distribution:")
    for category, count in sorted(category_counts.items()):
        logger.info(f"  {category}: {count}")

    # Count by node_type
    type_counts = {}
    for node in all_nodes:
        node_type = node.metadata.get("node_type", "unknown")
        type_counts[node_type] = type_counts.get(node_type, 0) + 1

    logger.info("Node type distribution:")
    for node_type, count in sorted(type_counts.items()):
        logger.info(f"  {node_type}: {count}")

    # Verify we can filter by each category
    for category in category_counts:
        if category != "unknown":
            filtered = await store.list(filters={"category": category}, limit=100)
            logger.info(f"✓ Filter by category '{category}': {len(filtered)} results")

    logger.info("✓ Metadata statistics test passed")


async def test_update_metadata_only(store: BaseVectorStore, _store_name: str):
    """Test updating only metadata without changing content."""
    logger.info("=" * 20 + " UPDATE METADATA ONLY TEST " + "=" * 20)

    # Get an existing node
    original = await store.get("test_node1")
    original_content = original.content

    # Update with same content but different metadata
    updated_node = VectorNode(
        vector_id="test_node1",
        content=original_content,  # Keep same content
        metadata={
            **original.metadata,
            "metadata_updated": "true",
            "update_count": "1",
            "last_modified": "2024-12-31",
        },
    )

    await store.update(updated_node)
    logger.info("✓ Updated metadata without changing content")

    # Verify update
    result = await store.get("test_node1")
    assert result.content == original_content, "Content should remain unchanged"
    assert result.metadata.get("metadata_updated") == "true", "Metadata should be updated"
    logger.info("✓ Verified metadata-only update")

    # Update metadata again
    updated_node_2 = VectorNode(
        vector_id="test_node1",
        content=original_content,
        metadata={
            **result.metadata,
            "update_count": "2",
            "last_modified": "2024-12-31T12:00:00",
        },
    )
    await store.update(updated_node_2)

    result_2 = await store.get("test_node1")
    assert result_2.metadata.get("update_count") == "2", "Metadata should be updated again"
    logger.info("✓ Multiple metadata updates successful")

    logger.info("✓ Update metadata only test passed")


async def test_filter_combinations(store: BaseVectorStore, _store_name: str):
    """Test various filter combinations and edge cases."""
    logger.info("=" * 20 + " FILTER COMBINATIONS TEST " + "=" * 20)

    # Test 1: Empty filter (should return all results)
    results_1 = await store.search(query="technology", filters={}, limit=10)
    logger.info(f"Test 1 - Empty filter: {len(results_1)} results")

    # Test 2: Single value filter
    results_2 = await store.search(
        query="technology",
        filters={"node_type": "tech"},
        limit=10,
    )
    logger.info(f"Test 2 - Single value filter: {len(results_2)} results")
    for r in results_2:
        assert r.metadata.get("node_type") == "tech"

    # Test 3: List filter with single item
    results_3 = await store.search(
        query="technology",
        filters={"node_type": ["tech"]},
        limit=10,
    )
    logger.info(f"Test 3 - List filter (single item): {len(results_3)} results")

    # Test 4: List filter with multiple items
    results_4 = await store.search(
        query="technology",
        filters={"category": ["AI", "ML", "DL"]},
        limit=10,
    )
    logger.info(f"Test 4 - List filter (multiple items): {len(results_4)} results")
    for r in results_4:
        assert r.metadata.get("category") in ["AI", "ML", "DL"]

    # Test 5: Multiple filters (AND operation)
    results_5 = await store.search(
        query="technology",
        filters={
            "node_type": ["tech", "tech_new"],
            "source": "research",
            "priority": "high",
        },
        limit=10,
    )
    logger.info(f"Test 5 - Multiple filters (AND): {len(results_5)} results")
    for r in results_5:
        assert r.metadata.get("node_type") in ["tech", "tech_new"]
        assert r.metadata.get("source") == "research"
        assert r.metadata.get("priority") == "high"

    # Test 6: Filter with non-existent value
    results_6 = await store.search(
        query="technology",
        filters={"category": "NON_EXISTENT_CATEGORY"},
        limit=10,
    )
    logger.info(f"Test 6 - Non-existent filter value: {len(results_6)} results")
    assert len(results_6) == 0, "Should return no results for non-existent filter value"

    # Test 7: List operation with filters
    list_results = await store.list(
        filters={"node_type": "tech", "priority": "high"},
        limit=20,
    )
    logger.info(f"Test 7 - List with filters: {len(list_results)} results")
    for r in list_results:
        assert r.metadata.get("node_type") == "tech"
        assert r.metadata.get("priority") == "high"

    logger.info("✓ Filter combinations test passed")


# ==================== Test Runner ====================


async def run_all_tests_for_store(store_type: str, store_name: str):
    """Run all tests for a specific vector store type.

    Args:
        store_type: Type of vector store ("local" or "es")
        store_name: Display name for the vector store
    """
    logger.info(f"\n\n{'#' * 60}")
    logger.info(f"# Running all tests for: {store_name}")
    logger.info(f"{'#' * 60}")

    config = TestConfig()
    collection_name = f"{config.TEST_COLLECTION_PREFIX}_{store_type}_main"

    # Create vector store instance
    store = create_vector_store(store_type, collection_name)

    try:
        # Run cosine similarity test first (only for LocalVectorStore)
        await test_cosine_similarity(store_name)

        # ========== Basic Tests ==========
        logger.info(f"\n{'#' * 60}")
        logger.info("# BASIC FUNCTIONALITY TESTS")
        logger.info(f"{'#' * 60}")

        await test_create_collection(store, store_name)
        await test_insert(store, store_name)
        await test_search(store, store_name)
        await test_search_with_single_filter(store, store_name)
        await test_search_with_list_filter(store, store_name)
        await test_search_with_multiple_filters(store, store_name)
        await test_get_by_id(store, store_name)
        await test_list_all(store, store_name)
        await test_list_with_filters(store, store_name)
        await test_update(store, store_name)
        await test_delete(store, store_name)

        # ========== Advanced Tests ==========
        logger.info(f"\n{'#' * 60}")
        logger.info("# ADVANCED FUNCTIONALITY TESTS")
        logger.info(f"{'#' * 60}")

        await test_batch_operations(store, store_name)
        await test_complex_metadata_queries(store, store_name)
        await test_edge_cases(store, store_name)
        await test_concurrent_operations(store, store_name)
        await test_search_relevance_ranking(store, store_name)
        await test_metadata_statistics(store, store_name)
        await test_update_metadata_only(store, store_name)
        await test_filter_combinations(store, store_name)

        # ========== Collection Management Tests ==========
        logger.info(f"\n{'#' * 60}")
        logger.info("# COLLECTION MANAGEMENT TESTS")
        logger.info(f"{'#' * 60}")

        await test_list_collections(store, store_name)
        await test_copy_collection(store, store_name)
        await test_delete_collection(store, store_name)

        logger.info(f"\n{'=' * 60}")
        logger.info(f"✓ All tests passed for {store_name}!")
        logger.info(f"{'=' * 60}")

    except Exception as e:
        logger.error(f"Test failed: {e}")
        raise
    finally:
        # Cleanup
        await cleanup_store(store, store_type)


async def cleanup_store(store: BaseVectorStore, store_type: str):
    """Clean up test resources for a vector store.

    Args:
        store: Vector store instance
        store_type: Type of vector store ("local" or "es")
    """
    logger.info("=" * 20 + " CLEANUP " + "=" * 20)

    try:
        # Clean up test collections
        config = TestConfig()
        collections = await store.list_collections()
        test_collections = [c for c in collections if c.startswith(config.TEST_COLLECTION_PREFIX)]

        for collection in test_collections:
            try:
                await store.delete_collection(collection)
                logger.info(f"Deleted test collection: {collection}")
            except Exception as e:
                logger.warning(f"Failed to delete collection {collection}: {e}")

        # Close connections
        await store.close()

        # Clean up local directory if LocalVectorStore
        if store_type == "local":
            test_dir = Path(config.LOCAL_ROOT_PATH)
            if test_dir.exists():
                shutil.rmtree(test_dir)
                logger.info(f"Cleaned up local directory: {config.LOCAL_ROOT_PATH}")

        # Clean up local directory if ChromaVectorStore
        if store_type == "chroma" and config.CHROMA_PATH:
            test_dir = Path(config.CHROMA_PATH)
            if test_dir.exists():
                shutil.rmtree(test_dir)
                logger.info(f"Cleaned up chroma directory: {config.CHROMA_PATH}")

        logger.info("✓ Cleanup completed")
    except Exception as e:
        logger.error(f"Cleanup error: {e}")


# ==================== Main Entry Point ====================


async def main():
    """Main entry point for running tests."""
    parser = argparse.ArgumentParser(
        description="Run vector store tests",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python test_vector_store.py --local      # Test LocalVectorStore only
  python test_vector_store.py --es         # Test ESVectorStore only
  python test_vector_store.py --pgvector   # Test PGVectorStore only
  python test_vector_store.py --qdrant     # Test QdrantVectorStore only
  python test_vector_store.py --chroma     # Test ChromaVectorStore only
  python test_vector_store.py --all        # Test all vector stores
        """,
    )
    parser.add_argument(
        "--local",
        action="store_true",
        help="Test LocalVectorStore",
    )
    parser.add_argument(
        "--es",
        action="store_true",
        help="Test ESVectorStore",
    )
    parser.add_argument(
        "--qdrant",
        action="store_true",
        help="Test QdrantVectorStore",
    )
    parser.add_argument(
        "--pgvector",
        action="store_true",
        help="Test PGVectorStore",
    )
    parser.add_argument(
        "--chroma",
        action="store_true",
        help="Test ChromaVectorStore",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run tests for all available vector stores",
    )

    args = parser.parse_args()

    # Determine which vector stores to test
    stores_to_test = []

    if args.all:
        stores_to_test = [
            ("local", "LocalVectorStore"),
            ("es", "ESVectorStore"),
            ("pgvector", "PGVectorStore"),
            ("qdrant", "QdrantVectorStore"),
            ("chroma", "ChromaVectorStore"),
        ]
    else:
        # Build list based on individual flags
        if args.local:
            stores_to_test.append(("local", "LocalVectorStore"))
        if args.es:
            stores_to_test.append(("es", "ESVectorStore"))
        if args.pgvector:
            stores_to_test.append(("pgvector", "PGVectorStore"))
        if args.qdrant:
            stores_to_test.append(("qdrant", "QdrantVectorStore"))
        if args.chroma:
            stores_to_test.append(("chroma", "ChromaVectorStore"))

        if not stores_to_test:
            # Default to all vector stores if no argument provided
            stores_to_test = [
                ("local", "LocalVectorStore"),
                ("es", "ESVectorStore"),
                ("pgvector", "PGVectorStore"),
                ("qdrant", "QdrantVectorStore"),
                ("chroma", "ChromaVectorStore"),
            ]
            print("No vector store specified, defaulting to test all vector stores")
            print(
                "Use --local/--es/--pgvector/--qdrant/--chroma to test specific ones\n",
            )

    # Run tests for each vector store
    for store_type, store_name in stores_to_test:
        try:
            await run_all_tests_for_store(store_type, store_name)
        except Exception as e:
            logger.error(f"\n✗ FAILED: {store_name} tests failed with error:")
            logger.error(f"  {type(e).__name__}: {e}")
            raise

    # Final summary
    print(f"\n\n{'#' * 60}")
    print("# TEST SUMMARY")
    print(f"{'#' * 60}")
    print(f"✓ All tests passed for {len(stores_to_test)} vector store(s):")
    for _, store_name in stores_to_test:
        print(f"  - {store_name}")
    print(f"{'#' * 60}\n")


if __name__ == "__main__":
    asyncio.run(main())
