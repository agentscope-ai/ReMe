"""Unified test suite for vector store implementations.

This module provides comprehensive test coverage for LocalVectorStore, ESVectorStore,
and QdrantVectorStore implementations. Tests can be run for specific vector stores or all implementations.

Usage:
    python test_vector_store.py --local      # Test LocalVectorStore only
    python test_vector_store.py --es         # Test ESVectorStore only
    python test_vector_store.py --qdrant     # Test QdrantVectorStore only
    python test_vector_store.py --all        # Test all vector stores

"""

import asyncio
import argparse
import shutil
from pathlib import Path
from typing import Type, List

from loguru import logger

from reme_ai.core.embedding import OpenAIEmbeddingModel
from reme_ai.core.schema import VectorNode
from reme_ai.core.vector_store import (
    BaseVectorStore,
    LocalVectorStore,
    ESVectorStore,
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
        str: Type identifier ("local", "es", or "qdrant")
    """
    if isinstance(store, LocalVectorStore):
        return "local"
    elif isinstance(store, QdrantVectorStore):
        return "qdrant"
    elif isinstance(store, ESVectorStore):
        return "es"
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
    else:
        raise ValueError(f"Unknown store type: {store_type}")


# ==================== Test Functions ====================


async def test_create_collection(store: BaseVectorStore, store_name: str):
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


async def test_insert(store: BaseVectorStore, store_name: str) -> List[VectorNode]:
    """Test node insertion."""
    logger.info("=" * 20 + " INSERT TEST " + "=" * 20)
    
    sample_nodes = SampleDataGenerator.create_sample_nodes("test")
    await store.insert(sample_nodes)
    
    logger.info(f"✓ Inserted {len(sample_nodes)} nodes")
    return sample_nodes


async def test_search(store: BaseVectorStore, store_name: str):
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


async def test_search_with_single_filter(store: BaseVectorStore, store_name: str):
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
        assert node_type == "tech", f"Result should have node_type='tech'"
    
    logger.info("✓ Single filter search test passed")


async def test_search_with_list_filter(store: BaseVectorStore, store_name: str):
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
        assert node_type in ["tech", "tech_new"], f"Result should have node_type in [tech, tech_new]"
    
    logger.info("✓ List filter search test passed")


async def test_search_with_multiple_filters(store: BaseVectorStore, store_name: str):
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
        f"Multi-filter search (node_type IN [tech, tech_new] AND source=research) "
        f"returned {len(results)} results"
    )
    for i, r in enumerate(results, 1):
        node_type = r.metadata.get("node_type")
        source = r.metadata.get("source")
        logger.info(f"  Result {i}: type={node_type}, source={source}, content={r.content[:40]}...")
        assert node_type in ["tech", "tech_new"], f"Result should have node_type in [tech, tech_new]"
        assert source == "research", f"Result should have source='research'"
    
    logger.info("✓ Multiple filters search test passed")


async def test_get_by_id(store: BaseVectorStore, store_name: str):
    """Test retrieving nodes by vector_id."""
    logger.info("=" * 20 + " GET BY ID TEST " + "=" * 20)
    
    # Test single ID retrieval
    target_id = "test_node1"
    result = await store.get(target_id)
    
    assert isinstance(result, VectorNode), "Should return a VectorNode for single ID"
    assert result.vector_id == target_id, f"Result should have vector_id={target_id}"
    logger.info(f"✓ Retrieved single node: {result.vector_id}")
    
    # Test multiple IDs retrieval
    target_ids = ["test_node1", "test_node2"]
    results = await store.get(target_ids)
    
    assert isinstance(results, list), "Should return a list for multiple IDs"
    assert len(results) == 2, f"Should return 2 results, got {len(results)}"
    result_ids = {r.vector_id for r in results}
    assert result_ids == set(target_ids), f"Result IDs should match {target_ids}"
    logger.info(f"✓ Retrieved {len(results)} nodes by IDs")


async def test_list_all(store: BaseVectorStore, store_name: str):
    """Test listing all nodes in collection."""
    logger.info("=" * 20 + " LIST ALL TEST " + "=" * 20)
    
    results = await store.list(limit=10)
    
    logger.info(f"Collection contains {len(results)} nodes")
    for i, node in enumerate(results, 1):
        logger.info(f"  Node {i}: id={node.vector_id}, content={node.content[:50]}...")
    
    assert len(results) > 0, "Collection should contain nodes"
    logger.info("✓ List all nodes test passed")


async def test_list_with_filters(store: BaseVectorStore, store_name: str):
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


async def test_update(store: BaseVectorStore, store_name: str):
    """Test updating existing nodes."""
    logger.info("=" * 20 + " UPDATE TEST " + "=" * 20)
    
    # Update node content and metadata
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
    
    # Verify update
    result = await store.get("test_node2")
    assert "updated" in result.metadata, "Updated metadata should be present"
    logger.info(f"✓ Updated node: {result.vector_id}")
    logger.info(f"  New content: {result.content[:60]}...")


async def test_delete(store: BaseVectorStore, store_name: str):
    """Test deleting nodes."""
    logger.info("=" * 20 + " DELETE TEST " + "=" * 20)
    
    node_to_delete = "test_node4"
    await store.delete(node_to_delete)
    
    # Verify deletion - try to get the deleted node
    try:
        result = await store.get(node_to_delete)
        # If result is empty list or None, deletion was successful
        if isinstance(result, list):
            assert len(result) == 0, "Deleted node should not be retrievable"
        else:
            assert result is None, "Deleted node should not be retrievable"
    except Exception:
        pass  # Expected if node doesn't exist
    
    logger.info(f"✓ Deleted node: {node_to_delete}")


async def test_copy_collection(store: BaseVectorStore, store_name: str):
    """Test copying a collection."""
    logger.info("=" * 20 + " COPY COLLECTION TEST " + "=" * 20)
    
    config = TestConfig()
    copy_collection_name = f"{config.TEST_COLLECTION_PREFIX}_{store_name}_copy"
    
    # Elasticsearch requires lowercase index names
    store_type = get_store_type(store)
    if store_type == "es":
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
    logger.info(f"✓ Cleaned up copied collection")


async def test_list_collections(store: BaseVectorStore, store_name: str):
    """Test listing all collections."""
    logger.info("=" * 20 + " LIST COLLECTIONS TEST " + "=" * 20)
    
    collections = await store.list_collections()
    
    logger.info(f"Found {len(collections)} collections")
    config = TestConfig()
    test_collections = [c for c in collections if c.startswith(config.TEST_COLLECTION_PREFIX)]
    logger.info(f"  Test collections: {test_collections}")
    
    assert store.collection_name in collections, "Main test collection should be listed"
    logger.info("✓ List collections test passed")


async def test_delete_collection(store: BaseVectorStore, store_name: str):
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
    sim1 = LocalVectorStore._cosine_similarity(vec1, vec2)
    logger.info(f"Similarity between perpendicular vectors: {sim1:.4f}")
    assert abs(sim1) < 0.0001, "Perpendicular vectors should have similarity close to 0"
    
    # Test identical vectors (similarity = 1)
    sim2 = LocalVectorStore._cosine_similarity(vec1, vec3)
    logger.info(f"Similarity between identical vectors: {sim2:.4f}")
    assert abs(sim2 - 1.0) < 0.0001, "Identical vectors should have similarity close to 1"
    
    # Test with real-world like vectors
    vec4 = [0.5, 0.5, 0.5]
    vec5 = [0.6, 0.4, 0.5]
    sim3 = LocalVectorStore._cosine_similarity(vec4, vec5)
    logger.info(f"Similarity between similar vectors: {sim3:.4f}")
    assert sim3 > 0.9, "Similar vectors should have high similarity"
    
    logger.info("✓ Cosine similarity tests passed")


# ==================== Test Runner ====================


async def run_all_tests_for_store(store_type: str, store_name: str):
    """Run all tests for a specific vector store type.
    
    Args:
        store_type: Type of vector store ("local" or "es")
        store_name: Display name for the vector store
    """
    logger.info(f"\n\n{'#'*60}")
    logger.info(f"# Running all tests for: {store_name}")
    logger.info(f"{'#'*60}")
    
    config = TestConfig()
    collection_name = f"{config.TEST_COLLECTION_PREFIX}_{store_type}_main"
    
    # Create vector store instance
    store = create_vector_store(store_type, collection_name)
    
    try:
        # Run cosine similarity test first (only for LocalVectorStore)
        await test_cosine_similarity(store_name)
        
        # Run all other tests
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
        await test_list_collections(store, store_name)
        await test_copy_collection(store, store_name)
        await test_delete_collection(store, store_name)
        
        logger.info(f"\n{'='*60}")
        logger.info(f"✓ All tests passed for {store_name}!")
        logger.info(f"{'='*60}")
        
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
        test_collections = [
            c for c in collections 
            if c.startswith(config.TEST_COLLECTION_PREFIX)
        ]
        
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
  python test_vector_store.py --all        # Test both vector stores
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
            ("qdrant", "QdrantVectorStore"),
        ]
    elif args.local and args.es and args.qdrant:
        stores_to_test = [
            ("local", "LocalVectorStore"),
            ("es", "ESVectorStore"),
            ("qdrant", "QdrantVectorStore"),
        ]
    elif args.local and args.es:
        stores_to_test = [
            ("local", "LocalVectorStore"),
            ("es", "ESVectorStore"),
        ]
    elif args.local and args.qdrant:
        stores_to_test = [
            ("local", "LocalVectorStore"),
            ("qdrant", "QdrantVectorStore"),
        ]
    elif args.es and args.qdrant:
        stores_to_test = [
            ("es", "ESVectorStore"),
            ("qdrant", "QdrantVectorStore"),
        ]
    elif args.local:
        stores_to_test = [("local", "LocalVectorStore")]
    elif args.es:
        stores_to_test = [("es", "ESVectorStore")]
    elif args.qdrant:
        stores_to_test = [("qdrant", "QdrantVectorStore")]
    else:
        # Default to LocalVectorStore if no argument provided
        stores_to_test = [("local", "LocalVectorStore")]
        print("No vector store specified, defaulting to LocalVectorStore")
        print("Use --all to test all vector stores, or --local/--es/--qdrant to test a specific one\n")

    # Run tests for each vector store
    for store_type, store_name in stores_to_test:
        try:
            await run_all_tests_for_store(store_type, store_name)
        except Exception as e:
            logger.error(f"\n✗ FAILED: {store_name} tests failed with error:")
            logger.error(f"  {type(e).__name__}: {e}")
            raise

    # Final summary
    print(f"\n\n{'#'*60}")
    print(f"# TEST SUMMARY")
    print(f"{'#'*60}")
    print(f"✓ All tests passed for {len(stores_to_test)} vector store(s):")
    for _, store_name in stores_to_test:
        print(f"  - {store_name}")
    print(f"{'#'*60}\n")


if __name__ == "__main__":
    asyncio.run(main())

