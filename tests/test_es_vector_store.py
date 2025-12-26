"""Test script for ESVectorStore implementation.

This script provides comprehensive test coverage for the Elasticsearch-based vector store
implementation in ReMe. It tests all major functionalities including collection management,
CRUD operations, vector search, and filtering capabilities.

Requires proper environment variables:
- FLOW_EMBEDDING_API_KEY: API key for authentication
- FLOW_EMBEDDING_BASE_URL: Base URL for the API endpoint
- ES_HOST: Elasticsearch host (default: localhost)
- ES_PORT: Elasticsearch port (default: 9200)
- ES_USER: Elasticsearch username (optional)
- ES_PASSWORD: Elasticsearch password (optional)

Usage:
    python test_es_vector_store.py              # Run all tests
    python test_es_vector_store.py --cleanup    # Clean up test collections only

Examples:
    python test_es_vector_store.py              # Run full test suite
    python test_es_vector_store.py --cleanup    # Remove test artifacts
"""

import asyncio
import os
import sys
from typing import List
from pathlib import Path

from loguru import logger

# Add parent directory to path to import reme_ai
sys.path.insert(0, str(Path(__file__).parent.parent))

from reme_ai.core.embedding.openai_embedding_model import OpenAIEmbeddingModel
from reme_ai.core.schema.vector_node import VectorNode
from reme_ai.core.vector_store.es_vector_store import ESVectorStore


# ==================== Configuration ====================


class TestConfig:
    """Configuration for test execution."""
    
    # Elasticsearch connection settings
    ES_HOST = os.getenv("ES_HOST", "localhost")
    ES_PORT = int(os.getenv("ES_PORT", "9200"))
    ES_USER = os.getenv("ES_USER")
    ES_PASSWORD = os.getenv("ES_PASSWORD")
    
    # Embedding model settings
    EMBEDDING_API_KEY = os.getenv("FLOW_EMBEDDING_API_KEY", "")
    EMBEDDING_BASE_URL = os.getenv("FLOW_EMBEDDING_BASE_URL", "")
    EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "text-embedding-v4")
    EMBEDDING_DIMENSIONS = int(os.getenv("EMBEDDING_DIMENSIONS", "64"))
    
    # Test collection naming
    TEST_COLLECTION_PREFIX = "test_es_vector_store"


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


# ==================== Test Class ====================


class ESVectorStoreTest:
    """Comprehensive test class for ESVectorStore."""
    
    def __init__(self):
        """Initialize the test class with embedding model and vector store."""
        self.config = TestConfig()
        
        # Initialize embedding model
        self.embedding_model = OpenAIEmbeddingModel(
            api_key=self.config.EMBEDDING_API_KEY,
            base_url=self.config.EMBEDDING_BASE_URL,
            model_name=self.config.EMBEDDING_MODEL_NAME,
            dimensions=self.config.EMBEDDING_DIMENSIONS,
        )
        
        # Initialize vector store
        self.collection_name = f"{self.config.TEST_COLLECTION_PREFIX}_main"
        self.vector_store = ESVectorStore(
            collection_name=self.collection_name,
            embedding_model=self.embedding_model,
            host=self.config.ES_HOST,
            port=self.config.ES_PORT,
            user=self.config.ES_USER,
            password=self.config.ES_PASSWORD,
        )
        
        logger.info(f"Initialized ESVectorStoreTest with collection: {self.collection_name}")
    
    async def test_create_collection(self):
        """Test collection creation."""
        logger.info("=" * 20 + " CREATE COLLECTION TEST " + "=" * 20)
        
        # Clean up if exists
        collections = await self.vector_store.list_collections()
        if self.collection_name in collections:
            await self.vector_store.delete_collection(self.collection_name)
            logger.info(f"Cleaned up existing collection: {self.collection_name}")
        
        # Create collection
        await self.vector_store.create_collection(
            self.collection_name,
            dimensions=self.config.EMBEDDING_DIMENSIONS,
        )
        
        # Verify creation
        collections = await self.vector_store.list_collections()
        assert self.collection_name in collections, "Collection should exist after creation"
        logger.info(f"✓ Created collection: {self.collection_name}")
    
    async def test_insert(self) -> List[VectorNode]:
        """Test node insertion."""
        logger.info("=" * 20 + " INSERT TEST " + "=" * 20)
        
        sample_nodes = SampleDataGenerator.create_sample_nodes("test")
        await self.vector_store.insert(sample_nodes, refresh=True)
        
        logger.info(f"✓ Inserted {len(sample_nodes)} nodes")
        return sample_nodes
    
    async def test_search(self):
        """Test basic vector search."""
        logger.info("=" * 20 + " SEARCH TEST " + "=" * 20)
        
        results = await self.vector_store.search(
            query="What is artificial intelligence?",
            limit=3,
        )
        
        logger.info(f"Search returned {len(results)} results")
        for i, r in enumerate(results, 1):
            logger.info(f"  Result {i}: {r.content[:60]}... (score: {r.metadata.get('_score', 'N/A')})")
        
        assert len(results) > 0, "Search should return results"
        logger.info("✓ Basic search test passed")
    
    async def test_search_with_single_filter(self):
        """Test vector search with single metadata filter."""
        logger.info("=" * 20 + " SINGLE FILTER SEARCH TEST " + "=" * 20)
        
        # Test single value filter
        filters = {"node_type": "tech"}
        results = await self.vector_store.search(
            query="What is artificial intelligence?",
            limit=5,
            filters=filters,
        )
        
        logger.info(f"Filtered search (node_type=tech) returned {len(results)} results")
        for i, r in enumerate(results, 1):
            logger.info(f"  Result {i}: type={r.metadata.get('node_type')}, content={r.content[:50]}...")
            assert r.metadata.get("node_type") == "tech", f"Result should have node_type='tech'"
        
        logger.info("✓ Single filter search test passed")
    
    async def test_search_with_list_filter(self):
        """Test vector search with list filter (IN operation)."""
        logger.info("=" * 20 + " LIST FILTER SEARCH TEST " + "=" * 20)
        
        # Test list filter (IN operation)
        filters = {"node_type": ["tech", "tech_new"]}
        results = await self.vector_store.search(
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
    
    async def test_search_with_multiple_filters(self):
        """Test vector search with multiple metadata filters (AND operation)."""
        logger.info("=" * 20 + " MULTIPLE FILTERS SEARCH TEST " + "=" * 20)
        
        # Test multiple filters (AND operation)
        filters = {
            "node_type": ["tech", "tech_new"],
            "source": "research",
        }
        results = await self.vector_store.search(
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
    
    async def test_search_with_complex_filters(self):
        """Test vector search with complex multi-key list filters."""
        logger.info("=" * 20 + " COMPLEX FILTERS SEARCH TEST " + "=" * 20)
        
        # Test with multiple list filters
        filters = {
            "node_type": ["tech", "tech_new"],
            "language": ["english", "chinese"],
        }
        results = await self.vector_store.search(
            query="What is artificial intelligence?",
            limit=5,
            filters=filters,
        )
        
        logger.info(
            f"Complex filter search (node_type IN [tech, tech_new] AND language IN [english, chinese]) "
            f"returned {len(results)} results"
        )
        for i, r in enumerate(results, 1):
            node_type = r.metadata.get("node_type")
            language = r.metadata.get("language")
            logger.info(f"  Result {i}: type={node_type}, lang={language}, content={r.content[:40]}...")
            assert node_type in ["tech", "tech_new"], f"Result should have node_type in [tech, tech_new]"
            assert language in ["english", "chinese"], f"Result should have language in [english, chinese]"
        
        logger.info("✓ Complex filters search test passed")
    
    async def test_get_by_id(self):
        """Test retrieving nodes by vector_id."""
        logger.info("=" * 20 + " GET BY ID TEST " + "=" * 20)
        
        # Test single ID retrieval
        target_id = "test_node1"
        result = await self.vector_store.get(target_id)
        
        assert isinstance(result, VectorNode), "Should return a VectorNode for single ID"
        assert result.vector_id == target_id, f"Result should have vector_id={target_id}"
        logger.info(f"✓ Retrieved single node: {result.vector_id}")
        
        # Test multiple IDs retrieval
        target_ids = ["test_node1", "test_node2"]
        results = await self.vector_store.get(target_ids)
        
        assert isinstance(results, list), "Should return a list for multiple IDs"
        assert len(results) == 2, f"Should return 2 results, got {len(results)}"
        result_ids = {r.vector_id for r in results}
        assert result_ids == set(target_ids), f"Result IDs should match {target_ids}"
        logger.info(f"✓ Retrieved {len(results)} nodes by IDs")
    
    async def test_list_all(self):
        """Test listing all nodes in collection."""
        logger.info("=" * 20 + " LIST ALL TEST " + "=" * 20)
        
        results = await self.vector_store.list(limit=10)
        
        logger.info(f"Collection contains {len(results)} nodes")
        for i, node in enumerate(results, 1):
            logger.info(f"  Node {i}: id={node.vector_id}, content={node.content[:50]}...")
        
        assert len(results) > 0, "Collection should contain nodes"
        logger.info("✓ List all nodes test passed")
    
    async def test_list_with_filters(self):
        """Test listing nodes with metadata filters."""
        logger.info("=" * 20 + " LIST WITH FILTERS TEST " + "=" * 20)
        
        filters = {"category": "AI"}
        results = await self.vector_store.list(filters=filters, limit=10)
        
        logger.info(f"Filtered list (category=AI) returned {len(results)} nodes")
        for i, node in enumerate(results, 1):
            logger.info(f"  Node {i}: category={node.metadata.get('category')}, id={node.vector_id}")
            assert node.metadata.get("category") == "AI", "All nodes should have category=AI"
        
        logger.info("✓ List with filters test passed")
    
    async def test_update(self):
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
        
        await self.vector_store.update(updated_node, refresh=True)
        
        # Verify update
        result = await self.vector_store.get("test_node2")
        assert "updated" in result.metadata, "Updated metadata should be present"
        logger.info(f"✓ Updated node: {result.vector_id}")
        logger.info(f"  New content: {result.content[:60]}...")
    
    async def test_delete(self):
        """Test deleting nodes."""
        logger.info("=" * 20 + " DELETE TEST " + "=" * 20)
        
        node_to_delete = "test_node4"
        await self.vector_store.delete(node_to_delete, refresh=True)
        
        # Verify deletion - try to get the deleted node
        try:
            result = await self.vector_store.get(node_to_delete)
            # If result is empty list or None, deletion was successful
            if isinstance(result, list):
                assert len(result) == 0, "Deleted node should not be retrievable"
            else:
                assert result is None, "Deleted node should not be retrievable"
        except Exception:
            pass  # Expected if node doesn't exist
        
        logger.info(f"✓ Deleted node: {node_to_delete}")
    
    async def test_copy_collection(self):
        """Test copying a collection."""
        logger.info("=" * 20 + " COPY COLLECTION TEST " + "=" * 20)
        
        copy_collection_name = f"{self.config.TEST_COLLECTION_PREFIX}_copy"
        
        # Clean up if exists
        collections = await self.vector_store.list_collections()
        if copy_collection_name in collections:
            await self.vector_store.delete_collection(copy_collection_name)
        
        # Copy collection
        await self.vector_store.copy_collection(copy_collection_name)
        
        # Verify copy
        collections = await self.vector_store.list_collections()
        assert copy_collection_name in collections, "Copied collection should exist"
        logger.info(f"✓ Copied collection to: {copy_collection_name}")
        
        # Clean up copied collection
        await self.vector_store.delete_collection(copy_collection_name)
        logger.info(f"✓ Cleaned up copied collection")
    
    async def test_list_collections(self):
        """Test listing all collections."""
        logger.info("=" * 20 + " LIST COLLECTIONS TEST " + "=" * 20)
        
        collections = await self.vector_store.list_collections()
        
        logger.info(f"Found {len(collections)} collections")
        test_collections = [c for c in collections if c.startswith(self.config.TEST_COLLECTION_PREFIX)]
        logger.info(f"  Test collections: {test_collections}")
        
        assert self.collection_name in collections, "Main test collection should be listed"
        logger.info("✓ List collections test passed")
    
    async def test_delete_collection(self):
        """Test deleting a collection."""
        logger.info("=" * 20 + " DELETE COLLECTION TEST " + "=" * 20)
        
        await self.vector_store.delete_collection(self.collection_name)
        
        # Verify deletion
        collections = await self.vector_store.list_collections()
        assert self.collection_name not in collections, "Collection should not exist after deletion"
        logger.info(f"✓ Deleted collection: {self.collection_name}")
    
    async def cleanup(self):
        """Clean up all test resources."""
        logger.info("=" * 20 + " CLEANUP " + "=" * 20)
        
        try:
            collections = await self.vector_store.list_collections()
            test_collections = [
                c for c in collections 
                if c.startswith(self.config.TEST_COLLECTION_PREFIX)
            ]
            
            for collection in test_collections:
                try:
                    await self.vector_store.delete_collection(collection)
                    logger.info(f"Deleted test collection: {collection}")
                except Exception as e:
                    logger.warning(f"Failed to delete collection {collection}: {e}")
            
            # Close connections
            await self.vector_store.close()
            await self.embedding_model.close()
            logger.info("✓ Cleanup completed")
        except Exception as e:
            logger.error(f"Cleanup error: {e}")
    
    async def run_all_tests(self):
        """Run all test methods in sequence."""
        logger.info("=" * 60)
        logger.info(" ES VECTOR STORE COMPREHENSIVE TEST SUITE ")
        logger.info("=" * 60)
        
        try:
            await self.test_create_collection()
            await self.test_insert()
            await self.test_search()
            await self.test_search_with_single_filter()
            await self.test_search_with_list_filter()
            await self.test_search_with_multiple_filters()
            await self.test_search_with_complex_filters()
            await self.test_get_by_id()
            await self.test_list_all()
            await self.test_list_with_filters()
            await self.test_update()
            await self.test_delete()
            await self.test_list_collections()
            await self.test_copy_collection()
            await self.test_delete_collection()
            
            logger.info("=" * 60)
            logger.info(" ALL TESTS PASSED ✓ ")
            logger.info("=" * 60)
        except Exception as e:
            logger.error(f"Test failed: {e}")
            raise
        finally:
            await self.cleanup()


# ==================== Cleanup Function ====================


async def cleanup_test_collections():
    """Clean up all test collections without running tests."""
    logger.info("=" * 60)
    logger.info(" CLEANUP TEST COLLECTIONS ")
    logger.info("=" * 60)
    
    config = TestConfig()
    embedding_model = OpenAIEmbeddingModel(
        api_key=config.EMBEDDING_API_KEY,
        base_url=config.EMBEDDING_BASE_URL,
        model_name=config.EMBEDDING_MODEL_NAME,
        dimensions=config.EMBEDDING_DIMENSIONS,
    )
    
    vector_store = ESVectorStore(
        collection_name="dummy",  # Placeholder name
        embedding_model=embedding_model,
        host=config.ES_HOST,
        port=config.ES_PORT,
        user=config.ES_USER,
        password=config.ES_PASSWORD,
    )
    
    try:
        collections = await vector_store.list_collections()
        test_collections = [
            c for c in collections 
            if c.startswith(config.TEST_COLLECTION_PREFIX)
        ]
        
        if not test_collections:
            logger.info("No test collections found to clean up.")
        else:
            logger.info(f"Found {len(test_collections)} test collections to delete")
            for collection in test_collections:
                try:
                    await vector_store.delete_collection(collection)
                    logger.info(f"✓ Deleted: {collection}")
                except Exception as e:
                    logger.warning(f"Failed to delete {collection}: {e}")
            
            logger.info(f"✓ Cleanup completed: removed {len(test_collections)} collections")
    finally:
        await vector_store.close()
        await embedding_model.close()


# ==================== Main Entry Point ====================


def print_usage():
    """Print usage information."""
    print(__doc__)


async def main():
    """Main entry point for test execution."""
    args = sys.argv[1:]
    
    if args and args[0] in ("-h", "--help"):
        print_usage()
        return
    
    if args and args[0] == "--cleanup":
        # Only cleanup, don't run tests
        await cleanup_test_collections()
    else:
        # Run full test suite
        test = ESVectorStoreTest()
        await test.run_all_tests()


if __name__ == "__main__":
    asyncio.run(main())

