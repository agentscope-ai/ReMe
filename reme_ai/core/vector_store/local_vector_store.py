"""Local file system vector store implementation for ReMe.

This module provides a simple file-based vector store that implements the BaseVectorStore
interface. It stores vectors as JSON files and manually calculates cosine similarity for search.
"""

import json
from pathlib import Path
from typing import List

from loguru import logger

from .base_vector_store import BaseVectorStore
from ..context import C
from ..embedding import BaseEmbeddingModel
from ..schema import VectorNode


@C.register_vector_store("local")
class LocalVectorStore(BaseVectorStore):
    """Local file system-based vector store implementation.
    
    This class provides a simple vector storage solution using JSON files on the local
    file system. It manually calculates cosine similarity for vector search operations.
    
    Each collection is stored as a separate directory, with each vector node stored
    as an individual JSON file named by its vector_id.
    """

    def __init__(
        self,
        collection_name: str,
        embedding_model: BaseEmbeddingModel,
        root_path: str = "./local_vector_store",
        **kwargs,
    ):
        """Initialize the local vector store.
        
        Args:
            collection_name: Name of the collection (directory) to use.
            embedding_model: Embedding model for generating embeddings. Cannot be None.
            root_path: Root directory path for storing collections (default: "./local_vector_store").
            **kwargs: Additional configuration parameters.
        """
        super().__init__(collection_name=collection_name, embedding_model=embedding_model, **kwargs)
        self.root_path = Path(root_path)
        self.collection_path = self.root_path / collection_name
        
        # Ensure root directory exists
        self.root_path.mkdir(parents=True, exist_ok=True)

    def _get_collection_path(self, collection_name: str) -> Path:
        """Get the path for a specific collection.
        
        Args:
            collection_name: Name of the collection.
            
        Returns:
            Path object for the collection directory.
        """
        return self.root_path / collection_name

    def _get_node_file_path(self, vector_id: str, collection_name: str | None = None) -> Path:
        """Get the file path for a specific vector node.
        
        Args:
            vector_id: ID of the vector node.
            collection_name: Optional collection name (uses self.collection_name if not provided).
            
        Returns:
            Path object for the node's JSON file.
        """
        col_path = self._get_collection_path(collection_name or self.collection_name)
        return col_path / f"{vector_id}.json"

    def _save_node(self, node: VectorNode, collection_name: str | None = None):
        """Save a vector node to a JSON file.
        
        Args:
            node: The vector node to save.
            collection_name: Optional collection name (uses self.collection_name if not provided).
        """
        file_path = self._get_node_file_path(node.vector_id, collection_name)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(node.model_dump(), f, ensure_ascii=False, indent=2)

    def _load_node(self, vector_id: str, collection_name: str | None = None) -> VectorNode | None:
        """Load a vector node from a JSON file.
        
        Args:
            vector_id: ID of the vector node to load.
            collection_name: Optional collection name (uses self.collection_name if not provided).
            
        Returns:
            The loaded VectorNode or None if not found.
        """
        file_path = self._get_node_file_path(vector_id, collection_name)
        
        if not file_path.exists():
            return None
        
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            return VectorNode(**data)

    def _load_all_nodes(self, collection_name: str | None = None) -> List[VectorNode]:
        """Load all vector nodes from a collection.
        
        Args:
            collection_name: Optional collection name (uses self.collection_name if not provided).
            
        Returns:
            List of all VectorNodes in the collection.
        """
        col_path = self._get_collection_path(collection_name or self.collection_name)
        
        if not col_path.exists():
            return []
        
        nodes = []
        for file_path in col_path.glob("*.json"):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    nodes.append(VectorNode(**data))
            except Exception as e:
                logger.warning(f"Failed to load node from {file_path}: {e}")
        
        return nodes

    @staticmethod
    def _cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors.
        
        Args:
            vec1: First vector.
            vec2: Second vector.
            
        Returns:
            Cosine similarity score between -1 and 1.
        """
        if len(vec1) != len(vec2):
            raise ValueError(f"Vectors must have same length: {len(vec1)} != {len(vec2)}")
        
        # Calculate dot product
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        
        # Calculate magnitudes
        magnitude1 = sum(a * a for a in vec1) ** 0.5
        magnitude2 = sum(b * b for b in vec2) ** 0.5
        
        # Avoid division by zero
        if magnitude1 == 0 or magnitude2 == 0:
            return 0.0
        
        return dot_product / (magnitude1 * magnitude2)

    def _match_filters(self, node: VectorNode, filters: dict | None) -> bool:
        """Check if a node matches the given filters.
        
        Args:
            node: The vector node to check.
            filters: Dictionary of metadata filters. 
                Supports exact match: {"key": "value"} and IN operation: {"key": ["v1", "v2"]}.
            
        Returns:
            True if the node matches all filters, False otherwise.
        """
        if not filters:
            return True
        
        for key, value in filters.items():
            node_value = node.metadata.get(key)
            
            if isinstance(value, list):
                # IN operation: node value must be in the list
                if node_value not in value:
                    return False
            else:
                # Exact match
                if node_value != value:
                    return False
        
        return True

    async def list_collections(self) -> List[str]:
        """List all available collections in the local vector store.
        
        Returns:
            A list of collection names (directory names).
        """
        if not self.root_path.exists():
            return []
        
        collections = [
            d.name for d in self.root_path.iterdir()
            if d.is_dir() and not d.name.startswith('.')
        ]
        
        return collections

    async def create_collection(self, collection_name: str, **kwargs):
        """Create a new collection (directory) in the local vector store.
        
        Args:
            collection_name: Name of the collection to create.
            **kwargs: Additional collection-specific configuration parameters (ignored for local store).
        """
        col_path = self._get_collection_path(collection_name)
        col_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Created collection {collection_name} at {col_path}")

    async def delete_collection(self, collection_name: str, **kwargs):
        """Delete a collection (directory) from the local vector store.
        
        Args:
            collection_name: Name of the collection to delete.
            **kwargs: Additional parameters for deletion operation (ignored for local store).
        """
        col_path = self._get_collection_path(collection_name)
        
        if not col_path.exists():
            logger.warning(f"Collection {collection_name} does not exist")
            return
        
        # Delete all files in the collection
        for file_path in col_path.glob("*.json"):
            file_path.unlink()
        
        # Delete the directory
        col_path.rmdir()
        logger.info(f"Deleted collection {collection_name}")

    async def copy_collection(self, collection_name: str, **kwargs):
        """Copy the current collection to a new collection.
        
        Args:
            collection_name: Name for the new copied collection.
            **kwargs: Additional parameters for the copy operation (ignored for local store).
        """
        source_path = self._get_collection_path(self.collection_name)
        target_path = self._get_collection_path(collection_name)
        
        if not source_path.exists():
            logger.warning(f"Source collection {self.collection_name} does not exist")
            return
        
        # Create target directory
        target_path.mkdir(parents=True, exist_ok=True)
        
        # Copy all JSON files
        for file_path in source_path.glob("*.json"):
            target_file = target_path / file_path.name
            with open(file_path, 'r', encoding='utf-8') as src:
                with open(target_file, 'w', encoding='utf-8') as dst:
                    dst.write(src.read())
        
        logger.info(f"Copied collection {self.collection_name} to {collection_name}")

    async def insert(self, nodes: VectorNode | List[VectorNode], **kwargs):
        """Insert one or more vector nodes into the collection.
        
        Args:
            nodes: A single VectorNode or list of VectorNodes to insert.
            **kwargs: Additional parameters (ignored for local store).
        """
        # Normalize to list
        if isinstance(nodes, VectorNode):
            nodes = [nodes]
        
        # Generate embeddings if needed
        nodes_to_insert = []
        for node in nodes:
            if node.vector is None:
                node = await self.get_node_embeddings(node)
            nodes_to_insert.append(node)
        
        # Save nodes to files
        for node in nodes_to_insert:
            self._save_node(node)
        
        logger.info(f"Inserted {len(nodes_to_insert)} nodes into {self.collection_name}")

    async def search(
        self,
        query: str,
        limit: int = 5,
        filters: dict | None = None,
        **kwargs
    ) -> List[VectorNode]:
        """Search for the most similar vectors to the given query.
        
        Args:
            query: The text query to search for.
            limit: Maximum number of results to return (default: 5).
            filters: Optional metadata filters to apply. 
                Supports exact match: {"key": "value"} and IN operation: {"key": ["v1", "v2"]}.
            **kwargs: Additional search parameters (e.g., score_threshold).
            
        Returns:
            A list of VectorNodes ordered by similarity score (most similar first).
        """
        # Generate query embedding
        query_vector = await self.get_embeddings(query)
        
        # Load all nodes
        all_nodes = self._load_all_nodes()
        
        # Filter nodes based on metadata
        filtered_nodes = [node for node in all_nodes if self._match_filters(node, filters)]
        
        # Calculate similarity scores
        scored_nodes = []
        for node in filtered_nodes:
            if node.vector is None:
                logger.warning(f"Node {node.vector_id} has no vector, skipping")
                continue
            
            try:
                score = self._cosine_similarity(query_vector, node.vector)
                scored_nodes.append((node, score))
            except ValueError as e:
                logger.warning(f"Failed to calculate similarity for node {node.vector_id}: {e}")
        
        # Sort by score in descending order
        scored_nodes.sort(key=lambda x: x[1], reverse=True)
        
        # Apply score threshold if provided
        score_threshold = kwargs.get("score_threshold")
        if score_threshold is not None:
            scored_nodes = [(node, score) for node, score in scored_nodes if score >= score_threshold]
        
        # Limit results
        scored_nodes = scored_nodes[:limit]
        
        # Add scores to metadata and return nodes
        results = []
        for node, score in scored_nodes:
            node.metadata["_score"] = str(score)  # Store as string to match VectorNode schema
            results.append(node)
        
        return results

    async def delete(self, vector_ids: str | List[str], **kwargs):
        """Delete one or more vectors by their IDs.
        
        Args:
            vector_ids: A single vector ID or list of IDs to delete.
            **kwargs: Additional parameters (ignored for local store).
        """
        # Normalize to list
        if isinstance(vector_ids, str):
            vector_ids = [vector_ids]
        
        deleted_count = 0
        for vector_id in vector_ids:
            file_path = self._get_node_file_path(vector_id)
            if file_path.exists():
                file_path.unlink()
                deleted_count += 1
            else:
                logger.warning(f"Node {vector_id} does not exist")
        
        logger.info(f"Deleted {deleted_count} nodes from {self.collection_name}")

    async def update(self, nodes: VectorNode | List[VectorNode], **kwargs):
        """Update one or more existing vectors in the collection.
        
        Args:
            nodes: A single VectorNode or list of VectorNodes with updated data.
            **kwargs: Additional parameters (ignored for local store).
        """
        # Normalize to list
        if isinstance(nodes, VectorNode):
            nodes = [nodes]
        
        # Generate embeddings for nodes that need them
        nodes_to_update = []
        for node in nodes:
            if node.vector is None and node.content:
                node = await self.get_node_embeddings(node)
            nodes_to_update.append(node)
        
        # Update nodes (overwrite existing files)
        updated_count = 0
        for node in nodes_to_update:
            file_path = self._get_node_file_path(node.vector_id)
            if file_path.exists():
                self._save_node(node)
                updated_count += 1
            else:
                logger.warning(f"Node {node.vector_id} does not exist, skipping update")
        
        logger.info(f"Updated {updated_count} nodes in {self.collection_name}")

    async def get(self, vector_ids: str | List[str]) -> VectorNode | List[VectorNode]:
        """Retrieve one or more vectors by their IDs.
        
        Args:
            vector_ids: A single vector ID or list of IDs to retrieve.
            
        Returns:
            A single VectorNode (if single ID provided) or list of VectorNodes
            (if list of IDs provided).
        """
        single_result = isinstance(vector_ids, str)
        if single_result:
            vector_ids = [vector_ids]
        
        results = []
        for vector_id in vector_ids:
            node = self._load_node(vector_id)
            if node:
                results.append(node)
            else:
                logger.warning(f"Node {vector_id} not found")
        
        return results[0] if single_result and results else results

    async def list(
        self,
        filters: dict | None = None,
        limit: int | None = None
    ) -> List[VectorNode]:
        """List vectors in the collection with optional filtering.
        
        Args:
            filters: Optional metadata filters. 
                Supports exact match: {"key": "value"} and IN operation: {"key": ["v1", "v2"]}.
            limit: Optional maximum number of vectors to return.
                
        Returns:
            A list of VectorNodes matching the filters.
        """
        # Load all nodes
        all_nodes = self._load_all_nodes()
        
        # Filter nodes based on metadata
        filtered_nodes = [node for node in all_nodes if self._match_filters(node, filters)]
        
        # Apply limit if specified
        if limit is not None:
            filtered_nodes = filtered_nodes[:limit]
        
        return filtered_nodes

    async def close(self):
        """Close the vector store and release resources.
        
        For local file system store, no cleanup is needed.
        """
        logger.info("Local vector store closed")
