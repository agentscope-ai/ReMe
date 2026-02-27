"""Local file system vector store implementation for ReMe."""

import json
from pathlib import Path

import numpy as np
from loguru import logger

from .base_vector_store import BaseVectorStore
from ..embedding import BaseEmbeddingModel
from ..schema import VectorNode
from ..utils import batch_cosine_similarity


class LocalVectorStore(BaseVectorStore):
    """Local file system-based vector store with in-memory caching.

    All operations are performed in memory after start().
    Changes are persisted to disk on close().
    """

    def __init__(
        self,
        collection_name: str,
        db_path: str | Path,
        embedding_model: BaseEmbeddingModel,
        **kwargs,
    ):
        """Initialize the local vector store with a db_path and collection name."""
        super().__init__(
            collection_name=collection_name,
            db_path=db_path,
            embedding_model=embedding_model,
            **kwargs,
        )
        # In-memory cache: vector_id -> VectorNode
        self._cache: dict[str, VectorNode] = {}
        self._dirty: bool = False  # Track if cache has unsaved changes

    def _get_collection_path(self, collection_name: str) -> Path:
        """Get the file system path for a specific collection."""
        return self.db_path / collection_name

    def _get_node_file_path(self, vector_id: str, collection_name: str | None = None) -> Path:
        """Get the JSON file path for a specific vector node."""
        col_path = self._get_collection_path(collection_name or self.collection_name)
        return col_path / f"{vector_id}.json"

    def _save_node_to_disk(self, node: VectorNode):
        """Save a vector node to a JSON file on disk."""
        file_path = self._get_node_file_path(node.vector_id)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(node.model_dump(), f, ensure_ascii=False, indent=2)

    def _load_all_from_disk(self) -> dict[str, VectorNode]:
        """Load all vector nodes from disk into a dictionary."""
        col_path = self._get_collection_path(self.collection_name)
        if not col_path.exists():
            return {}

        nodes = {}
        for file_path in col_path.glob("*.json"):
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    node = VectorNode(**data)
                    nodes[node.vector_id] = node
            except Exception as e:
                logger.warning(f"Failed to load node from {file_path}: {e}")
        return nodes

    def _flush_to_disk(self):
        """Persist all cached nodes to disk."""
        col_path = self._get_collection_path(self.collection_name)
        col_path.mkdir(parents=True, exist_ok=True)

        # Remove files that are no longer in cache
        existing_files = set(col_path.glob("*.json"))
        cached_ids = set(self._cache.keys())
        for file_path in existing_files:
            vector_id = file_path.stem
            if vector_id not in cached_ids:
                file_path.unlink()

        # Write all cached nodes
        for node in self._cache.values():
            self._save_node_to_disk(node)

        self._dirty = False
        logger.info(f"Flushed {len(self._cache)} nodes to disk")

    @staticmethod
    def _match_filters(node: VectorNode, filters: dict | None) -> bool:
        """Check if a vector node matches the provided metadata filters.

        Supports two filter formats:
        1. Range query: {"field": [start_value, end_value]} - filters for field >= start_value AND field <= end_value
        2. Exact match: {"field": value} - filters for field == value
        """
        if not filters:
            return True

        for key, value in filters.items():
            node_value = node.metadata.get(key)

            # New syntax: [start, end] represents a range query
            if isinstance(value, list) and len(value) == 2:
                # Range query: field >= value[0] AND field <= value[1]
                if node_value is None:
                    return False
                try:
                    # Try numeric comparison
                    if not value[0] <= node_value <= value[1]:
                        return False
                except TypeError:
                    # If comparison fails, the filter doesn't match
                    return False
            else:
                # Exact match
                if node_value != value:
                    return False

        return True

    async def list_collections(self) -> list[str]:
        """List all collection directories in the db_path."""
        if not self.db_path.exists():
            return []

        return [d.name for d in self.db_path.iterdir() if d.is_dir() and not d.name.startswith(".")]

    async def create_collection(self, collection_name: str, **kwargs):
        """Create a new collection directory."""
        col_path = self._get_collection_path(collection_name)
        col_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Created collection {collection_name} at {col_path}")

    async def delete_collection(self, collection_name: str, **kwargs):
        """Delete a collection directory and all its JSON files."""
        col_path = self._get_collection_path(collection_name)

        if not col_path.exists():
            logger.warning(f"Collection {collection_name} does not exist")
            return

        for file_path in col_path.glob("*.json"):
            file_path.unlink()

        col_path.rmdir()
        logger.info(f"Deleted collection {collection_name}")

    async def copy_collection(self, collection_name: str, **kwargs):
        """Copy all nodes from the current collection to a new one."""
        source_path = self._get_collection_path(self.collection_name)
        target_path = self._get_collection_path(collection_name)

        if not source_path.exists():
            logger.warning(f"Source collection {self.collection_name} does not exist")
            return

        target_path.mkdir(parents=True, exist_ok=True)

        for file_path in source_path.glob("*.json"):
            target_file = target_path / file_path.name
            target_file.write_text(file_path.read_text(encoding="utf-8"), encoding="utf-8")

        logger.info(f"Copied collection {self.collection_name} to {collection_name}")

    async def insert(self, nodes: VectorNode | list[VectorNode], **kwargs):
        """Insert vector nodes into the cache, generating embeddings if necessary."""
        if isinstance(nodes, VectorNode):
            nodes = [nodes]

        nodes_without_vectors = [node for node in nodes if node.vector is None]
        if nodes_without_vectors:
            nodes_with_vectors = await self.get_node_embeddings(nodes_without_vectors)
            vector_map = {n.vector_id: n for n in nodes_with_vectors}
            nodes_to_insert = [vector_map.get(n.vector_id, n) if n.vector is None else n for n in nodes]
        else:
            nodes_to_insert = nodes

        for node in nodes_to_insert:
            self._cache[node.vector_id] = node

        self._dirty = True
        logger.info(f"Inserted {len(nodes_to_insert)} nodes into {self.collection_name}")

    async def search(
        self,
        query: str,
        limit: int = 5,
        filters: dict | None = None,
        **kwargs,
    ) -> list[VectorNode]:
        """Search for nodes similar to the query using batch cosine similarity."""
        query_vector = await self.get_embedding(query)

        # Filter nodes from cache
        filtered_nodes = [node for node in self._cache.values() if self._match_filters(node, filters)]

        # Separate nodes with and without vectors
        nodes_with_vectors = [node for node in filtered_nodes if node.vector is not None]
        if not nodes_with_vectors:
            return []

        # Build matrix for batch similarity computation
        node_vectors = np.array([node.vector for node in nodes_with_vectors])
        query_matrix = np.array([query_vector])

        # Compute similarities in batch: shape (1, num_nodes) -> flatten to (num_nodes,)
        similarities = batch_cosine_similarity(query_matrix, node_vectors).flatten()

        # Apply score threshold if specified
        score_threshold = kwargs.get("score_threshold")

        # Pair nodes with scores and filter/sort
        scored_nodes = list(zip(nodes_with_vectors, similarities))
        if score_threshold is not None:
            scored_nodes = [(node, score) for node, score in scored_nodes if score >= score_threshold]

        scored_nodes.sort(key=lambda x: x[1], reverse=True)
        scored_nodes = scored_nodes[:limit]

        # Attach scores to metadata
        results = []
        for node, score in scored_nodes:
            node.metadata["score"] = float(score)
            results.append(node)

        return results

    async def delete(self, vector_ids: str | list[str], **kwargs):
        """Delete specific vector nodes by their IDs from cache."""
        if isinstance(vector_ids, str):
            vector_ids = [vector_ids]

        deleted_count = 0
        for vector_id in vector_ids:
            if vector_id in self._cache:
                del self._cache[vector_id]
                deleted_count += 1
            else:
                logger.warning(f"Node {vector_id} does not exist")

        if deleted_count > 0:
            self._dirty = True
        logger.info(f"Deleted {deleted_count} nodes from {self.collection_name}")

    async def delete_all(self, **kwargs):
        """Remove all vectors from the cache."""
        count = len(self._cache)
        self._cache.clear()
        self._dirty = True
        logger.info(f"Deleted all {count} nodes from {self.collection_name}")

    async def update(self, nodes: VectorNode | list[VectorNode], **kwargs):
        """Update existing vector nodes in the cache."""
        if isinstance(nodes, VectorNode):
            nodes = [nodes]

        nodes_without_vectors = [node for node in nodes if node.vector is None and node.content]
        if nodes_without_vectors:
            nodes_with_vectors = await self.get_node_embeddings(nodes_without_vectors)
            vector_map = {n.vector_id: n for n in nodes_with_vectors}
            nodes_to_update = [vector_map.get(n.vector_id, n) if n.vector is None and n.content else n for n in nodes]
        else:
            nodes_to_update = nodes

        updated_count = 0
        for node in nodes_to_update:
            if node.vector_id in self._cache:
                self._cache[node.vector_id] = node
                updated_count += 1
            else:
                logger.warning(f"Node {node.vector_id} does not exist, skipping update")

        if updated_count > 0:
            self._dirty = True
        logger.info(f"Updated {updated_count} nodes in {self.collection_name}")

    async def get(self, vector_ids: str | list[str]) -> VectorNode | list[VectorNode]:
        """Retrieve one or more vector nodes from cache by their unique IDs."""
        is_single = isinstance(vector_ids, str)
        ids = [vector_ids] if is_single else vector_ids

        results = []
        for vector_id in ids:
            node = self._cache.get(vector_id)
            if node:
                results.append(node)
            else:
                logger.warning(f"Node {vector_id} not found")

        return results[0] if is_single and results else results

    async def list(
        self,
        filters: dict | None = None,
        limit: int | None = None,
        sort_key: str | None = None,
        reverse: bool = True,
    ) -> list[VectorNode]:
        """List vector nodes from cache with optional filtering and limits.

        Args:
            filters: Dictionary of filter conditions to match vectors
            limit: Maximum number of vectors to return
            sort_key: Key to sort the results by (e.g., field name in metadata). None for no sorting
            reverse: If True, sort in descending order; if False, sort in ascending order
        """
        filtered_nodes = [node for node in self._cache.values() if self._match_filters(node, filters)]

        # Apply sorting if sort_key is provided
        if sort_key:

            def sort_key_func(node):
                value = node.metadata.get(sort_key)
                if value is None:
                    return float("-inf") if not reverse else float("inf")
                return value

            filtered_nodes.sort(key=sort_key_func, reverse=reverse)

        if limit is not None:
            filtered_nodes = filtered_nodes[:limit]

        return filtered_nodes

    async def start(self) -> None:
        """Initialize the local vector store and load all nodes into memory."""
        await super().start()
        self._cache = self._load_all_from_disk()
        self._dirty = False
        logger.info(f"Local vector store loaded {len(self._cache)} nodes from {self.collection_name}")

    async def close(self):
        """Persist all cached data to disk and close the vector store."""
        if self._dirty:
            self._flush_to_disk()
        logger.info("Local vector store closed")
