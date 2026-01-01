"""Local file system vector store implementation for ReMe."""

import json
from pathlib import Path

from loguru import logger

from .base_vector_store import BaseVectorStore
from ..context import C
from ..embedding import BaseEmbeddingModel
from ..schema import VectorNode


@C.register_vector_store("local")
class LocalVectorStore(BaseVectorStore):
    """Local file system-based vector store using JSON files and manual cosine similarity."""

    def __init__(
        self,
        collection_name: str,
        embedding_model: BaseEmbeddingModel,
        root_path: str = "./local_vector_store",
        **kwargs,
    ):
        """Initialize the local vector store with a root path and collection name."""
        super().__init__(collection_name=collection_name, embedding_model=embedding_model, **kwargs)
        self.root_path = Path(root_path)
        self.collection_path = self.root_path / collection_name
        self.root_path.mkdir(parents=True, exist_ok=True)

    def _get_collection_path(self, collection_name: str) -> Path:
        """Get the file system path for a specific collection."""
        return self.root_path / collection_name

    def _get_node_file_path(self, vector_id: str, collection_name: str | None = None) -> Path:
        """Get the JSON file path for a specific vector node."""
        col_path = self._get_collection_path(collection_name or self.collection_name)
        return col_path / f"{vector_id}.json"

    def _save_node(self, node: VectorNode, collection_name: str | None = None):
        """Save a vector node to a JSON file on disk."""
        file_path = self._get_node_file_path(node.vector_id, collection_name)
        file_path.parent.mkdir(parents=True, exist_ok=True)

        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(node.model_dump(), f, ensure_ascii=False, indent=2)

    def _load_node(self, vector_id: str, collection_name: str | None = None) -> VectorNode | None:
        """Load a vector node from a JSON file."""
        file_path = self._get_node_file_path(vector_id, collection_name)

        if not file_path.exists():
            return None

        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            return VectorNode(**data)

    def _load_all_nodes(self, collection_name: str | None = None) -> list[VectorNode]:
        """Load all vector nodes existing in a collection."""
        col_path = self._get_collection_path(collection_name or self.collection_name)

        if not col_path.exists():
            return []

        nodes = []
        for file_path in col_path.glob("*.json"):
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    nodes.append(VectorNode(**data))
            except Exception as e:
                logger.warning(f"Failed to load node from {file_path}: {e}")

        return nodes

    @staticmethod
    def _cosine_similarity(vec1: list[float], vec2: list[float]) -> float:
        """Calculate the cosine similarity between two numeric vectors."""
        if len(vec1) != len(vec2):
            raise ValueError(f"Vectors must have same length: {len(vec1)} != {len(vec2)}")

        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        magnitude1 = sum(a * a for a in vec1) ** 0.5
        magnitude2 = sum(b * b for b in vec2) ** 0.5

        if magnitude1 == 0 or magnitude2 == 0:
            return 0.0

        return dot_product / (magnitude1 * magnitude2)

    @staticmethod
    def _match_filters(node: VectorNode, filters: dict | None) -> bool:
        """Check if a vector node matches the provided metadata filters."""
        if not filters:
            return True

        for key, value in filters.items():
            node_value = node.metadata.get(key)

            if isinstance(value, list):
                if node_value not in value:
                    return False
            else:
                if node_value != value:
                    return False

        return True

    async def list_collections(self) -> list[str]:
        """List all collection directories in the root path."""
        if not self.root_path.exists():
            return []

        return [d.name for d in self.root_path.iterdir() if d.is_dir() and not d.name.startswith(".")]

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
        """Insert vector nodes into the local store, generating embeddings if necessary."""
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
            self._save_node(node)

        logger.info(f"Inserted {len(nodes_to_insert)} nodes into {self.collection_name}")

    async def search(
        self,
        query: str,
        limit: int = 5,
        filters: dict | None = None,
        **kwargs,
    ) -> list[VectorNode]:
        """Search for nodes similar to the query using brute-force cosine similarity."""
        query_vector = await self.get_embedding(query)
        all_nodes = self._load_all_nodes()
        filtered_nodes = [node for node in all_nodes if self._match_filters(node, filters)]

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

        scored_nodes.sort(key=lambda x: x[1], reverse=True)

        score_threshold = kwargs.get("score_threshold")
        if score_threshold is not None:
            scored_nodes = [(node, score) for node, score in scored_nodes if score >= score_threshold]

        scored_nodes = scored_nodes[:limit]
        results = []
        for node, score in scored_nodes:
            node.metadata["_score"] = score
            results.append(node)

        return results

    async def delete(self, vector_ids: str | list[str], **kwargs):
        """Delete specific vector nodes by their IDs."""
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

    async def update(self, nodes: VectorNode | list[VectorNode], **kwargs):
        """Update existing vector nodes with new data or embeddings."""
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
            file_path = self._get_node_file_path(node.vector_id)
            if file_path.exists():
                self._save_node(node)
                updated_count += 1
            else:
                logger.warning(f"Node {node.vector_id} does not exist, skipping update")

        logger.info(f"Updated {updated_count} nodes in {self.collection_name}")

    async def get(self, vector_ids: str | list[str]) -> VectorNode | list[VectorNode]:
        """Retrieve one or more vector nodes by their unique IDs."""
        is_single = isinstance(vector_ids, str)
        ids = [vector_ids] if is_single else vector_ids

        results = []
        for vector_id in ids:
            node = self._load_node(vector_id)
            if node:
                results.append(node)
            else:
                logger.warning(f"Node {vector_id} not found")

        return results[0] if is_single and results else results

    async def list(
        self,
        filters: dict | None = None,
        limit: int | None = None,
    ) -> list[VectorNode]:
        """List vector nodes in the collection with optional filtering and limits."""
        all_nodes = self._load_all_nodes()
        filtered_nodes = [node for node in all_nodes if self._match_filters(node, filters)]

        if limit is not None:
            filtered_nodes = filtered_nodes[:limit]

        return filtered_nodes

    async def close(self):
        """Close the vector store (no-op for local file system)."""
        logger.info("Local vector store closed")
