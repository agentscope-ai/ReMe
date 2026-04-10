"""Memory handler"""

import numpy as np
from loguru import logger

from ....core import ServiceContext
from ....core.enumeration import MemoryType
from ....core.schema import MemoryNode
from ....core.utils.common_utils import batch_cosine_similarity
from ....core.vector_store import BaseVectorStore


class MemoryHandler:
    """Handler for managing memory nodes in the vector store."""

    def __init__(self, memory_target: str, service_context: ServiceContext):
        self.memory_target: str = memory_target
        self.memory_type: MemoryType | None = service_context.memory_target_type_mapping.get(memory_target, None)
        self.vector_store: BaseVectorStore = service_context.vector_stores["default"]

    async def add_batch(self, memories: list[dict]) -> list[MemoryNode]:
        """Add multiple memory nodes and return their memory_ids."""
        # First, delete existing memory nodes if memory_ids are provided
        memory_ids_to_delete = [mem.get("memory_id") for mem in memories if mem.get("memory_id")]
        if memory_ids_to_delete:
            await self.vector_store.delete(memory_ids_to_delete)

        # Create MemoryNode objects
        memory_nodes = [
            MemoryNode(
                memory_type=self.memory_type,
                memory_target=self.memory_target,
                content=mem.get("content", ""),
                when_to_use=mem.get("when_to_use", ""),
                message_time=mem.get("message_time", ""),
                ref_memory_id=mem.get("ref_memory_id", ""),
                author=mem.get("author", ""),
                score=mem.get("score", 0.0),
                metadata=mem.get("metadata", {}),
            )
            for mem in memories
        ]

        # Deduplicate memory_nodes by content (keep last occurrence)
        memory_dict = {node.content: node for node in memory_nodes}
        memory_nodes = list(memory_dict.values())

        # Convert to VectorNodes and insert
        vector_nodes = [node.to_vector_node() for node in memory_nodes]
        await self.vector_store.insert(vector_nodes)
        return memory_nodes

    async def add(
        self,
        content: str,
        when_to_use: str = "",
        message_time: str = "",
        ref_memory_id: str = "",
        author: str = "",
        score: float = 0.0,
        **kwargs,
    ) -> MemoryNode:
        """Add a single memory node and return its memory_id."""
        memory_dict = {
            "content": content,
            "when_to_use": when_to_use,
            "message_time": message_time,
            "ref_memory_id": ref_memory_id,
            "author": author,
            "score": score,
            "metadata": kwargs,
        }
        memory_nodes = await self.add_batch([memory_dict])
        return memory_nodes[0]

    async def get(self, memory_ids: str | list[str]) -> MemoryNode | list[MemoryNode]:
        """Get one or more memory nodes by their memory_ids."""
        # Call vector_store.get with the memory_ids
        vector_nodes = await self.vector_store.get(memory_ids)

        # Convert VectorNode(s) to MemoryNode(s)
        if isinstance(vector_nodes, list):
            return [MemoryNode.from_vector_node(node) for node in vector_nodes]
        else:
            return MemoryNode.from_vector_node(vector_nodes)

    async def delete(self, memory_ids: str | list[str]):
        """Delete multiple memory nodes by their memory_ids."""
        # Deduplicate if input is a list
        if isinstance(memory_ids, list):
            memory_ids = list(dict.fromkeys(memory_ids))
        await self.vector_store.delete(memory_ids)

    async def delete_all(self):
        """Delete all memory nodes."""
        await self.vector_store.delete_all()

    async def update_batch(self, updates: list[dict]) -> list[MemoryNode]:
        """Update multiple memory nodes with their memory_ids and new values using delete + add."""
        # Deduplicate updates by memory_id (keep last occurrence)
        updates_dict = {upd["memory_id"]: upd for upd in updates}
        updates = list(updates_dict.values())
        memory_ids = list(updates_dict.keys())

        # Get existing nodes
        vector_nodes = await self.vector_store.get(memory_ids)
        if not isinstance(vector_nodes, list):
            vector_nodes = [vector_nodes]

        # Update and convert back
        updated_nodes: list[MemoryNode] = []
        for vector_node, update in zip(vector_nodes, updates):
            memory_node = MemoryNode.from_vector_node(vector_node)
            memory_node.memory_target = self.memory_target
            memory_node.memory_type = self.memory_type

            if "content" in update:
                memory_node.content = update["content"]
            if "when_to_use" in update:
                memory_node.when_to_use = update["when_to_use"]
            if "message_time" in update:
                memory_node.message_time = update["message_time"]
            if "ref_memory_id" in update:
                memory_node.ref_memory_id = update["ref_memory_id"]
            if "author" in update:
                memory_node.author = update["author"]
            if "score" in update:
                memory_node.score = update["score"]
            if "metadata" in update:
                memory_node.metadata.update(update["metadata"])
            updated_nodes.append(memory_node)

        # Delete old nodes first
        await self.vector_store.delete(memory_ids)

        # Then add updated nodes
        vector_nodes = [node.to_vector_node() for node in updated_nodes]
        await self.vector_store.insert(vector_nodes)

        return updated_nodes

    async def update(
        self,
        memory_id: str,
        content: str | None = None,
        when_to_use: str | None = None,
        message_time: str | None = None,
        ref_memory_id: str | None = None,
        author: str | None = None,
        score: float | None = None,
        **kwargs,
    ) -> MemoryNode:
        """Update a memory node's content, when_to_use, or other fields."""
        update_dict: dict = {"memory_id": memory_id}
        if content is not None:
            update_dict["content"] = content
        if when_to_use is not None:
            update_dict["when_to_use"] = when_to_use
        if message_time is not None:
            update_dict["message_time"] = message_time
        if ref_memory_id is not None:
            update_dict["ref_memory_id"] = ref_memory_id
        if author is not None:
            update_dict["author"] = author
        if score is not None:
            update_dict["score"] = score
        if kwargs is not None:
            update_dict["metadata"] = kwargs

        memory_nodes = await self.update_batch([update_dict])
        return memory_nodes[0]

    async def search(
        self,
        query: str | list[str],
        limit: int = 5,
        filters: dict | None = None,
        **kwargs,
    ) -> list[MemoryNode]:
        """Search for similar memory nodes based on query text."""
        filters = filters or {}
        filters["memory_type"] = self.memory_type.value
        filters["memory_target"] = self.memory_target

        # Handle single query
        if isinstance(query, str):
            vector_nodes = await self.vector_store.search(query, limit=limit, filters=filters, **kwargs)
            return [MemoryNode.from_vector_node(node) for node in vector_nodes]

        # Handle multiple queries: search each query with the same limit
        seen_ids: dict[str, MemoryNode] = {}

        for q in query:
            vector_nodes = await self.vector_store.search(q, limit=limit, filters=filters, **kwargs)
            for vector_node in vector_nodes:
                memory_node = MemoryNode.from_vector_node(vector_node)
                if memory_node.memory_id not in seen_ids:
                    seen_ids[memory_node.memory_id] = memory_node

        return list(seen_ids.values())

    async def batch_search(self, searches: list[dict], hybrid_threshold: float = None) -> list[MemoryNode]:
        """Execute multiple search queries in batch and return deduplicated results."""
        if hybrid_threshold is not None:
            # Extract query list from searches
            query_list: list[str] = [search["query"] for search in searches]

            # Step 1: Get embeddings for all queries using the embedding model
            # Shape: [query_size X emb_size]
            embedding_model = self.vector_store.embedding_model
            query_embeddings_list: list[list[float]] = await embedding_model.get_embeddings(query_list)
            query_embeddings = np.array(query_embeddings_list)  # Convert to numpy array

            # Step 2: Use self.search to get search results for each query and deduplicate
            seen_ids: dict[str, MemoryNode] = {}
            for search_params in searches:
                search_result = await self.search(**search_params)
                for memory_node in search_result:
                    if memory_node.memory_id not in seen_ids:
                        seen_ids[memory_node.memory_id] = memory_node

            # Step 3: Get deduplicated results
            deduplicated_results = list(seen_ids.values())

            # If no results, return empty list
            if not deduplicated_results:
                return []

            # Step 4: Extract embeddings from results
            # Shape: [result_size X emb_size]
            result_embeddings_list = [node.vector for node in deduplicated_results if node.vector]

            # Filter out nodes without embeddings
            results_with_embeddings = [node for node in deduplicated_results if node.vector]

            if not result_embeddings_list:
                logger.warning("No results with embeddings found")
                return deduplicated_results

            result_embeddings = np.array(result_embeddings_list)

            # Step 5: Compute cosine similarity matrix
            # Shape: [query_size X result_size]
            similarity_matrix = batch_cosine_similarity(query_embeddings, result_embeddings)

            # Step 6: Calculate average score for each result across all queries
            # Shape: [result_size]
            avg_scores = np.mean(similarity_matrix, axis=0)

            # Step 7: Filter results by hybrid_threshold and sort by average score
            filtered_results = []
            for idx, node in enumerate(results_with_embeddings):
                if avg_scores[idx] >= hybrid_threshold:
                    node.score = float(avg_scores[idx])
                    filtered_results.append(node)

            # Sort by score in descending order
            filtered_results.sort(key=lambda x: x.score, reverse=True)

            return filtered_results

        else:
            # Original behavior: simple deduplication without hybrid scoring
            seen_ids: dict[str, MemoryNode] = {}

            for search_params in searches:
                search_result = await self.search(**search_params)
                for memory_node in search_result:
                    if memory_node.memory_id not in seen_ids:
                        seen_ids[memory_node.memory_id] = memory_node

            return list(seen_ids.values())

    async def list(
        self,
        filters: dict | None = None,
        limit: int | None = None,
        sort_key: str | None = None,
        reverse: bool = True,
    ) -> list[MemoryNode]:
        """List memory nodes with optional filtering and sorting."""
        filters = filters or {}
        filters["memory_type"] = self.memory_type.value
        filters["memory_target"] = self.memory_target

        vector_nodes = await self.vector_store.list(filters=filters, limit=limit, sort_key=sort_key, reverse=reverse)
        return [MemoryNode.from_vector_node(node) for node in vector_nodes]
