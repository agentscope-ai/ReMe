"""ChromaDB vector store implementation for ReMe.

This module provides a ChromaDB-based vector store that implements the BaseVectorStore
interface. It supports both local persistent storage and remote ChromaDB server connections.
"""

from typing import Dict, List, Optional

from loguru import logger

try:
    import chromadb
    from chromadb.config import Settings
except ImportError:
    raise ImportError(
        "The 'chromadb' library is required. Please install it using 'pip install chromadb'."
    )

from .base_vector_store import BaseVectorStore
from ..context import C
from ..embedding import BaseEmbeddingModel
from ..schema import VectorNode


@C.register_vector_store("chroma")
class ChromaVectorStore(BaseVectorStore):
    """ChromaDB-based vector store implementation.
    
    This class provides vector storage using ChromaDB, supporting both local persistent
    storage and remote server connections. All database operations are wrapped to run
    asynchronously in a thread pool executor.
    
    Attributes:
        client: ChromaDB client instance.
        collection: Current active collection.
    """

    def __init__(
            self,
            collection_name: str,
            embedding_model: BaseEmbeddingModel,
            client: Optional[chromadb.Client] = None,
            host: Optional[str] = None,
            port: Optional[int] = None,
            path: Optional[str] = None,
            api_key: Optional[str] = None,
            tenant: Optional[str] = None,
            database: Optional[str] = None,
            **kwargs,
    ):
        """Initialize the ChromaDB vector store.
        
        Args:
            collection_name: Name of the collection to use.
            embedding_model: Embedding model for generating embeddings. Cannot be None.
            client: Existing ChromaDB client instance (optional).
            host: Host address for ChromaDB server (optional).
            port: Port for ChromaDB server (optional).
            path: Path for local persistent ChromaDB database (default: "./chroma_db").
            api_key: ChromaDB Cloud API key (optional).
            tenant: ChromaDB Cloud tenant ID (optional).
            database: ChromaDB Cloud database name (optional, default: "default").
            **kwargs: Additional configuration parameters.
        """
        super().__init__(
            collection_name=collection_name,
            embedding_model=embedding_model,
            **kwargs
        )

        self.client: chromadb.Client
        self.collection: chromadb.Collection

        if client:
            # Use provided client
            self.client = client
        elif api_key and tenant:
            # Initialize ChromaDB Cloud client
            logger.info("Initializing ChromaDB Cloud client")
            self.client = chromadb.CloudClient(
                api_key=api_key,
                tenant=tenant,
                database=database or "default",
            )
        elif host and port:
            # Initialize HTTP client for remote server
            logger.info(f"Initializing ChromaDB HTTP client at {host}:{port}")
            self.client = chromadb.HttpClient(host=host, port=port)
        else:
            # Initialize local persistent client
            if path is None:
                path = "./chroma_db"
            logger.info(f"Initializing local ChromaDB at {path}")
            self.client = chromadb.PersistentClient(
                path=path,
                settings=Settings(anonymized_telemetry=False),
            )

        # Get or create the collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},  # Use cosine similarity
        )

    def _parse_results(
            self,
            results: Dict,
            include_score: bool = False
    ) -> List[VectorNode]:
        """Parse ChromaDB query results into VectorNode list.
        
        Args:
            results: ChromaDB query result dictionary.
            include_score: Whether to include similarity scores in metadata.
            
        Returns:
            List of VectorNode objects.
        """
        nodes = []

        # Handle nested list structure from query results
        ids = results.get("ids", [])
        documents = results.get("documents", [])
        metadatas = results.get("metadatas", [])
        embeddings = results.get("embeddings")
        if embeddings is None:
            embeddings = []
        distances = results.get("distances")
        if distances is None:
            distances = []

        # Flatten if nested (query returns nested lists)
        if ids and isinstance(ids[0], list):
            ids = ids[0] if ids else []
            documents = documents[0] if documents else []
            metadatas = metadatas[0] if metadatas else []
            embeddings = embeddings[0] if embeddings is not None and len(embeddings) > 0 else []
            distances = distances[0] if distances is not None and len(distances) > 0 else []

        for i, vector_id in enumerate(ids):
            metadata = metadatas[i] if i < len(metadatas) and metadatas[i] else {}

            # Add score to metadata if available
            if include_score and distances and i < len(distances):
                # ChromaDB returns distances, convert to similarity score (1 - distance for cosine)
                metadata["_score"] = 1.0 - distances[i]

            node = VectorNode(
                vector_id=vector_id,
                content=documents[i] if i < len(documents) and documents[i] else "",
                vector=embeddings[i] if len(embeddings) > i else None,
                metadata=metadata,
            )
            nodes.append(node)

        return nodes

    @staticmethod
    def _generate_where_clause(filters: Optional[Dict]) -> Optional[Dict]:
        """Generate a properly formatted where clause for ChromaDB.
        
        Converts universal filter format to ChromaDB's filter format.
        
        Args:
            filters: The filter conditions in universal format.
            
        Returns:
            Properly formatted where clause for ChromaDB, or None if no filters.
        """
        if not filters:
            return None

        def convert_condition(key: str, value) -> Optional[Dict]:
            """Convert a single filter condition to ChromaDB format."""
            if value == "*":
                # Wildcard - match any value (skip this filter)
                return None
            elif isinstance(value, dict):
                # Handle comparison operators
                chroma_condition = {}
                for op, val in value.items():
                    if op == "eq":
                        chroma_condition[key] = {"$eq": val}
                    elif op == "ne":
                        chroma_condition[key] = {"$ne": val}
                    elif op == "gt":
                        chroma_condition[key] = {"$gt": val}
                    elif op == "gte":
                        chroma_condition[key] = {"$gte": val}
                    elif op == "lt":
                        chroma_condition[key] = {"$lt": val}
                    elif op == "lte":
                        chroma_condition[key] = {"$lte": val}
                    elif op == "in":
                        chroma_condition[key] = {"$in": val}
                    elif op == "nin":
                        chroma_condition[key] = {"$nin": val}
                    elif op in ["contains", "icontains"]:
                        # ChromaDB doesn't support contains, fallback to equality
                        chroma_condition[key] = {"$eq": val}
                    else:
                        # Unknown operator, treat as equality
                        chroma_condition[key] = {"$eq": val}
                return chroma_condition
            elif isinstance(value, list):
                # IN operation
                return {key: {"$in": value}}
            else:
                # Simple equality
                return {key: {"$eq": value}}

        processed_filters = []

        for key, value in filters.items():
            if key == "$or":
                # Handle OR conditions
                or_conditions = []
                for condition in value:
                    or_condition = {}
                    for sub_key, sub_value in condition.items():
                        converted = convert_condition(sub_key, sub_value)
                        if converted:
                            or_condition.update(converted)
                    if or_condition:
                        or_conditions.append(or_condition)

                if len(or_conditions) > 1:
                    processed_filters.append({"$or": or_conditions})
                elif len(or_conditions) == 1:
                    processed_filters.append(or_conditions[0])

            elif key == "$and":
                # Handle AND conditions
                and_conditions = []
                for condition in value:
                    for sub_key, sub_value in condition.items():
                        converted = convert_condition(sub_key, sub_value)
                        if converted:
                            and_conditions.append(converted)

                if and_conditions:
                    processed_filters.extend(and_conditions)

            elif key == "$not":
                # Handle NOT conditions - skip for now as ChromaDB has limited support
                continue

            else:
                # Regular condition
                converted = convert_condition(key, value)
                if converted:
                    processed_filters.append(converted)

        # Return appropriate format based on number of conditions
        if len(processed_filters) == 0:
            return None
        elif len(processed_filters) == 1:
            return processed_filters[0]
        else:
            return {"$and": processed_filters}

    async def list_collections(self) -> List[str]:
        """List all available collections in the ChromaDB.
        
        Returns:
            A list of collection names.
        """

        def _list():
            collections = self.client.list_collections()
            return [col.name for col in collections]

        return await self._run_sync_in_executor(_list)

    async def create_collection(self, collection_name: str, **kwargs):
        """Create a new collection in ChromaDB.
        
        Args:
            collection_name: Name of the collection to create.
            **kwargs: Additional collection-specific configuration parameters.
                Supported: metadata (dict), distance_metric (str, default: "cosine").
        """

        def _create():
            distance_metric = kwargs.get("distance_metric", "cosine")
            metadata = kwargs.get("metadata", {})
            metadata["hnsw:space"] = distance_metric

            new_collection = self.client.get_or_create_collection(
                name=collection_name,
                metadata=metadata,
            )
            return new_collection

        new_collection = await self._run_sync_in_executor(_create)

        # Update self.collection if we're creating our own collection
        if collection_name == self.collection_name:
            self.collection = new_collection

        logger.info(f"Created collection {collection_name}")

    async def delete_collection(self, collection_name: str, **kwargs):
        """Delete a collection from ChromaDB.
        
        Args:
            collection_name: Name of the collection to delete.
            **kwargs: Additional parameters (ignored for ChromaDB).
        """

        def _delete():
            try:
                self.client.delete_collection(name=collection_name)
                return True
            except Exception as e:
                logger.warning(f"Failed to delete collection {collection_name}: {e}")
                return False

        deleted = await self._run_sync_in_executor(_delete)

        # Invalidate self.collection if we deleted our own collection
        if deleted and collection_name == self.collection_name:
            self.collection = None

        logger.info(f"Deleted collection {collection_name}")

    async def copy_collection(self, collection_name: str, **kwargs):
        """Copy the current collection to a new collection.
        
        Args:
            collection_name: Name for the new copied collection.
            **kwargs: Additional parameters (ignored for ChromaDB).
        """

        def _copy():
            # Get all data from source collection
            source_data = self.collection.get(include=["documents", "metadatas", "embeddings"])

            if not source_data["ids"]:
                logger.warning(f"Source collection {self.collection_name} is empty")
                return

            # Create target collection
            target_collection = self.client.get_or_create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"},
            )

            # Add data to target collection
            target_collection.add(
                ids=source_data["ids"],
                documents=source_data["documents"],
                metadatas=source_data["metadatas"],
                embeddings=source_data["embeddings"],
            )

        await self._run_sync_in_executor(_copy)
        logger.info(f"Copied collection {self.collection_name} to {collection_name}")

    async def insert(self, nodes: VectorNode | List[VectorNode], **kwargs):
        """Insert one or more vector nodes into the collection.
        
        Args:
            nodes: A single VectorNode or list of VectorNodes to insert.
            **kwargs: Additional parameters.
                Supported: batch_size (int, default: 100).
        """
        # Normalize to list
        if isinstance(nodes, VectorNode):
            nodes = [nodes]

        if not nodes:
            return

        # Generate embeddings if needed
        nodes_to_insert = []
        for node in nodes:
            if node.vector is None:
                node = await self.get_node_embeddings(node)
            nodes_to_insert.append(node)

        batch_size = kwargs.get("batch_size", 100)

        def _insert_batch(batch_nodes: List[VectorNode]):
            ids = [node.vector_id for node in batch_nodes]
            documents = [node.content for node in batch_nodes]
            embeddings = [node.vector for node in batch_nodes]
            metadatas = [node.metadata for node in batch_nodes]

            self.collection.add(
                ids=ids,
                documents=documents,
                embeddings=embeddings,
                metadatas=metadatas,
            )

        # Insert in batches
        for i in range(0, len(nodes_to_insert), batch_size):
            batch = nodes_to_insert[i:i + batch_size]
            await self._run_sync_in_executor(_insert_batch, batch)

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
            **kwargs: Additional search parameters.
                Supported: score_threshold (float), include_embeddings (bool).
            
        Returns:
            A list of VectorNodes ordered by similarity score (most similar first).
        """
        # Generate query embedding
        query_vector = await self.get_embeddings(query)

        where_clause = self._generate_where_clause(filters)
        include_embeddings = kwargs.get("include_embeddings", False)

        def _search():
            include = ["documents", "metadatas", "distances"]
            if include_embeddings:
                include.append("embeddings")

            results = self.collection.query(
                query_embeddings=[query_vector],
                n_results=limit,
                where=where_clause,
                include=include,
            )
            return results

        results = await self._run_sync_in_executor(_search)
        nodes = self._parse_results(results, include_score=True)

        # Apply score threshold if provided
        score_threshold = kwargs.get("score_threshold")
        if score_threshold is not None:
            nodes = [
                node for node in nodes
                if node.metadata.get("_score", 0) >= score_threshold
            ]

        return nodes

    async def delete(self, vector_ids: str | List[str], **kwargs):
        """Delete one or more vectors by their IDs.
        
        Args:
            vector_ids: A single vector ID or list of IDs to delete.
            **kwargs: Additional parameters (ignored for ChromaDB).
        """
        # Normalize to list
        if isinstance(vector_ids, str):
            vector_ids = [vector_ids]

        if not vector_ids:
            return

        def _delete():
            self.collection.delete(ids=vector_ids)

        await self._run_sync_in_executor(_delete)
        logger.info(f"Deleted {len(vector_ids)} nodes from {self.collection_name}")

    async def update(self, nodes: VectorNode | List[VectorNode], **kwargs):
        """Update one or more existing vectors in the collection.
        
        Args:
            nodes: A single VectorNode or list of VectorNodes with updated data.
            **kwargs: Additional parameters (ignored for ChromaDB).
        """
        # Normalize to list
        if isinstance(nodes, VectorNode):
            nodes = [nodes]

        if not nodes:
            return

        # Generate embeddings for nodes that need them
        nodes_to_update = []
        for node in nodes:
            if node.vector is None and node.content:
                node = await self.get_node_embeddings(node)
            nodes_to_update.append(node)

        def _update():
            ids = [node.vector_id for node in nodes_to_update]
            documents = [node.content for node in nodes_to_update]
            embeddings = [node.vector for node in nodes_to_update if node.vector]
            metadatas = [node.metadata for node in nodes_to_update]

            # Use upsert for update (ChromaDB's update requires the item to exist)
            self.collection.upsert(
                ids=ids,
                documents=documents,
                embeddings=embeddings if embeddings else None,
                metadatas=metadatas,
            )

        await self._run_sync_in_executor(_update)
        logger.info(f"Updated {len(nodes_to_update)} nodes in {self.collection_name}")

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

        def _get():
            results = self.collection.get(
                ids=vector_ids,
                include=["documents", "metadatas", "embeddings"],
            )
            return results

        results = await self._run_sync_in_executor(_get)
        nodes = self._parse_results(results)

        if single_result:
            return nodes[0] if nodes else None

        return nodes

    async def list(
            self,
            filters: dict | None = None,
            limit: int | None = None
    ) -> List[VectorNode]:
        """List vectors in the collection with optional filtering.
        
        Args:
            filters: Optional metadata filters to apply.
            limit: Optional maximum number of vectors to return.
                
        Returns:
            A list of VectorNodes matching the filters.
        """
        where_clause = self._generate_where_clause(filters)

        def _list():
            results = self.collection.get(
                where=where_clause,
                limit=limit,
                include=["documents", "metadatas", "embeddings"],
            )
            return results

        results = await self._run_sync_in_executor(_list)
        return self._parse_results(results)

    async def count(self) -> int:
        """Get the number of vectors in the collection.
        
        Returns:
            Number of vectors in the collection.
        """

        def _count():
            return self.collection.count()

        return await self._run_sync_in_executor(_count)

    async def reset(self):
        """Reset the collection by deleting and recreating it."""
        logger.warning(f"Resetting collection {self.collection_name}...")

        await self.delete_collection(self.collection_name)

        def _recreate():
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"},
            )

        await self._run_sync_in_executor(_recreate)
        logger.info(f"Collection {self.collection_name} has been reset")

    async def close(self):
        """Close the vector store and release resources.
        
        For ChromaDB, cleanup may not be strictly necessary but we log it.
        """
        logger.info(f"ChromaDB vector store for collection {self.collection_name} closed")
