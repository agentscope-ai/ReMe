"""PGVector vector store implementation for ReMe.

This module provides a PostgreSQL + pgvector based vector store that implements
the BaseVectorStore interface. It supports dense vector storage and retrieval
using pgvector's efficient ANN search capabilities with native async operations.
"""

import json
from typing import Any, List

from loguru import logger

_ASYNCPG_IMPORT_ERROR = None

try:
    import asyncpg
    from asyncpg import Pool
except ImportError as e:
    _ASYNCPG_IMPORT_ERROR = e
    asyncpg = None
    Pool = None

from ..embedding import BaseEmbeddingModel
from ..schema import VectorNode
from .base_vector_store import BaseVectorStore
from ..context import C


@C.register_vector_store("pgvector")
class PGVectorStore(BaseVectorStore):
    """PostgreSQL + pgvector based vector store implementation.
    
    This class provides a high-performance vector storage solution using PostgreSQL
    with the pgvector extension. It supports both HNSW and DiskANN indexes for
    efficient approximate nearest neighbor search with full async operations.
    
    Attributes:
        pool: asyncpg connection pool for database operations.
        embedding_model_dims: Dimension of the embedding vectors.
        use_hnsw: Whether to use HNSW index for faster search.
        use_diskann: Whether to use DiskANN index for faster search.
    """

    def __init__(
            self,
            collection_name: str,
            embedding_model: BaseEmbeddingModel,
            host: str = "localhost",
            port: int = 5432,
            database: str = "postgres",
            user: str = "postgres",
            password: str = "",
            min_size: int = 1,
            max_size: int = 10,
            dsn: str | None = None,
            use_hnsw: bool = True,
            use_diskann: bool = False,
            **kwargs,
    ):
        """Initialize the PGVector vector store.
        
        Args:
            collection_name: Name of the collection (table) to use for storing vectors.
            embedding_model: Embedding model for generating embeddings. Cannot be None.
            host: PostgreSQL server host address (default: "localhost").
            port: PostgreSQL server port (default: 5432).
            database: Database name (default: "postgres").
            user: Database user (default: "postgres").
            password: Database password (default: "").
            min_size: Minimum number of connections in the pool (default: 1).
            max_size: Maximum number of connections in the pool (default: 10).
            dsn: Full DSN connection string (overrides individual connection parameters).
            use_hnsw: Whether to create HNSW index for faster search (default: True).
            use_diskann: Whether to create DiskANN index for faster search (default: False).
                Note: DiskANN requires the vectorscale extension.
            **kwargs: Additional configuration parameters for asyncpg pool.
        """
        if _ASYNCPG_IMPORT_ERROR is not None:
            raise ImportError(
                "PGVector requires asyncpg. Install with `pip install asyncpg pgvector`"
            ) from _ASYNCPG_IMPORT_ERROR

        super().__init__(collection_name=collection_name, embedding_model=embedding_model, **kwargs)

        # Connection parameters
        self.dsn = dsn
        self.host = host
        self.port = port
        self.database = database
        self.user = user
        self.password = password
        self.min_size = min_size
        self.max_size = max_size

        # Index options
        self.use_hnsw = use_hnsw
        self.use_diskann = use_diskann

        # Pool will be initialized lazily
        self._pool: Pool | None = None

        # Embedding dimensions
        self.embedding_model_dims = embedding_model.dimensions

    async def _get_pool(self) -> Pool:
        """Get or create the connection pool.
        
        Returns:
            The asyncpg connection pool.
        """
        if self._pool is None:
            if self.dsn:
                self._pool = await asyncpg.create_pool(
                    dsn=self.dsn,
                    min_size=self.min_size,
                    max_size=self.max_size,
                )
            else:
                self._pool = await asyncpg.create_pool(
                    host=self.host,
                    port=self.port,
                    database=self.database,
                    user=self.user,
                    password=self.password,
                    min_size=self.min_size,
                    max_size=self.max_size,
                )

            # Initialize pgvector extension
            async with self._pool.acquire() as conn:
                await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")

            logger.info(f"PGVector connection pool created for database {self.database}")

        return self._pool

    async def _ensure_collection_exists(self):
        """Ensure the collection table exists, creating it if necessary."""
        collections = await self.list_collections()
        if self.collection_name not in collections:
            await self.create_collection(self.collection_name)

    async def list_collections(self) -> List[str]:
        """List all available collections (tables) in the database.
        
        Returns:
            A list of collection/table names.
        """
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT table_name FROM information_schema.tables WHERE table_schema = 'public'"
            )
            return [row["table_name"] for row in rows]

    async def create_collection(self, collection_name: str, **kwargs):
        """Create a new collection (table) in PostgreSQL with vector support.
        
        Args:
            collection_name: Name of the collection to create.
            **kwargs: Additional collection settings (e.g., dimensions).
                If dimensions is not provided, it will be obtained from embedding_model.
        """
        pool = await self._get_pool()
        dimensions = kwargs.get("dimensions", self.embedding_model_dims)

        async with pool.acquire() as conn:
            # Create table with vector column
            await conn.execute(f"""
                CREATE TABLE IF NOT EXISTS {collection_name} (
                    id TEXT PRIMARY KEY,
                    content TEXT,
                    vector vector({dimensions}),
                    metadata JSONB
                )
            """)

            # Create index based on configuration
            if self.use_diskann and dimensions < 2000:
                # Check if vectorscale extension is available
                result = await conn.fetchval(
                    "SELECT 1 FROM pg_extension WHERE extname = 'vectorscale'"
                )
                if result:
                    await conn.execute(f"""
                        CREATE INDEX IF NOT EXISTS {collection_name}_diskann_idx
                        ON {collection_name}
                        USING diskann (vector)
                    """)
                    logger.info(f"Created DiskANN index for collection {collection_name}")
                else:
                    logger.warning("vectorscale extension not available, skipping DiskANN index")
            elif self.use_hnsw:
                await conn.execute(f"""
                    CREATE INDEX IF NOT EXISTS {collection_name}_hnsw_idx
                    ON {collection_name}
                    USING hnsw (vector vector_cosine_ops)
                """)
                logger.info(f"Created HNSW index for collection {collection_name}")

        logger.info(f"Created collection {collection_name} with dimensions={dimensions}")

    async def delete_collection(self, collection_name: str, **kwargs):
        """Delete a collection (table) from PostgreSQL.
        
        Args:
            collection_name: Name of the collection to delete.
            **kwargs: Additional parameters for deletion operation.
        """
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            await conn.execute(f"DROP TABLE IF EXISTS {collection_name}")
        logger.info(f"Deleted collection {collection_name}")

    async def copy_collection(self, collection_name: str, **kwargs):
        """Copy the current collection to a new collection.
        
        Creates a duplicate of self.collection_name with the name specified
        by collection_name parameter.
        
        Args:
            collection_name: Name for the new copied collection.
            **kwargs: Additional parameters for the copy operation.
        """
        pool = await self._get_pool()

        async with pool.acquire() as conn:
            # Get column info from source table
            columns = await conn.fetch(f"""
                SELECT column_name, data_type, udt_name
                FROM information_schema.columns
                WHERE table_name = $1 AND table_schema = 'public'
            """, self.collection_name)

            if not columns:
                raise ValueError(f"Source collection {self.collection_name} does not exist")

            # Create new table as copy of source structure
            await conn.execute(f"""
                CREATE TABLE {collection_name} AS TABLE {self.collection_name}
            """)

            # Add primary key constraint
            await conn.execute(f"""
                ALTER TABLE {collection_name} ADD PRIMARY KEY (id)
            """)

            # Create index on new table
            if self.use_hnsw:
                await conn.execute(f"""
                    CREATE INDEX IF NOT EXISTS {collection_name}_hnsw_idx
                    ON {collection_name}
                    USING hnsw (vector vector_cosine_ops)
                """)

        logger.info(f"Copied collection {self.collection_name} to {collection_name}")

    async def insert(self, nodes: VectorNode | List[VectorNode], **kwargs):
        """Insert one or more vector nodes into the collection.
        
        Args:
            nodes: A single VectorNode or list of VectorNodes to insert.
            **kwargs: Additional parameters (e.g., on_conflict for upsert behavior).
        """
        await self._ensure_collection_exists()

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

        pool = await self._get_pool()

        # Prepare data for batch insert
        data = [
            (
                node.vector_id,
                node.content,
                f"[{','.join(map(str, node.vector))}]",  # Format vector for pgvector
                json.dumps(node.metadata),
            )
            for node in nodes_to_insert
        ]

        async with pool.acquire() as conn:
            # Use INSERT ... ON CONFLICT for upsert behavior
            on_conflict = kwargs.get("on_conflict", "update")

            if on_conflict == "update":
                await conn.executemany(f"""
                    INSERT INTO {self.collection_name} (id, content, vector, metadata)
                    VALUES ($1, $2, $3::vector, $4::jsonb)
                    ON CONFLICT (id) DO UPDATE SET
                        content = EXCLUDED.content,
                        vector = EXCLUDED.vector,
                        metadata = EXCLUDED.metadata
                """, data)
            elif on_conflict == "ignore":
                await conn.executemany(f"""
                    INSERT INTO {self.collection_name} (id, content, vector, metadata)
                    VALUES ($1, $2, $3::vector, $4::jsonb)
                    ON CONFLICT (id) DO NOTHING
                """, data)
            else:
                await conn.executemany(f"""
                    INSERT INTO {self.collection_name} (id, content, vector, metadata)
                    VALUES ($1, $2, $3::vector, $4::jsonb)
                """, data)

        logger.info(f"Inserted {len(nodes_to_insert)} documents into {self.collection_name}")

    def _build_filter_clause(self, filters: dict | None) -> tuple[str, list]:
        """Build SQL WHERE clause from filters.
        
        Args:
            filters: Dictionary of filters to apply.
                Supports exact match: {"key": "value"}
                Supports IN operation: {"key": ["v1", "v2"]}
                
        Returns:
            Tuple of (filter_clause, filter_params).
        """
        if not filters:
            return "", []

        conditions = []
        params = []
        param_idx = 1

        for key, value in filters.items():
            if isinstance(value, list):
                # IN operation
                placeholders = ", ".join([f"${param_idx + i}" for i in range(len(value))])
                conditions.append(f"metadata->>'{key}' IN ({placeholders})")
                params.extend([str(v) for v in value])
                param_idx += len(value)
            else:
                # Exact match
                conditions.append(f"metadata->>'{key}' = ${param_idx}")
                params.append(str(value))
                param_idx += 1

        filter_clause = "WHERE " + " AND ".join(conditions) if conditions else ""
        return filter_clause, params

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
        await self._ensure_collection_exists()

        # Generate query embedding
        query_vector = await self.get_embeddings(query)
        vector_str = f"[{','.join(map(str, query_vector))}]"

        pool = await self._get_pool()

        # Build filter clause
        filter_clause, filter_params = self._build_filter_clause(filters)

        # Adjust parameter indices for filter params (query vector is $1)
        if filter_clause:
            # Shift parameter indices by 1 (since $1 is the vector)
            for i in range(len(filter_params)):
                old_idx = i + 1
                new_idx = i + 2
                filter_clause = filter_clause.replace(f"${old_idx}", f"${new_idx}", 1)

        async with pool.acquire() as conn:
            # Use cosine distance (<=> operator)
            sql = f"""
                SELECT id, content, vector, metadata, vector <=> $1::vector AS distance
                FROM {self.collection_name}
                {filter_clause}
                ORDER BY distance
                LIMIT ${len(filter_params) + 2}
            """

            rows = await conn.fetch(sql, vector_str, *filter_params, limit)

        # Convert results to VectorNodes
        results = []
        score_threshold = kwargs.get("score_threshold")

        for row in rows:
            distance = row["distance"]

            # Filter by score threshold if provided
            if score_threshold is not None and distance > score_threshold:
                continue

            # Parse vector from string representation
            vector_data = None
            if row["vector"]:
                vector_str_raw = str(row["vector"])
                if vector_str_raw.startswith("[") and vector_str_raw.endswith("]"):
                    vector_data = [float(x) for x in vector_str_raw[1:-1].split(",")]

            # Parse metadata
            metadata = row["metadata"] if row["metadata"] else {}
            if isinstance(metadata, str):
                metadata = json.loads(metadata)

            # Add score to metadata
            metadata["_score"] = 1 - distance  # Convert distance to similarity score
            metadata["_distance"] = distance

            node = VectorNode(
                vector_id=row["id"],
                content=row["content"] or "",
                vector=vector_data,
                metadata=metadata,
            )
            results.append(node)

        return results

    async def delete(self, vector_ids: str | List[str], **kwargs):
        """Delete one or more vectors by their IDs.
        
        Args:
            vector_ids: A single vector ID or list of IDs to delete.
            **kwargs: Additional parameters for the deletion operation.
        """
        await self._ensure_collection_exists()

        # Normalize to list
        if isinstance(vector_ids, str):
            vector_ids = [vector_ids]

        if not vector_ids:
            return

        pool = await self._get_pool()

        async with pool.acquire() as conn:
            placeholders = ", ".join([f"${i + 1}" for i in range(len(vector_ids))])
            await conn.execute(
                f"DELETE FROM {self.collection_name} WHERE id IN ({placeholders})",
                *vector_ids
            )

        logger.info(f"Deleted {len(vector_ids)} documents from {self.collection_name}")

    async def update(self, nodes: VectorNode | List[VectorNode], **kwargs):
        """Update one or more existing vectors in the collection.
        
        Args:
            nodes: A single VectorNode or list of VectorNodes with updated data.
            **kwargs: Additional parameters for the update operation.
        """
        await self._ensure_collection_exists()

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

        pool = await self._get_pool()

        async with pool.acquire() as conn:
            for node in nodes_to_update:
                update_fields = []
                params = []
                param_idx = 1

                if node.content:
                    update_fields.append(f"content = ${param_idx}")
                    params.append(node.content)
                    param_idx += 1

                if node.vector:
                    vector_str = f"[{','.join(map(str, node.vector))}]"
                    update_fields.append(f"vector = ${param_idx}::vector")
                    params.append(vector_str)
                    param_idx += 1

                if node.metadata:
                    update_fields.append(f"metadata = ${param_idx}::jsonb")
                    params.append(json.dumps(node.metadata))
                    param_idx += 1

                if update_fields:
                    params.append(node.vector_id)
                    await conn.execute(
                        f"UPDATE {self.collection_name} SET {', '.join(update_fields)} WHERE id = ${param_idx}",
                        *params
                    )

        logger.info(f"Updated {len(nodes_to_update)} documents in {self.collection_name}")

    async def get(self, vector_ids: str | List[str]) -> VectorNode | List[VectorNode]:
        """Retrieve one or more vectors by their IDs.
        
        Args:
            vector_ids: A single vector ID or list of IDs to retrieve.
            
        Returns:
            A single VectorNode (if single ID provided) or list of VectorNodes
            (if list of IDs provided).
        """
        await self._ensure_collection_exists()

        single_result = isinstance(vector_ids, str)
        if single_result:
            vector_ids = [vector_ids]

        if not vector_ids:
            return [] if not single_result else None

        pool = await self._get_pool()

        async with pool.acquire() as conn:
            placeholders = ", ".join([f"${i + 1}" for i in range(len(vector_ids))])
            rows = await conn.fetch(
                f"SELECT id, content, vector, metadata FROM {self.collection_name} WHERE id IN ({placeholders})",
                *vector_ids
            )

        # Convert results to VectorNodes
        results = []
        for row in rows:
            # Parse vector from string representation
            vector_data = None
            if row["vector"]:
                vector_str_raw = str(row["vector"])
                if vector_str_raw.startswith("[") and vector_str_raw.endswith("]"):
                    vector_data = [float(x) for x in vector_str_raw[1:-1].split(",")]

            # Parse metadata
            metadata = row["metadata"] if row["metadata"] else {}
            if isinstance(metadata, str):
                metadata = json.loads(metadata)

            node = VectorNode(
                vector_id=row["id"],
                content=row["content"] or "",
                vector=vector_data,
                metadata=metadata,
            )
            results.append(node)

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
        await self._ensure_collection_exists()

        pool = await self._get_pool()

        # Build filter clause
        filter_clause, filter_params = self._build_filter_clause(filters)

        # Build limit clause
        limit_clause = ""
        if limit:
            limit_clause = f"LIMIT ${len(filter_params) + 1}"
            filter_params.append(limit)

        async with pool.acquire() as conn:
            sql = f"""
                SELECT id, content, vector, metadata
                FROM {self.collection_name}
                {filter_clause}
                {limit_clause}
            """
            rows = await conn.fetch(sql, *filter_params)

        # Convert results to VectorNodes
        results = []
        for row in rows:
            # Parse vector from string representation
            vector_data = None
            if row["vector"]:
                vector_str_raw = str(row["vector"])
                if vector_str_raw.startswith("[") and vector_str_raw.endswith("]"):
                    vector_data = [float(x) for x in vector_str_raw[1:-1].split(",")]

            # Parse metadata
            metadata = row["metadata"] if row["metadata"] else {}
            if isinstance(metadata, str):
                metadata = json.loads(metadata)

            node = VectorNode(
                vector_id=row["id"],
                content=row["content"] or "",
                vector=vector_data,
                metadata=metadata,
            )
            results.append(node)

        return results

    async def collection_info(self) -> dict[str, Any]:
        """Get information about the current collection.
        
        Returns:
            Dictionary containing collection name, row count, and total size.
        """
        pool = await self._get_pool()

        async with pool.acquire() as conn:
            row = await conn.fetchrow(f"""
                SELECT
                    '{self.collection_name}' as name,
                    (SELECT COUNT(*) FROM {self.collection_name}) as row_count,
                    pg_size_pretty(pg_total_relation_size('{self.collection_name}')) as total_size
            """)

        return {
            "name": row["name"],
            "count": row["row_count"],
            "size": row["total_size"],
        }

    async def reset(self):
        """Reset the collection by deleting and recreating it."""
        logger.warning(f"Resetting collection {self.collection_name}...")
        await self.delete_collection(self.collection_name)
        await self.create_collection(self.collection_name)

    async def close(self):
        """Close the database connection pool and release resources."""
        if self._pool is not None:
            await self._pool.close()
            self._pool = None
            logger.info("PGVector connection pool closed")
