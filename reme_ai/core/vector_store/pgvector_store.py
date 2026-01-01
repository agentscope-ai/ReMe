"""PostgreSQL pgvector implementation for vector storage and retrieval."""

import json
from typing import Any

from loguru import logger

from .base_vector_store import BaseVectorStore
from ..context import C
from ..embedding import BaseEmbeddingModel
from ..schema import VectorNode

_ASYNCPG_IMPORT_ERROR = None

try:
    import asyncpg
    from asyncpg import Pool
except ImportError as e:
    _ASYNCPG_IMPORT_ERROR = e
    asyncpg = None
    Pool = None


@C.register_vector_store("pgvector")
class PGVectorStore(BaseVectorStore):
    """Vector store implementation using PostgreSQL and pgvector for efficient similarity search."""

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
        """Initialize the PGVector store with connection parameters and index settings."""
        if _ASYNCPG_IMPORT_ERROR is not None:
            raise ImportError(
                "PGVector requires extra dependencies. Install with `pip install asyncpg pgvector`",
            ) from _ASYNCPG_IMPORT_ERROR

        super().__init__(collection_name=collection_name, embedding_model=embedding_model, **kwargs)

        self.dsn = dsn
        self.host = host
        self.port = port
        self.database = database
        self.user = user
        self.password = password
        self.min_size = min_size
        self.max_size = max_size
        self.use_hnsw = use_hnsw
        self.use_diskann = use_diskann
        self._pool: Pool | None = None
        self.embedding_model_dims = embedding_model.dimensions

    async def _get_pool(self) -> Pool:
        """Create or return the existing asyncpg connection pool."""
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

            async with self._pool.acquire() as conn:
                await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")

            logger.info(f"PGVector connection pool created for database {self.database}")

        return self._pool

    async def _ensure_collection_exists(self):
        """Check if the collection table exists and create it if missing."""
        collections = await self.list_collections()
        if self.collection_name not in collections:
            await self.create_collection(self.collection_name)

    async def list_collections(self) -> list[str]:
        """List all available table names in the current database."""
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT table_name FROM information_schema.tables WHERE table_schema = 'public'",
            )
            return [row["table_name"] for row in rows]

    async def create_collection(self, collection_name: str, **kwargs):
        """Create a new PostgreSQL table with vector support and appropriate indexing."""
        pool = await self._get_pool()
        dimensions = kwargs.get("dimensions", self.embedding_model_dims)

        async with pool.acquire() as conn:
            await conn.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {collection_name} (
                    id TEXT PRIMARY KEY,
                    content TEXT,
                    vector vector({dimensions}),
                    metadata JSONB
                )
            """,
            )

            if self.use_diskann and dimensions < 2000:
                result = await conn.fetchval(
                    "SELECT 1 FROM pg_extension WHERE extname = 'vectorscale'",
                )
                if result:
                    await conn.execute(
                        f"""
                        CREATE INDEX IF NOT EXISTS {collection_name}_diskann_idx
                        ON {collection_name}
                        USING diskann (vector)
                    """,
                    )
                    logger.info(f"Created DiskANN index for collection {collection_name}")
                else:
                    logger.warning("vectorscale extension not available, skipping DiskANN index")
            elif self.use_hnsw:
                await conn.execute(
                    f"""
                    CREATE INDEX IF NOT EXISTS {collection_name}_hnsw_idx
                    ON {collection_name}
                    USING hnsw (vector vector_cosine_ops)
                """,
                )
                logger.info(f"Created HNSW index for collection {collection_name}")

        logger.info(f"Created collection {collection_name} with dimensions={dimensions}")

    async def delete_collection(self, collection_name: str, **kwargs):
        """Remove the specified collection table from the database."""
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            await conn.execute(f"DROP TABLE IF EXISTS {collection_name}")
        logger.info(f"Deleted collection {collection_name}")

    async def copy_collection(self, collection_name: str, **kwargs):
        """Duplicate the structure and content of the current collection to a new table."""
        pool = await self._get_pool()

        async with pool.acquire() as conn:
            columns = await conn.fetch(
                """
                SELECT column_name, data_type, udt_name
                FROM information_schema.columns
                WHERE table_name = $1 AND table_schema = 'public'
            """,
                self.collection_name,
            )

            if not columns:
                raise ValueError(f"Source collection {self.collection_name} does not exist")

            await conn.execute(f"CREATE TABLE {collection_name} AS TABLE {self.collection_name}")
            await conn.execute(f"ALTER TABLE {collection_name} ADD PRIMARY KEY (id)")

            if self.use_hnsw:
                await conn.execute(
                    f"""
                    CREATE INDEX IF NOT EXISTS {collection_name}_hnsw_idx
                    ON {collection_name}
                    USING hnsw (vector vector_cosine_ops)
                """,
                )

        logger.info(f"Copied collection {self.collection_name} to {collection_name}")

    async def insert(self, nodes: VectorNode | list[VectorNode], **kwargs):
        """Insert or upsert vector nodes into the PostgreSQL collection."""
        await self._ensure_collection_exists()

        if isinstance(nodes, VectorNode):
            nodes = [nodes]

        if not nodes:
            return

        nodes_without_vectors = [node for node in nodes if node.vector is None]
        if nodes_without_vectors:
            nodes_with_vectors = await self.get_node_embeddings(nodes_without_vectors)
            vector_map = {n.vector_id: n for n in nodes_with_vectors}
            nodes_to_insert = [vector_map.get(n.vector_id, n) if n.vector is None else n for n in nodes]
        else:
            nodes_to_insert = nodes

        pool = await self._get_pool()
        data = [
            (
                node.vector_id,
                node.content,
                f"[{','.join(map(str, node.vector))}]",
                json.dumps(node.metadata),
            )
            for node in nodes_to_insert
        ]

        async with pool.acquire() as conn:
            on_conflict = kwargs.get("on_conflict", "update")

            if on_conflict == "update":
                await conn.executemany(
                    f"""
                    INSERT INTO {self.collection_name} (id, content, vector, metadata)
                    VALUES ($1, $2, $3::vector, $4::jsonb)
                    ON CONFLICT (id) DO UPDATE SET
                        content = EXCLUDED.content,
                        vector = EXCLUDED.vector,
                        metadata = EXCLUDED.metadata
                """,
                    data,
                )
            elif on_conflict == "ignore":
                await conn.executemany(
                    f"""
                    INSERT INTO {self.collection_name} (id, content, vector, metadata)
                    VALUES ($1, $2, $3::vector, $4::jsonb)
                    ON CONFLICT (id) DO NOTHING
                """,
                    data,
                )
            else:
                await conn.executemany(
                    f"""
                    INSERT INTO {self.collection_name} (id, content, vector, metadata)
                    VALUES ($1, $2, $3::vector, $4::jsonb)
                """,
                    data,
                )

        logger.info(f"Inserted {len(nodes_to_insert)} documents into {self.collection_name}")

    @staticmethod
    def _build_filter_clause(filters: dict | None) -> tuple[str, list]:
        """Generate an SQL WHERE clause and parameter list from a filter dictionary."""
        if not filters:
            return "", []

        conditions = []
        params = []
        param_idx = 1

        for key, value in filters.items():
            if isinstance(value, list):
                placeholders = ", ".join([f"${param_idx + i}" for i in range(len(value))])
                conditions.append(f"metadata->>'{key}' IN ({placeholders})")
                params.extend([str(v) for v in value])
                param_idx += len(value)
            else:
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
        **kwargs,
    ) -> list[VectorNode]:
        """Perform vector similarity search with optional metadata filtering."""
        await self._ensure_collection_exists()

        query_vector = await self.get_embedding(query)
        vector_str = f"[{','.join(map(str, query_vector))}]"
        pool = await self._get_pool()

        filter_clause, filter_params = self._build_filter_clause(filters)

        if filter_clause:
            for i in range(len(filter_params)):
                old_idx = i + 1
                new_idx = i + 2
                filter_clause = filter_clause.replace(f"${old_idx}", f"${new_idx}", 1)

        async with pool.acquire() as conn:
            sql = f"""
                SELECT id, content, vector, metadata, vector <=> $1::vector AS distance
                FROM {self.collection_name}
                {filter_clause}
                ORDER BY distance
                LIMIT ${len(filter_params) + 2}
            """
            rows = await conn.fetch(sql, vector_str, *filter_params, limit)

        results = []
        score_threshold = kwargs.get("score_threshold")

        for row in rows:
            distance = row["distance"]
            if score_threshold is not None and distance > score_threshold:
                continue

            vector_data = None
            if row["vector"]:
                vector_str_raw = str(row["vector"])
                if vector_str_raw.startswith("[") and vector_str_raw.endswith("]"):
                    vector_data = [float(x) for x in vector_str_raw[1:-1].split(",")]

            metadata = row["metadata"] if row["metadata"] else {}
            if isinstance(metadata, str):
                metadata = json.loads(metadata)

            metadata["_score"] = 1 - distance
            metadata["_distance"] = distance

            node = VectorNode(
                vector_id=row["id"],
                content=row["content"] or "",
                vector=vector_data,
                metadata=metadata,
            )
            results.append(node)

        return results

    async def delete(self, vector_ids: str | list[str], **kwargs):
        """Remove specific vector records from the collection by their IDs."""
        await self._ensure_collection_exists()

        if isinstance(vector_ids, str):
            vector_ids = [vector_ids]

        if not vector_ids:
            return

        pool = await self._get_pool()
        async with pool.acquire() as conn:
            placeholders = ", ".join([f"${i + 1}" for i in range(len(vector_ids))])
            await conn.execute(
                f"DELETE FROM {self.collection_name} WHERE id IN ({placeholders})",
                *vector_ids,
            )

        logger.info(f"Deleted {len(vector_ids)} documents from {self.collection_name}")

    async def update(self, nodes: VectorNode | list[VectorNode], **kwargs):
        """Update existing vector nodes with new content, embeddings, or metadata."""
        await self._ensure_collection_exists()

        if isinstance(nodes, VectorNode):
            nodes = [nodes]

        if not nodes:
            return

        nodes_without_vectors = [node for node in nodes if node.vector is None and node.content]
        if nodes_without_vectors:
            nodes_with_vectors = await self.get_node_embeddings(nodes_without_vectors)
            vector_map = {n.vector_id: n for n in nodes_with_vectors}
            nodes_to_update = [vector_map.get(n.vector_id, n) if n.vector is None and n.content else n for n in nodes]
        else:
            nodes_to_update = nodes

        pool = await self._get_pool()
        async with pool.acquire() as conn:
            for node in nodes_to_update:
                update_fields = []
                params = []
                idx = 1

                if node.content:
                    update_fields.append(f"content = ${idx}")
                    params.append(node.content)
                    idx += 1

                if node.vector:
                    vector_str = f"[{','.join(map(str, node.vector))}]"
                    update_fields.append(f"vector = ${idx}::vector")
                    params.append(vector_str)
                    idx += 1

                if node.metadata:
                    update_fields.append(f"metadata = ${idx}::jsonb")
                    params.append(json.dumps(node.metadata))
                    idx += 1

                if update_fields:
                    params.append(node.vector_id)
                    await conn.execute(
                        f"UPDATE {self.collection_name} SET {', '.join(update_fields)} WHERE id = ${idx}",
                        *params,
                    )

        logger.info(f"Updated {len(nodes_to_update)} documents in {self.collection_name}")

    async def get(self, vector_ids: str | list[str]) -> VectorNode | list[VectorNode] | None:
        """Retrieve vector nodes by their unique identifiers."""
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
                *vector_ids,
            )

        results = []
        for row in rows:
            vector_data = None
            if row["vector"]:
                vector_str_raw = str(row["vector"])
                if vector_str_raw.startswith("[") and vector_str_raw.endswith("]"):
                    vector_data = [float(x) for x in vector_str_raw[1:-1].split(",")]

            metadata = row["metadata"] if row["metadata"] else {}
            if isinstance(metadata, str):
                metadata = json.loads(metadata)

            results.append(
                VectorNode(
                    vector_id=row["id"],
                    content=row["content"] or "",
                    vector=vector_data,
                    metadata=metadata,
                ),
            )

        if single_result:
            return results[0] if results else None
        return results

    async def list(
        self,
        filters: dict | None = None,
        limit: int | None = None,
    ) -> list[VectorNode]:
        """Return a list of vector nodes matching the provided filters and limit."""
        await self._ensure_collection_exists()

        pool = await self._get_pool()
        filter_clause, filter_params = self._build_filter_clause(filters)

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

        results = []
        for row in rows:
            vector_data = None
            if row["vector"]:
                vector_str_raw = str(row["vector"])
                if vector_str_raw.startswith("[") and vector_str_raw.endswith("]"):
                    vector_data = [float(x) for x in vector_str_raw[1:-1].split(",")]

            metadata = row["metadata"] if row["metadata"] else {}
            if isinstance(metadata, str):
                metadata = json.loads(metadata)

            results.append(
                VectorNode(
                    vector_id=row["id"],
                    content=row["content"] or "",
                    vector=vector_data,
                    metadata=metadata,
                ),
            )

        return results

    async def collection_info(self) -> dict[str, Any]:
        """Fetch metadata including record count and disk usage for the collection."""
        pool = await self._get_pool()

        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                f"""
                SELECT
                    '{self.collection_name}' as name,
                    (SELECT COUNT(*) FROM {self.collection_name}) as row_count,
                    pg_size_pretty(pg_total_relation_size('{self.collection_name}')) as total_size
            """,
            )

        return {
            "name": row["name"],
            "count": row["row_count"],
            "size": row["total_size"],
        }

    async def reset(self):
        """Purge all data by dropping and recreating the collection table."""
        logger.warning(f"Resetting collection {self.collection_name}...")
        await self.delete_collection(self.collection_name)
        await self.create_collection(self.collection_name)

    async def close(self):
        """Terminate the database connection pool and release associated resources."""
        if self._pool is not None:
            await self._pool.close()
            self._pool = None
            logger.info("PGVector connection pool closed")
