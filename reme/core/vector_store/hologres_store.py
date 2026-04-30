"""Hologres implementation for vector storage and retrieval."""

import json
import re
from pathlib import Path
from typing import Any

from loguru import logger

from .base_vector_store import BaseVectorStore
from ..embedding import BaseEmbeddingModel
from ..schema import VectorNode

_ASYNCPG_IMPORT_ERROR: Exception | None = None

try:
    import asyncpg
    from asyncpg import Pool
except Exception as e:
    _ASYNCPG_IMPORT_ERROR = e
    asyncpg = None
    Pool = None


class HologresVectorStore(BaseVectorStore):
    """Vector store implementation using Hologres for efficient similarity search.

    Hologres uses native float4[] arrays for vector storage with built-in
    HGraph index for approximate nearest neighbor search, unlike pgvector
    which requires an extension.
    """

    @staticmethod
    def _validate_table_name(name: str) -> None:
        """Validate table name to prevent SQL injection."""
        if not name:
            raise ValueError("Table name cannot be empty")
        if len(name) > 63:
            raise ValueError(f"Table name too long: {len(name)} characters (max 63)")
        if not re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*$", name):
            raise ValueError(
                f"Invalid table name: {name}. Must start with letter or underscore, "
                "and contain only alphanumeric characters and underscores.",
            )

    def __init__(
        self,
        collection_name: str,
        db_path: str | Path,
        embedding_model: BaseEmbeddingModel,
        host: str = "localhost",
        port: int = 80,
        database: str = "postgres",
        user: str = "postgres",
        password: str = "",
        schema: str = "public",
        min_size: int = 1,
        max_size: int = 10,
        dsn: str | None = None,
        distance_method: str = "Cosine",
        **kwargs,
    ):
        """Initialize the Hologres vector store with connection parameters.

        Args:
            collection_name: Name of the collection (table).
            db_path: Database path (used by base class).
            embedding_model: Embedding model for generating vectors.
            host: Hologres host address.
            port: Hologres port (default 80 for Hologres).
            database: Database name.
            user: Database user.
            password: Database password.
            schema: PostgreSQL schema name (default "public").
            min_size: Minimum connections in pool.
            max_size: Maximum connections in pool.
            dsn: Full DSN connection string (overrides individual params).
            distance_method: Distance method for HGraph index (Cosine, InnerProduct, Euclidean).
        """
        if _ASYNCPG_IMPORT_ERROR is not None:
            raise ImportError(
                "Hologres vector store requires asyncpg. Install with `pip install asyncpg`",
            ) from _ASYNCPG_IMPORT_ERROR

        self._validate_table_name(collection_name)
        self._validate_table_name(schema)

        super().__init__(
            collection_name=collection_name,
            db_path=db_path,
            embedding_model=embedding_model,
            **kwargs,
        )

        self.dsn = dsn
        self.host = host
        self.port = port
        self.database = database
        self.user = user
        self.password = password
        self.schema = schema
        self.min_size = min_size
        self.max_size = max_size
        self.distance_method = distance_method
        self._pool: Pool | None = None
        self.embedding_model_dims = embedding_model.dimensions

    @property
    def _qualified_name(self) -> str:
        """Return the schema-qualified table name (e.g. 'my_schema.my_table')."""
        return f"{self.schema}.{self.collection_name}"

    def _qualify(self, table_name: str) -> str:
        """Return a schema-qualified name for an arbitrary table."""
        return f"{self.schema}.{table_name}"

    @staticmethod
    async def _hologres_reset(conn):
        """Custom reset for Hologres connections."""
        await conn.execute(
            """
            SELECT pg_advisory_unlock_all();
            CLOSE ALL;
            RESET ALL;
        """,
        )

    async def _get_pool(self) -> Pool:
        """Create or return the existing asyncpg connection pool."""
        if self._pool is None:
            if self.dsn:
                self._pool = await asyncpg.create_pool(
                    dsn=self.dsn,
                    min_size=self.min_size,
                    max_size=self.max_size,
                    reset=self._hologres_reset,
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
                    reset=self._hologres_reset,
                )

            # Ensure schema exists
            async with self._pool.acquire() as conn:
                await conn.execute(f"CREATE SCHEMA IF NOT EXISTS {self.schema}")

            logger.info(f"Hologres connection pool created for database {self.database}")

        return self._pool

    @staticmethod
    def _vector_to_pg_array(vector: list[float]) -> str:
        """Convert a Python list of floats to PostgreSQL array literal format."""
        return "{" + ",".join(map(str, vector)) + "}"

    @staticmethod
    def _pg_array_to_vector(pg_array) -> list[float] | None:
        """Convert a PostgreSQL array result to a Python list of floats."""
        if pg_array is None:
            return None
        if isinstance(pg_array, list):
            return [float(x) for x in pg_array]
        # Handle string format like {1.0,2.0,3.0}
        raw = str(pg_array)
        if raw.startswith("{") and raw.endswith("}"):
            return [float(x) for x in raw[1:-1].split(",")]
        return None

    async def list_collections(self) -> list[str]:
        """List all available table names in the current schema."""
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT table_name FROM information_schema.tables WHERE table_schema = $1",
                self.schema,
            )
            return [row["table_name"] for row in rows]

    async def create_collection(self, collection_name: str, **kwargs):
        """Create a new Hologres table with vector support and HGraph index."""
        self._validate_table_name(collection_name)
        pool = await self._get_pool()
        dimensions = kwargs.get("dimensions", self.embedding_model_dims)
        qualified = self._qualify(collection_name)

        async with pool.acquire() as conn:
            create_sql = f"""
                CREATE TABLE IF NOT EXISTS {qualified} (
                    id TEXT PRIMARY KEY,
                    content TEXT,
                    vector float4[] CHECK (array_ndims(vector) = 1 AND array_length(vector, 1) = {dimensions}),
                    metadata JSONB
                )
                WITH (
                    vectors = '{{
                        "vector": {{
                            "algorithm": "HGraph",
                            "distance_method": "{self.distance_method}",
                            "builder_params": {{
                                "base_quantization_type": "rabitq",
                                "rabitq_use_fht":true,
                                "graph_storage_type": "compressed",
                                "max_total_size_to_merge_mb": 4096,
                                "max_degree": 64,
                                "ef_construction": 400,
                                "precise_quantization_type": "fp32",
                                "use_reorder": true
                            }}
                        }}
                    }}'
                )
            """
            await conn.execute(create_sql)

        logger.info(f"Created Hologres collection {qualified} with dimensions={dimensions}")

    async def delete_collection(self, collection_name: str, **kwargs):
        """Remove the specified collection table from the database."""
        self._validate_table_name(collection_name)
        pool = await self._get_pool()
        qualified = self._qualify(collection_name)
        async with pool.acquire() as conn:
            await conn.execute(f"DROP TABLE IF EXISTS {qualified}")
        logger.info(f"Deleted collection {qualified}")

    async def copy_collection(self, collection_name: str, **kwargs):
        """Duplicate the structure and content of the current collection to a new table."""
        self._validate_table_name(collection_name)
        pool = await self._get_pool()
        qualified_src = self._qualified_name
        qualified_dst = self._qualify(collection_name)

        async with pool.acquire() as conn:
            columns = await conn.fetch(
                """
                SELECT column_name, data_type, udt_name
                FROM information_schema.columns
                WHERE table_name = $1 AND table_schema = $2
            """,
                self.collection_name,
                self.schema,
            )

            if not columns:
                raise ValueError(f"Source collection {qualified_src} does not exist")

            # Create new table with primary key, then add data
            await conn.execute(
                f"""
                               SET hg_experimental_enable_create_table_like_properties = true;
                               CALL hg_create_table_like('{qualified_dst}', 'select * from {qualified_src}')
                               """,
            )
            await conn.execute(f"INSERT INTO {qualified_dst} SELECT * FROM {qualified_src} ;")

        logger.info(f"Copied collection {qualified_src} to {qualified_dst}")

    async def insert(self, nodes: VectorNode | list[VectorNode], **kwargs):
        """Insert or upsert vector nodes into the Hologres collection."""
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
                node.vector,
                json.dumps(node.metadata),
            )
            for node in nodes_to_insert
        ]

        async with pool.acquire() as conn:
            on_conflict = kwargs.get("on_conflict", "update")

            if on_conflict == "update":
                await conn.executemany(
                    f"""
                    INSERT INTO {self._qualified_name} (id, content, vector, metadata)
                    VALUES ($1, $2, $3::float4[], $4::jsonb)
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
                    INSERT INTO {self._qualified_name} (id, content, vector, metadata)
                    VALUES ($1, $2, $3::float4[], $4::jsonb)
                    ON CONFLICT (id) DO NOTHING
                """,
                    data,
                )
            else:
                await conn.executemany(
                    f"""
                    INSERT INTO {self._qualified_name} (id, content, vector, metadata)
                    VALUES ($1, $2, $3::float4[], $4::jsonb)
                """,
                    data,
                )

        logger.info(f"Inserted {len(nodes_to_insert)} documents into {self._qualified_name}")

    @staticmethod
    def _build_filter_clause(filters: dict | None) -> tuple[str, list]:
        """Generate an SQL WHERE clause and parameter list from a filter dictionary.

        Supports two filter formats:
        1. Range query: {"field": [start_value, end_value]}
        2. Exact match: {"field": value}
        """
        if not filters:
            return "", []

        conditions = []
        params = []
        param_idx = 1

        for key, value in filters.items():
            if not key.replace("_", "").replace(".", "").isalnum():
                raise ValueError(
                    f"Invalid metadata key: {key}. Only alphanumeric characters, underscore and dot are allowed.",
                )

            if isinstance(value, list) and len(value) == 2:
                if isinstance(value[0], (int, float)) and isinstance(value[1], (int, float)):
                    conditions.append(
                        f"(metadata->>'{key}')::numeric >= ${param_idx} AND "
                        f"(metadata->>'{key}')::numeric <= ${param_idx + 1}",
                    )
                else:
                    conditions.append(f"metadata->>'{key}' >= ${param_idx} AND metadata->>'{key}' <= ${param_idx + 1}")
                params.extend([value[0], value[1]])
                param_idx += 2
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
        """Perform vector similarity search using Hologres approx_cosine_distance."""
        query_vector = await self.get_embedding(query)
        vector_str = self._vector_to_pg_array(query_vector)
        pool = await self._get_pool()

        filter_clause, filter_params = self._build_filter_clause(filters)

        # filter_params use $1..$N, limit uses $(N+1)
        limit_placeholder = f"${len(filter_params) + 1}"

        async with pool.acquire() as conn:
            sql = f"""
                SELECT id, content, vector, metadata,
                       approx_cosine_distance(vector, '{vector_str}') AS distance
                FROM {self._qualified_name}
                {filter_clause}
                ORDER BY distance DESC
                LIMIT {limit_placeholder}
            """
            rows = await conn.fetch(sql, *filter_params, limit)

        results = []
        score_threshold = kwargs.get("score_threshold")

        for row in rows:
            distance = float(row["distance"])
            # approx_cosine_distance returns cosine similarity (higher = more similar)
            score = distance
            if score_threshold is not None and score < score_threshold:
                continue

            vector_data = self._pg_array_to_vector(row["vector"])

            metadata = row["metadata"] if row["metadata"] else {}
            if isinstance(metadata, str):
                metadata = json.loads(metadata)

            metadata["score"] = score
            metadata["_distance"] = 1 - score

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
        if isinstance(vector_ids, str):
            vector_ids = [vector_ids]

        if not vector_ids:
            return

        pool = await self._get_pool()
        async with pool.acquire() as conn:
            placeholders = ", ".join([f"${i + 1}" for i in range(len(vector_ids))])
            await conn.execute(
                f"DELETE FROM {self._qualified_name} WHERE id IN ({placeholders})",
                *vector_ids,
            )

        logger.info(f"Deleted {len(vector_ids)} documents from {self._qualified_name}")

    async def delete_all(self, **kwargs):
        """Remove all vectors from the collection."""
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            result = await conn.execute(f"DELETE FROM {self._qualified_name}")

        logger.info(f"Deleted all documents from {self._qualified_name} result={result}")

    async def update(self, nodes: VectorNode | list[VectorNode], **kwargs):
        """Update existing vector nodes with new content, embeddings, or metadata."""
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
                    update_fields.append(f"vector = ${idx}::float4[]")
                    params.append(node.vector)
                    idx += 1

                if node.metadata:
                    update_fields.append(f"metadata = ${idx}::jsonb")
                    params.append(json.dumps(node.metadata))
                    idx += 1

                if update_fields:
                    params.append(node.vector_id)
                    await conn.execute(
                        f"UPDATE {self._qualified_name} SET {', '.join(update_fields)} WHERE id = ${idx}",
                        *params,
                    )

        logger.info(f"Updated {len(nodes_to_update)} documents in {self._qualified_name}")

    async def get(self, vector_ids: str | list[str]) -> VectorNode | list[VectorNode] | None:
        """Retrieve vector nodes by their unique identifiers."""
        single_result = isinstance(vector_ids, str)
        if single_result:
            vector_ids = [vector_ids]

        if not vector_ids:
            return [] if not single_result else None

        pool = await self._get_pool()
        async with pool.acquire() as conn:
            placeholders = ", ".join([f"${i + 1}" for i in range(len(vector_ids))])
            rows = await conn.fetch(
                f"SELECT id, content, vector, metadata FROM {self._qualified_name} WHERE id IN ({placeholders})",
                *vector_ids,
            )

        results = []
        for row in rows:
            vector_data = self._pg_array_to_vector(row["vector"])

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
        sort_key: str | None = None,
        reverse: bool = False,
    ) -> list[VectorNode]:
        """Return a list of vector nodes matching the provided filters and limit.

        Args:
            filters: Dictionary of filter conditions to match vectors
            limit: Maximum number of vectors to return
            sort_key: Key to sort the results by (e.g., field name in metadata). None for no sorting
            reverse: If True, sort in descending order; if False, sort in ascending order
        """
        pool = await self._get_pool()
        filter_clause, filter_params = self._build_filter_clause(filters)

        order_clause = ""
        if sort_key:
            order_direction = "DESC" if reverse else "ASC"
            order_clause = f"ORDER BY metadata->>'{sort_key}' {order_direction}"

        limit_clause = ""
        if limit:
            limit_clause = f"LIMIT ${len(filter_params) + 1}"
            filter_params.append(limit)

        async with pool.acquire() as conn:
            sql = f"""
                SELECT id, content, vector, metadata
                FROM {self._qualified_name}
                {filter_clause}
                {order_clause}
                {limit_clause}
            """
            rows = await conn.fetch(sql, *filter_params)

        results = []
        for row in rows:
            vector_data = self._pg_array_to_vector(row["vector"])

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
        qualified = self._qualified_name

        async with pool.acquire() as conn:
            count = await conn.fetchval(f"SELECT COUNT(*) FROM {qualified}")
            size = await conn.fetchval(f"SELECT pg_size_pretty(pg_total_relation_size('{qualified}'))")

        return {
            "name": qualified,
            "count": count,
            "size": size,
        }

    async def reset(self):
        """Purge all data by dropping and recreating the collection table."""
        logger.warning(f"Resetting collection {self._qualified_name}...")
        await self.delete_collection(self.collection_name)
        await self.create_collection(self.collection_name)

    async def reset_collection(self, collection_name: str):
        """Reset collection with table name validation."""
        self._validate_table_name(collection_name)
        self.collection_name = collection_name
        await self.create_collection(collection_name)
        logger.info(f"Collection reset to {self._qualified_name}")

    async def start(self) -> None:
        """Initialize the PGVector store.

        Creates the connection pool and ensures the collection table exists.
        """
        await self._get_pool()
        await super().start()
        logger.info(f"Hologres collection {self._qualified_name} initialized")

    async def close(self):
        """Terminate the database connection pool."""
        if self._pool is not None:
            await self._pool.close()
            self._pool = None
            logger.info("Hologres connection pool closed")
