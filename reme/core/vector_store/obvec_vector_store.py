"""OceanBase vector store implementation for ReMe.

This module provides an OceanBase-based vector store that implements the BaseVectorStore
interface for high-performance dense vector storage and retrieval using pyobvector.
"""

import json
from pathlib import Path
from typing import Any

from loguru import logger

from .base_vector_store import BaseVectorStore
from ..embedding import BaseEmbeddingModel
from ..schema import VectorNode

_OBVECTOR_IMPORT_ERROR: Exception | None = None

try:
    from pyobvector import IndexParams, ObVecClient, VecIndexType, VECTOR
except Exception as e:
    _OBVECTOR_IMPORT_ERROR = e
    IndexParams = None
    ObVecClient = None
    VecIndexType = None
    VECTOR = None


class ObVecVectorStore(BaseVectorStore):
    """OceanBase-based vector store for dense vector storage and kNN search."""

    @staticmethod
    def _coerce_db_vector(raw: Any) -> list[float] | None:
        if raw is None:
            return None
        if isinstance(raw, list):
            return [float(x) for x in raw]
        if isinstance(raw, str):
            try:
                parsed = json.loads(raw)
                if isinstance(parsed, list):
                    return [float(x) for x in parsed]
            except (json.JSONDecodeError, TypeError, ValueError):
                pass
        return None

    @staticmethod
    def _coerce_db_metadata(raw: Any) -> dict[str, Any]:
        if raw is None:
            return {}
        if isinstance(raw, dict):
            return raw
        if isinstance(raw, str):
            try:
                parsed = json.loads(raw)
                if isinstance(parsed, dict):
                    return parsed
            except (json.JSONDecodeError, TypeError):
                pass
        return {}

    def __init__(
        self,
        collection_name: str,
        db_path: str | Path,
        embedding_model: BaseEmbeddingModel,
        uri: str = "127.0.0.1:2881",
        user: str = "root",
        password: str = "",
        database: str = "test",
        index_type: str = "HNSW",
        index_metric: str = "cosine",
        index_ef_search: int = 100,
        **kwargs,
    ):
        """Initialize the OceanBase vector store with connection parameters.

        Args:
            collection_name: Name of the collection (table).
            db_path: Database path (not used for remote OceanBase, kept for API consistency).
            embedding_model: Model instance used to generate vector embeddings.
            uri: Connection URI for OceanBase server.
            user: Username for authentication.
            password: Password for authentication.
            database: Database name to use.
            index_type: Type of vector index (HNSW).
            index_metric: Distance metric (cosine, l2, ip).
            index_ef_search: HNSW ef_search parameter.
            **kwargs: Additional configuration passed to the base class.
        """
        if _OBVECTOR_IMPORT_ERROR is not None:
            raise ImportError(
                "ObVecVectorStore requires extra dependencies. Install with `pip install pyobvector`",
            ) from _OBVECTOR_IMPORT_ERROR

        super().__init__(
            collection_name=collection_name,
            db_path=db_path,
            embedding_model=embedding_model,
            **kwargs,
        )

        self.uri = uri
        self.user = user
        self.password = password
        self.database = database
        self.index_type = index_type.upper()
        self.index_metric = index_metric.lower()
        self.index_ef_search = index_ef_search

        self.client: ObVecClient | None = None
        self.embedding_model_dims = embedding_model.dimensions

    async def list_collections(self) -> list[str]:
        """List all available table names in the current database."""
        # OceanBase doesn't have a direct list collections API
        # We'll return the current collection name if it exists
        if self.client is None:
            return []
        try:
            result = self.client.perform_raw_text_sql(
                f"SHOW TABLES FROM `{self.database}`",
            )
            rows = result.fetchall()
            return [row[0] for row in rows if row]
        except Exception as e:
            logger.warning(f"Failed to list collections: {e}")
            return []

    async def create_collection(self, collection_name: str, **kwargs):
        """Create a new table with vector support and appropriate indexing."""
        dimensions = kwargs.get("dimensions", self.embedding_model_dims)

        if self.client is None:
            logger.warning("Client not initialized, skipping collection creation")
            return

        # Check if table already exists
        if self.client.check_table_exists(collection_name):
            logger.info(f"Collection {collection_name} already exists")
            return

        from sqlalchemy import Column, String, Text, JSON, create_engine
        from sqlalchemy.dialects.mysql import LONGTEXT

        # Create columns
        columns = [
            Column("id", String(255), primary_key=True),
            Column("content", LONGTEXT),
            Column("vector", VECTOR(dimensions)),
            Column("metadata", JSON),
        ]

        vidxs: IndexParams | None = None
        if self.index_type == "HNSW":
            metric_map = {
                "cosine": "cosine",
                "l2": "l2_distance",
                "ip": "inner_product",
            }
            metric = metric_map.get(self.index_metric, "cosine")
            vidxs = IndexParams()
            vidxs.add_index(
                "vector",
                VecIndexType.HNSW,
                f"{collection_name}_vidx",
                metric_type=metric,
                params={"efSearch": self.index_ef_search},
            )

        try:
            # Create table with index
            self.client.create_table_with_index_params(
                table_name=collection_name,
                columns=columns,
                vidxs=vidxs,
            )
            logger.info(f"Created collection {collection_name} with dimensions={dimensions}")
        except Exception as e:
            logger.error(f"Failed to create collection {collection_name}: {e}")
            raise

    async def delete_collection(self, collection_name: str, **kwargs):
        """Permanently delete a collection table from the database."""
        if self.client is None:
            logger.warning("Client not initialized, skipping collection deletion")
            return

        try:
            self.client.drop_table_if_exist(collection_name)
            logger.info(f"Deleted collection {collection_name}")
        except Exception as e:
            logger.error(f"Failed to delete collection {collection_name}: {e}")
            raise

    async def copy_collection(self, collection_name: str, **kwargs):
        """Duplicate the current collection to a new one with the given name."""
        if self.client is None:
            logger.warning("Client not initialized, skipping collection copy")
            return

        # Check if source collection exists
        if not self.client.check_table_exists(self.collection_name):
            raise ValueError(f"Source collection {self.collection_name} does not exist")

        # Create new collection with same structure
        await self.create_collection(collection_name)

        # Copy data from source to target
        try:
            # Get all data from source
            source_data = await self.list(limit=None)

            # Insert into new collection
            if source_data:
                await self.insert(source_data, collection_name=collection_name)

            logger.info(f"Copied collection {self.collection_name} to {collection_name}")
        except Exception as e:
            # Clean up the created collection if copy fails
            try:
                self.client.drop_table_if_exist(collection_name)
            except:
                pass
            raise

    async def insert(self, nodes: VectorNode | list[VectorNode], **kwargs):
        """Insert or upsert vector nodes into the OceanBase collection."""
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

        # Prepare data for insertion
        data = []
        for node in nodes_to_insert:
            # Convert vector to list if it's not already
            vector_list = node.vector if node.vector is not None else []

            data.append({
                "id": node.vector_id,
                "content": node.content,
                "vector": vector_list,
                "metadata": node.metadata if node.metadata else {},
            })

        # Determine which collection to use
        target_collection = kwargs.get("collection_name", self.collection_name)

        try:
            self.client.insert(
                table_name=target_collection,
                data=data,
            )
            logger.info(f"Inserted {len(nodes_to_insert)} documents into {target_collection}")
        except Exception as e:
            logger.error(f"Failed to insert documents: {e}")
            raise

    def _build_filter_clause(self, filters: dict | None) -> str:
        """Generate a WHERE clause from filter dictionary.

        Supports two filter formats:
        1. Range query: {"field": [start_value, end_value]} - filters for field >= start_value AND field <= end_value
        2. Exact match: {"field": value} - filters for field == value
        """
        if not filters:
            return ""

        conditions = []
        for key, value in filters.items():
            # Sanitize key to prevent SQL injection
            if not key.replace("_", "").replace(".", "").isalnum():
                continue

            # New syntax: [start, end] represents a range query
            if isinstance(value, list) and len(value) == 2:
                lo, hi = value[0], value[1]
                if isinstance(lo, (int, float)) and isinstance(hi, (int, float)):
                    conditions.append(
                        f"(JSON_EXTRACT(metadata, '$.{key}') >= {lo} AND "
                        f"JSON_EXTRACT(metadata, '$.{key}') <= {hi})",
                    )
                else:
                    conditions.append(
                        f"(JSON_EXTRACT(metadata, '$.{key}') >= '{lo}' AND "
                        f"JSON_EXTRACT(metadata, '$.{key}') <= '{hi}')",
                    )
            else:
                if isinstance(value, (int, float)):
                    conditions.append(f"JSON_EXTRACT(metadata, '$.{key}') = {value}")
                else:
                    conditions.append(f"JSON_EXTRACT(metadata, '$.{key}') = '{value}'")

        return " AND ".join(conditions)

    async def search(
        self,
        query: str,
        limit: int = 5,
        filters: dict | None = None,
        **kwargs,
    ) -> list[VectorNode]:
        """Perform a kNN similarity search based on a text query."""
        query_vector = await self.get_embedding(query)
        if hasattr(query_vector, "tolist"):
            query_vector = query_vector.tolist()
        else:
            query_vector = [float(x) for x in query_vector]

        # Determine distance function based on metric
        if self.index_metric == "cosine":
            from pyobvector import cosine_distance
            distance_func = cosine_distance
        elif self.index_metric == "l2":
            from pyobvector import l2_distance
            distance_func = l2_distance
        else:  # ip (inner product)
            from pyobvector import inner_product
            distance_func = inner_product

        # ann_search passes where_clause to SQLAlchemy where(*...); use text(), not raw str.
        from sqlalchemy import text as sa_text

        filter_sql = self._build_filter_clause(filters)
        where_parts = [sa_text(filter_sql)] if filter_sql else None

        try:
            # pyobvector expects vec_data as a flat list of floats, not [vector].
            results = self.client.ann_search(
                table_name=self.collection_name,
                vec_data=query_vector,
                vec_column_name="vector",
                distance_func=distance_func,
                with_dist=True,
                topk=limit,
                output_column_names=["id", "content", "metadata"],
                where_clause=where_parts,
            )

            search_results = []
            score_threshold = kwargs.get("score_threshold")

            for row in results:
                # row format: (id, content, metadata, distance)
                if len(row) >= 4:
                    vector_id = row[0]
                    content = row[1]
                    metadata_str = row[2]
                    distance = row[3]

                    # Convert distance to similarity score
                    if self.index_metric == "cosine":
                        score = max(0.0, 1.0 - distance / 2.0)
                    elif self.index_metric == "l2":
                        # For L2, we need to normalize - this is a rough approximation
                        score = max(0.0, 1.0 - distance / 2.0)
                    else:  # inner product
                        # For IP, higher is better, no conversion needed
                        score = max(0.0, distance)

                    if score_threshold is not None and score < score_threshold:
                        continue

                    # Parse metadata
                    metadata = {}
                    if metadata_str:
                        metadata = self._coerce_db_metadata(metadata_str)

                    metadata["score"] = score
                    metadata["_distance"] = distance

                    search_results.append(
                        VectorNode(
                            vector_id=vector_id,
                            content=content or "",
                            vector=None,  # Don't return vector in search results
                            metadata=metadata,
                        ),
                    )

            return search_results
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []

    async def delete(self, vector_ids: str | list[str], **kwargs):
        """Remove specific vector records from the collection by their IDs."""
        if isinstance(vector_ids, str):
            vector_ids = [vector_ids]

        if not vector_ids:
            return

        try:
            self.client.delete(self.collection_name, ids=vector_ids)
            logger.info(f"Deleted {len(vector_ids)} documents from {self.collection_name}")
        except Exception as e:
            logger.error(f"Failed to delete documents: {e}")
            raise

    async def delete_all(self, **kwargs):
        """Remove all vectors from the collection."""
        try:
            self.client.delete(self.collection_name)
            logger.info(f"Deleted all documents from {self.collection_name}")
        except Exception as e:
            logger.error(f"Failed to delete all documents: {e}")
            raise

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

        try:
            from sqlalchemy import text

            for node in nodes_to_update:
                updates: list[str] = []
                params: dict[str, Any] = {}

                if node.content is not None:
                    updates.append("content = :content")
                    params["content"] = node.content

                if node.vector is not None:
                    updates.append("vector = :vector")
                    # VECTOR column expects a single literal, not a Python list bound as multi-column
                    params["vector"] = "[" + ",".join(str(float(v)) for v in node.vector) + "]"

                if node.metadata is not None:
                    updates.append("metadata = :metadata")
                    params["metadata"] = json.dumps(node.metadata)

                if updates:
                    params["vid"] = node.vector_id
                    update_sql = (
                        f"UPDATE `{self.collection_name}` SET {', '.join(updates)} WHERE id = :vid"
                    )
                    with self.client.engine.connect() as conn:
                        with conn.begin():
                            conn.execute(text(update_sql), params)

            logger.info(f"Updated {len(nodes_to_update)} documents in {self.collection_name}")
        except Exception as e:
            logger.error(f"Failed to update documents: {e}")
            raise

    async def get(self, vector_ids: str | list[str]) -> VectorNode | list[VectorNode] | None:
        """Retrieve vector nodes by their unique identifiers."""
        single_result = isinstance(vector_ids, str)
        if single_result:
            vector_ids = [vector_ids]

        if not vector_ids:
            return [] if not single_result else None

        try:
            ids_str = "', '".join(vector_ids)
            select_sql = (
                f"SELECT id, content, vector, metadata FROM `{self.collection_name}` "
                f"WHERE id IN ('{ids_str}')"
            )

            result = self.client.perform_raw_text_sql(select_sql)
            rows = result.fetchall()

            results = []
            for row in rows:
                if row:
                    results.append(
                        VectorNode(
                            vector_id=row[0],
                            content=row[1] or "",
                            vector=self._coerce_db_vector(row[2]),
                            metadata=self._coerce_db_metadata(row[3]),
                        ),
                    )

            if single_result:
                return results[0] if results else None
            return results
        except Exception as e:
            logger.error(f"Failed to get documents: {e}")
            return [] if not single_result else None

    async def list(
        self,
        filters: dict | None = None,
        limit: int | None = None,
        sort_key: str | None = None,
        reverse: bool = False,
    ) -> list[VectorNode]:
        """Retrieve a list of vector nodes matching the provided filters and limit.

        Args:
            filters: Dictionary of filter conditions to match vectors
            limit: Maximum number of vectors to return
            sort_key: Key to sort the results by (e.g., field name in metadata). None for no sorting
            reverse: If True, sort in descending order; if False, sort in ascending order
        """
        try:
            # Build select SQL
            select_sql = f"SELECT id, content, vector, metadata FROM `{self.collection_name}`"

            where_clause = self._build_filter_clause(filters)
            if where_clause:
                select_sql += f" WHERE {where_clause}"

            # Add sorting
            if sort_key:
                order = "DESC" if reverse else "ASC"
                select_sql += f" ORDER BY JSON_EXTRACT(metadata, '$.{sort_key}') {order}"

            # Add limit
            if limit:
                select_sql += f" LIMIT {limit}"

            result = self.client.perform_raw_text_sql(select_sql)
            rows = result.fetchall()

            results = []
            for row in rows:
                if row:
                    results.append(
                        VectorNode(
                            vector_id=row[0],
                            content=row[1] or "",
                            vector=self._coerce_db_vector(row[2]),
                            metadata=self._coerce_db_metadata(row[3]),
                        ),
                    )

            return results
        except Exception as e:
            logger.error(f"Failed to list documents: {e}")
            return []

    async def collection_info(self) -> dict[str, Any]:
        """Fetch metadata including record count for the collection."""
        try:
            count_sql = f"SELECT COUNT(*) FROM `{self.collection_name}`"
            result = self.client.perform_raw_text_sql(count_sql)
            row = result.fetchone()
            count = row[0] if row else 0

            return {
                "name": self.collection_name,
                "count": count,
            }
        except Exception as e:
            logger.error(f"Failed to get collection info: {e}")
            return {"name": self.collection_name, "count": 0}

    async def reset(self):
        """Purge all data by dropping and recreating the collection table."""
        logger.warning(f"Resetting collection {self.collection_name}...")
        await self.delete_collection(self.collection_name)
        await self.create_collection(self.collection_name)

    async def reset_collection(self, collection_name: str):
        """Reset collection with the given name."""
        self.collection_name = collection_name
        await self.create_collection(collection_name)
        logger.info(f"Collection reset to {collection_name}")

    async def start(self) -> None:
        """Initialize the OceanBase client and ensure the collection exists.

        Creates the collection table if it doesn't exist.
        """
        # Initialize the client
        self.client = ObVecClient(
            uri=self.uri,
            user=self.user,
            password=self.password,
            db_name=self.database,
        )

        await super().start()
        logger.info(f"OceanBase collection {self.collection_name} initialized")

    async def close(self):
        """Terminate the OceanBase client connection and release resources."""
        # Note: ObVecClient may not have a close method
        # We'll just log the closure
        self.client = None
        logger.info("OceanBase client connection closed")