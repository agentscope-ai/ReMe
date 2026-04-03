"""OceanBase / seekdb vector store for ReMe (pyobvector).

Dense vectors, kNN via ``ObVecClient.ann_search``, JSON metadata filters, and
helpers for coercion / metrics live in this module (pyobvector-aligned).
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from loguru import logger
from sqlalchemy import Column, JSON, String, text as sa_text
from sqlalchemy.dialects.mysql import LONGTEXT

from .base_vector_store import BaseVectorStore
from ..embedding import BaseEmbeddingModel
from ..schema import VectorNode

_OBVECTOR_IMPORT_ERROR: Exception | None = None
_DISTANCE_BY_METRIC: dict[str, Callable[..., Any]] | None = None

try:
    from pyobvector import IndexParams, ObVecClient, VecIndexType, VECTOR
    from pyobvector import cosine_distance, inner_product
except Exception as e:
    _OBVECTOR_IMPORT_ERROR = e
    IndexParams = None  # type: ignore[misc, assignment]
    ObVecClient = None  # type: ignore[misc, assignment]
    VecIndexType = None  # type: ignore[misc, assignment]
    VECTOR = None  # type: ignore[misc, assignment]
else:
    _DISTANCE_BY_METRIC = {
        "cosine": cosine_distance,
        "ip": inner_product,
    }

# ann_search with ``with_dist=True`` yields id, content, metadata, distance.
_ANN_ROW_MIN_COLUMNS = 4

_COL_SELECT = "id, content, vector, metadata"

# HNSW index metric strings for pyobvector ``IndexParam`` (metric_type / distance).
_METRIC_TO_INDEX_DISTANCE: dict[str, str] = {
    "cosine": "cosine",
    "ip": "inner_product",
}


def _is_safe_metadata_key(key: str) -> bool:
    return bool(key.replace("_", "").replace(".", "").isalnum())


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


def _build_metadata_filter_sql(filters: dict[str, Any] | None) -> str:
    if not filters:
        return ""

    parts: list[str] = []
    for key, value in filters.items():
        if not _is_safe_metadata_key(key):
            continue

        path = f"$.{key}"
        if isinstance(value, list) and len(value) == 2:
            lo, hi = value[0], value[1]
            if isinstance(lo, (int, float)) and isinstance(hi, (int, float)):
                parts.append(
                    f"(JSON_EXTRACT(metadata, '{path}') >= {lo} AND "
                    f"JSON_EXTRACT(metadata, '{path}') <= {hi})",
                )
            else:
                parts.append(
                    f"(JSON_EXTRACT(metadata, '{path}') >= '{lo}' AND "
                    f"JSON_EXTRACT(metadata, '{path}') <= '{hi}')",
                )
        elif isinstance(value, (int, float)):
            parts.append(f"JSON_EXTRACT(metadata, '{path}') = {value}")
        else:
            parts.append(f"JSON_EXTRACT(metadata, '{path}') = '{value}'")

    return " AND ".join(parts)


def _format_vector_sql_literal(vector: list[float]) -> str:
    return "[" + ",".join(str(float(v)) for v in vector) + "]"


def _normalize_embedding_for_ann(raw: Any) -> list[float]:
    if hasattr(raw, "tolist"):
        raw = raw.tolist()
    return [float(x) for x in raw]


def _distance_function(metric: str) -> Callable[..., Any]:
    if _DISTANCE_BY_METRIC is None:
        raise RuntimeError("pyobvector distance functions are unavailable")
    return _DISTANCE_BY_METRIC.get(metric.lower(), _DISTANCE_BY_METRIC["cosine"])


def _similarity_from_distance(metric: str, distance: float) -> float:
    m = metric.lower()
    if m == "cosine":
        return max(0.0, 1.0 - distance / 2.0)
    return max(0.0, float(distance))


def _vector_node_from_db_row(row: tuple[Any, ...]) -> VectorNode:
    return VectorNode(
        vector_id=row[0],
        content=row[1] or "",
        vector=_coerce_db_vector(row[2]),
        metadata=_coerce_db_metadata(row[3]),
    )


def _normalize_nodes(nodes: VectorNode | list[VectorNode]) -> list[VectorNode]:
    return [nodes] if isinstance(nodes, VectorNode) else list(nodes)


def _search_result_metadata(metadata_raw: Any, score: float, distance: Any) -> dict[str, Any]:
    meta = _coerce_db_metadata(metadata_raw) if metadata_raw is not None else {}
    meta["score"] = score
    meta["_score"] = score
    meta["_distance"] = distance
    return meta


def _sql_table(name: str) -> str:
    return f"`{name}`"


class ObVecVectorStore(BaseVectorStore):
    """OceanBase or seekdb vector store for dense vectors and kNN search."""

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
        if _OBVECTOR_IMPORT_ERROR is not None:
            raise ImportError(
                "ObVecVectorStore requires pyobvector. Install with `pip install pyobvector`",
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

    def _require_client(self) -> ObVecClient:
        if self.client is None:
            raise RuntimeError("ObVecVectorStore.start() must be called before this operation")
        return self.client

    async def _nodes_with_filled_embeddings(
        self,
        nodes: list[VectorNode],
        embed_if: Callable[[VectorNode], bool],
    ) -> list[VectorNode]:
        need = [n for n in nodes if embed_if(n)]
        if not need:
            return nodes
        filled = await self.get_node_embeddings(need)
        by_id = {n.vector_id: n for n in filled}
        return [by_id.get(n.vector_id, n) if embed_if(n) else n for n in nodes]

    async def list_collections(self) -> list[str]:
        if self.client is None:
            return []
        try:
            result = self.client.perform_raw_text_sql(f"SHOW TABLES FROM {_sql_table(self.database)}")
            rows = result.fetchall()
            return [row[0] for row in rows if row]
        except Exception as e:
            logger.warning("Failed to list collections: {}", e)
            return []

    def _table_columns_for_create(self, dimensions: int) -> list[Any]:
        return [
            Column("id", String(255), primary_key=True),
            Column("content", LONGTEXT),
            Column("vector", VECTOR(dimensions)),
            Column("metadata", JSON),
        ]

    def _hnsw_index_params(self, collection_name: str) -> IndexParams | None:
        if self.index_type != "HNSW":
            return None
        metric = _METRIC_TO_INDEX_DISTANCE.get(self.index_metric, "cosine")
        vidxs = IndexParams()
        vidxs.add_index(
            "vector",
            VecIndexType.HNSW,
            f"{collection_name}_vidx",
            metric_type=metric,
            params={"efSearch": self.index_ef_search},
        )
        return vidxs

    async def create_collection(self, collection_name: str, **kwargs):
        client = self._require_client()
        dimensions = kwargs.get("dimensions", self.embedding_model_dims)

        if client.check_table_exists(collection_name):
            logger.info("Collection {} already exists", collection_name)
            return

        columns = self._table_columns_for_create(dimensions)
        vidxs = self._hnsw_index_params(collection_name)

        client.create_table_with_index_params(
            table_name=collection_name,
            columns=columns,
            vidxs=vidxs,
        )
        logger.info("Created collection {} with dimensions={}", collection_name, dimensions)

    async def delete_collection(self, collection_name: str, **kwargs):
        client = self._require_client()
        client.drop_table_if_exist(collection_name)
        logger.info("Deleted collection {}", collection_name)

    async def copy_collection(self, collection_name: str, **kwargs):
        client = self._require_client()

        if not client.check_table_exists(self.collection_name):
            raise ValueError(f"Source collection {self.collection_name} does not exist")

        await self.create_collection(collection_name)

        try:
            source_data = await self.list(limit=None)
            if source_data:
                await self.insert(source_data, collection_name=collection_name)
            logger.info("Copied collection {} to {}", self.collection_name, collection_name)
        except Exception:
            try:
                client.drop_table_if_exist(collection_name)
            except Exception as cleanup_err:
                logger.warning("Cleanup after failed copy failed: {}", cleanup_err)
            raise

    async def insert(self, nodes: VectorNode | list[VectorNode], **kwargs):
        nodes = _normalize_nodes(nodes)
        if not nodes:
            return

        client = self._require_client()

        nodes_to_insert = await self._nodes_with_filled_embeddings(
            nodes,
            embed_if=lambda n: n.vector is None,
        )

        data = [
            {
                "id": node.vector_id,
                "content": node.content,
                "vector": node.vector if node.vector is not None else [],
                "metadata": node.metadata if node.metadata else {},
            }
            for node in nodes_to_insert
        ]
        target = kwargs.get("collection_name", self.collection_name)
        client.insert(table_name=target, data=data)
        logger.info("Inserted {} documents into {}", len(nodes_to_insert), target)

    async def search(
        self,
        query: str,
        limit: int = 5,
        filters: dict | None = None,
        **kwargs,
    ) -> list[VectorNode]:
        client = self._require_client()
        raw_vec = await self.get_embedding(query)
        query_vector = _normalize_embedding_for_ann(raw_vec)
        dist_fn = _distance_function(self.index_metric)

        filter_sql = _build_metadata_filter_sql(filters)
        where_parts = [sa_text(filter_sql)] if filter_sql else None

        results = client.ann_search(
            table_name=self.collection_name,
            vec_data=query_vector,
            vec_column_name="vector",
            distance_func=dist_fn,
            with_dist=True,
            topk=limit,
            output_column_names=["id", "content", "metadata"],
            where_clause=where_parts,
        )

        score_threshold = kwargs.get("score_threshold")
        out: list[VectorNode] = []
        for row in results:
            if len(row) < _ANN_ROW_MIN_COLUMNS:
                logger.warning(
                    "ann_search row has unexpected width: len={} (expected >= {})",
                    len(row),
                    _ANN_ROW_MIN_COLUMNS,
                )
                continue
            vid, content, metadata_raw, distance = row[0], row[1], row[2], row[3]
            score = _similarity_from_distance(self.index_metric, float(distance))
            if score_threshold is not None and score < score_threshold:
                continue
            meta = _search_result_metadata(metadata_raw, score, distance)
            out.append(
                VectorNode(
                    vector_id=vid,
                    content=content or "",
                    vector=None,
                    metadata=meta,
                ),
            )
        return out

    async def delete(self, vector_ids: str | list[str], **kwargs):
        if isinstance(vector_ids, str):
            vector_ids = [vector_ids]
        if not vector_ids:
            return
        client = self._require_client()
        client.delete(self.collection_name, ids=vector_ids)
        logger.info("Deleted {} documents from {}", len(vector_ids), self.collection_name)

    async def delete_all(self, **kwargs):
        client = self._require_client()
        client.delete(self.collection_name)
        logger.info("Deleted all documents from {}", self.collection_name)

    async def update(self, nodes: VectorNode | list[VectorNode], **kwargs):
        nodes = _normalize_nodes(nodes)
        if not nodes:
            return

        client = self._require_client()
        nodes_to_update = await self._nodes_with_filled_embeddings(
            nodes,
            embed_if=lambda n: n.vector is None and bool(n.content),
        )

        for node in nodes_to_update:
            updates: list[str] = []
            params: dict[str, Any] = {}

            if node.content is not None:
                updates.append("content = :content")
                params["content"] = node.content

            if node.vector is not None:
                updates.append("vector = :vector")
                params["vector"] = _format_vector_sql_literal(node.vector)

            if node.metadata is not None:
                updates.append("metadata = :metadata")
                params["metadata"] = json.dumps(node.metadata)

            if not updates:
                continue

            params["vid"] = node.vector_id
            update_sql = f"UPDATE {_sql_table(self.collection_name)} SET {', '.join(updates)} WHERE id = :vid"
            with client.engine.connect() as conn:
                with conn.begin():
                    conn.execute(sa_text(update_sql), params)

        logger.info("Updated {} documents in {}", len(nodes_to_update), self.collection_name)

    async def get(self, vector_ids: str | list[str]) -> VectorNode | list[VectorNode] | None:
        single = isinstance(vector_ids, str)
        if single:
            vector_ids = [vector_ids]
        if not vector_ids:
            return [] if not single else None

        client = self._require_client()
        try:
            ids_str = "', '".join(vector_ids)
            select_sql = (
                f"SELECT {_COL_SELECT} FROM {_sql_table(self.collection_name)} "
                f"WHERE id IN ('{ids_str}')"
            )
            result = client.perform_raw_text_sql(select_sql)
            rows = result.fetchall()
            parsed = [_vector_node_from_db_row(row) for row in rows if row]
            if single:
                return parsed[0] if parsed else None
            return parsed
        except Exception as e:
            logger.error("Failed to get documents: {}", e)
            return [] if not single else None

    async def list(
        self,
        filters: dict | None = None,
        limit: int | None = None,
        sort_key: str | None = None,
        reverse: bool = False,
    ) -> list[VectorNode]:
        client = self._require_client()
        try:
            select_sql = f"SELECT {_COL_SELECT} FROM {_sql_table(self.collection_name)}"
            where_clause = _build_metadata_filter_sql(filters)
            if where_clause:
                select_sql += f" WHERE {where_clause}"
            if sort_key and _is_safe_metadata_key(sort_key):
                order = "DESC" if reverse else "ASC"
                select_sql += f" ORDER BY JSON_EXTRACT(metadata, '$.{sort_key}') {order}"
            if limit is not None:
                select_sql += f" LIMIT {limit}"
            result = client.perform_raw_text_sql(select_sql)
            rows = result.fetchall()
            return [_vector_node_from_db_row(row) for row in rows if row]
        except Exception as e:
            logger.error("Failed to list documents: {}", e)
            return []

    async def collection_info(self) -> dict[str, Any]:
        client = self._require_client()
        try:
            count_sql = f"SELECT COUNT(*) FROM {_sql_table(self.collection_name)}"
            result = client.perform_raw_text_sql(count_sql)
            row = result.fetchone()
            count = row[0] if row else 0
            return {"name": self.collection_name, "count": count}
        except Exception as e:
            logger.error("Failed to get collection info: {}", e)
            return {"name": self.collection_name, "count": 0}

    async def reset(self):
        logger.warning("Resetting collection {}...", self.collection_name)
        await self.delete_collection(self.collection_name)
        await self.create_collection(self.collection_name)

    async def reset_collection(self, collection_name: str):
        self.collection_name = collection_name
        await self.create_collection(collection_name)
        logger.info("Collection reset to {}", collection_name)

    async def start(self) -> None:
        self.client = ObVecClient(
            uri=self.uri,
            user=self.user,
            password=self.password,
            db_name=self.database,
        )

        await super().start()
        logger.info("seekdb / OceanBase vector table {} ready", self.collection_name)

    async def close(self):
        self.client = None
        logger.info("ObVec client connection closed")
