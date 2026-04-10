"""OceanBase / seekdb vector store for ReMe (pyobvector)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from loguru import logger
from sqlalchemy import Column, JSON, String, text as sa_text
from sqlalchemy.dialects.mysql import LONGTEXT

from .base_vector_store import BaseVectorStore
from ..embedding import BaseEmbeddingModel
from ..schema import VectorNode

_OBVECTOR_IMPORT_ERROR: Exception | None = None

try:
    from pyobvector import IndexParams, ObVecClient, VecIndexType, VECTOR
    from pyobvector import cosine_distance, inner_product
except Exception as e:
    _OBVECTOR_IMPORT_ERROR = e
    IndexParams = None  # type: ignore[misc, assignment]
    ObVecClient = None  # type: ignore[misc, assignment]
    VecIndexType = None  # type: ignore[misc, assignment]
    VECTOR = None  # type: ignore[misc, assignment]

_COL_SELECT = "id, content, vector, metadata"


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
                    f"(JSON_EXTRACT(metadata, '{path}') >= {lo} AND " f"JSON_EXTRACT(metadata, '{path}') <= {hi})",
                )
            else:
                parts.append(
                    f"(JSON_EXTRACT(metadata, '{path}') >= '{lo}' AND " f"JSON_EXTRACT(metadata, '{path}') <= '{hi}')",
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


def _vector_node_from_db_row(row: tuple[Any, ...]) -> VectorNode:
    return VectorNode(
        vector_id=row[0],
        content=row[1] or "",
        vector=_coerce_db_vector(row[2]),
        metadata=_coerce_db_metadata(row[3]),
    )


def _normalize_nodes(nodes: VectorNode | list[VectorNode]) -> list[VectorNode]:
    return [nodes] if isinstance(nodes, VectorNode) else list(nodes)


def _sql_table(name: str) -> str:
    return f"`{name}`"


class ObVecVectorStore(BaseVectorStore):
    """OceanBase or seekdb vector store for dense vectors and kNN search.

    Args:
        index_metric: ``cosine`` or ``ip`` (inner product). Invalid values raise
            ``ValueError``; unsupported strings are not mapped to another metric.
    """

    def __init__(
        self,
        collection_name: str,
        db_path: str | Path,
        embedding_model: BaseEmbeddingModel,
        uri: str = "127.0.0.1:2881",
        user: str = "root",
        password: str = "",
        database: str = "test",
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

        key = index_metric.strip().lower()
        if key not in ("cosine", "ip"):
            raise ValueError(
                f"ObVecVectorStore index_metric must be 'cosine' or 'ip', got {index_metric!r}",
            )
        self.uri = uri
        self.user = user
        self.password = password
        self.database = database
        self.index_metric = key
        self.index_ef_search = index_ef_search

        self.client: ObVecClient | None = None
        self.embedding_model_dims = embedding_model.dimensions

    def _require_client(self) -> ObVecClient:
        if self.client is None:
            raise RuntimeError("ObVecVectorStore.start() must be called before this operation")
        return self.client

    async def list_collections(self) -> list[str]:
        client = self._require_client()
        result = client.perform_raw_text_sql(f"SHOW TABLES FROM {_sql_table(self.database)}")
        rows = result.fetchall()
        return [row[0] for row in rows if row]

    def _table_columns_for_create(self, dimensions: int) -> list[Any]:
        return [
            Column("id", String(255), primary_key=True),
            Column("content", LONGTEXT),
            Column("vector", VECTOR(dimensions)),
            Column("metadata", JSON),
        ]

    def _hnsw_index_params(self, collection_name: str) -> IndexParams:
        metric = "cosine" if self.index_metric == "cosine" else "inner_product"
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

        need_emb = [n for n in nodes if n.vector is None]
        if need_emb:
            filled = await self.get_node_embeddings(need_emb)
            by_id = {n.vector_id: n for n in filled}
            nodes_to_insert = [by_id.get(n.vector_id, n) for n in nodes]
        else:
            nodes_to_insert = nodes

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
        dist_fn = cosine_distance if self.index_metric == "cosine" else inner_product

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
            if len(row) < 4:
                raise RuntimeError(
                    "ann_search row must have id, content, metadata, distance " f"(got {len(row)} columns)",
                )
            vid, content, metadata_raw, distance = row[0], row[1], row[2], row[3]
            dist_f = float(distance)
            if self.index_metric == "cosine":
                score = max(0.0, 1.0 - dist_f / 2.0)
            else:
                score = max(0.0, dist_f)
            if score_threshold is not None and score < score_threshold:
                continue
            meta = _coerce_db_metadata(metadata_raw) if metadata_raw is not None else {}
            meta["score"] = score
            meta["_distance"] = dist_f
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
        need_emb = [n for n in nodes if n.vector is None and bool(n.content)]
        if need_emb:
            filled = await self.get_node_embeddings(need_emb)
            by_id = {n.vector_id: n for n in filled}
            nodes_to_update = [
                by_id.get(n.vector_id, n) if (n.vector is None and bool(n.content)) else n for n in nodes
            ]
        else:
            nodes_to_update = nodes

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
        ids_str = "', '".join(vector_ids)
        select_sql = f"SELECT {_COL_SELECT} FROM {_sql_table(self.collection_name)} " f"WHERE id IN ('{ids_str}')"
        result = client.perform_raw_text_sql(select_sql)
        rows = result.fetchall()
        parsed = [_vector_node_from_db_row(row) for row in rows if row]
        if single:
            return parsed[0] if parsed else None
        return parsed

    async def list(
        self,
        filters: dict | None = None,
        limit: int | None = None,
        sort_key: str | None = None,
        reverse: bool = False,
    ) -> list[VectorNode]:
        client = self._require_client()
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

    async def collection_info(self) -> dict[str, Any]:
        """Return collection name and row count."""
        client = self._require_client()
        count_sql = f"SELECT COUNT(*) FROM {_sql_table(self.collection_name)}"
        result = client.perform_raw_text_sql(count_sql)
        row = result.fetchone()
        count = row[0] if row else 0
        return {"name": self.collection_name, "count": count}

    async def reset(self):
        """Drop and recreate the current collection table."""
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
