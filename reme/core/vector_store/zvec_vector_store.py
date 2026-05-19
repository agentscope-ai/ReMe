"""Zvec vector store implementation for the ReMe framework."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from loguru import logger

from .base_vector_store import BaseVectorStore
from ..embedding import BaseEmbeddingModel
from ..schema import VectorNode

_ZVEC_IMPORT_ERROR: Exception | None = None

try:
    import zvec  # type: ignore[import-untyped]
    from zvec import (
        CollectionOption,
        CollectionSchema,
        DataType,
        Doc,
        FieldSchema,
        HnswIndexParam,
        InvertIndexParam,
        VectorQuery,
        VectorSchema,
    )
    from zvec.typing import MetricType
except Exception as e:
    _ZVEC_IMPORT_ERROR = e
    zvec = None  # type: ignore[assignment]


# Default vector field name used inside zvec collections
_DEFAULT_VECTOR_FIELD = "embedding"

# Default scalar content field name for storing text
_CONTENT_FIELD = "content"

# Field name for JSON-serialized metadata
_METADATA_FIELD = "metadata"

# Metadata fields promoted to top-level zvec schema columns for native filtering.
# These are the most commonly filtered keys in ReMe's memory system.
# Defining them as independent schema columns allows zvec to perform
# filtering at the database level instead of Python post-filtering.
# Format: {metadata_key: (zvec_data_type_str, has_inverted_index)}
_PROMOTED_FIELD_SPECS: dict[str, tuple[str, bool]] = {
    "memory_type": ("STRING", True),  # Inverted index for exact match filtering
    "memory_target": ("STRING", True),  # Inverted index for exact match filtering
    "author": ("STRING", False),
    "time_int": ("INT64", False),  # Numeric for range queries
}

# zvec data-type string → DataType enum mapping (populated after import)
_DATATYPE_MAP: dict[str, Any] = {}  # filled in _build_collection_schema


def _escape_zvec_string(value: str) -> str:
    """Escape a string value for use in zvec filter expressions."""
    return value.replace("'", "\\'")


def _build_zvec_filter(
    filters: dict | None,
    promoted_fields: set[str],
) -> tuple[str | None, dict | None]:
    """Split ReMe filter dict into a zvec native filter expression and remaining post-filters.

    For filter keys that correspond to promoted schema fields, native
    zvec filter expressions are generated.  Non-promoted keys are
    kept for Python post-filtering.

    Args:
        filters: ReMe-style filter dictionary.
        promoted_fields: Set of metadata keys that exist as top-level schema columns.

    Returns:
        (native_filter_expr, post_filter_dict) — either may be None.
    """
    if not filters:
        return None, None

    native_conditions: list[str] = []
    post_filters: dict = {}

    for key, value in filters.items():
        if key.startswith("$"):
            # Compound operators ($or, $and, $not) — keep for post-filtering
            post_filters[key] = value
            continue

        if key not in promoted_fields:
            # Not a promoted field — use post-filtering
            post_filters[key] = value
            continue

        # Build native filter condition for promoted fields
        field_type = _PROMOTED_FIELD_SPECS.get(key, ("STRING", False))[0]

        if isinstance(value, list) and len(value) == 2:
            # Range query: [start, end]
            if field_type == "INT64":
                native_conditions.append(f"{key} >= {value[0]} AND {key} <= {value[1]}")
            else:
                # STRING range — use >= and <= with string escaping
                native_conditions.append(
                    f"{key} >= '{_escape_zvec_string(str(value[0]))}' "
                    f"AND {key} <= '{_escape_zvec_string(str(value[1]))}'",
                )
        elif isinstance(value, bool):
            native_conditions.append(f"{key} = {str(value).upper()}")
        elif isinstance(value, (int, float)):
            native_conditions.append(f"{key} = {value}")
        elif isinstance(value, str):
            native_conditions.append(f"{key} = '{_escape_zvec_string(value)}'")
        else:
            # Unsupported type — fall back to post-filtering
            post_filters[key] = value

    native_filter = " AND ".join(native_conditions) if native_conditions else None
    return native_filter, post_filters if post_filters else None


def _metric_type_from_str(metric: str) -> Any:
    """Convert a string metric name to zvec MetricType enum value."""
    if zvec is None:
        return None
    mapping = {
        "cosine": MetricType.COSINE,
        "l2": MetricType.L2,
        "ip": MetricType.IP,
    }
    return mapping.get(metric.lower(), MetricType.COSINE)


def _build_collection_schema(
    name: str,
    dimension: int,
    metric: str = "cosine",
) -> CollectionSchema:
    """Build a zvec CollectionSchema for ReMe usage.

    The schema contains:
    - "content" (STRING, inverted index) — text content
    - "metadata" (STRING) — JSON-serialized metadata dictionary
    - Promoted metadata fields (STRING / INT64) — for native zvec filtering
    - "embedding" (VECTOR_FP32, dimension, HNSW index) — the vector field

    Promoted fields are commonly filtered metadata keys defined as top-level
    schema columns so that zvec can perform filtering natively instead of
    Python post-filtering.  The full metadata is still stored as JSON in the
    "metadata" field for complete round-trip serialization.

    zvec automatically manages the document ID (string type); we do NOT
    define an "id" field in the schema.
    """
    # Populate the DataType map on first call
    if not _DATATYPE_MAP:
        _DATATYPE_MAP.update(
            {
                "STRING": DataType.STRING,
                "INT64": DataType.INT64,
            },
        )

    distance = _metric_type_from_str(metric)

    # Base fields
    fields = [
        FieldSchema("content", DataType.STRING, nullable=True, index_param=InvertIndexParam()),
        FieldSchema("metadata", DataType.STRING, nullable=True),
    ]

    # Add promoted metadata fields as top-level schema columns
    for field_name, (type_str, has_inv_index) in _PROMOTED_FIELD_SPECS.items():
        dt = _DATATYPE_MAP[type_str]
        idx_param = InvertIndexParam() if has_inv_index else None
        fields.append(FieldSchema(field_name, dt, nullable=True, index_param=idx_param))

    return CollectionSchema(
        name=name,
        fields=fields,
        vectors=[
            VectorSchema(
                name=_DEFAULT_VECTOR_FIELD,
                data_type=DataType.VECTOR_FP32,
                dimension=dimension,
                index_param=HnswIndexParam(metric_type=distance),
            ),
        ],
    )


def _vector_node_to_doc(node: VectorNode) -> Doc:
    """Convert a ReMe VectorNode to a zvec Doc.

    Metadata is serialized as a JSON string into the "metadata" field.
    The "score" key is excluded since it is a computed value, not stored data.
    Promoted metadata fields are also extracted as top-level Doc fields
    for native zvec filtering.
    The vector is placed under the default vector field name.
    The zvec Doc id must be a string.
    """
    # Filter out computed score before serialization
    meta_to_store = {k: v for k, v in node.metadata.items() if k != "score"}

    fields: dict[str, Any] = {
        "content": node.content,
        "metadata": json.dumps(meta_to_store) if meta_to_store else "{}",
    }

    # Extract promoted metadata fields as top-level schema columns
    for field_name, (type_str, _) in _PROMOTED_FIELD_SPECS.items():
        value = meta_to_store.get(field_name)
        if value is not None:
            # Ensure correct type: INT64 fields must be int
            if type_str == "INT64" and not isinstance(value, int):
                try:
                    value = int(value)
                except (ValueError, TypeError):
                    continue
            fields[field_name] = value

    vectors: dict[str, Any] = {}
    if node.vector is not None:
        vectors[_DEFAULT_VECTOR_FIELD] = node.vector

    return Doc(id=str(node.vector_id), fields=fields, vectors=vectors)


def _doc_to_vector_node(doc: Doc, include_score: bool = False) -> VectorNode:
    """Convert a zvec Doc back to a ReMe VectorNode.

    The "metadata" field is parsed from JSON. The "content" field becomes
    the node content.  If ``include_score`` is True, the search score is
    added to the metadata dictionary.
    """
    metadata: dict[str, str | bool | int | float] = {}

    # Parse JSON metadata
    raw_metadata = doc.field("metadata")
    if raw_metadata:
        try:
            parsed = json.loads(raw_metadata)
            if isinstance(parsed, dict):
                metadata.update(parsed)
        except (json.JSONDecodeError, TypeError):
            logger.warning(f"Failed to parse metadata JSON: {raw_metadata}")

    if include_score and doc.score is not None:
        metadata["score"] = doc.score

    # Extract vector — doc.vector() returns list or empty dict
    raw_vector = doc.vector(_DEFAULT_VECTOR_FIELD)
    vector = raw_vector if isinstance(raw_vector, list) and len(raw_vector) > 0 else None

    content = doc.field("content") or ""

    return VectorNode(
        vector_id=str(doc.id),
        content=str(content),
        vector=vector,
        metadata=metadata,
    )


def _apply_filters_post(nodes: list[VectorNode], filters: dict | None) -> list[VectorNode]:
    """Apply ReMe-style filter dict as post-filtering on metadata.

    Used as a fallback for metadata keys that are NOT promoted to top-level
    schema columns (and thus cannot be filtered natively by zvec).  Promoted
    fields are handled by zvec's native ``filter`` parameter instead.

    Supports:
    - Exact match: {"field": value}
    - Range query: {"field": [start, end]}
    """
    if not filters:
        return nodes

    filtered = []
    for node in nodes:
        match = True
        for key, value in filters.items():
            if key.startswith("$"):
                # Skip compound operators for post-filtering
                continue
            node_value = node.metadata.get(key)

            # Range query: [start, end]
            if isinstance(value, list) and len(value) == 2:
                if node_value is None:
                    match = False
                    break
                try:
                    if not value[0] <= node_value <= value[1]:
                        match = False
                        break
                except TypeError:
                    match = False
                    break
            else:
                # Exact match
                if node_value != value:
                    match = False
                    break

        if match:
            filtered.append(node)

    return filtered


class ZvecVectorStore(BaseVectorStore):
    """Zvec-based vector store implementation.

    Zvec is a high-performance vector database. This adapter bridges the
    ReMe ``BaseVectorStore`` interface with zvec's Python API.

    Supports local persistent storage via ``db_path``.

    Args:
        collection_name: Name of the vector collection.
        db_path: Local storage path for persistent mode.
        embedding_model: Model used for generating vector embeddings.
        dimension: Dimensionality of the embedding vectors (default: 1024).
        distance: Distance metric — cosine / l2 / ip (default: cosine).
        **kwargs: Additional zvec-specific configuration.
    """

    def __init__(
        self,
        collection_name: str,
        db_path: str | Path,
        embedding_model: BaseEmbeddingModel,
        dimension: int = 1024,
        distance: str = "cosine",
        **kwargs: Any,
    ):
        """Initialize the Zvec vector store."""
        if _ZVEC_IMPORT_ERROR is not None:
            raise ImportError(
                "Zvec requires extra dependencies. Install with `pip install zvec`",
            ) from _ZVEC_IMPORT_ERROR

        super().__init__(
            collection_name=collection_name,
            db_path=db_path,
            embedding_model=embedding_model,
            **kwargs,
        )

        self.dimension = dimension
        self.distance = distance
        self._collection = None
        self._initialized = False
        # Set of promoted field names that exist in the current collection's schema.
        # Populated during start() by inspecting the schema.  Only fields present
        # in the schema can use native zvec filtering; the rest fall back to
        # Python post-filtering.
        self._promoted_fields_in_schema: set[str] = set()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Initialize the Zvec engine and open the collection.

        Calls ``zvec.init()`` once, then tries to ``zvec.open()`` an existing
        collection or ``zvec.create_and_open()`` a new one.
        After opening, detects which promoted fields exist in the schema
        and attempts to add missing numeric fields via ``add_column``.
        """
        if not self._initialized:
            try:
                zvec.init()
            except RuntimeError:
                # Already initialized — safe to ignore
                pass
            self._initialized = True

        self.db_path.mkdir(parents=True, exist_ok=True)
        collection_path = str(self.db_path / self.collection_name)

        option = CollectionOption(read_only=False, enable_mmap=True)

        try:
            # Try opening an existing collection first
            self._collection = zvec.open(collection_path, option)
            logger.info(f"Opened existing Zvec collection at {collection_path}")
        except Exception:
            # Collection doesn't exist — create it
            schema = _build_collection_schema(
                name=self.collection_name,
                dimension=self.dimension,
                metric=self.distance,
            )
            self._collection = zvec.create_and_open(
                path=collection_path,
                schema=schema,
                option=option,
            )
            logger.info(f"Created new Zvec collection at {collection_path}")

        # Detect which promoted fields exist in the current schema
        self._detect_promoted_fields()

        # Try to add missing numeric promoted fields to existing collections
        # (zvec's add_column only supports numeric types: INT64, FLOAT, etc.)
        self._ensure_numeric_promoted_columns()

    async def close(self) -> None:
        """Flush pending writes and release the collection handle."""
        if self._collection is not None:
            try:
                self._collection.flush()
            except Exception as e:
                logger.warning(f"Failed to flush collection on close: {e}")
            self._collection = None
        logger.info(f"Zvec vector store for collection {self.collection_name} closed")

    # ------------------------------------------------------------------
    # Collection management
    # ------------------------------------------------------------------

    async def list_collections(self) -> list[str]:
        """Retrieve a list of collection names in the db_path directory.

        Zvec doesn't have a global ``list_collections`` API; we scan the
        db_path directory for zvec collection folders.
        """
        if not self.db_path.exists():
            return []
        collections = []
        for child in self.db_path.iterdir():
            if child.is_dir():
                collections.append(child.name)
        return collections

    async def create_collection(self, collection_name: str, **kwargs) -> None:
        """Create a new collection with the specified name and distance metric."""
        if not self._initialized:
            try:
                zvec.init()
            except RuntimeError:
                pass
            self._initialized = True

        self.db_path.mkdir(parents=True, exist_ok=True)
        collection_path = str(self.db_path / collection_name)

        dimension = kwargs.get("dimension", self.dimension)
        metric = kwargs.get("distance_metric", self.distance)

        schema = _build_collection_schema(
            name=collection_name,
            dimension=dimension,
            metric=metric,
        )
        option = CollectionOption(read_only=False, enable_mmap=True)

        collection = zvec.create_and_open(path=collection_path, schema=schema, option=option)
        if collection_name == self.collection_name:
            self._collection = collection
        logger.info(f"Created collection `{collection_name}`")

    async def delete_collection(self, collection_name: str, **kwargs) -> None:
        """Permanently remove a collection from disk."""
        # If it's the active collection, destroy it via zvec API
        if self._collection is not None and collection_name == self.collection_name:
            try:
                self._collection.destroy()
                self._collection = None
                deleted = True
            except Exception as _e:
                logger.warning(f"Failed to destroy collection {collection_name}: {_e}")
                deleted = False
        else:
            # For non-active collections, remove the directory
            collection_path = self.db_path / collection_name
            if collection_path.exists():
                import shutil

                shutil.rmtree(collection_path, ignore_errors=True)
                deleted = True
            else:
                deleted = False

        logger.info(f"Deleted collection {collection_name}: {deleted}")

    async def copy_collection(self, collection_name: str, **kwargs) -> None:
        """Duplicate the current collection to a new one with the given name.

        Uses ``shutil.copytree`` to directly copy the collection directory on
        disk, which is both faster and complete — it avoids the topk limit of
        ``list()`` (max 1024 docs) that would cause data loss for large
        collections.

        The source collection is flushed before copying to ensure all
        pending writes are persisted to disk.
        """
        import shutil

        # Flush source collection so all data is on disk
        if self._collection is not None:
            self._collection.flush()

        src_path = self.db_path / self.collection_name
        dst_path = self.db_path / collection_name

        if not src_path.exists():
            logger.warning(f"Source collection directory not found: {src_path}")
            return

        if dst_path.exists():
            logger.warning(f"Target collection already exists: {dst_path}, removing it first")
            shutil.rmtree(dst_path, ignore_errors=True)

        shutil.copytree(src_path, dst_path)
        logger.info(
            f"Copied collection {self.collection_name} to {collection_name} "
            f"(directory copy: {src_path} -> {dst_path})",
        )

    # ------------------------------------------------------------------
    # CRUD operations
    # ------------------------------------------------------------------

    async def insert(self, nodes: VectorNode | list[VectorNode], **kwargs) -> None:
        """Add one or more vector nodes into the current collection.

        Automatically generates embeddings for nodes that lack vectors.
        """
        if isinstance(nodes, VectorNode):
            nodes = [nodes]
        if not nodes:
            return

        # Batch generate embeddings for nodes that need them
        nodes_without_vectors = [n for n in nodes if n.vector is None]
        if nodes_without_vectors:
            nodes_with_vectors = await self.get_node_embeddings(nodes_without_vectors)
            vector_map = {n.vector_id: n for n in nodes_with_vectors}
            nodes_to_insert = [vector_map.get(n.vector_id, n) if n.vector is None else n for n in nodes]
        else:
            nodes_to_insert = nodes

        batch_size = kwargs.get("batch_size", 100)

        for i in range(0, len(nodes_to_insert), batch_size):
            batch = nodes_to_insert[i : i + batch_size]
            docs = [_vector_node_to_doc(n) for n in batch]
            self._collection.insert(docs)

        logger.info(f"Inserted {len(nodes_to_insert)} nodes into {self.collection_name}")

    async def search(
        self,
        query: str,
        limit: int = 5,
        filters: dict | None = None,
        **kwargs,
    ) -> list[VectorNode]:
        """Find the most similar vector nodes based on a text query.

        Uses zvec's ``query()`` method with a ``VectorQuery`` built from the
        embedding of the query text.  Promoted metadata fields are filtered
        natively via zvec's ``filter`` parameter; remaining filters are
        applied as post-filtering in Python.
        """
        query_vector = await self.get_embedding(query)

        vq = VectorQuery(
            field_name=_DEFAULT_VECTOR_FIELD,
            vector=query_vector,
        )
        include_vector = kwargs.get("include_embeddings", False)

        # Split filters: native zvec filter vs Python post-filter
        native_filter, post_filters = _build_zvec_filter(filters, self._promoted_fields_in_schema)

        # Over-fetch to compensate for post-filtering
        _ZVEC_MAX_TOPK = 1024
        # When post-filters remain, we need to fetch more results because
        # many may be filtered out. Use the maximum allowed to minimize misses.
        fetch_limit = _ZVEC_MAX_TOPK if post_filters else min(limit, _ZVEC_MAX_TOPK)

        results = self._collection.query(
            vectors=vq,
            topk=fetch_limit,
            filter=native_filter,
            include_vector=include_vector,
        )

        nodes = [_doc_to_vector_node(doc, include_score=True) for doc in results]

        # Post-filter on non-promoted metadata fields
        nodes = _apply_filters_post(nodes, post_filters)

        score_threshold = kwargs.get("score_threshold")
        if score_threshold is not None:
            nodes = [n for n in nodes if n.metadata.get("score", 0) >= score_threshold]

        return nodes[:limit]

    async def delete(self, vector_ids: str | list[str], **kwargs) -> None:
        """Remove specific vectors from the collection using their identifiers."""
        if isinstance(vector_ids, str):
            vector_ids = [vector_ids]
        if not vector_ids:
            return

        self._collection.delete(vector_ids)
        logger.info(f"Deleted {len(vector_ids)} nodes from {self.collection_name}")

    async def delete_all(self, **kwargs) -> None:
        """Remove all vectors from the collection.

        Uses zvec's ``delete_by_filter`` with a condition that matches all
        documents (content is not empty), or falls back to query + delete
        in batches (zvec topk max is 1024).
        """
        stats = self._collection.stats
        count = stats.doc_count if stats else 0
        if count > 0:
            try:
                # Use delete_by_filter for efficiency
                self._collection.delete_by_filter("content!=''")
            except Exception:
                # Fallback: fetch all IDs in batches then delete
                _ZVEC_MAX_TOPK = 1024
                remaining = count
                while remaining > 0:
                    all_docs = self._collection.query(topk=min(remaining, _ZVEC_MAX_TOPK), include_vector=False)
                    if not all_docs:
                        break
                    ids = [doc.id for doc in all_docs]
                    self._collection.delete(ids)
                    remaining -= len(ids)
        logger.info(f"Deleted all {count} nodes from {self.collection_name}")

    async def update(self, nodes: VectorNode | list[VectorNode], **kwargs) -> None:
        """Update existing vectors using zvec's ``upsert``.

        Automatically regenerates embeddings for nodes whose content changed
        but lack an updated vector.
        """
        if isinstance(nodes, VectorNode):
            nodes = [nodes]
        if not nodes:
            return

        # Batch generate embeddings for nodes that need them
        nodes_without_vectors = [n for n in nodes if n.vector is None and n.content]
        if nodes_without_vectors:
            nodes_with_vectors = await self.get_node_embeddings(nodes_without_vectors)
            vector_map = {n.vector_id: n for n in nodes_with_vectors}
            nodes_to_update = [vector_map.get(n.vector_id, n) if n.vector is None and n.content else n for n in nodes]
        else:
            nodes_to_update = nodes

        docs = [_vector_node_to_doc(n) for n in nodes_to_update]
        self._collection.upsert(docs)
        logger.info(f"Updated {len(nodes_to_update)} nodes in {self.collection_name}")

    async def get(self, vector_ids: str | list[str]) -> VectorNode | list[VectorNode]:
        """Fetch specific vector nodes from the collection by their IDs."""
        is_single = isinstance(vector_ids, str)
        ids = [vector_ids] if is_single else vector_ids

        result_dict = self._collection.fetch(ids)
        nodes = [_doc_to_vector_node(doc) for doc in result_dict.values()]
        return nodes[0] if is_single and nodes else (nodes if not is_single else None)

    async def list(
        self,
        filters: dict | None = None,
        limit: int | None = None,
        sort_key: str | None = None,
        reverse: bool = True,
    ) -> list[VectorNode]:
        """Retrieve vectors matching optional metadata filters.

        Uses zvec's ``query()`` without a vector query to list all documents.
        Promoted metadata fields are filtered natively via zvec's ``filter``
        parameter; remaining filters are applied as post-filtering in Python.

        Args:
            filters: Dictionary of filter conditions to match vectors.
            limit: Maximum number of vectors to return.
            sort_key: Key to sort the results by (in metadata).
            reverse: If True, sort in descending order; otherwise ascending.
        """
        # Split filters: native zvec filter vs Python post-filter
        native_filter, post_filters = _build_zvec_filter(filters, self._promoted_fields_in_schema)

        # Determine fetch limit — zvec max topk is 1024 (will be lifted to 100,000 in zvec v0.3.2+)
        _ZVEC_MAX_TOPK = 1024
        fetch_limit = min(limit or _ZVEC_MAX_TOPK, _ZVEC_MAX_TOPK)
        if sort_key or post_filters:
            fetch_limit = _ZVEC_MAX_TOPK  # fetch max and sort/filter in Python

        results = self._collection.query(
            topk=fetch_limit,
            filter=native_filter,
            include_vector=True,
        )

        nodes = [_doc_to_vector_node(doc) for doc in results]

        # Post-filter on non-promoted metadata fields
        nodes = _apply_filters_post(nodes, post_filters)

        # Apply sorting if sort_key is provided
        if sort_key:

            def _sort_key_func(node: VectorNode):
                value = node.metadata.get(sort_key)
                if value is None:
                    return float("-inf") if not reverse else float("inf")
                return value

            nodes.sort(key=_sort_key_func, reverse=reverse)

        if limit is not None:
            nodes = nodes[:limit]

        return nodes

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _detect_promoted_fields(self) -> None:
        """Detect which promoted fields exist in the current collection's schema.

        Compares the set of promoted field names against the actual schema
        and populates ``_promoted_fields_in_schema`` accordingly.  Only fields
        present in the schema can use native zvec filtering.
        """
        if self._collection is None:
            return

        try:
            schema = self._collection.schema
            existing_fields = {f.name for f in schema.fields} if schema.fields else set()
        except Exception as e:
            logger.warning(f"Failed to read collection schema: {e}")
            existing_fields = set()

        self._promoted_fields_in_schema = set(_PROMOTED_FIELD_SPECS.keys()) & existing_fields

        missing = set(_PROMOTED_FIELD_SPECS.keys()) - existing_fields
        if missing:
            logger.info(
                f"Promoted fields not in schema (will use post-filtering): {missing}",
            )

    def _ensure_numeric_promoted_columns(self) -> None:
        """Add missing numeric promoted fields to existing collections.

        zvec's ``add_column`` only supports numeric types (INT64, FLOAT, etc.).
        STRING fields cannot be added via ``add_column`` and must be defined
        at collection creation time.  For those, we fall back to post-filtering.
        """
        if self._collection is None:
            return

        missing = set(_PROMOTED_FIELD_SPECS.keys()) - self._promoted_fields_in_schema
        if not missing:
            return

        # Populate the DataType map if needed
        if not _DATATYPE_MAP:
            _DATATYPE_MAP.update(
                {
                    "STRING": DataType.STRING,
                    "INT64": DataType.INT64,
                },
            )

        for field_name in missing:
            type_str, _ = _PROMOTED_FIELD_SPECS[field_name]
            # Only numeric types can be added via add_column
            if type_str not in ("INT64", "INT32", "FLOAT", "DOUBLE"):
                continue
            try:
                dt = _DATATYPE_MAP[type_str]
                self._collection.add_column(FieldSchema(field_name, dt, nullable=True))
                self._promoted_fields_in_schema.add(field_name)
                logger.info(f"Added promoted column '{field_name}' to existing collection")
            except Exception as e:
                logger.warning(f"Failed to add column '{field_name}': {e}")

    async def count(self) -> int:
        """Return the total number of documents in the current collection."""
        stats = self._collection.stats
        return stats.doc_count if stats else 0

    async def reset(self):
        """Reset the current collection by destroying and recreating it."""
        logger.warning(f"Resetting collection {self.collection_name}...")
        await self.delete_collection(self.collection_name)
        await self.create_collection(self.collection_name)
        logger.info(f"Collection {self.collection_name} has been reset")
