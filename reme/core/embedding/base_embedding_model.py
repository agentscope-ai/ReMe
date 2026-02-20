"""Base embedding model interface for ReMe.

Defines the abstract base class and standard API for all embedding model implementations.
"""

import asyncio
import hashlib
import json
import os
import time
from abc import ABC
from collections import OrderedDict
from pathlib import Path

from loguru import logger

from ..schema import VectorNode
from ..schema.memory_chunk import MemoryChunk


class BaseEmbeddingModel(ABC):
    """Abstract base class for embedding model implementations.

    Provides a standard interface for text-to-vector generation with
    built-in batching, retry logic, and error handling.
    """

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model_name: str = "",
        dimensions: int | None = 1024,
        max_batch_size: int = 10,
        max_retries: int = 3,
        raise_exception: bool = True,
        max_input_length: int = 8192,
        cache_dir: str | Path = ".reme",
        max_cache_size: int = 2000,
        **kwargs,
    ):
        """Initialize model configuration and parameters.

        Args:
            api_key: API key for the embedding service
            base_url: Base URL for the embedding service
            model_name: Name of the embedding model
            dimensions: Vector dimensions of the embeddings
            max_batch_size: Maximum batch size for embedding requests
            max_retries: Maximum number of retry attempts on failure
            raise_exception: Whether to raise exceptions on failure
            max_input_length: Maximum input text length
            max_cache_size: Maximum number of embeddings to cache in memory (LRU)
            **kwargs: Additional model-specific parameters
        """
        self._api_key: str = api_key
        self._base_url: str = base_url
        self.model_name = model_name
        self.dimensions = dimensions
        self.max_batch_size = max_batch_size
        self.max_retries = max_retries
        self.raise_exception = raise_exception
        self.max_input_length = max_input_length
        self.cache_dir = cache_dir
        self.max_cache_size = max_cache_size
        self.kwargs = kwargs

        # Initialize LRU cache for embeddings
        self._embedding_cache: OrderedDict[str, list[float]] = OrderedDict()
        self._cache_hits = 0
        self._cache_misses = 0

        self.cache_path: Path = Path(self.cache_dir)
        self.cache_path.mkdir(parents=True, exist_ok=True)

        # Load cache from disk if available
        self._load_cache()

    @property
    def api_key(self) -> str | None:
        """Get API key from environment variable."""
        return os.getenv("REME_EMBEDDING_API_KEY") or self._api_key

    @property
    def base_url(self) -> str | None:
        """Get base URL from environment variable."""
        return os.getenv("REME_EMBEDDING_BASE_URL") or self._base_url

    def _truncate_text(self, text: str) -> str:
        """Truncate text to max_input_length if it exceeds the limit."""
        if len(text) > self.max_input_length:
            logger.warning(
                f"Text length {len(text)} exceeds max_input_length {self.max_input_length}, truncating",
            )
            return text[: self.max_input_length]
        return text

    def _truncate_texts(self, texts: list[str]) -> list[str]:
        """Truncate a list of texts to max_input_length."""
        return [self._truncate_text(text) for text in texts]

    def _get_cache_key(self, text: str) -> str:
        """Generate a cache key by hashing text + model_name + dimensions.

        This ensures that the same text produces different cache keys when
        using different models or dimensions.

        Args:
            text: Input text to hash

        Returns:
            SHA256 hash combining text, model name, and dimensions
        """
        # Combine text, model_name, and dimensions to create unique cache key
        cache_string = f"{text}|{self.model_name}|{self.dimensions}"
        return hashlib.sha256(cache_string.encode("utf-8")).hexdigest()

    def _get_cache_file_path(self) -> Path:
        """Get the path to the cache file.

        Returns:
            Path to the embedding cache JSONL file
        """
        return self.cache_path / "embedding_cache.jsonl"

    def _load_cache(self) -> None:
        """Load embedding cache from disk (JSONL format).

        Each line in the JSONL file contains a JSON object with:
        - key: the cache key (SHA256 hash)
        - embedding: the embedding vector (list of floats)

        Loads in reverse order (newest first) to prioritize recent embeddings
        when max_cache_size is smaller than the file content.
        """
        cache_file = self._get_cache_file_path()
        if not cache_file.exists():
            logger.info(f"No cache file found at {cache_file}, starting with empty cache")
            return

        try:
            # Read all lines first (to load in reverse order)
            with open(cache_file, "r", encoding="utf-8") as f:
                lines = f.readlines()

            loaded_count = 0
            # Load in reverse order (newest entries first)
            for _, line in enumerate(reversed(lines), 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    cache_key = data.get("key")
                    embedding = data.get("embedding")

                    if cache_key and embedding:
                        # Skip if already loaded (keep the newest)
                        if cache_key in self._embedding_cache:
                            continue

                        # Respect max_cache_size during loading
                        if len(self._embedding_cache) >= self.max_cache_size:
                            logger.info(
                                f"Cache size limit reached ({self.max_cache_size}), "
                                f"loaded {loaded_count} newest entries",
                            )
                            break
                        self._embedding_cache[cache_key] = embedding
                        loaded_count += 1
                except json.JSONDecodeError as e:
                    logger.warning(f"Failed to parse line in cache file: {e}")
                    continue

            logger.info(f"Loaded {loaded_count} embeddings from cache file: {cache_file}")
        except Exception as e:
            logger.error(f"Failed to load cache from {cache_file}: {e}")

    def _save_cache(self) -> None:
        """Save embedding cache to disk (JSONL format).

        Each line contains a JSON object with the cache key and embedding vector.
        Only saves if cache is non-empty.
        """
        logger.info(f"Attempting to save cache, current size: {len(self._embedding_cache)}")
        if not self._embedding_cache:
            logger.info("Cache is empty, skipping save")
            return

        cache_file = self._get_cache_file_path()
        try:
            with open(cache_file, "w", encoding="utf-8") as f:
                for cache_key, embedding in self._embedding_cache.items():
                    cache_entry = {"key": cache_key, "embedding": embedding}
                    f.write(json.dumps(cache_entry, ensure_ascii=False) + "\n")

            logger.info(f"Saved {len(self._embedding_cache)} embeddings to cache file: {cache_file}")
        except Exception as e:
            logger.error(f"Failed to save cache to {cache_file}: {e}")

    def _get_from_cache(self, text: str) -> list[float] | None:
        """Retrieve embedding from cache if it exists.

        Args:
            text: Input text to look up

        Returns:
            Cached embedding vector or None if not found
        """
        cache_key = self._get_cache_key(text)
        if cache_key in self._embedding_cache:
            # Move to end (most recently used)
            self._embedding_cache.move_to_end(cache_key)
            self._cache_hits += 1
            text_preview = text[:50] + "..." if len(text) > 50 else text
            logger.info(
                f"Cache hit for text: '{text_preview}' (hits: {self._cache_hits}, misses: {self._cache_misses})",
            )
            return self._embedding_cache[cache_key]
        self._cache_misses += 1
        return None

    def _put_to_cache(self, text: str, embedding: list[float]) -> None:
        """Store embedding in cache with LRU eviction.

        Args:
            text: Input text used as cache key
            embedding: Embedding vector to cache
        """
        if self.max_cache_size <= 0:
            return

        cache_key = self._get_cache_key(text)

        # Remove oldest entry if cache is full
        if len(self._embedding_cache) >= self.max_cache_size and cache_key not in self._embedding_cache:
            self._embedding_cache.popitem(last=False)

        self._embedding_cache[cache_key] = embedding
        self._embedding_cache.move_to_end(cache_key)

    def get_cache_stats(self) -> dict[str, int]:
        """Get cache statistics.

        Returns:
            Dictionary with cache size, hits, misses, and hit rate
        """
        total_requests = self._cache_hits + self._cache_misses
        hit_rate = self._cache_hits / total_requests if total_requests > 0 else 0.0
        return {
            "cache_size": len(self._embedding_cache),
            "max_cache_size": self.max_cache_size,
            "cache_hits": self._cache_hits,
            "cache_misses": self._cache_misses,
            "hit_rate": hit_rate,
        }

    def clear_cache(self) -> None:
        """Clear the embedding cache and reset statistics."""
        self._embedding_cache.clear()
        self._cache_hits = 0
        self._cache_misses = 0

    async def _get_embeddings(self, input_text: list[str], **kwargs) -> list[list[float]]:
        """Internal async implementation for calling the embedding API with batch input."""

    def _get_embeddings_sync(self, input_text: list[str], **kwargs) -> list[list[float]]:
        """Internal synchronous implementation for calling the embedding API with batch input."""

    async def get_embedding(self, input_text: str, **kwargs) -> list[float]:
        """Async get embedding for a single text with exponential backoff retries."""
        truncated_text = self._truncate_text(input_text)

        # Check cache first
        cached_embedding = self._get_from_cache(truncated_text)
        if cached_embedding is not None:
            return cached_embedding

        # Cache miss - compute embedding
        for i in range(self.max_retries):
            try:
                result = await self._get_embeddings([truncated_text], **kwargs)
                embedding = result[0]
                # Store in cache
                self._put_to_cache(truncated_text, embedding)
                return embedding
            except Exception as e:
                logger.error(f"Model {self.model_name} failed: {e}")
                if i == self.max_retries - 1:
                    if self.raise_exception:
                        raise
                    return []
                await asyncio.sleep(i + 1)
        return []

    async def get_embeddings(self, input_text: list[str], **kwargs) -> list[list[float]]:
        """Async get embeddings with automatic batching and exponential backoff retries."""
        # Truncate all input texts first
        truncated_texts = self._truncate_texts(input_text)

        # Check cache for each text and separate cached vs uncached
        results: list[list[float] | None] = [None] * len(truncated_texts)
        texts_to_compute: list[tuple[int, str]] = []  # (original_index, text)

        for idx, text in enumerate(truncated_texts):
            cached = self._get_from_cache(text)
            if cached is not None:
                results[idx] = cached
            else:
                texts_to_compute.append((idx, text))

        # If all texts were cached, return early
        if not texts_to_compute:
            return [r for r in results if r is not None]

        # Compute embeddings for uncached texts in batches
        uncached_texts = [text for _, text in texts_to_compute]
        for i in range(0, len(uncached_texts), self.max_batch_size):
            batch_texts = uncached_texts[i : i + self.max_batch_size]
            batch_indices = [idx for idx, _ in texts_to_compute[i : i + self.max_batch_size]]

            # Process each batch with retry logic
            for retry in range(self.max_retries):
                try:
                    batch_embeddings = await self._get_embeddings(batch_texts, **kwargs)
                    if batch_embeddings:
                        # Store results and cache them
                        for orig_idx, text, embedding in zip(batch_indices, batch_texts, batch_embeddings):
                            results[orig_idx] = embedding
                            self._put_to_cache(text, embedding)
                    break
                except Exception as e:
                    logger.error(f"Model {self.model_name} batch failed: {e}")
                    if retry == self.max_retries - 1:
                        if self.raise_exception:
                            raise
                    else:
                        await asyncio.sleep(retry + 1)

        return [r for r in results if r is not None]

    def get_embedding_sync(self, input_text: str, **kwargs) -> list[float]:
        """Synchronous get embedding for a single text with retry logic."""
        truncated_text = self._truncate_text(input_text)

        # Check cache first
        cached_embedding = self._get_from_cache(truncated_text)
        if cached_embedding is not None:
            return cached_embedding

        # Cache miss - compute embedding
        for i in range(self.max_retries):
            try:
                result = self._get_embeddings_sync([truncated_text], **kwargs)
                embedding = result[0]
                # Store in cache
                self._put_to_cache(truncated_text, embedding)
                return embedding
            except Exception as exc:
                logger.error(f"Model {self.model_name} failed: {exc}")
                if i == self.max_retries - 1:
                    if self.raise_exception:
                        raise
                    return []
                time.sleep(i + 1)
        return []

    def get_embeddings_sync(self, input_text: list[str], **kwargs) -> list[list[float]]:
        """Synchronous get embeddings with automatic batching and retry logic."""
        # Truncate all input texts first
        truncated_texts = self._truncate_texts(input_text)

        # Check cache for each text and separate cached vs uncached
        results: list[list[float] | None] = [None] * len(truncated_texts)
        texts_to_compute: list[tuple[int, str]] = []  # (original_index, text)

        for idx, text in enumerate(truncated_texts):
            cached = self._get_from_cache(text)
            if cached is not None:
                results[idx] = cached
            else:
                texts_to_compute.append((idx, text))

        # If all texts were cached, return early
        if not texts_to_compute:
            return [r for r in results if r is not None]

        # Compute embeddings for uncached texts in batches
        uncached_texts = [text for _, text in texts_to_compute]
        for i in range(0, len(uncached_texts), self.max_batch_size):
            batch_texts = uncached_texts[i : i + self.max_batch_size]
            batch_indices = [idx for idx, _ in texts_to_compute[i : i + self.max_batch_size]]

            # Process each batch with retry logic
            for retry in range(self.max_retries):
                try:
                    batch_embeddings = self._get_embeddings_sync(batch_texts, **kwargs)
                    if batch_embeddings:
                        # Store results and cache them
                        for orig_idx, text, embedding in zip(batch_indices, batch_texts, batch_embeddings):
                            results[orig_idx] = embedding
                            self._put_to_cache(text, embedding)
                    break
                except Exception as exc:
                    logger.error(f"Model {self.model_name} batch failed: {exc}")
                    if retry == self.max_retries - 1:
                        if self.raise_exception:
                            raise
                    else:
                        time.sleep(retry + 1)

        return [r for r in results if r is not None]

    async def get_node_embedding(self, node: VectorNode, **kwargs) -> VectorNode:
        """Async generate and populate vector field for a single VectorNode object."""
        node.vector = await self.get_embedding(node.content, **kwargs)
        return node

    async def get_node_embeddings(self, nodes: list[VectorNode], **kwargs) -> list[VectorNode]:
        """Async generate and populate vector fields for a batch of VectorNode objects."""
        contents = [node.content for node in nodes]
        embeddings: list[list[float]] = await self.get_embeddings(contents, **kwargs)

        if len(embeddings) == len(nodes):
            for node, vec in zip(nodes, embeddings):
                node.vector = vec
        else:
            logger.warning(f"Mismatch: got {len(embeddings)} vectors for {len(nodes)} nodes")
        return nodes

    def get_node_embedding_sync(self, node: VectorNode, **kwargs) -> VectorNode:
        """Synchronously generate and populate vector field for a single VectorNode object."""
        node.vector = self.get_embedding_sync(node.content, **kwargs)
        return node

    def get_node_embeddings_sync(self, nodes: list[VectorNode], **kwargs) -> list[VectorNode]:
        """Synchronously generate and populate vector fields for a batch of VectorNode objects."""
        contents = [node.content for node in nodes]
        embeddings: list[list[float]] = self.get_embeddings_sync(contents, **kwargs)

        if len(embeddings) == len(nodes):
            for node, vec in zip(nodes, embeddings):
                node.vector = vec
        else:
            logger.warning(f"Mismatch: got {len(embeddings)} vectors for {len(nodes)} nodes")
        return nodes

    async def get_chunk_embedding(self, chunk: MemoryChunk, **kwargs) -> MemoryChunk:
        """Async generate and populate embedding field for a single MemoryChunk object.

        Args:
            chunk: MemoryChunk object containing text to embed
            **kwargs: Additional arguments passed to the embedding model

        Returns:
            The same MemoryChunk object with populated embedding field
        """
        chunk.embedding = await self.get_embedding(chunk.text, **kwargs)
        return chunk

    async def get_chunk_embeddings(self, chunks: list[MemoryChunk], **kwargs) -> list[MemoryChunk]:
        """Async generate and populate embedding fields for a batch of MemoryChunk objects.

        Args:
            chunks: List of MemoryChunk objects containing text to embed
            **kwargs: Additional arguments passed to the embedding model

        Returns:
            The same list of MemoryChunk objects with populated embedding fields
        """
        texts = [chunk.text for chunk in chunks]
        embeddings: list[list[float]] = await self.get_embeddings(texts, **kwargs)

        if len(embeddings) == len(chunks):
            for chunk, vec in zip(chunks, embeddings):
                chunk.embedding = vec
        else:
            logger.warning(f"Mismatch: got {len(embeddings)} vectors for {len(chunks)} chunks")
        return chunks

    def get_chunk_embedding_sync(self, chunk: MemoryChunk, **kwargs) -> MemoryChunk:
        """Synchronously generate and populate embedding field for a single MemoryChunk object.

        Args:
            chunk: MemoryChunk object containing text to embed
            **kwargs: Additional arguments passed to the embedding model

        Returns:
            The same MemoryChunk object with populated embedding field
        """
        chunk.embedding = self.get_embedding_sync(chunk.text, **kwargs)
        return chunk

    def get_chunk_embeddings_sync(self, chunks: list[MemoryChunk], **kwargs) -> list[MemoryChunk]:
        """Synchronously generate embeddings for a batch of MemoryChunk objects.

        Args:
            chunks: List of MemoryChunk objects containing text to embed
            **kwargs: Additional arguments passed to the embedding model

        Returns:
            The same list of MemoryChunk objects with populated embedding fields
        """
        texts = [chunk.text for chunk in chunks]
        embeddings: list[list[float]] = self.get_embeddings_sync(texts, **kwargs)

        if len(embeddings) == len(chunks):
            for chunk, vec in zip(chunks, embeddings):
                chunk.embedding = vec
        else:
            logger.warning(f"Mismatch: got {len(embeddings)} vectors for {len(chunks)} chunks")
        return chunks

    def close_sync(self):
        """Synchronously release resources and close connections."""
        self._save_cache()

    async def close(self):
        """Asynchronously release resources and close connections."""
        self._save_cache()
