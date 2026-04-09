"""Base embedding model interface for ReMe.

Defines the abstract base class and standard API for all embedding model implementations.
"""

import asyncio
import hashlib
import json
import time
from abc import abstractmethod
from collections import OrderedDict
from pathlib import Path

from ..base_component import BaseComponent
from ...schema import BaseNode


class BaseEmbeddingModel(BaseComponent):
    """Abstract base class for embedding model implementations.

    Provides a standard interface for text-to-vector generation with
    built-in batching, retry logic, and error handling.
    """

    def __init__(
            self,
            api_key: str | None = None,
            base_url: str | None = None,
            model_name: str = "",
            dimensions: int = 1024,
            use_dimensions: bool = False,
            max_batch_size: int = 10,
            max_retries: int = 3,
            raise_exception: bool = True,
            max_input_length: int = 8192,
            cache_dir: str | Path = ".reme",
            max_cache_size: int = 2000,
            enable_cache: bool = True,
            encoding: str = "utf-8",
            **kwargs,
    ):
        """Initialize model configuration and parameters.

        Args:
            api_key: API key for the embedding service
            base_url: Base URL for the embedding service
            model_name: Name of the embedding model
            dimensions: Vector dimensions of the embeddings
            use_dimensions: Whether to pass dimensions parameter to API (some APIs don't support it)
            max_batch_size: Maximum batch size for embedding requests
            max_retries: Maximum number of retry attempts on failure
            raise_exception: Whether to raise exceptions on failure
            max_input_length: Maximum input text length
            max_cache_size: Maximum number of embeddings to cache in memory (LRU)
            enable_cache: Whether to enable embedding cache
            encoding: Text encoding for cache file operations
            **kwargs: Additional model-specific parameters
        """
        super().__init__(**kwargs)
        self.api_key: str | None = api_key
        self.base_url: str | None = base_url
        self.model_name = model_name
        self.dimensions = dimensions
        self.use_dimensions = use_dimensions
        self.max_batch_size = max_batch_size
        self.max_retries = max_retries
        self.raise_exception = raise_exception
        self.max_input_length = max_input_length
        self.cache_dir = cache_dir
        self.max_cache_size = max_cache_size
        self.enable_cache = enable_cache
        self.encoding = encoding

        self._embedding_cache: OrderedDict[str, list[float]] = OrderedDict()
        self._cache_hits = 0
        self._cache_misses = 0

        self.cache_path: Path = Path(self.cache_dir)

    def _truncate_text(self, text: str) -> str:
        return text[: self.max_input_length] if len(text) > self.max_input_length else text

    def _validate_and_adjust_embedding(self, embedding: list[float]) -> list[float]:
        """Validate and adjust embedding dimensions to match expected dimensions.

        Args:
            embedding: The embedding vector to validate

        Returns:
            Embedding vector adjusted to match self.dimensions
        """
        actual_len = len(embedding)
        if actual_len == self.dimensions:
            return embedding

        elif actual_len < self.dimensions:
            self.logger.warning(
                f"[ACTUAL_EMB_LENGTH]Embedding dimensions {actual_len} is less than expected {self.dimensions}, "
                f"padding with zeros",
            )
            return embedding + [0.0] * (self.dimensions - actual_len)

        else:
            self.logger.warning(
                f"[ACTUAL_EMB_LENGTH]Embedding dimensions {actual_len} is greater than expected {self.dimensions}, "
                f"truncating to {self.dimensions}",
            )
            return embedding[: self.dimensions]

    def _get_cache_key(self, text: str) -> str:
        """Generate a cache key by hashing text + model_name + dimensions."""
        cache_string = f"{text}|{self.model_name}|{self.dimensions}"
        return hashlib.sha256(cache_string.encode(self.encoding)).hexdigest()

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
        if not self.enable_cache:
            return

        self.cache_path.mkdir(parents=True, exist_ok=True)
        cache_file = self._get_cache_file_path()
        if not cache_file.exists():
            self.logger.info(f"No cache file found at {cache_file}, starting with empty cache")
            return

        try:
            load_start = time.time()
            # Read all lines first (to load in reverse order)
            with open(cache_file, "r", encoding=self.encoding) as f:
                lines = f.readlines()

            loaded_count = 0
            # Load in reverse order (newest entries first)
            for _, line in enumerate(reversed(lines), 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError as e:
                    self.logger.warning(f"Failed to parse line in cache file: {e}")
                    continue

                if not data:
                    continue
                # Each line is {cache_key: embedding}
                cache_key, embedding = next(iter(data.items()))

                if cache_key and embedding and isinstance(embedding, list):
                    # Skip if already loaded (keep the newest)
                    if cache_key in self._embedding_cache:
                        continue

                    if len(embedding) != self.dimensions:
                        self.logger.warning(
                            f"Embedding dimensions mismatch for cache key {cache_key}, "
                            f"expected {self.dimensions}, got {len(embedding)}",
                        )
                        continue

                    # Respect max_cache_size during loading
                    if len(self._embedding_cache) >= self.max_cache_size:
                        self.logger.info(
                            f"Cache size limit reached ({self.max_cache_size}), "
                            f"loaded {loaded_count} newest entries",
                        )
                        break
                    self._embedding_cache[cache_key] = embedding
                    loaded_count += 1

            self.logger.info(
                f"Loaded {loaded_count} embeddings from cache file: {cache_file} in {time.time() - load_start:.2f}s",
            )
        except Exception as e:
            self.logger.error(f"Failed to load cache from {cache_file}: {e}, deleting cache file")
            try:
                cache_file.unlink()
                self.logger.info(f"Deleted corrupted cache file: {cache_file}")
            except Exception as del_e:
                self.logger.error(f"Failed to delete cache file {cache_file}: {del_e}")

    def _save_cache(self) -> None:
        """Save embedding cache to disk (JSONL format).

        Each line contains a JSON object with the cache key and embedding vector.
        Only saves if cache is non-empty.
        """
        if not self.enable_cache:
            return

        self.logger.info(f"Attempting to save cache, current size: {len(self._embedding_cache)}")
        if not self._embedding_cache:
            self.logger.info("Cache is empty, skipping save")
            return

        cache_file = self._get_cache_file_path()
        try:
            with open(cache_file, "w", encoding=self.encoding) as f:
                for cache_key, embedding in self._embedding_cache.items():
                    if len(embedding) != self.dimensions:
                        self.logger.warning(
                            f"Embedding dimensions mismatch for cache key {cache_key}, "
                            f"expected {self.dimensions}, got {len(embedding)}",
                        )
                        continue
                    cache_entry = {cache_key: embedding}
                    f.write(json.dumps(cache_entry, ensure_ascii=False) + "\n")

            self.logger.info(f"Saved {len(self._embedding_cache)} embeddings to cache file: {cache_file}")
        except Exception as e:
            self.logger.error(f"Failed to save cache to {cache_file}: {e}")

    def _get_from_cache(self, text: str) -> list[float] | None:
        if not self.enable_cache:
            return None

        cache_key = self._get_cache_key(text)
        if cache_key not in self._embedding_cache:
            self._cache_misses += 1
            return None

        embeddings = self._embedding_cache[cache_key]
        if len(embeddings) != self.dimensions:
            self.logger.warning(
                f"Cached embedding dimensions mismatch: expected {self.dimensions}, "
                f"got {len(embeddings)}. Removing invalid cache entry.",
            )
            del self._embedding_cache[cache_key]
            self._cache_misses += 1
            return None

        self._embedding_cache.move_to_end(cache_key)
        self._cache_hits += 1
        text_preview = text[:50] + "..." if len(text) > 50 else text
        self.logger.info(f"Cache hit for text: {text_preview} (hits: {self._cache_hits}, misses: {self._cache_misses})")
        return embeddings

    def _put_to_cache(self, text: str, embedding: list[float]) -> None:
        if not self.enable_cache or self.max_cache_size <= 0:
            return

        cache_key = self._get_cache_key(text)
        if len(embedding) != self.dimensions:
            self.logger.warning(
                f"[PUT_TO_CACHE] Embedding dimensions mismatch for cache key {cache_key}, "
                f"expected {self.dimensions}, got real length {len(embedding)}",
            )
            return

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

    @abstractmethod
    async def _get_embeddings(self, input_text: list[str], **kwargs) -> list[list[float]]:
        """Internal async implementation for calling the embedding API with batch input."""

    async def get_embedding(self, input_text: str, **kwargs) -> list[float]:
        truncated_text = self._truncate_text(input_text)
        cached_embedding = self._get_from_cache(truncated_text)
        if cached_embedding is not None:
            return cached_embedding

        for retry in range(self.max_retries):
            try:
                result = await self._get_embeddings([truncated_text], **kwargs)
                if result and len(result) == 1:
                    embedding = self._validate_and_adjust_embedding(result[0])
                    self._put_to_cache(truncated_text, embedding)
                    return embedding
                # Empty or mismatched result, treat as failure for retry
                self.logger.warning(
                    f"Model {self.model_name} returned {len(result) if result else 0} results, expected 1"
                )
                if retry == self.max_retries - 1:
                    if self.raise_exception:
                        raise RuntimeError("Embedding API returned empty result")
                    return []
                await asyncio.sleep(retry + 1)
            except Exception as e:
                self.logger.error(f"Model {self.model_name} failed: {e}")
                if retry == self.max_retries - 1:
                    if self.raise_exception:
                        raise
                    return []
                await asyncio.sleep(retry + 1)
        return []

    async def get_embeddings(self, input_text: list[str], **kwargs) -> list[list[float]]:
        truncated_texts = [self._truncate_text(t) for t in input_text]
        results: list[list[float] | None] = [None] * len(truncated_texts)
        texts_to_compute: list[tuple[int, str]] = []

        for idx, text in enumerate(truncated_texts):
            cached = self._get_from_cache(text)
            if cached is not None:
                results[idx] = cached
            else:
                texts_to_compute.append((idx, text))

        if texts_to_compute:
            uncached_texts = [text for _, text in texts_to_compute]
            for i in range(0, len(uncached_texts), self.max_batch_size):
                batch_texts = uncached_texts[i: i + self.max_batch_size]
                batch_indices = [idx for idx, _ in texts_to_compute[i: i + self.max_batch_size]]

                for retry in range(self.max_retries):
                    try:
                        batch_embeddings = await self._get_embeddings(batch_texts, **kwargs)
                        if batch_embeddings and len(batch_embeddings) == len(batch_texts):
                            for orig_idx, text, embedding in zip(batch_indices, batch_texts, batch_embeddings):
                                adjusted_embedding = self._validate_and_adjust_embedding(embedding)
                                results[orig_idx] = adjusted_embedding
                                self._put_to_cache(text, adjusted_embedding)
                            break  # Success, exit retry loop
                        else:
                            self.logger.warning(
                                f"Batch embedding returned {len(batch_embeddings) if batch_embeddings else 0} results "
                                f"for {len(batch_texts)} inputs"
                            )
                            if retry == self.max_retries - 1:
                                if self.raise_exception:
                                    raise RuntimeError(
                                        f"Batch embedding returned {len(batch_embeddings) if batch_embeddings else 0} "
                                        f"results for {len(batch_texts)} inputs after {self.max_retries} retries"
                                    )
                                # Fill failed positions with empty lists
                                for orig_idx in batch_indices:
                                    if results[orig_idx] is None:
                                        results[orig_idx] = []
                            else:
                                await asyncio.sleep(retry + 1)
                    except Exception as e:
                        self.logger.error(f"Model {self.model_name} batch failed: {e}")
                        if retry == self.max_retries - 1:
                            if self.raise_exception:
                                raise
                            # Fill failed positions with empty lists
                            for orig_idx in batch_indices:
                                if results[orig_idx] is None:
                                    results[orig_idx] = []
                        else:
                            await asyncio.sleep(retry + 1)

        return [r if r is not None else [] for r in results]

    async def get_node_embeddings(self, nodes: list[BaseNode], **kwargs) -> list[BaseNode]:
        texts = [node.text for node in nodes]
        embeddings = await self.get_embeddings(texts, **kwargs)

        if len(embeddings) == len(nodes):
            for node, vec in zip(nodes, embeddings):
                node.embedding = vec
        else:
            self.logger.warning(
                f"Mismatch: got {len(embeddings)} vectors for {len(nodes)} nodes, "
                f"skipping embedding assignment"
            )
        return nodes

    async def _start(self, app_context=None) -> None:
        """Initialize resources and load cache."""
        self._load_cache()

    async def _close(self) -> None:
        """Release resources and save cache."""
        self._save_cache()
