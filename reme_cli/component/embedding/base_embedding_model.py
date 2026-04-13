"""Base embedding model with caching, batching, and retry support."""

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
    """Abstract base class for embedding models with LRU cache and retry logic.

    Provides:
        - LRU in-memory cache with disk persistence (JSONL)
        - Automatic text truncation to max_input_length
        - Retry logic with exponential backoff
        - Batch embedding support
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
        """Initialize embedding model configuration.

        Args:
            api_key: API key for the embedding service.
            base_url: Base URL for the embedding service.
            model_name: Name of the embedding model.
            dimensions: Vector dimensions.
            use_dimensions: Whether to pass dimensions parameter to API.
            max_batch_size: Maximum batch size for embedding requests.
            max_retries: Maximum retry attempts on failure.
            raise_exception: Whether to raise exceptions on failure.
            max_input_length: Maximum input text length.
            cache_dir: Directory for cache storage.
            max_cache_size: Maximum LRU cache size.
            enable_cache: Whether to enable caching.
            encoding: Text encoding for cache file operations.
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
        """Truncate text to max_input_length."""
        return text[: self.max_input_length] if len(text) > self.max_input_length else text

    def _validate_and_adjust_embedding(self, embedding: list[float]) -> list[float]:
        """Adjust embedding dimensions to match expected dimensions."""
        actual_len = len(embedding)
        if actual_len == self.dimensions:
            return embedding

        if actual_len < self.dimensions:
            self.logger.warning(
                f"[ACTUAL_EMB_LENGTH] Embedding {actual_len} < expected {self.dimensions}, padding with zeros"
            )
            return embedding + [0.0] * (self.dimensions - actual_len)

        self.logger.warning(
            f"[ACTUAL_EMB_LENGTH] Embedding {actual_len} > expected {self.dimensions}, truncating"
        )
        return embedding[: self.dimensions]

    def _get_cache_key(self, text: str) -> str:
        """Generate cache key from text + model_name + dimensions."""
        cache_string = f"{text}|{self.model_name}|{self.dimensions}"
        return hashlib.sha256(cache_string.encode(self.encoding)).hexdigest()

    def _get_cache_file_path(self) -> Path:
        """Return path to the cache JSONL file."""
        return self.cache_path / "embedding_cache.jsonl"

    def _load_cache(self) -> None:
        """Load embedding cache from disk (JSONL format)."""
        if not self.enable_cache:
            return

        self.cache_path.mkdir(parents=True, exist_ok=True)
        cache_file = self._get_cache_file_path()
        if not cache_file.exists():
            self.logger.info(f"No cache file at {cache_file}, starting empty")
            return

        try:
            load_start = time.time()
            with open(cache_file, "r", encoding=self.encoding) as f:
                lines = f.readlines()

            loaded_count = 0
            for line in reversed(lines):
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError as e:
                    self.logger.warning(f"Failed to parse cache line: {e}")
                    continue

                if not data:
                    continue

                cache_key, embedding = next(iter(data.items()))
                if cache_key and embedding and isinstance(embedding, list):
                    if cache_key in self._embedding_cache:
                        continue
                    if len(embedding) != self.dimensions:
                        self.logger.warning(
                            f"Cache dimension mismatch for {cache_key}: "
                            f"expected {self.dimensions}, got {len(embedding)}"
                        )
                        continue
                    if len(self._embedding_cache) >= self.max_cache_size:
                        self.logger.info(f"Cache limit reached ({self.max_cache_size}), loaded {loaded_count}")
                        break
                    self._embedding_cache[cache_key] = embedding
                    loaded_count += 1

            self.logger.info(f"Loaded {loaded_count} embeddings from {cache_file} in {time.time() - load_start:.2f}s")
        except Exception as e:
            self.logger.error(f"Failed to load cache from {cache_file}: {e}, deleting file")
            try:
                cache_file.unlink()
            except Exception as del_e:
                self.logger.error(f"Failed to delete cache file: {del_e}")

    def _save_cache(self) -> None:
        """Save embedding cache to disk (JSONL format)."""
        if not self.enable_cache or not self._embedding_cache:
            return

        cache_file = self._get_cache_file_path()
        try:
            with open(cache_file, "w", encoding=self.encoding) as f:
                for cache_key, embedding in self._embedding_cache.items():
                    if len(embedding) != self.dimensions:
                        self.logger.warning(f"Cache dimension mismatch for {cache_key}")
                        continue
                    f.write(json.dumps({cache_key: embedding}, ensure_ascii=False) + "\n")
            self.logger.info(f"Saved {len(self._embedding_cache)} embeddings to {cache_file}")
        except Exception as e:
            self.logger.error(f"Failed to save cache to {cache_file}: {e}")

    def _get_from_cache(self, text: str) -> list[float] | None:
        """Retrieve embedding from cache if available."""
        if not self.enable_cache:
            return None

        cache_key = self._get_cache_key(text)
        if cache_key not in self._embedding_cache:
            self._cache_misses += 1
            return None

        embeddings = self._embedding_cache[cache_key]
        if len(embeddings) != self.dimensions:
            self.logger.warning(f"Cached embedding dimension mismatch, removing entry")
            del self._embedding_cache[cache_key]
            self._cache_misses += 1
            return None

        self._embedding_cache.move_to_end(cache_key)
        self._cache_hits += 1
        preview = text[:50] + "..." if len(text) > 50 else text
        self.logger.info(f"Cache hit: {preview} (hits: {self._cache_hits}, misses: {self._cache_misses})")
        return embeddings

    def _put_to_cache(self, text: str, embedding: list[float]) -> None:
        """Store embedding in cache with LRU eviction."""
        if not self.enable_cache or self.max_cache_size <= 0:
            return

        cache_key = self._get_cache_key(text)
        if len(embedding) != self.dimensions:
            self.logger.warning(f"[PUT_TO_CACHE] Dimension mismatch for {cache_key}")
            return

        if len(self._embedding_cache) >= self.max_cache_size and cache_key not in self._embedding_cache:
            self._embedding_cache.popitem(last=False)

        self._embedding_cache[cache_key] = embedding
        self._embedding_cache.move_to_end(cache_key)

    def get_cache_stats(self) -> dict[str, int | float]:
        """Return cache statistics: size, hits, misses, hit_rate."""
        total = self._cache_hits + self._cache_misses
        hit_rate = self._cache_hits / total if total > 0 else 0.0
        return {
            "cache_size": len(self._embedding_cache),
            "max_cache_size": self.max_cache_size,
            "cache_hits": self._cache_hits,
            "cache_misses": self._cache_misses,
            "hit_rate": hit_rate,
        }

    def clear_cache(self) -> None:
        """Clear in-memory cache and reset statistics."""
        self._embedding_cache.clear()
        self._cache_hits = 0
        self._cache_misses = 0

    @abstractmethod
    async def _get_embeddings(self, input_text: list[str], **kwargs) -> list[list[float]]:
        """Fetch embeddings for a batch of texts. Override in subclasses."""

    async def get_embedding(self, input_text: str, **kwargs) -> list[float]:
        """Get embedding for a single text with cache and retry."""
        truncated_text = self._truncate_text(input_text)
        cached = self._get_from_cache(truncated_text)
        if cached is not None:
            return cached

        for retry in range(self.max_retries):
            try:
                result = await self._get_embeddings([truncated_text], **kwargs)
                if result and len(result) == 1:
                    embedding = self._validate_and_adjust_embedding(result[0])
                    self._put_to_cache(truncated_text, embedding)
                    return embedding
                self.logger.warning(
                    f"Model {self.model_name} returned {len(result) if result else 0} results, expected 1")
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
        """Get embeddings for multiple texts with cache and batching."""
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
                                adjusted = self._validate_and_adjust_embedding(embedding)
                                results[orig_idx] = adjusted
                                self._put_to_cache(text, adjusted)
                            break
                        self.logger.warning(
                            f"Batch returned {len(batch_embeddings) if batch_embeddings else 0} "
                            f"results for {len(batch_texts)} inputs"
                        )
                        if retry == self.max_retries - 1:
                            if self.raise_exception:
                                raise RuntimeError(f"Batch embedding failed after {self.max_retries} retries")
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
                            for orig_idx in batch_indices:
                                if results[orig_idx] is None:
                                    results[orig_idx] = []
                        else:
                            await asyncio.sleep(retry + 1)

        return [r if r is not None else [] for r in results]

    async def get_node_embeddings(self, nodes: list[BaseNode], **kwargs) -> list[BaseNode]:
        """Get embeddings for a list of nodes and assign to node.embedding."""
        texts = [node.text for node in nodes]
        embeddings = await self.get_embeddings(texts, **kwargs)

        if len(embeddings) == len(nodes):
            for node, vec in zip(nodes, embeddings):
                node.embedding = vec
        else:
            self.logger.warning(f"Mismatch: {len(embeddings)} vectors for {len(nodes)} nodes, skipping assignment")
        return nodes

    async def _start(self, app_context=None) -> None:
        """Load cache on start."""
        self._load_cache()

    async def _close(self) -> None:
        """Save cache on close."""
        self._save_cache()
