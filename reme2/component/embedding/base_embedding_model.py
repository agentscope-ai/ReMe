"""Base embedding model with caching and batching support."""

import hashlib
import os
import time
from abc import abstractmethod
from collections import OrderedDict
from pathlib import Path

import numpy as np

from ..base_component import BaseComponent
from ...enumeration import ComponentEnum
from ...schema import BaseNode


class BaseEmbeddingModel(BaseComponent):
    """Abstract base class for embedding models with LRU cache.

    Provides:
        - LRU in-memory cache with disk persistence (npz)
        - Automatic text truncation to max_input_length
        - Batch embedding support
    """

    component_type = ComponentEnum.EMBEDDING_MODEL

    def __init__(
            self,
            api_key: str | None = None,
            base_url: str | None = None,
            model_name: str = "",
            dimensions: int = 1024,
            pass_dimensions: bool = False,
            max_batch_size: int = 10,
            max_input_length: int = 8192,
            max_cache_size: int = 2000,
            enable_cache: bool = True,
            **kwargs,
    ):
        super().__init__(**kwargs)
        self.api_key: str = api_key or os.environ.get("EMBEDDING_API_KEY", "")
        self.base_url: str = base_url or os.environ.get("EMBEDDING_BASE_URL", "")
        self.model_name = model_name
        self.dimensions = dimensions
        self.pass_dimensions = pass_dimensions
        self.max_batch_size = max_batch_size
        self.max_input_length = max_input_length
        self.max_cache_size = max_cache_size
        self.enable_cache = enable_cache

        self._embedding_cache: OrderedDict[str, list[float]] = OrderedDict()
        self._cache_hits = 0
        self._cache_misses = 0
        self.cache_path: Path = Path()

    def clear_cache(self) -> None:
        """Clear in-memory cache and reset statistics."""
        self._embedding_cache.clear()
        self._cache_hits = 0
        self._cache_misses = 0

    async def _start(self) -> None:
        """Load cache on start."""
        assert self.app_context is not None, "app_context must be provided"
        self.clear_cache()
        working_path = Path(self.app_context.app_config.working_dir)
        self.cache_path = working_path / "embedding_cache" / f"{self.name}.npz"
        self._load_cache()

    async def _close(self) -> None:
        """Save cache on close."""
        self._save_cache()

    def _validate_and_adjust_embedding(self, embedding: list[float]) -> list[float]:
        """Adjust embedding dimensions to match expected dimensions."""
        actual_len = len(embedding)
        if actual_len == self.dimensions:
            return embedding
        if actual_len < self.dimensions:
            self.logger.warning(f"Embedding dim {actual_len} < expected {self.dimensions}, padding")
            return embedding + [0.0] * (self.dimensions - actual_len)
        self.logger.warning(f"Embedding dim {actual_len} > expected {self.dimensions}, truncating")
        return embedding[:self.dimensions]

    def _get_cache_key(self, text: str) -> str:
        """Generate cache key from text + model_name + dimensions."""
        return hashlib.sha256(f"{text}|{self.model_name}|{self.dimensions}".encode()).hexdigest()

    def _load_cache(self) -> None:
        """Load embedding cache from disk (npz format)."""
        if not self.enable_cache:
            return

        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.cache_path.exists():
            self.logger.info(f"No cache file at {self.cache_path}, starting empty")
            return

        load_start = time.time()
        try:
            data = np.load(self.cache_path)
        except Exception:
            self.logger.exception(f"Failed to load cache from {self.cache_path}, deleting file")
            self.cache_path.unlink(missing_ok=True)
            return

        loaded_count = 0
        for key, emb in zip(data["keys"], data["embeddings"]):
            emb_list = emb.tolist()
            if len(emb_list) != self.dimensions:
                self.logger.warning(f"Cache dimension mismatch for {key}: expected {self.dimensions}, got {len(emb_list)}")
                continue
            if len(self._embedding_cache) >= self.max_cache_size:
                self.logger.info(f"Cache limit reached ({self.max_cache_size}), loaded {loaded_count}")
                break
            self._embedding_cache[str(key)] = emb_list
            loaded_count += 1

        self.logger.info(f"Loaded {loaded_count} embeddings from {self.cache_path} in {time.time() - load_start:.2f}s")

    def _save_cache(self) -> None:
        """Save embedding cache to disk (npz format)."""
        if not self.enable_cache or not self._embedding_cache:
            return

        keys, embeddings = [], []
        for cache_key, embedding in self._embedding_cache.items():
            keys.append(cache_key)
            embeddings.append(embedding)

        try:
            np.savez(self.cache_path, keys=np.array(keys, dtype=str), embeddings=np.array(embeddings, dtype=np.float32))
        except Exception as e:
            self.logger.error(f"Failed to save cache to {self.cache_path}: {e}")
            return

        self.logger.info(f"Saved {len(keys)} embeddings to {self.cache_path}")

    def _get_from_cache(self, text: str) -> list[float] | None:
        """Retrieve embedding from cache if available."""
        if not self.enable_cache:
            return None

        cache_key = self._get_cache_key(text)
        if cache_key not in self._embedding_cache:
            self._cache_misses += 1
            return None

        self._embedding_cache.move_to_end(cache_key)
        self._cache_hits += 1
        return self._embedding_cache[cache_key]

    def _put_to_cache(self, text: str, embedding: list[float]) -> None:
        """Store embedding in cache with LRU eviction."""
        if not self.enable_cache or self.max_cache_size <= 0:
            return
        if len(embedding) != self.dimensions:
            return

        cache_key = self._get_cache_key(text)
        if len(self._embedding_cache) >= self.max_cache_size and cache_key not in self._embedding_cache:
            self._embedding_cache.popitem(last=False)

        self._embedding_cache[cache_key] = embedding
        self._embedding_cache.move_to_end(cache_key)

    def get_cache_stats(self) -> dict[str, int | float]:
        """Return cache statistics: size, hits, misses, hit_rate."""
        total = self._cache_hits + self._cache_misses
        return {
            "cache_size": len(self._embedding_cache),
            "max_cache_size": self.max_cache_size,
            "cache_hits": self._cache_hits,
            "cache_misses": self._cache_misses,
            "hit_rate": self._cache_hits / total if total > 0 else 0.0,
        }

    @abstractmethod
    async def _get_embeddings(self, input_text: list[str], **kwargs) -> list[list[float] | None]:
        """Fetch embeddings for a batch of texts. Override in subclasses."""

    async def get_embedding(self, input_text: str, **kwargs) -> list[float] | None:
        """Get embedding for a single text with cache. Returns None on failure."""
        results = await self.get_embeddings([input_text], **kwargs)
        return results[0] if results else None

    async def get_embeddings(self, input_text: list[str], **kwargs) -> list[list[float] | None]:
        """Get embeddings for multiple texts with cache and batching."""
        # TODO change to bytes instead of str
        truncated_texts = [t[:self.max_input_length] for t in input_text]
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
                batch = texts_to_compute[i:i + self.max_batch_size]
                batch_indices = [idx for idx, _ in batch]
                batch_texts = [text for _, text in batch]

                try:
                    batch_embeddings = await self._get_embeddings(batch_texts, **kwargs)
                    if batch_embeddings and len(batch_embeddings) == len(batch_texts):
                        for orig_idx, text, embedding in zip(batch_indices, batch_texts, batch_embeddings):
                            adjusted = self._validate_and_adjust_embedding(embedding)
                            results[orig_idx] = adjusted
                            self._put_to_cache(text, adjusted)
                    else:
                        self.logger.warning(
                            f"Batch returned {len(batch_embeddings) if batch_embeddings else 0} "
                            f"results for {len(batch_texts)} inputs",
                        )
                except Exception as e:
                    self.logger.error(f"Model {self.model_name} batch failed: {e}")

        return results

    async def get_node_embeddings(self, nodes: list[BaseNode], **kwargs) -> list[BaseNode]:
        """Get embeddings for a list of nodes and assign to node.embedding."""
        texts = [node.text for node in nodes]
        embeddings = await self.get_embeddings(texts, **kwargs)
        if len(embeddings) == len(nodes):
            for node, vec in zip(nodes, embeddings):
                if vec is not None:
                    node.embedding = vec
                else:
                    self.logger.warning(f"Embedding failed for node, skipping assignment")
        else:
            self.logger.warning(f"Mismatch: {len(embeddings)} vectors for {len(nodes)} nodes, skipping assignment")
        return nodes
