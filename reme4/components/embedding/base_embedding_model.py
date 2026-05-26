"""Base embedding model with LRU cache and disk persistence."""

import asyncio
import hashlib
import os
from abc import abstractmethod
from collections import OrderedDict
from pathlib import Path

import numpy as np

from ..base_component import BaseComponent
from ...enumeration import ComponentEnum
from ...schema import EmbNode

Miss = tuple[int, str, str]  # (result_index, text, cache_key)


class BaseEmbeddingModel(BaseComponent):
    """Embedding model with LRU cache, disk persistence, and concurrent batching."""

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
        max_cache_size: int = 10000,
        max_concurrency: int = 2,
        enable_cache: bool = True,
        cache_version: str = "v1",
        max_retries: int = 3,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.api_key = api_key or os.environ.get("EMBEDDING_API_KEY", "")
        self.base_url = base_url or os.environ.get("EMBEDDING_BASE_URL", "")
        self.model_name = model_name
        self.dimensions = dimensions
        self.pass_dimensions = pass_dimensions
        self.max_batch_size = max_batch_size
        self.max_input_length = max_input_length
        self.max_cache_size = max_cache_size
        self.max_concurrency = max_concurrency
        self.enable_cache = enable_cache
        self.cache_version = cache_version
        self.max_retries = max_retries
        self._cache: OrderedDict[str, np.ndarray] = OrderedDict()
        self._key_suffix = f"|{model_name}|{dimensions}".encode()
        self.is_healthy: bool = True

    @property
    def cache_path(self) -> Path:
        return self.vault_metadata_path / "embedding_cache" / f"{self.name}_{self.cache_version}.npz"

    async def _start(self) -> None:
        await self.load()

    async def _close(self) -> None:
        await self.dump()

    async def health_check(self, timeout: float = 2.0) -> bool:
        """Probe the provider; sets and returns is_healthy."""
        tag = f"[EMBEDDING HEALTH CHECK] name={self.name} model={self.model_name}"
        try:
            result = await asyncio.wait_for(self._get_embeddings(["ping"]), timeout=timeout)
            if not result or result[0] is None:
                raise RuntimeError("empty embedding")
            self.is_healthy = True
            self.logger.info(f"{tag} -> OK")
        except asyncio.TimeoutError:
            self.is_healthy = False
            self.logger.error(f"{tag} -> FAIL timeout({timeout}s)")
        except Exception as e:
            self.is_healthy = False
            self.logger.error(f"{tag} -> FAIL {type(e).__name__}: {e}")
        return self.is_healthy

    # -- Public API --

    async def get_embedding(self, input_text: str, **kwargs) -> np.ndarray | None:
        results = await self.get_embeddings([input_text], **kwargs)
        return results[0] if results else None

    async def get_embeddings(self, input_text: list[str], **kwargs) -> list[np.ndarray | None]:
        """Get embeddings for texts. Cache hits return immediately; misses run concurrently."""
        texts = [self._truncate(t) for t in input_text]
        results, misses = self._partition_by_cache(texts)
        if misses:
            await self._fill_misses(misses, results, **kwargs)
        return results

    async def get_node_embeddings(self, nodes: list[EmbNode], **kwargs) -> list[EmbNode]:
        embeddings = await self.get_embeddings([n.text for n in nodes], **kwargs)
        if len(embeddings) == len(nodes):
            for node, vec in zip(nodes, embeddings):
                if vec is not None:
                    node.embedding = vec
        return nodes

    @abstractmethod
    async def _get_embeddings(self, input_text: list[str], **kwargs) -> list[list[float] | None]:
        """Get raw embeddings from the underlying provider."""

    # -- Batching --

    def _truncate(self, text: str) -> str:
        return text if len(text) <= self.max_input_length else text[: self.max_input_length]

    def _partition_by_cache(self, texts: list[str]) -> tuple[list[np.ndarray | None], list[Miss]]:
        """Split texts into pre-filled results (hits) and a miss list to compute."""
        results: list[np.ndarray | None] = [None] * len(texts)
        misses: list[Miss] = []
        for idx, text in enumerate(texts):
            key = self._cache_key(text)
            hit = self._cache_get(key)
            if hit is not None:
                results[idx] = hit
            else:
                misses.append((idx, text, key))
        return results, misses

    async def _fill_misses(self, misses: list[Miss], results: list[np.ndarray | None], **kwargs) -> None:
        """Compute miss embeddings in concurrent batches and write into results + cache."""
        size = self.max_batch_size
        batches = [misses[i : i + size] for i in range(0, len(misses), size)]
        sem = asyncio.Semaphore(self.max_concurrency)

        async def run(batch: list[Miss]) -> list[tuple[int, str, np.ndarray]]:
            async with sem:
                return await self._compute_batch(batch, **kwargs)

        for done in await asyncio.gather(*(run(b) for b in batches)):
            for idx, key, emb in done:
                results[idx] = emb
                self._cache_put(key, emb)

    async def _compute_batch(self, batch: list[Miss], **kwargs) -> list[tuple[int, str, np.ndarray]]:
        """Call provider for one batch with retry; returns [(idx, key, embedding)]."""
        texts = [text for _, text, _ in batch]
        embeddings = await self._call_with_retry(texts, **kwargs)
        if not embeddings or len(embeddings) != len(texts):
            return []
        out: list[tuple[int, str, np.ndarray]] = []
        for (idx, _text, key), raw in zip(batch, embeddings):
            if raw is None:
                continue
            emb = self._normalize_dim(np.asarray(raw, dtype=np.float16))
            out.append((idx, key, emb))
        return out

    async def _call_with_retry(self, texts: list[str], **kwargs) -> list[list[float] | None] | None:
        """Call provider with exponential backoff on transient errors."""
        for attempt in range(self.max_retries):
            try:
                result = await self._get_embeddings(texts, **kwargs)
                if result and len(result) == len(texts):
                    return result
            except (TimeoutError, ConnectionError, OSError):
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(2**attempt)
            except Exception:
                self.logger.exception("Embedding request failed")
                return None
        return None

    def _normalize_dim(self, emb: np.ndarray) -> np.ndarray:
        if len(emb) == self.dimensions:
            return emb
        if len(emb) < self.dimensions:
            return np.pad(emb, (0, self.dimensions - len(emb)))
        return emb[: self.dimensions]

    # -- Cache --

    def _cache_key(self, text: str) -> str:
        return hashlib.sha256(text.encode() + self._key_suffix).hexdigest()

    def _cache_get(self, key: str) -> np.ndarray | None:
        if not self.enable_cache or key not in self._cache:
            return None
        self._cache.move_to_end(key)
        return self._cache[key]

    def _cache_put(self, key: str, embedding: np.ndarray) -> None:
        if not self.enable_cache or self.max_cache_size <= 0 or len(embedding) != self.dimensions:
            return
        cache = self._cache
        if key in cache:
            cache.move_to_end(key)
            cache[key] = embedding
            return
        if len(cache) >= self.max_cache_size:
            cache.popitem(last=False)
        cache[key] = embedding

    # -- Persistence --

    async def load(self) -> None:
        """Load cached embeddings from disk (npz); replaces in-memory cache."""
        self._cache.clear()
        if not self.enable_cache or not self.cache_path.exists():
            return
        await asyncio.to_thread(self._load_sync)

    def _load_sync(self) -> None:
        try:
            data = np.load(self.cache_path)
        except Exception:
            self.logger.exception("Failed to load embedding cache, removing")
            self.cache_path.unlink(missing_ok=True)
            return
        for key, emb in zip(data["keys"], data["embeddings"]):
            if len(emb) != self.dimensions:
                continue
            if len(self._cache) >= self.max_cache_size:
                break
            self._cache[str(key)] = emb.astype(np.float16)
        self.logger.info(f"Loaded {len(self._cache)} embeddings from {self.cache_path}")

    async def dump(self) -> None:
        """Persist in-memory cache to disk (npz)."""
        if not self.enable_cache or not self._cache:
            return
        await asyncio.to_thread(self._dump_sync)

    def _dump_sync(self) -> None:
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        keys = np.array(list(self._cache.keys()), dtype=str)
        embeddings = np.stack(list(self._cache.values()))
        try:
            np.savez(self.cache_path, keys=keys, embeddings=embeddings)
            self.logger.info(f"Saved {len(self._cache)} embeddings to {self.cache_path}")
        except Exception:
            self.logger.exception("Failed to save embedding cache")
