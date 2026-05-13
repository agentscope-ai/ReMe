import hashlib
import os
from abc import abstractmethod
from collections import OrderedDict
from pathlib import Path

import numpy as np

from ..base_component import BaseComponent
from ...enumeration import ComponentEnum
from ...schema import EmbNode


class BaseEmbeddingModel(BaseComponent):
    """Embedding model with LRU cache and disk persistence."""

    component_type = ComponentEnum.EMBEDDING_MODEL

    # ==================== Initialization ====================

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model_name: str = "",
        dimensions: int = 1024,
        pass_dimensions: bool = False,
        max_batch_size: int = 10,
        max_input_length: int = 8192,
        max_cache_size: int = 5000,
        enable_cache: bool = True,
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
        self.enable_cache = enable_cache
        self._embedding_cache: OrderedDict[str, list[float]] = OrderedDict()

    @property
    def cache_path(self) -> Path:
        working_dir = self.app_context.app_config.working_dir if self.app_context else ""
        return Path(working_dir) / "embedding_cache" / f"{self.name}.npz"

    async def _start(self) -> None:
        self._embedding_cache.clear()
        self._load_cache()

    async def _close(self) -> None:
        self._save_cache()

    # ==================== Public API ====================

    async def get_embedding(self, input_text: str, **kwargs) -> list[float] | None:
        results = await self.get_embeddings([input_text], **kwargs)
        return results[0] if results else None

    async def get_embeddings(self, input_text: list[str], **kwargs) -> list[list[float] | None]:
        truncated = [t[: self.max_input_length] for t in input_text]
        results: list[list[float] | None] = [None] * len(truncated)
        to_compute: list[tuple[int, str]] = []

        for idx, text in enumerate(truncated):
            cached = self._get_from_cache(text)
            if cached is not None:
                results[idx] = cached
            else:
                to_compute.append((idx, text))

        if to_compute:
            for i in range(0, len(to_compute), self.max_batch_size):
                batch = to_compute[i: i + self.max_batch_size]
                indices = [idx for idx, _ in batch]
                texts = [text for _, text in batch]

                try:
                    embeddings = await self._get_embeddings(texts, **kwargs)
                    if embeddings and len(embeddings) == len(texts):
                        for orig_idx, text, emb in zip(indices, texts, embeddings):
                            if emb is None:
                                continue
                            if len(emb) != self.dimensions:
                                if len(emb) < self.dimensions:
                                    emb = emb + [0.0] * (self.dimensions - len(emb))
                                else:
                                    emb = emb[: self.dimensions]
                            results[orig_idx] = emb
                            self._put_to_cache(text, emb)
                except Exception:
                    pass

        return results

    async def get_node_embeddings(self, nodes: list[EmbNode], **kwargs) -> list[EmbNode]:
        embeddings = await self.get_embeddings([n.text for n in nodes], **kwargs)
        if len(embeddings) == len(nodes):
            for node, vec in zip(nodes, embeddings):
                if vec is not None:
                    node.embedding = vec
        return nodes

    # ==================== Abstract Method ====================

    @abstractmethod
    async def _get_embeddings(self, input_text: list[str], **kwargs) -> list[list[float] | None]:
        """Get embeddings for input text."""

    # ==================== Cache Operations ====================

    def _get_from_cache(self, text: str) -> list[float] | None:
        if not self.enable_cache:
            return None

        key = self._get_cache_key(text)
        if key not in self._embedding_cache:
            return None

        self._embedding_cache.move_to_end(key)
        return self._embedding_cache[key]

    def _put_to_cache(self, text: str, embedding: list[float]) -> None:
        if not self.enable_cache or self.max_cache_size <= 0 or len(embedding) != self.dimensions:
            return

        key = self._get_cache_key(text)
        if len(self._embedding_cache) >= self.max_cache_size and key not in self._embedding_cache:
            self._embedding_cache.popitem(last=False)

        self._embedding_cache[key] = embedding
        self._embedding_cache.move_to_end(key)

    def _get_cache_key(self, text: str) -> str:
        return hashlib.sha256(f"{text}|{self.model_name}|{self.dimensions}".encode()).hexdigest()

    # ==================== Cache Persistence ====================

    def _load_cache(self) -> None:
        if not self.enable_cache:
            return

        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.cache_path.exists():
            return

        try:
            data = np.load(self.cache_path)
        except Exception:
            self.cache_path.unlink(missing_ok=True)
            return

        for key, emb in zip(data["keys"], data["embeddings"]):
            emb_list = emb.tolist()
            if len(emb_list) != self.dimensions:
                continue
            if len(self._embedding_cache) >= self.max_cache_size:
                break
            self._embedding_cache[str(key)] = emb_list

    def _save_cache(self) -> None:
        if not self.enable_cache or not self._embedding_cache:
            return

        keys, embeddings = [], []
        for k, v in self._embedding_cache.items():
            keys.append(k)
            embeddings.append(v)

        try:
            np.savez(self.cache_path, keys=np.array(keys, dtype=str), embeddings=np.array(embeddings, dtype=np.float32))
        except Exception:
            pass