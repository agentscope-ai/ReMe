"""Local embedding store with LRU cache and disk persistence."""

import asyncio
import hashlib
from collections import OrderedDict
from pathlib import Path

import numpy as np

from ._embedding_types import _EmbeddingError, _EmbeddingResult
from .base_embedding_store import BaseEmbeddingStore
from ..component_registry import R
from ..as_embedding import BaseAsEmbedding

Miss = tuple[int, str, str]  # (result_index, text, cache_key)
_MAX_VECTOR_SPACE_ATTEMPTS = 3

# Each input-length rejection may be retried with a strictly smaller request, up to this
# many rounds (also bounded by the remaining retry budget).
_MAX_INPUT_SHRINK_ROUNDS = 3


# Match SDK/httpx transport exceptions without importing optional providers.
_TRANSPORT_FAILURE_TYPES = frozenset(
    {
        "APIConnectionError",
        "APITimeoutError",
        "ConnectError",
        "ConnectTimeout",
        "PoolTimeout",
        "ReadError",
        "ReadTimeout",
        "RemoteProtocolError",
        "WriteError",
        "WriteTimeout",
    },
)

# A client retry budget only applies to the transport exceptions it retries.
_SDK_TRANSPORT_RETRY_TYPES = frozenset({"APIConnectionError", "APITimeoutError"})


@R.register("local")
class LocalEmbeddingStore(BaseEmbeddingStore):
    """Embedding store with LRU cache, disk persistence, and serial batching.

    Delegates actual embedding computation to a bound ``as_embedding`` component.
    """

    def __init__(
        self,
        as_embedding: str = "default",
        max_cache_size: int = 10000,
        enable_cache: bool = True,
        cache_version: str = "v1",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.as_embedding = self.bind(as_embedding, BaseAsEmbedding, optional=False)
        self.max_cache_size = max_cache_size
        self.enable_cache = enable_cache
        self.cache_version = cache_version
        self._cache: OrderedDict[str, np.ndarray] = OrderedDict()
        self._cache_space: str = ""
        self._cache_space_lock = asyncio.Lock()

    @property
    def dimensions(self) -> int:
        """Return the embedding dimension size."""
        assert self.as_embedding is not None, "embedding component not bound"
        return self.as_embedding.dimensions

    @property
    def vector_space_id(self) -> str:
        """Return the digest of the vector space the bound provider currently produces."""
        assert self.as_embedding is not None, "embedding component not bound"
        return self.as_embedding.vector_space_id

    @property
    def cache_path(self) -> Path:
        """Return the disk cache file for the current vector space.

        Each vector space owns its own file, so switching the embedding model cannot
        read or overwrite vectors that belong to a different model.
        """
        return self._cache_path(self.vector_space_id)

    def _cache_path(self, vector_space_id: str) -> Path:
        return self.component_metadata_path / f"{self.name}_{self.cache_version}_{vector_space_id}.npz"

    async def _start(self) -> None:
        await self.load()

    async def _close(self) -> None:
        if self.is_started:
            await self.dump()

    async def health_check(self, timeout: float | None = None) -> bool:
        timeout = self.health_check_timeout if timeout is None else timeout
        if not isinstance(timeout, (int, float)) or not np.isfinite(timeout) or timeout <= 0:
            raise ValueError("timeout must be finite and greater than 0")
        tag = f"[EMBEDDING HEALTH CHECK] name={self.name} workspace_dir={self.workspace_path}"
        started_at = asyncio.get_running_loop().time()
        try:
            # Provider construction may synchronously import an SDK and build
            # its HTTP client. Keep that one-time work outside the request
            # timeout so the full budget applies to the initialized provider
            # call instead of being consumed before a request can be sent.
            self.as_embedding.initialize_model()
            result = await asyncio.wait_for(self.as_embedding(["ping"]), timeout=timeout)
            if not result or result[0] is None:
                raise RuntimeError("empty embedding")
            if len(result[0]) != self.dimensions:
                raise RuntimeError(f"embedding dimension mismatch: {len(result[0])} != {self.dimensions}")
            self.is_healthy = True
            elapsed = asyncio.get_running_loop().time() - started_at
            self.logger.info(f"{tag} -> OK timeout={timeout}s elapsed={elapsed:.3f}s")
        except asyncio.TimeoutError:
            self.is_healthy = False
            elapsed = asyncio.get_running_loop().time() - started_at
            self.logger.error(f"{tag} -> FAIL timeout={timeout}s elapsed={elapsed:.3f}s error=timeout({timeout}s)")
        except Exception as exc:  # Provider SDKs expose many exception types.
            self.is_healthy = False
            elapsed = asyncio.get_running_loop().time() - started_at
            self.logger.error(
                f"{tag} -> FAIL timeout={timeout}s elapsed={elapsed:.3f}s error={type(exc).__name__}: {exc}",
            )
        return self.is_healthy

    # -- Public API --

    async def get_embeddings(self, input_text: list[str], **kwargs) -> list[np.ndarray | None]:
        texts = [self._truncate(t) for t in input_text]
        for attempt in range(1, _MAX_VECTOR_SPACE_ATTEMPTS + 1):
            await self._sync_cache_space()
            vector_space_id = self._cache_space
            results, misses = self._partition_by_cache(texts)
            stable = not misses or await self._fill_misses(misses, results, vector_space_id, **kwargs)
            if stable and vector_space_id == self.vector_space_id == self._cache_space:
                return results
            if attempt == _MAX_VECTOR_SPACE_ATTEMPTS:
                self.logger.warning(
                    f"Embedding vector space kept changing while computing a request; "
                    f"discarding all result(s) after {attempt} attempts",
                )
            else:
                self.logger.info(
                    f"Embedding vector space changed while computing a request; "
                    f"discarding all result(s) and retrying ({attempt}/{_MAX_VECTOR_SPACE_ATTEMPTS})",
                )
        return [None] * len(texts)

    # -- Batching --

    def _partition_by_cache(self, texts: list[str]) -> tuple[list[np.ndarray | None], list[Miss]]:
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

    async def _fill_misses(
        self,
        misses: list[Miss],
        results: list[np.ndarray | None],
        vector_space_id: str,
        **kwargs,
    ) -> bool:
        """Fill every miss only while the request remains in one vector space."""
        size = self.max_batch_size
        for start in range(0, len(misses), size):
            if vector_space_id != self.vector_space_id or vector_space_id != self._cache_space:
                return False
            batch = misses[start : start + size]
            computed = await self._compute_batch(batch, **kwargs)
            if vector_space_id != self.vector_space_id or vector_space_id != self._cache_space:
                return False
            for idx, key, emb in computed:
                results[idx] = emb
                self._cache_put(key, emb)
        return True

    async def _compute_batch(self, batch: list[Miss], **kwargs) -> list[tuple[int, str, np.ndarray]]:
        texts = [text for _, text, _ in batch]
        result = await self._call_with_retry(texts, **kwargs)
        embeddings = result.embeddings
        if not embeddings or len(embeddings) != len(texts):
            return []
        out: list[tuple[int, str, np.ndarray]] = []
        bad_dims: dict[int, int] = {}
        for (idx, _text, key), raw in zip(batch, embeddings):
            if raw is None:
                continue
            emb = np.asarray(raw, dtype=np.float16)
            if not self._validate_dim(emb):
                bad_dims[len(emb)] = bad_dims.get(len(emb), 0) + 1
                continue
            out.append((idx, key, emb))
        if bad_dims:
            details = ", ".join(f"{count} with dim {dim}" for dim, count in sorted(bad_dims.items()))
            self.logger.error(f"Embedding dimension mismatch in batch: expected {self.dimensions}; rejected {details}")
        # Retain partial vectors without hiding the outage that ended isolation.
        self.is_healthy = bool(out) and not result.terminal_failure
        return out

    async def _call_with_retry(self, texts: list[str], **kwargs) -> _EmbeddingResult:
        """Retry failures owned by the store and isolate input-length rejections."""
        shrink_rounds = 0
        for attempt in range(self.max_retries):
            try:
                embeddings = await self.as_embedding(texts, **kwargs)
                if embeddings and len(embeddings) == len(texts):
                    return _EmbeddingResult(embeddings)
            except Exception as error:
                if self._is_transport_failure(error):
                    # Preserve recovery where the provider has no matching retry policy;
                    # otherwise leave its budget intact instead of adding another loop.
                    if self._provider_owns_transport_retry(error):
                        self.logger.exception(
                            "Embedding request failed: transport error owned by the provider, not splitting",
                        )
                        break
                    if attempt < self.max_retries - 1:
                        await asyncio.sleep(2**attempt)
                        continue
                    self.logger.exception("Embedding request failed: transport error after exhausting retries")
                    break

                details = _EmbeddingError.from_exception(error)
                if details.is_rate_limited:
                    if attempt < self.max_retries - 1:
                        delay = 2**attempt
                        self.logger.warning(f"Embedding rate limited; retrying in {delay:.1f}s")
                        await asyncio.sleep(delay)
                        continue
                    self.logger.exception("Embedding request failed after exhausting rate-limit retries")
                    break
                if details.is_insufficient_quota:
                    if self.quota_retry_delay is not None and attempt < self.max_retries - 1:
                        self.logger.warning(f"Embedding quota exhausted; retrying in {self.quota_retry_delay:.1f}s")
                        await asyncio.sleep(self.quota_retry_delay)
                        continue
                    self.logger.exception("Embedding request failed: quota exhausted")
                    break
                if details.is_provider_wide:
                    # Server-error retries belong to the wrapper/SDK, not to item isolation.
                    self.logger.exception("Embedding request failed: provider-wide error, not splitting")
                    break
                if details.is_input_dependent:
                    # Retry the item with a strictly smaller request derived from the text the
                    # provider just rejected. A remaining attempt is required so exhausted input
                    # failures stay non-terminal, and the shrinking rounds are bounded.
                    can_shrink = (
                        len(texts) == 1 and attempt < self.max_retries - 1 and shrink_rounds < _MAX_INPUT_SHRINK_ROUNDS
                    )
                    if can_shrink:
                        shorter = self._shrink_rejected_input(texts[0])
                        if shorter is not None:
                            shrink_rounds += 1
                            self.logger.warning(
                                f"Embedding input rejected as too long; retrying with a smaller request "
                                f"(chars={len(texts[0])} -> {len(shorter)}, "
                                f"round {shrink_rounds}/{_MAX_INPUT_SHRINK_ROUNDS})",
                            )
                            texts = [shorter]
                            continue
                    self.logger.exception("Embedding request failed: input-dependent error")
                    self.is_healthy = False
                    if len(texts) > 1:
                        return await self._embed_individually(texts, **kwargs)
                    return _EmbeddingResult()
                self.logger.exception("Embedding request failed: unrecognized error, not splitting")
                break
        self.is_healthy = False
        return _EmbeddingResult(terminal_failure=True)

    @staticmethod
    def _shrink_rejected_input(text: str) -> str | None:
        """Return a strictly smaller request, or ``None`` when there is none.

        The rejected text is the only reliable starting point: a budget derived
        from the configured ``max_input_length`` can still sit above the
        provider's real per-item limit, or leave an already short input
        untouched so nothing is re-sent. Halving the *current* request keeps
        every round an actual reduction, and the caller bounds the number of
        rounds so an impossible limit cannot spin.
        """
        if len(text) <= 1:
            return None
        return text[: len(text) // 2]

    async def _embed_individually(self, texts: list[str], **kwargs) -> _EmbeddingResult:
        """Keep healthy siblings, stopping when a failure is no longer specific to an input."""
        embeddings: list[list[float] | None] = []
        for index, text in enumerate(texts):
            result = await self._call_with_retry([text], **kwargs)
            if result.embeddings:
                embeddings.append(result.embeddings[0])
                continue
            self.logger.error(f"Embedding request failed for a single item (chars={len(text)})")
            embeddings.append(None)
            if result.terminal_failure:
                self.logger.error(
                    f"Stopping per-item isolation after {index + 1}/{len(texts)} items: "
                    "the failure is not specific to any input",
                )
                embeddings.extend([None] * (len(texts) - index - 1))
                return _EmbeddingResult(embeddings, terminal_failure=True)
        return _EmbeddingResult(embeddings)

    @staticmethod
    def _is_transport_failure(error: Exception) -> bool:
        """Recognize transport errors that must never trigger per-item isolation."""
        if isinstance(error, (TimeoutError, ConnectionError, OSError)):
            return True
        return any(cls.__name__ in _TRANSPORT_FAILURE_TYPES for cls in type(error).__mro__)

    def _provider_owns_transport_retry(self, error: Exception) -> bool:
        """Require both a matching exception policy and an enabled retry budget."""
        model = getattr(self.as_embedding, "model", None)
        if model is None:
            return False
        policy = self._declared_retry_policy(model)
        if policy:
            max_retries = getattr(model, "max_retries", None)
            if isinstance(max_retries, int) and max_retries > 0 and self._policy_covers(policy, error):
                return True
        return self._sdk_owns_transport_retry(model, error)

    @staticmethod
    def _declared_retry_policy(model) -> tuple[type[Exception], ...] | None:
        """Return the model's declared retryable types, or ``None`` when it declares none."""
        retryable = getattr(model, "_get_retryable_exceptions", None)
        if not callable(retryable):
            return None
        try:
            return tuple(retryable())  # pylint: disable=not-callable
        except Exception:  # pragma: no cover - a broken policy owns nothing
            return None

    @staticmethod
    def _policy_covers(policy: tuple[type[Exception], ...], error: Exception) -> bool:
        """Return whether the declared policy actually retries this exception."""
        try:
            return isinstance(error, policy)
        except TypeError:  # pragma: no cover - a malformed policy covers nothing
            return False

    @staticmethod
    def _sdk_owns_transport_retry(model, error: Exception) -> bool:
        """Check the SDK's budget and supported transport errors independently of the model."""
        if not LocalEmbeddingStore._is_transport_failure(error):
            return False
        client = getattr(model, "client", None)
        if client is None:
            return False
        max_retries = getattr(client, "max_retries", None)
        if not isinstance(max_retries, int) or max_retries <= 0:
            return False
        return any(cls.__name__ in _SDK_TRANSPORT_RETRY_TYPES for cls in type(error).__mro__)

    def _validate_dim(self, emb: np.ndarray) -> bool:
        """Return whether an embedding exactly matches the configured dimension."""
        return len(emb) == self.dimensions

    # -- Cache --

    async def _sync_cache_space(self) -> None:
        """Persist the previous space and restore the newly active space."""
        space = self.vector_space_id
        if space == self._cache_space:
            return
        async with self._cache_space_lock:
            while True:
                space = self.vector_space_id
                if space == self._cache_space:
                    return
                previous = self._cache_space
                snapshot = list(self._cache.items())
                if previous and self.enable_cache and snapshot:
                    await asyncio.to_thread(self._dump_sync, previous, snapshot)
                if space != self.vector_space_id:
                    continue
                dimensions = self.dimensions
                cache: OrderedDict[str, np.ndarray] = OrderedDict()
                if self.enable_cache and self._cache_path(space).exists():
                    cache = await asyncio.to_thread(self._load_sync, space, dimensions)
                if space != self.vector_space_id:
                    continue
                self._cache = cache
                self._cache_space = space
                return

    def _cache_key(self, text: str) -> str:
        return hashlib.sha256(text.encode()).hexdigest()

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
        self._cache.clear()
        self._cache_space = ""
        await self._sync_cache_space()

    def _load_sync(self, vector_space_id: str, dimensions: int) -> OrderedDict[str, np.ndarray]:
        path = self._cache_path(vector_space_id)
        cache: OrderedDict[str, np.ndarray] = OrderedDict()
        try:
            with np.load(path) as data:
                for key, emb in zip(data["keys"], data["embeddings"]):
                    if len(emb) != dimensions:
                        continue
                    if len(cache) >= self.max_cache_size:
                        break
                    cache[str(key)] = emb.astype(np.float16)
        except Exception:
            self.logger.exception("Failed to load embedding cache, removing")
            path.unlink(missing_ok=True)
            return cache
        self.logger.info(f"Loaded {len(cache)} embeddings from {path}")
        return cache

    async def dump(self) -> None:
        await self._sync_cache_space()
        snapshot = list(self._cache.items())
        if not self.enable_cache or not snapshot:
            return
        await asyncio.to_thread(self._dump_sync, self._cache_space, snapshot)

    def _dump_sync(self, vector_space_id: str, cache: list[tuple[str, np.ndarray]]) -> None:
        path = self._cache_path(vector_space_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        keys = np.array([key for key, _ in cache], dtype=str)
        embeddings = np.stack([embedding for _, embedding in cache])
        try:
            np.savez(path, keys=keys, embeddings=embeddings)
            self.logger.info(f"Saved {len(cache)} embeddings to {path}")
        except Exception:
            self.logger.exception("Failed to save embedding cache")
