"""Local embedding store with LRU cache and disk persistence."""

import asyncio
import hashlib
import re
from collections import OrderedDict
from pathlib import Path

import numpy as np

from .base_embedding_store import BaseEmbeddingStore
from ..component_registry import R
from ..as_embedding import BaseAsEmbedding

Miss = tuple[int, str, str]  # (result_index, text, cache_key)
_MAX_VECTOR_SPACE_ATTEMPTS = 3

_PROVIDER_WIDE_STATUS_CODES = frozenset({401, 403, 404})

# Transport failures raised by the OpenAI SDK and httpx. They do not inherit from the built-in
# TimeoutError / ConnectionError / OSError, so the built-in check alone lets them slip into the
# generic handler and fan out into per-item requests. Matching on class names (across the MRO,
# so subclasses such as APITimeoutError are covered) keeps this module free of SDK imports.
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

# The status a provider reports can live in more than one place. ``status_code`` is the
# OpenAI-compatible attribute, Google GenAI exposes an integer ``code``, httpx-style wrappers
# keep it on ``response``, and AgentScope's DashScope implementation raises a plain
# ``RuntimeError`` whose message embeds the structured response, so the status has to be read
# back out of the message. A string ``code`` names an error (``insufficient_quota``,
# ``rate_limit_exceeded``, ``InvalidApiKey``) and is never a status.
_STATUS_IN_MESSAGE = re.compile(r'"status_code"\s*:\s*(\d{3})')

# The business error code travels the same way the status does: a string ``code`` on the
# OpenAI-compatible SDKs, ``error.code`` inside a structured body, and a serialized
# ``"code"`` in the message for AgentScope's DashScope ``RuntimeError``. Reading only the
# attribute loses the DashScope case and misclassifies a quota error as a rate limit.
_CODE_IN_MESSAGE = re.compile(r'"code"\s*:\s*"([^"]+)"')

# A rejected batch can only be paid back one item at a time when the provider blames an input.
# A context-length rejection is the one status an over-long item explains by itself, so it is
# the only error allowed to fan out into per-item requests.
_INPUT_DEPENDENT_STATUS_CODES = frozenset({400})

# A 413 is ambiguous. Providers such as Ollama v0.40.3 normalize an embedding input-limit
# rejection to 413, but a bare 413 can also describe an unrelated request-body limit that no
# single input explains. It therefore only qualifies as input-dependent when the error itself
# names the input length, which is why it is checked through the marker list instead of being
# accepted outright.
_INPUT_DEPENDENT_AMBIGUOUS_STATUS_CODES = frozenset({413})

_INPUT_DEPENDENT_ERROR_TYPES = frozenset(
    {
        "ContextLengthError",
        "ContextWindowExceededError",
        "InputLengthError",
        "TokenLimitError",
    },
)

_INPUT_DEPENDENT_MARKERS = (
    "context length",
    "context_length",
    "context window",
    "input length",
    "input is too long",
    "maximum context",
    "reduce the length",
    "token limit",
    "too many tokens",
    "too long",
)


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
        # A provider-wide failure can end a request after some of its vectors were already
        # recovered. Those vectors are still worth keeping, but they must not report the
        # provider as healthy again: the unhealthy-to-healthy transition is what schedules
        # missing-vector backfill, so a hidden outage would either trigger it while the
        # provider is still unavailable or suppress a later genuine recovery.
        terminal_failure: list[bool] = []
        embeddings = await self._call_with_retry(
            texts,
            terminal_failure=terminal_failure,
            **kwargs,
        )
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
        if out and not terminal_failure:
            self.is_healthy = True
        else:
            self.is_healthy = False
        return out

    async def _call_with_retry(
        self,
        texts: list[str],
        *,
        isolated: bool = False,
        terminal_failure: list[bool] | None = None,
        **kwargs,
    ) -> list[list[float] | None] | None:
        """Answer a batch, retrying transient failures and isolating input ones.

        ``isolated`` marks a single-item attempt made while isolating a rejected batch. Such an
        attempt keeps the store's transport backoff, because it is the last chance to keep that
        one vector; every other request leaves transport retries to the embedding wrapper and the
        provider SDK, which already own them.

        ``terminal_failure`` is an optional out-parameter. When the caller passes a list,
        it is set to ``[True]`` if this call gave up on a failure that no remaining item
        can explain (a provider-wide error, exhausted quota, or exhausted transient
        retries). The per-item isolation loop uses it to stop instead of asking for every
        remaining item and multiplying the rejected requests, and it forwards the same
        flag to its caller so a request that recovered some vectors before the failure is
        still not reported as healthy.
        """

        def stop() -> None:
            """Record that this failure ends the work for the whole batch."""
            if terminal_failure is not None:
                terminal_failure[:] = [True]

        for attempt in range(self.max_retries):
            try:
                result = await self.as_embedding(texts, **kwargs)
                if result and len(result) == len(texts):
                    return result
            except Exception as error:
                if self._is_transport_failure(error):
                    # A connection failure or a timeout describes the endpoint, not any single
                    # item, so this request is never split per item. Which layer retries it is
                    # explicit: an isolated single-item attempt keeps the store's backoff, while
                    # any other request is left to the layers below, which already own that retry
                    # -- the OpenAI model retries a batch max_retries + 1 times and the SDK retries
                    # each of those attempts, so a batch they answer after 12 attempts used to
                    # reach the provider 36 times with this store retrying on top of them.
                    if isolated:
                        if attempt < self.max_retries - 1:
                            await asyncio.sleep(2**attempt)
                        continue
                    self.logger.exception("Embedding request failed: transport error, not splitting")
                    self.is_healthy = False
                    stop()
                    break
                if self._is_rate_limited(error):
                    if attempt < self.max_retries - 1:
                        delay = 2**attempt
                        self.logger.warning(f"Embedding rate limited; retrying in {delay:.1f}s")
                        await asyncio.sleep(delay)
                        continue
                    self.logger.exception("Embedding request failed after exhausting rate-limit retries")
                    self.is_healthy = False
                    stop()
                    return None
                if (
                    self.quota_retry_delay is not None
                    and self._is_insufficient_quota(error)
                    and attempt < self.max_retries - 1
                ):
                    self.logger.warning(
                        f"Embedding quota exhausted; retrying in {self.quota_retry_delay:.1f}s",
                    )
                    await asyncio.sleep(self.quota_retry_delay)
                    continue
                if self._is_insufficient_quota(error):
                    # Quota is a service-level failure: no single item is to blame, so splitting the
                    # batch cannot isolate anything and would only multiply the rejected requests.
                    # This also covers the case where the quota retries above are exhausted.
                    self.logger.exception("Embedding request failed: quota exhausted")
                    self.is_healthy = False
                    stop()
                    return None
                if self._is_provider_wide_failure(error):
                    # Credentials, permissions and server outages describe the endpoint, not any
                    # single item: isolation cannot fix them, so give up on the whole batch
                    # instead of multiplying the rejected requests.
                    #
                    # A 5xx is deliberately *not* retried here. The embedding wrapper and the
                    # provider SDK already retry transient server errors, and every extra layer
                    # that retries multiplies the attempts and the waiting time of the layers
                    # below it -- a batch the SDK answered after 12 attempts became 36 once this
                    # store retried on top of it. Retry ownership therefore stays with those
                    # layers: a caller that wants a longer budget configures it there instead of
                    # stacking another silent loop here. A transport failure is left to them for
                    # the same reason; it only skips the per-item split.
                    self.logger.exception("Embedding request failed: provider-wide error, not splitting")
                    stop()
                    break
                if self._is_input_dependent_failure(error):
                    # A rejection a single input explains by itself is the one case the per-item
                    # split can resolve: isolating it keeps the remaining vectors usable.
                    self.logger.exception("Embedding request failed: input-dependent error")
                    self.is_healthy = False
                    if len(texts) > 1:
                        return await self._embed_individually(
                            texts,
                            terminal_failure=terminal_failure,
                            **kwargs,
                        )
                    return None
                # Anything else counts as provider-wide. Splitting cannot recover a vector unless
                # the failing input is identified, so an unrecognized error is never answered
                # with one request per item.
                self.logger.exception("Embedding request failed: unrecognized error, not splitting")
                stop()
                break
        self.is_healthy = False
        stop()
        return None

    async def _embed_individually(
        self,
        texts: list[str],
        *,
        terminal_failure: list[bool] | None = None,
        **kwargs,
    ) -> list[list[float] | None]:
        """Retry a rejected batch one item at a time so one bad item cannot drop the rest.

        Provider-side request errors, such as a per-item context-length limit, reject the
        whole batch and would otherwise cost every vector in it. Isolating each item bounds
        the damage to the item that actually failed and keeps the remaining vectors usable.
        Each single-item request goes back through ``_call_with_retry`` with ``isolated=True`` so
        a rate limit and a transient transport failure still get the store's backoff instead of
        losing the vector; a provider-wide failure ends the work for the whole batch rather than
        being asked for every remaining item, and a one-item request never splits again.
        When the loop stops on a provider-wide failure, ``terminal_failure`` is set for the
        caller so the vectors recovered before it are not read as a healthy provider.
        """
        results: list[list[float] | None] = []
        for index, text in enumerate(texts):
            terminal: list[bool] = []
            single = await self._call_with_retry(
                [text],
                isolated=True,
                terminal_failure=terminal,
                **kwargs,
            )
            if single:
                results.append(single[0])
                continue
            self.logger.error(f"Embedding request failed for a single item (chars={len(text)})")
            results.append(None)
            if terminal and terminal[0]:
                # No remaining item can explain a provider-wide failure, so asking for the
                # rest would only multiply the rejected requests. Keep the vectors already
                # recovered and leave the items that were never answered as None. The batch
                # caller has to learn about it as well: a partial success must not report
                # the provider as healthy again.
                if terminal_failure is not None:
                    terminal_failure[:] = [True]
                self.logger.error(
                    f"Stopping per-item isolation after {index + 1}/{len(texts)} items: "
                    "the failure is not specific to any input",
                )
                results.extend([None] * (len(texts) - index - 1))
                break
        return results

    @staticmethod
    def _is_transport_failure(error: Exception) -> bool:
        """Return whether an error describes a transport failure rather than a rejected input.

        Covers the built-in socket errors as well as the provider/httpx exceptions that describe
        a connection failure or a timeout. Transport failures are provider-wide: no individual
        input can be blamed, so the request is never split per item. Who retries them is explicit
        in the transport branch of ``_call_with_retry``: an isolated single-item attempt keeps the
        store's backoff because it is the last chance to keep that vector, while every other
        request leaves the retry to the layers below, which already own it.
        """
        if isinstance(error, (TimeoutError, ConnectionError, OSError)):
            return True
        return any(cls.__name__ in _TRANSPORT_FAILURE_TYPES for cls in type(error).__mro__)

    @staticmethod
    def _http_status(error: Exception) -> int | None:
        """Return the HTTP status an exception carries, across the provider SDK shapes.

        ``status_code`` covers the OpenAI-compatible SDKs, an integer ``code`` covers Google
        GenAI, ``response.status_code`` covers httpx-style wrappers, and the structured
        ``status_code`` embedded in the message covers AgentScope's DashScope ``RuntimeError``.
        Only integers are accepted: a string ``code`` is a provider error name such as
        ``insufficient_quota``, and reading it as a status would misclassify those errors.
        """
        for attribute in ("status_code", "code"):
            value = getattr(error, attribute, None)
            if isinstance(value, int) and not isinstance(value, bool):
                return value
        value = getattr(getattr(error, "response", None), "status_code", None)
        if isinstance(value, int) and not isinstance(value, bool):
            return value
        match = _STATUS_IN_MESSAGE.search(str(error))
        return int(match.group(1)) if match else None

    @staticmethod
    def _is_provider_wide_failure(error: Exception) -> bool:
        """Return whether an error describes the provider rather than a single input.

        Authentication, permission, missing model or route and server errors belong to the
        endpoint, the credentials or the deployment: no individual input can be blamed, so
        splitting the batch cannot isolate anything and would only multiply the rejected
        requests. Rate limits and quota exhaustion have their own branches above and never
        reach this classifier. A context-length ``400`` stays input-dependent on purpose,
        because that is the one status an over-limit item can explain by itself.
        """
        status = LocalEmbeddingStore._http_status(error)
        if status is not None and (status in _PROVIDER_WIDE_STATUS_CODES or 500 <= status < 600):
            return True
        return type(error).__name__ in (
            "AuthenticationError",
            "PermissionDeniedError",
            "NotFoundError",
            "InternalServerError",
        )

    @staticmethod
    def _is_input_dependent_failure(error: Exception) -> bool:
        """Return whether a single input can be blamed for the rejected request.

        Only a context-length rejection qualifies: it is the one error an over-long item
        explains by itself, and therefore the only one a per-item retry can resolve. It can
        arrive as a ``400`` or, from a provider that normalizes the rejection (Ollama reports
        ``413``), as an ambiguous status that is accepted only when the error names the input
        itself. Anything else stays on the provider-wide path, so a client never answers a
        service failure with one request per item.
        """
        status = LocalEmbeddingStore._http_status(error)
        if status is not None and status not in _INPUT_DEPENDENT_STATUS_CODES:
            if status not in _INPUT_DEPENDENT_AMBIGUOUS_STATUS_CODES:
                return False
            # The status alone does not name an input for an ambiguous code, so the error text
            # has to carry the signal; otherwise this is a request-level rejection.
            text = LocalEmbeddingStore._error_text(error)
            return any(marker in text for marker in _INPUT_DEPENDENT_MARKERS)
        if type(error).__name__ in _INPUT_DEPENDENT_ERROR_TYPES:
            return True
        text = LocalEmbeddingStore._error_text(error)
        return any(marker in text for marker in _INPUT_DEPENDENT_MARKERS)

    @staticmethod
    def _error_text(error: Exception) -> str:
        """Flatten an exception and its OpenAI-compatible body into one lowercase string."""
        parts = [str(error)]
        body = getattr(error, "body", None)
        if isinstance(body, dict):
            parts.append(str(body))
        return " ".join(parts).lower()

    @staticmethod
    def _error_code(error: Exception) -> str | None:
        """Return the provider's business error code, across the supported error shapes.

        ``code`` covers the OpenAI-compatible SDKs when the provider puts the error name
        there, ``error.code`` inside ``body`` covers the structured OpenAI payload, and the
        serialized message covers AgentScope's DashScope ``RuntimeError`` — the same shape
        ``_http_status`` already reads the status back out of. Without the last case a
        DashScope quota error is indistinguishable from an ordinary 429.
        """
        code = getattr(error, "code", None)
        if isinstance(code, str) and code:
            return code
        body = getattr(error, "body", None)
        if isinstance(body, dict):
            details = body.get("error", body)
            if isinstance(details, dict):
                inner = details.get("code")
                if isinstance(inner, str) and inner:
                    return inner
        match = _CODE_IN_MESSAGE.search(str(error))
        return match.group(1) if match else None

    @staticmethod
    def _is_rate_limited(error: Exception) -> bool:
        """Recognize an OpenAI-compatible 429 response without importing a provider SDK."""
        code = LocalEmbeddingStore._error_code(error)
        if code == "insufficient_quota":
            return False
        if code == "rate_limit_exceeded":
            return True
        return LocalEmbeddingStore._http_status(error) == 429

    @staticmethod
    def _is_insufficient_quota(error: Exception) -> bool:
        """Recognize OpenAI-compatible quota errors without importing a provider SDK."""
        return LocalEmbeddingStore._error_code(error) == "insufficient_quota"

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
