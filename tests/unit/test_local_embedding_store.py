"""Regression tests for LocalEmbeddingStore dimension and vector space handling."""

# pylint: disable=protected-access

import asyncio
import json
from types import SimpleNamespace

import httpx
import numpy as np
import openai

from reme.components.as_embedding import DashScopeAsEmbedding, OllamaAsEmbedding, OpenAIAsEmbedding
from reme.components.embedding_store.base_embedding_store import BaseEmbeddingStore
from reme.components.embedding_store.local_embedding_store import LocalEmbeddingStore
from reme.schema import EmbNode


class FakeAsEmbedding:
    """Fake AgentScope embedding component."""

    dimensions = 2
    vector_space_id = "fakespace000"

    def initialize_model(self):
        """Mirror the real component's idempotent initialization hook."""

    async def __call__(self, texts: list[str], **_kwargs):
        return [[1.0] if text == "bad" else [1.0, 0.0] for text in texts]


class BadHealthAsEmbedding:
    """Fake provider whose health probe returns the wrong dimension."""

    dimensions = 2
    vector_space_id = "fakespace000"

    def initialize_model(self):
        """Mirror the real component's idempotent initialization hook."""

    async def __call__(self, _texts: list[str], **_kwargs):
        return [[1.0]]


class FailingHealthAsEmbedding:
    """Fake provider that records a failed health-check attempt."""

    dimensions = 2
    vector_space_id = "fakespace000"

    def __init__(self):
        self.calls = 0

    def initialize_model(self):
        """Mirror the real component's idempotent initialization hook."""

    async def __call__(self, _texts: list[str], **_kwargs):
        self.calls += 1
        raise ConnectionError("not ready")


class FakeProviderModel:
    """Stand-in for a constructed AgentScope embedding model object."""

    def __init__(
        self,
        model: str,
        dimensions: int = 2,
        base_url: str = "https://api.openai.com/v1",
    ):
        self.model = model
        self.dimensions = dimensions
        self.credential = SimpleNamespace(base_url=base_url)


class InsufficientQuotaError(Exception):
    """OpenAI-compatible quota error used without importing the provider SDK."""

    status_code = 429
    body = {"error": {"code": "insufficient_quota"}}


class RateLimitError(Exception):
    """OpenAI-compatible 429 error used without importing the provider SDK."""

    status_code = 429


class RateLimitedThenSuccessAsEmbedding:
    """Fail once with a 429, then return a valid embedding."""

    dimensions = 2

    def __init__(self):
        self.calls = 0

    async def __call__(self, texts: list[str], **_kwargs):
        self.calls += 1
        if self.calls == 1:
            raise RateLimitError("Requests are too frequent")
        return [[1.0, 0.0] for _ in texts]


class AlwaysRateLimitedAsEmbedding:
    """Always fail with a 429."""

    dimensions = 2

    def __init__(self):
        self.calls = 0

    async def __call__(self, _texts: list[str], **_kwargs):
        self.calls += 1
        raise RateLimitError("Requests are too frequent")


class QuotaThenSuccessAsEmbedding:
    """Fail once for quota, then return a valid embedding."""

    dimensions = 2

    def __init__(self):
        self.calls = 0

    async def __call__(self, texts: list[str], **_kwargs):
        self.calls += 1
        if self.calls == 1:
            raise InsufficientQuotaError("quota exhausted")
        return [[1.0, 0.0] for _ in texts]


class ContextLengthError(Exception):
    """OpenAI-compatible 400 raised when one item exceeds the provider's per-item window."""

    status_code = 400
    body = {"error": {"message": "the input length exceeds the context length"}}


class PerItemLimitAsEmbedding:
    """Reject a whole request when any single text is over the provider's per-item limit.

    Mirrors an OpenAI-compatible gateway: one over-limit item makes the provider answer
    HTTP 400 for the entire batch, so a client that gives up on the batch loses every
    vector in it.
    """

    dimensions = 2
    vector_space_id = "fakespace000"
    max_chars = 32

    def __init__(self):
        self.batch_sizes: list[int] = []

    def initialize_model(self):
        """Mirror the real component's idempotent initialization hook."""

    async def __call__(self, texts: list[str], **_kwargs):
        self.batch_sizes.append(len(texts))
        if any(len(text) > self.max_chars for text in texts):
            raise ContextLengthError("the input length exceeds the context length")
        return [[1.0, 0.0] for _ in texts]


class AlwaysInsufficientQuotaAsEmbedding:
    """Always fail with quota exhaustion, recording every requested batch size."""

    dimensions = 2

    def __init__(self):
        self.batch_sizes: list[int] = []

    async def __call__(self, texts: list[str], **_kwargs):
        self.batch_sizes.append(len(texts))
        raise InsufficientQuotaError("quota exhausted")


class PerItemLimitThenTransientErrorAsEmbedding:
    """Reject an over-limit batch, then fail the first single-item request once.

    Mirrors the review case: the batch is rejected because one item exceeds the per-item
    window, and a sibling item then hits a transient provider failure while retried alone.
    """

    dimensions = 2
    vector_space_id = "fakespace000"
    max_chars = 32

    def __init__(self, transient: Exception):
        self.transient = transient
        self.batch_sizes: list[int] = []
        self.transient_failures = 0

    def initialize_model(self):
        """Mirror the real component's idempotent initialization hook."""

    async def __call__(self, texts: list[str], **_kwargs):
        self.batch_sizes.append(len(texts))
        if any(len(text) > self.max_chars for text in texts):
            raise ContextLengthError("the input length exceeds the context length")
        if len(texts) == 1 and self.transient_failures == 0:
            self.transient_failures += 1
            raise self.transient
        return [[1.0, 0.0] for _ in texts]


class PerItemLimitThenQuotaAsEmbedding:
    """Reject an over-limit batch, recover the first isolated item, then exhaust the quota.

    Mirrors the review case: isolation starts, the first single-item request comes back, and
    every later call then fails provider-wide. Asking for the remaining items would only
    multiply the rejected requests, because no single item explains a quota failure.
    """

    dimensions = 2
    vector_space_id = "fakespace000"
    max_chars = 32

    def __init__(self):
        self.batch_sizes: list[int] = []
        self.singles = 0

    def initialize_model(self):
        """Mirror the real component's idempotent initialization hook."""

    async def __call__(self, texts: list[str], **_kwargs):
        self.batch_sizes.append(len(texts))
        if any(len(text) > self.max_chars for text in texts):
            raise ContextLengthError("the input length exceeds the context length")
        self.singles += 1
        if self.singles > 1:
            raise InsufficientQuotaError("quota exhausted")
        return [[1.0, 0.0] for _ in texts]


class AuthenticationError(Exception):
    """OpenAI-compatible 401 error used without importing the provider SDK."""

    status_code = 401


class PermissionDeniedError(Exception):
    """OpenAI-compatible 403 error used without importing the provider SDK."""

    status_code = 403


class InternalServerError(Exception):
    """OpenAI-compatible 500 error used without importing the provider SDK."""

    status_code = 500


class NotFoundError(Exception):
    """OpenAI-compatible 404 error, e.g. a missing model or endpoint."""

    status_code = 404


class APIError(Exception):
    """Base of the OpenAI SDK error hierarchy, reproduced without importing the SDK."""


class APIConnectionError(APIError):
    """OpenAI SDK connection failure: it does not inherit from the built-in ConnectionError."""


class APITimeoutError(APIConnectionError):
    """OpenAI SDK timeout, a subclass of APIConnectionError."""


class AlwaysProviderWideFailureAsEmbedding:
    """Always fail with the same provider-wide error, recording every requested batch size."""

    dimensions = 2

    def __init__(self, error: Exception):
        self.error = error
        self.batch_sizes: list[int] = []

    async def __call__(self, texts: list[str], **_kwargs):
        self.batch_sizes.append(len(texts))
        raise self.error


class TransientThenSuccessAsEmbedding:
    """Fail the first `failures` calls with one error, then return valid embeddings."""

    dimensions = 2
    vector_space_id = "fakespace000"

    def __init__(self, error: Exception, failures: int = 1):
        self.error = error
        self.failures = failures
        self.batch_sizes: list[int] = []

    def initialize_model(self):
        """Mirror the real component's idempotent initialization hook."""

    async def __call__(self, texts: list[str], **_kwargs):
        self.batch_sizes.append(len(texts))
        if len(self.batch_sizes) <= self.failures:
            raise self.error
        return [[1.0, 0.0] for _ in texts]


_REAL_OPENAI_ASYNC_CLIENT = openai.AsyncClient
_OFFLINE_OPENAI_BASE_URL = "http://127.0.0.1:9/v1"


class CountingOpenAIWrapper(OpenAIAsEmbedding):
    """Real OpenAI wrapper that records how often the store asked it for embeddings."""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.calls = 0

    async def __call__(self, inputs, **kwargs):
        self.calls += 1
        return await super().__call__(inputs, **kwargs)


class OfflineOpenAIEndpoint:
    """Real openai.AsyncClient wired to an offline httpx transport that counts provider attempts.

    Gives the wrapper-and-SDK layers their real code path, so a regression test can assert how
    many attempts the provider would see instead of trusting the store's own loop.
    """

    def __init__(self, handler, sdk_max_retries: int = 2) -> None:
        self.requests = 0
        self.handler = handler
        self.client = _REAL_OPENAI_ASYNC_CLIENT(
            api_key="test-key",
            base_url=_OFFLINE_OPENAI_BASE_URL,
            http_client=httpx.AsyncClient(transport=httpx.MockTransport(self._handle)),
            max_retries=sdk_max_retries,
        )

    def _handle(self, request):
        self.requests += 1
        return self.handler(request)

    def install(self, monkeypatch) -> "OfflineOpenAIEndpoint":
        """Route every openai.AsyncClient built during this test to the offline client."""
        monkeypatch.setattr(openai, "AsyncClient", lambda **kwargs: self.client)
        return self


def offline_openai_endpoint(handler) -> OfflineOpenAIEndpoint:
    """Build an offline endpoint whose transport follows ``handler``."""
    return OfflineOpenAIEndpoint(handler)


def openai_wrapper(name: str) -> CountingOpenAIWrapper:
    """Build a real OpenAI embedding wrapper pointed at the offline endpoint."""
    return CountingOpenAIWrapper(
        name=name,
        model="text-embedding-v4",
        backend="openai",
        dimensions=2,
        credential={"base_url": _OFFLINE_OPENAI_BASE_URL, "api_key": "test-key"},
        retry_delay=0,
        pass_dimensions=False,
    )


def embedding_payload(request) -> httpx.Response:
    """Answer an embeddings request the way the provider would."""
    body = json.loads(request.content.decode("utf-8"))
    inputs = body["input"]
    if isinstance(inputs, str):
        inputs = [inputs]
    return httpx.Response(
        200,
        json={
            "object": "list",
            "data": [
                {"object": "embedding", "index": position, "embedding": [0.0, 1.0]} for position in range(len(inputs))
            ],
            "model": body.get("model", "text-embedding-v4"),
            "usage": {"prompt_tokens": len(inputs), "total_tokens": len(inputs)},
        },
    )


def context_length_rejection() -> httpx.Response:
    """Answer a batch the way an OpenAI-compatible provider reports an over-long input."""
    return httpx.Response(
        400,
        json={
            "error": {
                "message": (
                    "This model's maximum context length is 8191 tokens. " "Please reduce the length of the input."
                ),
                "type": "invalid_request_error",
                "param": None,
                "code": "context_length_exceeded",
            },
        },
    )


def read_timeout(request) -> httpx.Response:
    """Fail one attempt the way a read timeout reaches the SDK."""
    raise httpx.ReadTimeout("simulated read timeout", request=request)


class BadNodeEmbeddingStore(BaseEmbeddingStore):
    """Embedding store that returns wrong-dimensional vectors."""

    dimensions = 2

    async def health_check(self, timeout: float = 2.0) -> bool:
        return True

    async def get_embeddings(self, input_text: list[str], **_kwargs):
        return [np.array([1.0], dtype=np.float16) for _ in input_text]


def run(coro):
    """Run an async test body."""
    return asyncio.run(coro)


def test_truncate_uses_cjk_aware_integer_budget():
    """Truncation should preserve ASCII behavior and budget non-ASCII text."""
    store = BadNodeEmbeddingStore(name="t_base_embedding_truncate", max_input_length=10)

    assert store._truncate("abcdefghijk") == "abcdefghij"
    assert store._truncate("中文中文中文中文") == "中文中文中文"
    assert store._truncate("éabcdefghij") == "éabcdefgh"

    store.max_input_length = -1
    assert store._truncate("text") == ""
    assert store._truncate("中文") == ""


def test_compute_batch_rejects_embeddings_with_wrong_dimension():
    """Provider results with wrong dimensions are not padded, truncated, or cached."""

    async def go():
        store = LocalEmbeddingStore(name="t_local_embedding_dim")
        store.as_embedding = FakeAsEmbedding()

        results = await store._compute_batch(
            [
                (0, "ok", "ok-cache-key"),
                (1, "bad", "bad-cache-key"),
            ],
        )

        assert len(results) == 1
        assert results[0][0] == 0
        assert results[0][2].tolist() == [1.0, 0.0]
        assert isinstance(results[0][2], np.ndarray)

    run(go())


def test_base_get_node_embeddings_rejects_wrong_dimension():
    """Base node assignment should not accept wrong-dimensional vectors."""

    async def go():
        store = BadNodeEmbeddingStore(name="t_base_embedding_dim")
        node = EmbNode(text="bad")

        await store.get_node_embeddings([node])

        assert node.embedding is None

    run(go())


def test_health_check_rejects_wrong_dimension():
    """Health check should fail when the provider returns the wrong vector length."""

    async def go():
        store = LocalEmbeddingStore(name="t_local_embedding_health_dim")
        store.as_embedding = BadHealthAsEmbedding()

        assert await store.health_check() is False
        assert store.is_healthy is False

    run(go())


def test_health_check_starts_timeout_after_provider_initialization(monkeypatch):
    """One-time client construction must not consume the request timeout."""

    async def go():
        events = []

        class InitializingAsEmbedding(FakeAsEmbedding):
            """Record initialization and provider-call ordering."""

            def initialize_model(self):
                events.append("initialized")

            async def __call__(self, texts: list[str], **_kwargs):
                events.append("remote request")
                return [[1.0, 0.0] for _ in texts]

        original_wait_for = asyncio.wait_for

        async def checked_wait_for(awaitable, timeout):
            assert events == ["initialized"]
            assert timeout == 5.0
            return await original_wait_for(awaitable, timeout)

        store = LocalEmbeddingStore(name="t_local_embedding_health_timeout_scope")
        store.as_embedding = InitializingAsEmbedding()
        monkeypatch.setattr(asyncio, "wait_for", checked_wait_for)

        assert await store.health_check(timeout=5.0) is True
        assert events == ["initialized", "remote request"]

    run(go())


def test_health_check_makes_one_attempt():
    """A failed startup probe does not add hidden retries."""

    async def go():
        provider = FailingHealthAsEmbedding()
        store = LocalEmbeddingStore(
            name="t_local_embedding_health_retry",
            health_check_timeout=3.0,
        )
        store.as_embedding = provider

        assert await store.health_check() is False
        assert provider.calls == 1

    run(go())


def test_insufficient_quota_waits_sixty_seconds_before_retry(monkeypatch):
    """Quota exhaustion uses the dedicated delay before ReMe retries."""

    async def go():
        sleeps = []

        async def fake_sleep(delay):
            sleeps.append(delay)

        store = LocalEmbeddingStore(
            name="t_local_embedding_quota",
            max_retries=2,
            quota_retry_delay=60.0,
        )
        embedding = QuotaThenSuccessAsEmbedding()
        store.as_embedding = embedding
        monkeypatch.setattr(asyncio, "sleep", fake_sleep)

        result = await store._call_with_retry(["text"])

        assert result == [[1.0, 0.0]]
        assert embedding.calls == 2
        assert sleeps == [60.0]

    run(go())


def test_insufficient_quota_does_not_retry_without_opt_in(monkeypatch):
    """The default store behavior remains unchanged for embedded consumers."""

    async def go():
        sleeps = []

        async def fake_sleep(delay):
            sleeps.append(delay)

        store = LocalEmbeddingStore(name="t_local_embedding_default_quota", max_retries=2)
        embedding = QuotaThenSuccessAsEmbedding()
        store.as_embedding = embedding
        monkeypatch.setattr(asyncio, "sleep", fake_sleep)

        result = await store._call_with_retry(["text"])

        assert result is None
        assert embedding.calls == 1
        assert not sleeps

    run(go())


def test_rate_limit_retries_with_backoff_without_opt_in(monkeypatch):
    """A 429 must retry like a network error, with no quota_retry_delay required."""

    async def go():
        sleeps = []

        async def fake_sleep(delay):
            sleeps.append(delay)

        store = LocalEmbeddingStore(name="t_local_embedding_rate_limit", max_retries=2)
        embedding = RateLimitedThenSuccessAsEmbedding()
        store.as_embedding = embedding
        monkeypatch.setattr(asyncio, "sleep", fake_sleep)

        result = await store._call_with_retry(["text"])

        assert result == [[1.0, 0.0]]
        assert embedding.calls == 2
        assert sleeps == [1.0]

    run(go())


def test_is_rate_limited_recognizes_status_code_or_body_and_nothing_else():
    """A 429 is recognized either by status_code or a body-embedded code, never by guesswork."""

    class StatusCodeError(Exception):
        """A 429 identified by an HTTP status code attribute."""

        status_code = 429

    class BodyCodeError(Exception):
        """A 429 identified only by a body-embedded error code."""

        body = {"error": {"code": "rate_limit_exceeded"}}

    assert LocalEmbeddingStore._is_rate_limited(StatusCodeError())
    assert LocalEmbeddingStore._is_rate_limited(BodyCodeError())
    assert not LocalEmbeddingStore._is_rate_limited(ValueError("unrelated"))


def test_is_rate_limited_defers_to_insufficient_quota_on_status_code_429():
    """An insufficient_quota error carrying status_code=429 is not the generic rate-limit case."""

    assert not LocalEmbeddingStore._is_rate_limited(InsufficientQuotaError("quota exhausted"))
    assert LocalEmbeddingStore._is_insufficient_quota(InsufficientQuotaError("quota exhausted"))


def test_rate_limit_exhausts_retries_and_reports_unhealthy(monkeypatch):
    """Repeated 429s still give up after max_retries, unlike an unclassified error."""

    async def go():
        sleeps = []

        async def fake_sleep(delay):
            sleeps.append(delay)

        store = LocalEmbeddingStore(name="t_local_embedding_rate_limit_exhausted", max_retries=3)
        embedding = AlwaysRateLimitedAsEmbedding()
        store.as_embedding = embedding
        monkeypatch.setattr(asyncio, "sleep", fake_sleep)

        result = await store._call_with_retry(["text"])

        assert result is None
        assert store.is_healthy is False
        assert embedding.calls == 3
        assert sleeps == [1.0, 2.0]

    run(go())


def test_vector_space_id_separates_models_of_equal_dimension():
    """Two models of the same width must not claim the same vector space."""
    common = {"backend": "openai", "dimensions": 1024, "credential": {"base_url": "https://example.com/v1"}}
    v3 = OpenAIAsEmbedding(name="t_space_v3", model="text-embedding-v3", **common)
    v4 = OpenAIAsEmbedding(name="t_space_v4", model="text-embedding-v4", **common)

    assert v3.dimensions == v4.dimensions
    assert v3.vector_space_id != v4.vector_space_id


def test_vector_space_id_separates_endpoints_of_one_model_name():
    """The same model name served by two endpoints is two vector spaces."""
    common = {"backend": "openai", "model": "text-embedding-v4", "dimensions": 1024}
    official = OpenAIAsEmbedding(name="t_space_a", credential={"base_url": "https://example.com/v1"}, **common)
    self_hosted = OpenAIAsEmbedding(name="t_space_b", credential={"base_url": "http://127.0.0.1:8000/v1"}, **common)

    assert official.vector_space_id != self_hosted.vector_space_id


def test_vector_space_id_ignores_trailing_slash_and_api_key():
    """Cosmetic and secret credential changes must not invalidate stored vectors."""
    common = {"backend": "openai", "model": "text-embedding-v4", "dimensions": 1024}
    first = OpenAIAsEmbedding(
        name="t_space_c",
        credential={"base_url": "https://example.com/v1", "api_key": "key-one"},
        **common,
    )
    second = OpenAIAsEmbedding(
        name="t_space_d",
        credential={"base_url": "https://example.com/v1/", "api_key": "key-two"},
        **common,
    )

    assert first.vector_space_id == second.vector_space_id


def test_vector_space_id_follows_a_model_swapped_in_after_start():
    """A provider replaced at runtime must win over the original kwargs."""
    embedding = OpenAIAsEmbedding(name="t_space_swap", backend="openai", model="v3", dimensions=2)
    before = embedding.vector_space_id

    # Mirrors Application.update_component("as_embedding", "default", model=<new provider>).
    embedding.model = FakeProviderModel("v4")

    assert embedding.vector_space_id != before


def test_vector_space_id_is_stable_across_lazy_provider_construction():
    """Constructing the configured provider must not look like a model switch."""
    embedding = OpenAIAsEmbedding(
        name="t_space_lazy",
        backend="openai",
        model="v3",
        dimensions=2,
        credential={"base_url": "https://example.com/v1"},
    )
    before = embedding.vector_space_id

    embedding.model = FakeProviderModel("v3", base_url="https://example.com/v1")

    assert embedding.vector_space_id == before


def test_vector_space_id_resolves_default_endpoint_before_lazy_construction():
    """Credential defaults must not change the cache namespace on the first request."""
    embedding = DashScopeAsEmbedding(
        name="t_space_default_endpoint",
        model="text-embedding-v3",
        dimensions=1024,
        credential={"api_key": "test"},
    )
    before = embedding.vector_space_id

    embedding._ensure_model()

    assert embedding.vector_space_id == before


def test_openai_vector_space_id_uses_sdk_resolved_endpoint(monkeypatch):
    """OPENAI_BASE_URL must separate caches and stay stable after lazy construction."""
    clients = []
    ids = []
    try:
        for endpoint in ("https://provider-a.example/v1", "https://provider-b.example/v1"):
            monkeypatch.setenv("OPENAI_BASE_URL", endpoint)
            embedding = OpenAIAsEmbedding(
                name="t_space_openai_env",
                backend="openai",
                model="text-embedding-3-small",
                dimensions=1536,
                credential={"api_key": "test"},
            )
            before = embedding.vector_space_id

            embedding._ensure_model()
            clients.append(embedding.model.client)

            assert str(embedding.model.client.base_url).rstrip("/") == endpoint
            assert embedding.vector_space_id == before
            ids.append(before)

        assert ids[0] != ids[1]
    finally:
        for client in clients:
            run(client.close())


def test_openai_vector_space_id_uses_sdk_default_endpoint(monkeypatch):
    """The SDK default URL must not change the namespace on first construction."""
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    embedding = OpenAIAsEmbedding(
        name="t_space_openai_default",
        backend="openai",
        model="text-embedding-3-small",
        dimensions=1536,
        credential={"api_key": "test"},
    )
    before = embedding.vector_space_id

    embedding._ensure_model()
    try:
        assert str(embedding.model.client.base_url).rstrip("/") == "https://api.openai.com/v1"
        assert embedding.vector_space_id == before
    finally:
        run(embedding.model.client.close())


def test_ollama_vector_space_id_uses_sdk_resolved_endpoint(monkeypatch):
    """OLLAMA_HOST must separate caches and stay stable after lazy construction."""
    ids = []
    for endpoint in ("http://provider-a.example:11434", "http://provider-b.example:11434"):
        monkeypatch.setenv("OLLAMA_HOST", endpoint)
        embedding = OllamaAsEmbedding(
            name="t_space_ollama_env",
            backend="ollama",
            model="nomic-embed-text",
            dimensions=768,
            credential={},
        )
        before = embedding.vector_space_id

        embedding._ensure_model()

        assert embedding.model.host is None
        assert embedding.vector_space_id == before
        ids.append(before)

    assert ids[0] != ids[1]


def test_ollama_vector_space_id_uses_sdk_default_endpoint(monkeypatch):
    """The Ollama SDK default URL must remain stable after lazy construction."""
    monkeypatch.delenv("OLLAMA_HOST", raising=False)
    embedding = OllamaAsEmbedding(
        name="t_space_ollama_default",
        backend="ollama",
        model="nomic-embed-text",
        dimensions=768,
        credential={},
    )
    before = embedding.vector_space_id

    embedding._ensure_model()

    assert embedding.vector_space[-1] == "http://127.0.0.1:11434"
    assert embedding.vector_space_id == before


def test_cache_is_saved_and_restored_per_vector_space(monkeypatch, tmp_path):
    """Switching models persists the old cache and restores it when switched back."""

    async def go():
        monkeypatch.setattr(
            LocalEmbeddingStore,
            "component_metadata_path",
            property(lambda _self: tmp_path),
        )
        embedding = OpenAIAsEmbedding(name="t_space_store", backend="openai", model="v3", dimensions=2)
        store = LocalEmbeddingStore(name="t_local_space")
        store.as_embedding = embedding
        await store.load()

        key = store._cache_key("hello")
        store._cache_put(key, np.array([1.0, 0.0], dtype=np.float16))
        v3_path = store.cache_path

        embedding.model = FakeProviderModel("v4")
        await store._sync_cache_space()

        assert store._cache_key("hello") == key
        assert store.cache_path != v3_path
        assert store._cache_get(store._cache_key("hello")) is None
        assert v3_path.exists()

        embedding.model = FakeProviderModel("v3")
        await store._sync_cache_space()

        np.testing.assert_array_equal(store._cache_get(key), np.array([1.0, 0.0], dtype=np.float16))

    run(go())


def test_cache_space_is_rechecked_after_async_load(monkeypatch, tmp_path):
    """A provider switch during disk I/O must not publish the stale namespace."""

    async def go():
        monkeypatch.setattr(
            LocalEmbeddingStore,
            "component_metadata_path",
            property(lambda _self: tmp_path),
        )
        embedding = OpenAIAsEmbedding(name="t_space_load_race", backend="openai", model="v3", dimensions=2)
        store = LocalEmbeddingStore(name="t_local_load_race")
        store.as_embedding = embedding
        await store.load()

        embedding.model = FakeProviderModel("v4")
        v4_space = embedding.vector_space_id
        np.savez(
            store._cache_path(v4_space),
            keys=np.array([store._cache_key("hello")]),
            embeddings=np.array([[4.0, 0.0]], dtype=np.float16),
        )
        original_to_thread = asyncio.to_thread

        async def switch_during_load(func, *args):
            if getattr(func, "__name__", "") == "_load_sync":
                embedding.model = FakeProviderModel("v3")
            return await original_to_thread(func, *args)

        monkeypatch.setattr(asyncio, "to_thread", switch_during_load)
        await store._sync_cache_space()

        assert store._cache_space == embedding.vector_space_id
        assert not store._cache

    run(go())


def test_whole_request_retries_after_vector_space_changes_between_batches():
    """Completed batches must be discarded when a later batch changes vector space."""

    async def go():
        embedding = FakeAsEmbedding()
        embedding.vector_space_id = "v3"
        store = LocalEmbeddingStore(name="t_local_write_race", max_batch_size=1, enable_cache=False)
        store.as_embedding = embedding
        store._cache_space = embedding.vector_space_id
        calls = 0

        async def switch_during_second_batch(batch, **_kwargs):
            nonlocal calls
            calls += 1
            if calls == 2:
                embedding.vector_space_id = "v4"
            idx, _text, key = batch[0]
            version = 3.0 if calls < 3 else 4.0
            return [(idx, key, np.array([version, 0.0], dtype=np.float16))]

        store._compute_batch = switch_during_second_batch
        results = await store.get_embeddings(["first", "second"])

        assert calls == 4
        assert store._cache_space == embedding.vector_space_id
        for result in results:
            np.testing.assert_array_equal(result, np.array([4.0, 0.0], dtype=np.float16))

    run(go())


def test_whole_request_rereads_cache_after_vector_space_changes(monkeypatch, tmp_path):
    """A cache hit from the old space must not survive a later provider switch."""

    async def go():
        monkeypatch.setattr(
            LocalEmbeddingStore,
            "component_metadata_path",
            property(lambda _self: tmp_path),
        )
        embedding = FakeAsEmbedding()
        embedding.vector_space_id = "v3"
        store = LocalEmbeddingStore(name="t_local_cache_race", enable_cache=True)
        store.as_embedding = embedding
        store._cache_space = embedding.vector_space_id
        first_key = store._cache_key("first")
        store._cache[first_key] = np.array([3.0, 0.0], dtype=np.float16)
        calls = 0

        async def switch_on_miss(batch, **_kwargs):
            nonlocal calls
            calls += 1
            if calls == 1:
                embedding.vector_space_id = "v4"
            version = 3.0 if calls == 1 else 4.0
            return [(idx, key, np.array([version, 0.0], dtype=np.float16)) for idx, _text, key in batch]

        store._compute_batch = switch_on_miss
        results = await store.get_embeddings(["first", "second"])

        assert calls == 2
        for result in results:
            np.testing.assert_array_equal(result, np.array([4.0, 0.0], dtype=np.float16))

    run(go())


def test_whole_request_stops_retrying_when_vector_space_keeps_changing():
    """Continuous configuration churn must discard the whole request instead of blocking forever."""

    async def go():
        embedding = FakeAsEmbedding()
        embedding.vector_space_id = "v3"
        store = LocalEmbeddingStore(name="t_local_write_churn")
        store.as_embedding = embedding
        store._cache_space = embedding.vector_space_id
        calls = 0

        async def change_space_every_time(_batch, **_kwargs):
            nonlocal calls
            calls += 1
            embedding.vector_space_id = f"v{calls + 3}"
            return [(0, "key", np.array([float(calls), 0.0], dtype=np.float16))]

        store._compute_batch = change_space_every_time
        results = await store.get_embeddings(["text"])

        assert calls == 3
        assert results == [None]
        assert "key" not in store._cache

    run(go())


def test_start_ignores_cache_file_without_vector_space_tag(monkeypatch, tmp_path):
    """An unattributable legacy cache is ignored without deleting derived data."""

    async def go():
        monkeypatch.setattr(
            LocalEmbeddingStore,
            "component_metadata_path",
            property(lambda _self: tmp_path),
        )
        store = LocalEmbeddingStore(name="t_local_untagged")
        store.as_embedding = FakeAsEmbedding()
        untagged = tmp_path / f"{store.name}_{store.cache_version}.npz"
        untagged.write_bytes(b"vectors from an unknown model")

        await store._start()

        assert untagged.exists()
        assert not store._cache

    run(go())


def test_batch_failure_falls_back_to_per_item_embedding():
    """One over-limit item must cost one vector, not its whole batch."""

    async def go():
        provider = PerItemLimitAsEmbedding()
        store = LocalEmbeddingStore(name="t_local_embedding_poison_batch", enable_cache=False)
        store.as_embedding = provider

        results = await store._call_with_retry(["ok", "x" * 64, "also ok"])

        assert results == [[1.0, 0.0], None, [1.0, 0.0]]
        assert provider.batch_sizes == [3, 1, 1, 1]

    run(go())


def test_poison_chunk_keeps_its_batch_mates():
    """A rejected chunk must not clear the vectors of the chunks beside it."""

    async def go():
        store = LocalEmbeddingStore(name="t_local_embedding_poison_nodes", enable_cache=False)
        store.as_embedding = PerItemLimitAsEmbedding()
        nodes = [EmbNode(text="first"), EmbNode(text="y" * 64), EmbNode(text="third")]

        await store.get_node_embeddings(nodes)

        assert nodes[0].embedding is not None
        assert nodes[1].embedding is None
        assert nodes[2].embedding is not None

    run(go())


def test_single_item_failure_still_reports_a_failed_batch():
    """A one-item request has nothing to isolate, so the existing contract stands."""

    async def go():
        store = LocalEmbeddingStore(name="t_local_embedding_single_poison", enable_cache=False)
        store.as_embedding = PerItemLimitAsEmbedding()

        assert await store._call_with_retry(["x" * 64]) is None

    run(go())


def test_rate_limited_batch_is_not_split_into_per_item_requests(monkeypatch):
    """A 429 is a provider-level failure; splitting it would multiply the load."""

    async def go():
        sleeps = []

        async def fake_sleep(delay):
            sleeps.append(delay)

        store = LocalEmbeddingStore(name="t_local_embedding_rate_limit_batch", max_retries=2)
        embedding = AlwaysRateLimitedAsEmbedding()
        store.as_embedding = embedding
        monkeypatch.setattr(asyncio, "sleep", fake_sleep)

        assert await store._call_with_retry(["a", "b"]) is None
        assert embedding.calls == 2
        assert sleeps == [1.0]

    run(go())


def test_insufficient_quota_batch_is_not_split_into_per_item_requests(monkeypatch):
    """Quota exhaustion is a service-level failure; splitting it would multiply the load."""

    async def go():
        sleeps = []

        async def fake_sleep(delay):
            sleeps.append(delay)

        store = LocalEmbeddingStore(name="t_local_embedding_quota_batch", max_retries=2)
        embedding = AlwaysInsufficientQuotaAsEmbedding()
        store.as_embedding = embedding
        monkeypatch.setattr(asyncio, "sleep", fake_sleep)

        assert await store._call_with_retry(["a", "b", "c"]) is None
        assert embedding.batch_sizes == [3]
        assert store.is_healthy is False
        assert not sleeps

    run(go())


def test_exhausted_quota_retries_still_do_not_split_the_batch(monkeypatch):
    """Once the opt-in quota retries run out, the store gives up instead of splitting."""

    async def go():
        sleeps = []

        async def fake_sleep(delay):
            sleeps.append(delay)

        store = LocalEmbeddingStore(
            name="t_local_embedding_quota_batch_exhausted",
            max_retries=3,
            quota_retry_delay=5.0,
        )
        embedding = AlwaysInsufficientQuotaAsEmbedding()
        store.as_embedding = embedding
        monkeypatch.setattr(asyncio, "sleep", fake_sleep)

        assert await store._call_with_retry(["a", "b"]) is None
        assert embedding.batch_sizes == [2, 2, 2]
        assert sleeps == [5.0, 5.0]

    run(go())


def test_per_item_fallback_keeps_a_rate_limited_vector(monkeypatch):
    """A 429 on one isolated item still gets the store's backoff and keeps its vector.

    Rate limiting is the failure class the store still owns: no lower layer reports it as a
    provider-wide error worth its own budget, so the retry lives here. Connection errors and
    timeouts are left to the wrapper and the SDK instead; see the transport tests above.
    """

    async def go():
        sleeps = []

        async def fake_sleep(delay):
            sleeps.append(delay)

        provider = PerItemLimitThenTransientErrorAsEmbedding(
            RateLimitError("Requests are too frequent"),
        )
        store = LocalEmbeddingStore(
            name="t_local_embedding_fallback_rate_limit",
            max_retries=3,
            enable_cache=False,
        )
        store.as_embedding = provider
        monkeypatch.setattr(asyncio, "sleep", fake_sleep)

        results = await store._call_with_retry(["ok", "x" * 64, "also ok"])

        assert results == [[1.0, 0.0], None, [1.0, 0.0]]
        assert provider.batch_sizes == [3, 1, 1, 1, 1]
        assert provider.transient_failures == 1
        assert sleeps == [1.0]

    run(go())


def test_provider_wide_credentials_errors_are_not_split_into_per_item_requests():
    """401/403 belong to the credentials: splitting ten items only multiplies the failures."""

    async def go():
        for error in (
            AuthenticationError("invalid api key"),
            PermissionDeniedError("no access to model"),
        ):
            provider = AlwaysProviderWideFailureAsEmbedding(error)
            store = LocalEmbeddingStore(name="t_local_embedding_provider_wide", max_retries=3)
            store.as_embedding = provider

            assert await store._call_with_retry([f"item{i}" for i in range(10)]) is None
            assert provider.batch_sizes == [10]
            assert store.is_healthy is False

    run(go())


def test_provider_wide_server_errors_are_not_retried_without_splitting(monkeypatch):
    """A 500 is left to the layers that own retries, so this store asks exactly once."""

    async def go():
        sleeps = []

        async def fake_sleep(delay):
            sleeps.append(delay)

        provider = AlwaysProviderWideFailureAsEmbedding(InternalServerError("upstream unavailable"))
        store = LocalEmbeddingStore(name="t_local_embedding_provider_server", max_retries=3)
        store.as_embedding = provider
        monkeypatch.setattr(asyncio, "sleep", fake_sleep)

        assert await store._call_with_retry([f"item{i}" for i in range(10)]) is None
        # One request only: the wrapper and the SDK already retry a 5xx, and retrying here
        # as well multiplied their attempts (12 -> 36 in the reported reproduction).
        assert provider.batch_sizes == [10]
        assert sleeps == []
        assert store.is_healthy is False

    run(go())


def test_sdk_transport_errors_are_provider_wide_and_not_retried_here(monkeypatch):
    """SDK connection/timeout errors are provider-wide: never split, and not retried here.

    The wrapper and the SDK already retry a transport failure, so the store answers once and
    leaves the retry budget to those layers instead of multiplying their attempts.
    """

    async def go():
        sleeps = []

        async def fake_sleep(delay):
            sleeps.append(delay)

        monkeypatch.setattr(asyncio, "sleep", fake_sleep)

        for error in (APIConnectionError("connection reset"), APITimeoutError("timed out")):
            sleeps.clear()
            provider = AlwaysProviderWideFailureAsEmbedding(error)
            store = LocalEmbeddingStore(name="t_local_embedding_sdk_transport", max_retries=3)
            store.as_embedding = provider

            assert await store._call_with_retry([f"item{i}" for i in range(10)]) is None
            # One request only: the wrapper and the SDK already retry a transport failure
            # (4 wrapper attempts, each retried by the SDK), so another loop here only
            # multiplies them -- a persistent timeout used to cost 36 attempts instead of 12.
            assert provider.batch_sizes == [10]
            assert sleeps == []
            assert store.is_healthy is False

    run(go())


def test_isolated_item_keeps_the_store_backoff_for_a_transport_error(monkeypatch):
    """An isolated single-item attempt still gets the store's backoff and keeps its vector.

    This fake provider has no wrapper or SDK beneath it, so it pins the store's own budget: the
    isolated attempt is the last chance to keep that one vector. A whole-batch transport failure
    is left to the layers below instead; see the earlier transport tests.
    """

    async def go():
        sleeps = []

        async def fake_sleep(delay):
            sleeps.append(delay)

        for transient in (
            APIConnectionError("connection reset"),
            ConnectionError("connection reset"),
            TimeoutError("request timed out"),
        ):
            sleeps.clear()

            provider = PerItemLimitThenTransientErrorAsEmbedding(transient)
            store = LocalEmbeddingStore(
                name="t_local_embedding_sdk_isolated",
                max_retries=3,
                enable_cache=False,
            )
            store.as_embedding = provider
            monkeypatch.setattr(asyncio, "sleep", fake_sleep)

            results = await store._call_with_retry(["ok", "x" * 64, "also ok"])

            assert results == [[1.0, 0.0], None, [1.0, 0.0]]
            assert provider.batch_sizes == [3, 1, 1, 1, 1]
            assert sleeps == [1.0]

    run(go())


def test_a_persistent_transport_error_is_not_multiplied_by_the_store(monkeypatch):
    """A timeout is asked once: the wrapper and the SDK own the retry budget.

    The provider sees the 4 wrapper attempts x 3 SDK attempts those layers make, while the
    store's max_retries=3 no longer multiplies them -- a persistent timeout used to cost 36
    provider attempts instead of 12.
    """

    async def go():
        endpoint = offline_openai_endpoint(read_timeout).install(monkeypatch)
        provider = openai_wrapper("t_openai_transport_budget")
        store = LocalEmbeddingStore(
            name="t_local_embedding_sdk_transport_budget",
            max_retries=3,
            enable_cache=False,
        )
        store.as_embedding = provider

        assert await store._call_with_retry([f"item{i}" for i in range(3)]) is None
        assert provider.calls == 1
        assert endpoint.requests == 12
        assert store.is_healthy is False

    run(go())


def test_a_transient_transport_error_still_recovers_through_the_sdk(monkeypatch):
    """One timeout is answered by the SDK's own retry: the batch survives and never splits."""

    async def go():
        state = {"timeouts": 1}

        def handler(request):
            if state["timeouts"] > 0:
                state["timeouts"] -= 1
                return read_timeout(request)
            return embedding_payload(request)

        endpoint = offline_openai_endpoint(handler).install(monkeypatch)
        provider = openai_wrapper("t_openai_transport_recover")
        store = LocalEmbeddingStore(
            name="t_local_embedding_sdk_transport_recover",
            max_retries=3,
            enable_cache=False,
        )
        store.as_embedding = provider

        results = await store._call_with_retry([f"item{i}" for i in range(3)])

        assert results == [[0.0, 1.0]] * 3
        assert provider.calls == 1
        assert endpoint.requests == 2

    run(go())


def test_an_isolated_item_still_recovers_through_the_sdk(monkeypatch):
    """After a context-length rejection, a timeout on one remaining item is still recovered.

    The store asks the wrapper once per item; the retry that saves that item belongs to the SDK.
    """

    async def go():
        state = {"timeout_pending": True}

        def handler(request):
            body = json.loads(request.content.decode("utf-8"))
            inputs = body["input"]
            if isinstance(inputs, str):
                inputs = [inputs]
            if any(len(item) > 32 for item in inputs):
                return context_length_rejection()
            if len(inputs) == 1 and state["timeout_pending"]:
                state["timeout_pending"] = False
                return read_timeout(request)
            return embedding_payload(request)

        endpoint = offline_openai_endpoint(handler).install(monkeypatch)
        provider = openai_wrapper("t_openai_transport_isolated")
        store = LocalEmbeddingStore(
            name="t_local_embedding_sdk_transport_isolated",
            max_retries=3,
            enable_cache=False,
        )
        store.as_embedding = provider

        results = await store._call_with_retry(["ok", "x" * 64, "also ok"])

        assert results == [[0.0, 1.0], None, [0.0, 1.0]]
        assert provider.calls == 4
        assert endpoint.requests == 5

    run(go())


def test_provider_wide_statuses_exclude_input_dependent_bad_request():
    """404 is provider-wide, but an over-limit 400 must stay input-dependent."""

    async def go():
        store = LocalEmbeddingStore(name="t_local_embedding_classifier")
        for error in (
            AuthenticationError("bad key"),
            PermissionDeniedError("no access"),
            NotFoundError("model_not_found"),
            InternalServerError("boom"),
        ):
            assert store._is_provider_wide_failure(error) is True
        over_limit = ContextLengthError("the input length exceeds the context length")
        assert store._is_provider_wide_failure(over_limit) is False

    run(go())


def test_provider_wide_not_found_errors_are_not_split(monkeypatch):
    """A missing model or endpoint is provider-wide: one request, then failure, never a split."""

    async def go():
        sleeps = []

        async def fake_sleep(delay):
            sleeps.append(delay)

        provider = AlwaysProviderWideFailureAsEmbedding(NotFoundError("model_not_found"))
        store = LocalEmbeddingStore(name="t_local_embedding_not_found", max_retries=3)
        store.as_embedding = provider
        monkeypatch.setattr(asyncio, "sleep", fake_sleep)

        assert await store._call_with_retry([f"item{i}" for i in range(10)]) is None
        assert provider.batch_sizes == [10]
        assert not sleeps

    run(go())


class GenAiClientError(Exception):
    """Google GenAI client error: the HTTP status is an integer ``code``, not ``status_code``."""

    def __init__(self, code: int, message: str = "client error"):
        super().__init__(message)
        self.code = code


class GenAiServerError(Exception):
    """Google GenAI server error: the HTTP status is an integer ``code``, not ``status_code``."""

    def __init__(self, code: int, message: str = "server error"):
        super().__init__(message)
        self.code = code


class StringCodeError(Exception):
    """OpenAI-compatible error whose ``code`` names the error instead of carrying a status."""

    code = "invalid_api_key"


class BadRequestError(Exception):
    """OpenAI-compatible 400 that names no input, such as a malformed payload."""

    status_code = 400


class ResponseLike:
    """httpx-style wrapper that keeps the status on a ``response`` attribute."""

    def __init__(self, status_code: int):
        self.status_code = status_code


class WrappedTransportError(Exception):
    """Error that only exposes its status through ``response.status_code``."""

    def __init__(self, status_code: int):
        super().__init__("wrapped")
        self.response = ResponseLike(status_code)


class DashScopeRuntimeError(RuntimeError):
    """AgentScope turns every non-200 DashScope response into a plain ``RuntimeError``.

    The structured response is embedded in the message instead of being kept as an attribute,
    so a client that only reads exception attributes cannot classify it.
    """

    @staticmethod
    def for_status(status_code: int, code: str = "InvalidApiKey") -> "DashScopeRuntimeError":
        """Build the exception AgentScope raises for a non-200 DashScope response."""
        response = {
            "status_code": status_code,
            "request_id": "req-0",
            "code": code,
            "message": "request rejected",
            "output": None,
            "usage": None,
        }
        return DashScopeRuntimeError(f"Failed to get embedding from DashScope API: {json.dumps(response)}")


def test_http_status_reads_every_supported_exception_shape():
    """Read the status from status_code, an integer code, response.status_code or the message."""

    store = LocalEmbeddingStore(name="t_local_embedding_status_shapes")

    assert store._http_status(AuthenticationError("bad key")) == 401
    assert store._http_status(NotFoundError("model_not_found")) == 404
    assert store._http_status(GenAiClientError(401)) == 401
    assert store._http_status(GenAiServerError(503)) == 503
    assert store._http_status(WrappedTransportError(404)) == 404
    assert store._http_status(DashScopeRuntimeError.for_status(401)) == 401
    assert store._http_status(ValueError("no status here")) is None


def test_http_status_ignores_a_string_code():
    """A string ``code`` names the error; reading it as a status would misclassify quota errors."""

    store = LocalEmbeddingStore(name="t_local_embedding_status_string_code")

    assert store._http_status(StringCodeError("invalid key")) is None
    assert store._is_insufficient_quota(StringCodeError("invalid key")) is False
    assert store._http_status(InsufficientQuotaError("quota exhausted")) == 429
    assert store._is_insufficient_quota(InsufficientQuotaError("quota exhausted")) is True


def test_google_genai_service_failures_are_not_split_into_per_item_requests(monkeypatch):
    """A Gemini 401 / 429 / 503 describes the service, so the batch is never split per item.

    A 503 is answered once: the wrapper and the SDK own the retry for a 5xx, and retrying
    here as well multiplied their attempts. A 429 keeps its retry because the rate-limit
    branch above this classifier is the one place that reacts to it.
    """

    async def go():
        sleeps = []

        async def fake_sleep(delay):
            sleeps.append(delay)

        monkeypatch.setattr(asyncio, "sleep", fake_sleep)

        for error, expected_batches in (
            (GenAiClientError(401), [10]),
            (GenAiServerError(503), [10]),
            (GenAiClientError(429), [10, 10, 10]),
        ):
            sleeps.clear()
            provider = AlwaysProviderWideFailureAsEmbedding(error)
            store = LocalEmbeddingStore(
                name="t_local_embedding_genai",
                max_retries=3,
                enable_cache=False,
            )
            store.as_embedding = provider

            assert await store._call_with_retry([f"item{i}" for i in range(10)]) is None
            assert provider.batch_sizes == expected_batches

    run(go())


def test_dashscope_runtime_errors_are_not_split_into_per_item_requests(monkeypatch):
    """A DashScope failure arrives as a plain RuntimeError; the status lives in its message.

    As with the Gemini shapes, a 503 is answered once (the lower layers own that retry) and
    a 429 keeps the retry from the rate-limit branch.
    """

    async def go():
        sleeps = []

        async def fake_sleep(delay):
            sleeps.append(delay)

        monkeypatch.setattr(asyncio, "sleep", fake_sleep)

        for status, expected_batches in ((401, [10]), (503, [10]), (429, [10, 10, 10])):
            sleeps.clear()
            provider = AlwaysProviderWideFailureAsEmbedding(DashScopeRuntimeError.for_status(status))
            store = LocalEmbeddingStore(
                name="t_local_embedding_dashscope",
                max_retries=3,
                enable_cache=False,
            )
            store.as_embedding = provider

            assert await store._call_with_retry([f"item{i}" for i in range(10)]) is None
            assert provider.batch_sizes == expected_batches

    run(go())


def test_unrecognized_errors_are_not_split_into_per_item_requests():
    """Only an input-dependent error may split; anything unrecognized stays a batch failure."""

    async def go():
        provider = AlwaysProviderWideFailureAsEmbedding(RuntimeError("provider said no"))
        store = LocalEmbeddingStore(name="t_local_embedding_unrecognized", max_retries=3)
        store.as_embedding = provider

        assert await store._call_with_retry(["a", "b", "c"]) is None
        assert provider.batch_sizes == [3]

    run(go())


def test_input_dependent_classifier_pins_both_halves_of_the_boundary():
    """The isolation must keep working for the one error a single item can explain."""

    store = LocalEmbeddingStore(name="t_local_embedding_input_dependent")

    assert (
        store._is_input_dependent_failure(
            ContextLengthError("the input length exceeds the context length"),
        )
        is True
    )
    assert store._is_input_dependent_failure(ContextLengthError("too long")) is True
    assert store._is_input_dependent_failure(AuthenticationError("bad key")) is False
    assert store._is_input_dependent_failure(InternalServerError("boom")) is False
    assert store._is_input_dependent_failure(GenAiClientError(400)) is False
    assert store._is_input_dependent_failure(RuntimeError("provider said no")) is False


def test_bad_request_without_an_input_length_signal_is_not_split():
    """A bare 400 does not name an input, so the batch is not divided over it."""

    async def go():
        provider = AlwaysProviderWideFailureAsEmbedding(BadRequestError("malformed payload"))
        store = LocalEmbeddingStore(name="t_local_embedding_bare_400", max_retries=2)
        store.as_embedding = provider

        assert await store._call_with_retry(["a", "b", "c"]) is None
        assert provider.batch_sizes == [3]

    run(go())


def test_isolation_stops_when_a_provider_wide_failure_hits():
    """Once isolation meets a provider-wide failure, the remaining items are not requested.

    The vectors already recovered stay in the result: only the items that were never answered
    are dropped, and no further request is sent for them.
    """

    async def go():
        provider = PerItemLimitThenQuotaAsEmbedding()
        store = LocalEmbeddingStore(
            name="t_local_embedding_isolation_stops",
            max_retries=2,
            enable_cache=False,
        )
        store.as_embedding = provider

        results = await store._call_with_retry(["ok", "x" * 64, "also ok", "still ok"])

        assert results == [[1.0, 0.0], None, None, None]
        # One rejected batch, the recovered first item, the over-limit item, then the quota
        # failure. The fourth item is never requested once the quota failure is seen.
        assert provider.batch_sizes == [4, 1, 1, 1]

    run(go())


def test_partial_success_keeps_a_failed_provider_unhealthy():
    """Recovered vectors must not overwrite the health state of a provider-wide failure.

    Isolation can recover an early item and then meet a provider-wide failure such as
    exhausted quota. ``_call_with_retry`` marks the store unhealthy and stops, but the
    vectors recovered before it are still returned. Treating that partial success as a
    healthy provider would hide the outage, and it would break the unhealthy-to-healthy
    transition the file store uses to schedule missing-vector backfill. The existing
    isolation test drives ``_call_with_retry`` directly, so it never sees the overwrite.
    """

    async def go():
        provider = PerItemLimitThenQuotaAsEmbedding()
        store = LocalEmbeddingStore(
            name="t_local_embedding_partial_health",
            max_retries=2,
            enable_cache=False,
        )
        store.as_embedding = provider

        vectors = await store.get_embeddings(["ok", "x" * 64, "also ok", "still ok"])

        # The vector recovered before the outage is kept ...
        assert vectors[0] is not None
        assert vectors[1] is None and vectors[2] is None and vectors[3] is None
        # ... and the provider-wide failure still decides the health state.
        assert store.is_healthy is False
        # One rejected batch, the recovered first item, the over-limit item, then the quota
        # failure: the fourth item is never requested.
        assert provider.batch_sizes == [4, 1, 1, 1]

    run(go())


def test_isolation_keeps_going_while_only_inputs_fail():
    """An input-dependent failure is local to its item, so isolation must continue.

    This pins the other half of the boundary: stopping on every isolated failure would lose
    the vectors of the items that are perfectly embeddable on their own.
    """

    async def go():
        provider = PerItemLimitAsEmbedding()
        store = LocalEmbeddingStore(
            name="t_local_embedding_isolation_keeps_going",
            max_retries=2,
            enable_cache=False,
        )
        store.as_embedding = provider

        results = await store._call_with_retry(["ok", "x" * 64, "also ok", "still ok"])

        assert results == [[1.0, 0.0], None, [1.0, 0.0], [1.0, 0.0]]
        assert provider.batch_sizes == [4, 1, 1, 1, 1]

    run(go())


def test_dashscope_serialized_quota_errors_keep_their_quota_policy(monkeypatch):
    """A serialized DashScope quota error must classify as quota, not as a rate limit.

    AgentScope embeds the structured response in the message, so the business error code has to
    be read back out of it. Without that the opt-in quota retry policy is never applied and the
    error is retried with the generic backoff instead.
    """

    async def go():
        sleeps = []

        async def fake_sleep(delay):
            sleeps.append(delay)

        monkeypatch.setattr(asyncio, "sleep", fake_sleep)

        error = DashScopeRuntimeError.for_status(429, code="insufficient_quota")
        assert LocalEmbeddingStore._is_insufficient_quota(error) is True
        assert LocalEmbeddingStore._is_rate_limited(error) is False

        provider = AlwaysProviderWideFailureAsEmbedding(error)
        store = LocalEmbeddingStore(
            name="t_local_embedding_dashscope_quota",
            max_retries=3,
            quota_retry_delay=7.0,
            enable_cache=False,
        )
        store.as_embedding = provider

        assert await store._call_with_retry(["a", "b"]) is None
        assert provider.batch_sizes == [2, 2, 2]
        assert sleeps == [7.0, 7.0]

    run(go())


class ContextLength413Error(Exception):
    """An over-limit input reported as HTTP 413 instead of 400.

    Ollama v0.40.3 normalizes an embedding input-limit rejection to 413 and propagates that
    status through the embedding endpoint, so the same over-long item that arrives as a 400
    from an OpenAI-compatible gateway is a 413 for that provider.
    """

    status_code = 413
    body = {"error": {"message": "the input length exceeds the context length"}}


class RequestBodyTooLargeError(Exception):
    """A bare 413 that never names an input: the request body itself is too large."""

    status_code = 413


class PerItemLimitAs413Embedding:
    """Reject a whole request with HTTP 413 when any single text is over the per-item limit."""

    dimensions = 2
    vector_space_id = "fakespace000"
    max_chars = 32

    def __init__(self):
        self.batch_sizes: list[int] = []

    def initialize_model(self):
        """Mirror the real component's idempotent initialization hook."""

    async def __call__(self, texts: list[str], **_kwargs):
        self.batch_sizes.append(len(texts))
        if any(len(text) > self.max_chars for text in texts):
            raise ContextLength413Error("the input length exceeds the context length")
        return [[1.0, 0.0] for _ in texts]


def test_context_length_rejection_reported_as_http_413_is_isolated():
    """An over-limit input carried by a 413 must still reach the per-item isolation path.

    Ollama normalizes an embedding input-limit rejection to 413, so a classifier that only
    accepts 400 rejects the batch instead of isolating the item that caused it, and every
    healthy vector in the request is lost.
    """

    async def go():
        provider = PerItemLimitAs413Embedding()
        store = LocalEmbeddingStore(
            name="t_local_embedding_413_isolated",
            max_retries=2,
            enable_cache=False,
        )
        store.as_embedding = provider

        results = await store._call_with_retry(["ok", "x" * 64, "also ok"])

        assert results == [[1.0, 0.0], None, [1.0, 0.0]]
        assert provider.batch_sizes == [3, 1, 1, 1]

    run(go())


def test_bare_http_413_without_an_input_signal_is_not_split():
    """A 413 that names no input stays provider-wide, so the batch is not divided over it.

    The status alone is ambiguous: it can describe an unrelated request-body limit, which no
    single input explains, so only a 413 that blames the input itself may fan out per item.
    """

    async def go():
        provider = AlwaysProviderWideFailureAsEmbedding(
            RequestBodyTooLargeError("request entity too large"),
        )
        store = LocalEmbeddingStore(name="t_local_embedding_bare_413", max_retries=2)
        store.as_embedding = provider

        assert await store._call_with_retry(["a", "b", "c"]) is None
        assert provider.batch_sizes == [3]

    run(go())


def test_input_dependent_classifier_accepts_a_413_only_with_an_input_signal():
    """Pin both halves of the 413 boundary, and that the 400 behaviour is unchanged."""

    store = LocalEmbeddingStore(name="t_local_embedding_413_classifier")

    assert (
        store._is_input_dependent_failure(
            ContextLength413Error("the input length exceeds the context length"),
        )
        is True
    )
    assert store._is_input_dependent_failure(ContextLength413Error("too long")) is True
    assert store._is_input_dependent_failure(RequestBodyTooLargeError("request entity too large")) is False
    assert store._is_input_dependent_failure(ContextLengthError("too long")) is True
    assert store._is_input_dependent_failure(GenAiClientError(400)) is False
