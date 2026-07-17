"""Tests for lazy AgentScope embedding provider construction."""

import asyncio
from types import SimpleNamespace

from reme.components.as_embedding import BaseAsEmbedding


class FakeModel:
    """Minimal async embedding model used to observe construction."""

    constructions = 0

    class Parameters:
        """Accept arbitrary provider parameters."""

        def __init__(self, **kwargs):
            self.kwargs = kwargs

    def __init__(self, credential, dimensions, parameters=None, **kwargs):
        type(self).constructions += 1
        self.credential = credential
        self.dimensions = dimensions
        self.parameters = parameters
        self.kwargs = kwargs

    async def __call__(self, inputs, **_kwargs):
        return SimpleNamespace(embeddings=[[float(index)] * self.dimensions for index, _ in enumerate(inputs)])


class FakeCredential:
    """Credential boundary that resolves to ``FakeModel``."""

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    @staticmethod
    def get_embedding_model_class():
        """Return the fake provider model class."""
        return FakeModel


class LazyAsEmbedding(BaseAsEmbedding):
    """Concrete wrapper backed by the local fakes."""

    credential_cls = FakeCredential


def test_provider_is_constructed_once_on_first_call():
    """Start and dimension inspection stay local; the first request builds the model."""

    async def go():
        FakeModel.constructions = 0
        embedding = LazyAsEmbedding(dimensions=3, credential={"token": "test"}, parameters={"mode": "test"})

        await embedding.start()
        assert embedding.model is None
        assert embedding.dimensions == 3
        assert FakeModel.constructions == 0

        assert await embedding(["first"]) == [[0.0, 0.0, 0.0]]
        assert await embedding(["second"]) == [[0.0, 0.0, 0.0]]
        assert FakeModel.constructions == 1

        await embedding.close()

    asyncio.run(go())
