"""Tests for small utilities in ``reme.utils``."""

import asyncio
import sys

import numpy as np
import pytest

from reme.utils import common_utils
from reme.utils.similarity_utils import batch_cosine_similarity, cosine_similarity


def test_batch_cosine_similarity_rejects_1d_inputs():
    """1D vectors should fail with a clear validation error, not IndexError."""
    with pytest.raises(ValueError, match="Expected 2D arrays"):
        batch_cosine_similarity(np.array([1.0, 0.0]), np.array([[1.0, 0.0]]))


def test_batch_cosine_similarity_pairwise_matrix():
    """Batch cosine returns the full pairwise matrix for valid 2D inputs."""
    result = batch_cosine_similarity(
        np.array([[1.0, 0.0], [0.0, 1.0]]),
        np.array([[1.0, 0.0], [1.0, 1.0]]),
    )

    assert result.shape == (2, 2)
    np.testing.assert_allclose(result[0], [1.0, 2**-0.5])
    np.testing.assert_allclose(result[1], [0.0, 2**-0.5])


def test_cosine_similarity_rejects_mismatched_lengths():
    """Single-vector cosine validates dimensions before computing."""
    with pytest.raises(ValueError, match="Vectors must have same length"):
        cosine_similarity([1.0], [1.0, 2.0])


def test_mock_reme_server_uses_reme_entrypoint(monkeypatch):
    """The test server helper should spawn the reme CLI module, not legacy reme."""
    captured: dict[str, list[str]] = {}

    class DummyProcess:
        """Process stub returned by the patched Popen."""

        stdout = None

        def poll(self):
            """Return a successful process status."""
            return 0

    def fake_popen(cmd, **_kwargs):
        """Capture the spawned command."""
        captured["cmd"] = cmd
        return DummyProcess()

    async def fake_wait_ready(_host, _port, _timeout):
        return None

    monkeypatch.setattr(common_utils.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(common_utils, "_wait_reme_ready", fake_wait_ready)

    async def run():
        async with common_utils.mock_reme_server(port=45678, log_to_file=False, enable_logo=False):
            pass

    asyncio.run(run())

    assert captured["cmd"][:4] == [sys.executable, "-m", "reme.reme", "start"]
