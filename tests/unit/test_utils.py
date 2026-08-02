"""Tests for small utilities in ``reme.utils``."""

import asyncio
import sys
import threading

import numpy as np
import pytest

from reme.utils import common_utils
from reme.utils.counter import (
    COUNTER_LOCK_KEY,
    COUNTER_TREE_KEY,
    global_counter_add,
    global_counter_add_many,
    global_counter_get,
    global_counter_get_all,
    global_counter_inc,
)
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


def test_inc_returns_old_value_starting_at_zero():
    """First call returns 0, then values increase by 1 per call."""
    metadata: dict = {}

    assert global_counter_inc(metadata, ["a"]) == 0
    assert global_counter_inc(metadata, ["a"]) == 1
    assert global_counter_inc(metadata, ["a"]) == 2


def test_add_returns_old_value_and_adds_val():
    """``add`` is fetch-and-add: old value out, ``val`` added in."""
    metadata: dict = {}

    assert global_counter_add(metadata, ["a"], 10) == 0
    assert global_counter_add(metadata, ["a"], 5) == 10
    assert global_counter_inc(metadata, ["a"]) == 15
    assert global_counter_get(metadata, ["a"]) == 16


def test_add_many_returns_old_values_and_updates_all_paths():
    """``add_many`` updates sibling metrics under one counter-tree lock."""
    metadata: dict = {}
    global_counter_add(metadata, ["usage", "input"], 2)

    previous = global_counter_add_many(
        metadata,
        {
            ("usage", "input"): 3,
            ("usage", "output"): 4,
            ("usage", "total"): 7,
        },
    )

    assert previous == {
        ("usage", "input"): 2,
        ("usage", "output"): 0,
        ("usage", "total"): 0,
    }
    assert global_counter_get(metadata, ["usage", "input"]) == 5
    assert global_counter_get(metadata, ["usage", "output"]) == 4
    assert global_counter_get(metadata, ["usage", "total"]) == 7


def test_add_many_holds_the_counter_lock_once_for_the_batch():
    """A multi-metric update is one critical section, not several writes."""

    class CountingLock:
        """Lock test double that counts entered critical sections."""

        def __init__(self):
            self.lock = threading.Lock()
            self.entries = 0

        def __enter__(self):
            self.lock.acquire()
            self.entries += 1
            return self

        def __exit__(self, *_args):
            self.lock.release()

    lock = CountingLock()
    metadata = {COUNTER_LOCK_KEY: lock}

    global_counter_add_many(
        metadata,
        {
            ("usage", "input"): 3,
            ("usage", "output"): 4,
            ("usage", "total"): 7,
        },
    )

    assert lock.entries == 1


def test_add_many_validates_every_update_before_mutating():
    """One invalid update leaves every valid counter unchanged."""
    metadata: dict = {}

    with pytest.raises(TypeError, match="increments must be integers"):
        global_counter_add_many(
            metadata,
            {
                ("usage", "input"): 3,
                ("usage", "output"): "invalid",
            },
        )

    assert COUNTER_TREE_KEY not in metadata


def test_counters_are_isolated_by_key_path():
    """Sibling and nested keys, plus the root, hold independent counters."""
    metadata: dict = {}

    assert global_counter_inc(metadata, ["a"]) == 0
    assert global_counter_inc(metadata, ["b"]) == 0
    assert global_counter_inc(metadata, ["a", "child"]) == 0
    assert global_counter_inc(metadata, []) == 0

    assert global_counter_get(metadata, ["a"]) == 1
    assert global_counter_get(metadata, ["b"]) == 1
    assert global_counter_get(metadata, ["a", "child"]) == 1
    assert global_counter_get(metadata, []) == 1


def test_get_does_not_create_missing_nodes():
    """``get`` reports 0 for missing paths and leaves the tree untouched."""
    metadata: dict = {}

    assert global_counter_get(metadata, ["missing"]) == 0
    assert COUNTER_TREE_KEY not in metadata

    global_counter_inc(metadata, ["a"])
    assert global_counter_get(metadata, ["a", "missing"]) == 0
    assert "missing" not in metadata[COUNTER_TREE_KEY]["children"]["a"]["children"]


def test_get_all_returns_none_for_missing_key():
    """``get_all`` returns None when the tree or the path does not exist."""
    metadata: dict = {}

    assert global_counter_get_all(metadata, []) is None
    assert global_counter_get_all(metadata, ["missing"]) is None

    global_counter_inc(metadata, ["a"])
    assert global_counter_get_all(metadata, ["missing"]) is None
    assert global_counter_get_all(metadata, ["a", "missing"]) is None


def test_get_all_returns_subtree_and_whole_tree():
    """``get_all`` returns the node at ``key``; an empty key returns the root."""
    metadata: dict = {}
    global_counter_add(metadata, ["a"], 2)
    global_counter_add(metadata, ["a", "child"], 3)

    subtree = global_counter_get_all(metadata, ["a"])
    assert subtree == {"value": 2, "children": {"child": {"value": 3, "children": {}}}}

    root = global_counter_get_all(metadata, [])
    assert root["value"] == 0
    assert root["children"]["a"] == subtree


def test_get_all_returns_deep_copy():
    """Mutating the returned subtree must not affect the live counter tree."""
    metadata: dict = {}
    global_counter_add(metadata, ["a", "child"], 3)

    subtree = global_counter_get_all(metadata, ["a"])
    subtree["value"] = 999
    subtree["children"]["child"]["value"] = 999
    subtree["children"]["extra"] = {"value": 1, "children": {}}

    assert global_counter_get(metadata, ["a"]) == 0
    assert global_counter_get(metadata, ["a", "child"]) == 3
    assert global_counter_get_all(metadata, ["a", "extra"]) is None


def test_concurrent_inc_yields_unique_values():
    """Parallel increments on one key never return duplicate values."""
    metadata: dict = {}
    results: list[int] = []
    results_lock = threading.Lock()
    calls_per_thread = 200

    def worker():
        for _ in range(calls_per_thread):
            value = global_counter_inc(metadata, ["shared"])
            with results_lock:
                results.append(value)

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    total = len(threads) * calls_per_thread
    assert sorted(results) == list(range(total))
    assert global_counter_get(metadata, ["shared"]) == total
