"""Tests for explicit scoped index rebuilds."""

from unittest.mock import AsyncMock

import pytest

from reme.components.file_store import LocalFileStore
from reme.components.runtime_context import RuntimeContext
from reme.steps.index import ReindexStep


@pytest.mark.asyncio
@pytest.mark.parametrize("scope", ["bm25", "embedding"])
async def test_reindex_step_delegates_scope(scope):
    """The step forwards each individual scope without clearing the store."""
    store = LocalFileStore(name=f"test_reindex_{scope}", embedding_store="")
    store.reindex = AsyncMock(return_value={"indexed": 3, "scope": scope})
    store.clear = AsyncMock()

    response = await ReindexStep(file_store=store)(RuntimeContext(scope=scope))

    store.reindex.assert_awaited_once_with(scope)
    store.clear.assert_not_called()
    assert response.metadata == {"indexed": 3, "scope": scope}


@pytest.mark.asyncio
async def test_reindex_step_delegates_all_once():
    """The step delegates the composite scope exactly once."""
    store = LocalFileStore(name="test_reindex_all", embedding_store="")
    details = {
        "scope": "all",
        "bm25": {"scope": "bm25", "indexed": 3},
        "embedding": {"scope": "embedding", "indexed": 3},
    }
    store.reindex = AsyncMock(return_value=details)

    response = await ReindexStep(file_store=store)(RuntimeContext(scope="all"))

    store.reindex.assert_awaited_once_with("all")
    assert response.metadata == details
