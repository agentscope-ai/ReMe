"""Tests for explicit scoped index rebuilds."""

from unittest.mock import AsyncMock

import pytest

from reme.components.file_store import LocalFileStore
from reme.components.runtime_context import RuntimeContext
from reme.config import resolve_app_config
from reme.steps.index import ReindexStep


@pytest.mark.asyncio
@pytest.mark.parametrize("scope", ["bm25", "embedding", "tag"])
async def test_reindex_step_delegates_scope(scope):
    """The step forwards each individual scope without clearing the store."""
    store = LocalFileStore(name=f"test_reindex_{scope}", embedding_store="")
    store.reindex = AsyncMock(return_value={"indexed": 3, "scope": scope})
    store.clear = AsyncMock()

    response = await ReindexStep(file_store=store)(RuntimeContext(scope=scope))

    store.reindex.assert_awaited_once_with(scope)
    store.clear.assert_not_called()
    assert response.metadata == {"indexed": 3, "scope": scope}
    assert response.answer == f"Reindexed scope={scope}: 3 indexed"


@pytest.mark.asyncio
async def test_reindex_step_delegates_all_once():
    """The step delegates the composite scope exactly once."""
    store = LocalFileStore(name="test_reindex_all", embedding_store="")
    details = {
        "scope": "all",
        "bm25": {"scope": "bm25", "indexed": 3},
        "embedding": {"scope": "embedding", "indexed": 3},
        "tag": {"scope": "tag", "indexed": 3},
    }
    store.reindex = AsyncMock(return_value=details)

    response = await ReindexStep(file_store=store)(RuntimeContext(scope="all"))

    store.reindex.assert_awaited_once_with("all")
    assert response.metadata == details
    assert response.answer == "Reindexed scope=all: bm25=3, embedding=3, tag=3"


@pytest.mark.asyncio
async def test_reindex_answer_is_a_readable_summary_of_the_report():
    """The answer is a string, so Studio shows it instead of a generic message.

    ReMe Studio only displays ``response.answer`` when it is a string
    (``reme_studio/app/settings-center.tsx``); a dict answer silently fell back
    to the generic "index rebuilt" text and the operator never saw the counts.
    The full report stays in metadata either way.
    """
    store = LocalFileStore(name="test_reindex_answer", embedding_store="")
    store.reindex = AsyncMock(
        return_value={
            "scope": "all",
            "bm25": {"scope": "bm25", "indexed": 7},
            "embedding": {"scope": "embedding", "indexed": 5},
            "tag": {"scope": "tag", "indexed": 2},
        },
    )

    response = await ReindexStep(file_store=store)(RuntimeContext(scope="all"))

    assert isinstance(response.answer, str)
    assert response.answer == "Reindexed scope=all: bm25=7, embedding=5, tag=2"
    assert response.metadata["bm25"] == {"scope": "bm25", "indexed": 7}


def test_reindex_job_schema_exposes_tag_scope():
    """The public job contract accepts every scope implemented by the file store."""
    config = resolve_app_config(config="default", log_config=False)
    scope = config["jobs"]["reindex"]["parameters"]["properties"]["scope"]

    assert scope["enum"] == ["all", "bm25", "embedding", "tag"]
