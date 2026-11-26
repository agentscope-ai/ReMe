"""Tests for explicit scoped index rebuilds."""

# pylint: disable=protected-access

from unittest.mock import AsyncMock

import pytest

from reme.components.file_store import LocalFileStore
from reme.core.context import RuntimeContext
from reme.config import resolve_app_config
from reme.steps.index import ReindexStep
from reme.steps.index.reindex import _render_reindex


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
    """The step delegates the composite scope once; the answer summarizes it."""
    store = LocalFileStore(name="test_reindex_all", embedding_store="")
    details = {
        "scope": "all",
        "bm25": {"scope": "bm25", "indexed": 7},
        "embedding": {"scope": "embedding", "indexed": 5},
        "tag": {"scope": "tag", "indexed": 2},
    }
    store.reindex = AsyncMock(return_value=details)

    response = await ReindexStep(file_store=store)(RuntimeContext(scope="all"))

    store.reindex.assert_awaited_once_with("all")
    assert response.metadata == details
    assert response.answer == "Reindexed scope=all: bm25=7, embedding=5, tag=2"


def test_reindex_job_schema_exposes_tag_scope():
    """The public job contract accepts every scope implemented by the file store."""
    config = resolve_app_config(config="default", log_config=False)
    scope = config["jobs"]["reindex"]["parameters"]["properties"]["scope"]

    assert scope["enum"] == ["all", "bm25", "embedding", "tag"]


@pytest.mark.asyncio
async def test_reindex_all_keeps_other_indexes_when_tag_rebuild_fails():
    """A failing optional tag rebuild must not abort the keyword or vector rebuild."""
    store = LocalFileStore(name="test_reindex_tag_fail", embedding_store="")

    async def ok_bm25() -> dict:
        return {"indexed": 7, "scope": "bm25"}

    async def ok_embedding() -> dict:
        return {"indexed": 5, "scope": "embedding"}

    async def failing_tag(*_args) -> bool:
        return False

    store._reindex_bm25 = ok_bm25  # type: ignore[method-assign]
    store._reindex_embedding = ok_embedding  # type: ignore[method-assign]
    store._rebuild_tag_index = failing_tag  # type: ignore[method-assign]

    details = await store.reindex("all")

    assert details["bm25"] == {"indexed": 7, "scope": "bm25"}
    assert details["embedding"] == {"indexed": 5, "scope": "embedding"}
    assert details["tag"] == {"indexed": 0, "scope": "tag", "failed": True}


@pytest.mark.asyncio
async def test_reindex_tag_scope_reports_failure_without_raising():
    """An explicit tag-only rebuild reports the failure instead of raising."""
    store = LocalFileStore(name="test_reindex_tag_only_fail", embedding_store="")

    async def failing_tag(*_args) -> bool:
        return False

    store._rebuild_tag_index = failing_tag  # type: ignore[method-assign]

    details = await store.reindex("tag")

    assert details == {"indexed": 0, "scope": "tag", "failed": True}


def test_render_reindex_flags_a_failed_optional_index():
    """The summary line marks the degraded index instead of silently showing 0."""
    details = {
        "scope": "all",
        "bm25": {"scope": "bm25", "indexed": 7},
        "embedding": {"scope": "embedding", "indexed": 5},
        "tag": {"scope": "tag", "indexed": 0, "failed": True},
    }

    assert _render_reindex(details) == "Reindexed scope=all: bm25=7, embedding=5, tag=0 (failed)"


def test_render_reindex_flags_a_failed_flat_report():
    """A single-index report keeps its top-level failure marker."""
    details = {"indexed": 0, "scope": "tag", "failed": True}

    assert _render_reindex(details) == "Reindexed scope=tag: 0 indexed (failed)"


@pytest.mark.asyncio
async def test_reindex_step_surfaces_a_failed_tag_only_rebuild():
    """The step's answer must not read as a successful empty rebuild."""
    store = LocalFileStore(name="test_reindex_step_tag_fail", embedding_store="")
    store.reindex = AsyncMock(
        return_value={"indexed": 0, "scope": "tag", "failed": True},
    )

    response = await ReindexStep(file_store=store)(RuntimeContext(scope="tag"))

    assert response.answer == "Reindexed scope=tag: 0 indexed (failed)"
    assert response.metadata["failed"] is True


@pytest.mark.asyncio
async def test_reindex_tag_scope_survives_a_raising_tag_rebuild():
    """A dependency error raised before the rebuild's own try must not escape."""
    store = LocalFileStore(name="test_reindex_tag_raises", embedding_store="")

    async def raising_tag(*_args) -> bool:
        raise RuntimeError("tag index dependency unavailable")

    store._rebuild_tag_index = raising_tag  # type: ignore[method-assign]

    details = await store.reindex("tag")

    assert details == {"indexed": 0, "scope": "tag", "failed": True}


@pytest.mark.asyncio
async def test_reindex_all_survives_a_raising_tag_rebuild():
    """A raising tag rebuild still leaves the keyword and vector results intact."""
    store = LocalFileStore(name="test_reindex_all_tag_raises", embedding_store="")

    async def ok_bm25() -> dict:
        return {"indexed": 7, "scope": "bm25"}

    async def ok_embedding() -> dict:
        return {"indexed": 5, "scope": "embedding"}

    async def raising_tag(*_args) -> bool:
        raise RuntimeError("tag index dependency unavailable")

    store._reindex_bm25 = ok_bm25  # type: ignore[method-assign]
    store._reindex_embedding = ok_embedding  # type: ignore[method-assign]
    store._rebuild_tag_index = raising_tag  # type: ignore[method-assign]

    details = await store.reindex("all")

    assert details["bm25"] == {"indexed": 7, "scope": "bm25"}
    assert details["embedding"] == {"indexed": 5, "scope": "embedding"}
    assert details["tag"] == {"indexed": 0, "scope": "tag", "failed": True}
