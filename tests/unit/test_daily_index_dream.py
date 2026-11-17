"""Regression tests for daily-index errors during dream extraction."""

from unittest.mock import AsyncMock, Mock, patch

import pytest

from reme.components.application_context import ApplicationContext
from reme.components.file_catalog import BaseFileCatalog
from reme.components.file_store import BaseFileStore
from reme.components.runtime_context import RuntimeContext
from reme.steps.evolve.dream.extract import DreamExtractStep


@pytest.mark.asyncio
async def test_dream_extract_reports_unreadable_index_without_mutating_source(tmp_path):
    """Index refresh failure must not be mistaken for successful extraction."""
    note = tmp_path / "daily/2026-05-28/session.md"
    note.parent.mkdir(parents=True)
    note.write_text("source input", encoding="utf-8")
    catalog = Mock(spec=BaseFileCatalog)
    catalog.get_nodes = AsyncMock(return_value=[])
    catalog.delete = AsyncMock()
    store = Mock(spec=BaseFileStore)
    store.workspace_path = tmp_path
    step = DreamExtractStep(scan_days=1, app_context=ApplicationContext(workspace_dir=str(tmp_path)))

    with (
        patch(
            "reme.steps.evolve.dream.extract.refresh_day_index",
            new=AsyncMock(return_value={"error": "daily index is unreadable: malformed YAML"}),
        ),
        patch("reme.steps.evolve.dream.extract.llm_available", return_value=False),
    ):
        response = await step(RuntimeContext(date="2026-05-28", file_catalog=catalog, file_store=store))

    assert response.success is False
    assert "daily index refresh failed" in response.answer
    assert "daily index is unreadable" in response.answer
    assert any("daily index refresh failed" in error for error in response.metadata["dream"]["errors"])
    assert note.read_text(encoding="utf-8") == "source input"
