"""Tests for Application.start() graceful degradation when dependencies are missing.

Covers the scenario where embedding model is not configured, ensuring
file_store and file_watcher initialize without KeyError.

Usage:
    pytest tests/test_application_start.py -v
"""

import tempfile
from pathlib import Path

import pytest

from reme.reme_light import ReMeLight


@pytest.fixture
def temp_dir():
    """Create a temporary directory for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.mark.asyncio
async def test_start_without_embedding_config(temp_dir):
    """Application.start() should succeed when embedding model is not configured.

    This reproduces the bug where copaw worker starts without EMBEDDING_API_KEY,
    causing KeyError: 'default' during file_watcher initialization.
    """
    app = ReMeLight(
        working_dir=str(temp_dir),
        default_file_store_config={
            "backend": "chroma",
            "store_name": "copaw",
            "vector_enabled": False,
            "fts_enabled": True,
        },
    )

    # Should not raise KeyError
    await app.start()

    # file_store should be initialized (with embedding_model=None)
    assert "default" in app.service_context.file_stores
    # file_watcher should be initialized (with file_store resolved)
    assert "default" in app.service_context.file_watchers

    await app.close()


@pytest.mark.asyncio
async def test_start_without_any_optional_config(temp_dir):
    """Application.start() should succeed with minimal config (no embedding, no file_store)."""
    app = ReMeLight(
        working_dir=str(temp_dir),
    )

    await app.start()
    await app.close()
