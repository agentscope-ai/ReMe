"""Unit tests for Claude Code auto-memory session persistence."""

# pylint: disable=protected-access

from types import SimpleNamespace

import pytest

from reme.steps.evolve.auto_memory_cc import AutoMemoryCCStep


@pytest.mark.asyncio
async def test_reme_cc_store_preserves_existing_session_layout(tmp_path):
    """Existing transcript UUIDs remain visible at session/claude_code/<session_id>.jsonl."""
    step = AutoMemoryCCStep()
    step.file_store = SimpleNamespace(workspace_path=tmp_path)
    session_id = "session-1"
    session_path = tmp_path / "session" / "claude_code" / f"{session_id}.jsonl"
    session_path.parent.mkdir(parents=True)
    session_path.write_text('{"uuid":"existing"}\n', encoding="utf-8")

    increment = await step._save_cc_session(
        session_id,
        [{"uuid": "existing"}, {"uuid": "new"}],
    )

    assert increment == [{"uuid": "new"}]
    assert step._session_link(session_id) == f"[[session/claude_code/{session_id}.jsonl]]"
    assert not (tmp_path / "session" / "claude_code" / "claude_code").exists()
