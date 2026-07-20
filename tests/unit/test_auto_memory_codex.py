"""Unit tests for the auto_memory_codex step."""

# pylint: disable=missing-class-docstring,missing-function-docstring,protected-access

import json
from pathlib import Path

import pytest

from reme.steps.evolve.auto_memory_codex import AutoMemoryCodexStep

# ---- helpers ---------------------------------------------------------------


def _make_cc_entry(entry_type: str, role: str, text: str, uuid: str = "abc-001") -> dict:
    """Create a Claude Code-style transcript entry."""
    return {
        "type": entry_type,
        "uuid": uuid,
        "message": {"role": role, "content": [{"type": "text", "text": text}]},
    }


def _make_codex_entry(role: str, text: str, uuid: str = "abc-001") -> dict:
    """Create a Codex-style transcript entry (response_item, input_text blocks)."""
    return {
        "type": "response_item",
        "uuid": uuid,
        "message": {"role": role, "content": [{"type": "input_text", "input_text": text}]},
    }


# ---- _codex_entries_to_messages --------------------------------------------


class TestCodexEntriesToMessages:
    def test_codex_user_and_developer_entries_rendered(self):
        """Codex uses 'developer' role — should be mapped to 'assistant'."""
        entries = [
            _make_codex_entry("user", "Hello", uuid="u1"),
            _make_codex_entry("developer", "Hi there", uuid="u2"),
        ]
        messages = AutoMemoryCodexStep._codex_entries_to_messages(entries)
        assert len(messages) == 2
        assert messages[0] == {"role": "user", "name": "user", "content": "Hello"}
        assert messages[1] == {"role": "assistant", "name": "assistant", "content": "Hi there"}

    def test_non_conversation_codex_entries_filtered(self):
        """response_item with 'system' role or non-response_item types are skipped."""
        entries = [
            _make_codex_entry("system", "prompt", uuid="u1"),
            _make_codex_entry("user", "real message", uuid="u2"),
            _make_cc_entry("assistant", "assistant", "cc format — should be dropped", uuid="u3"),
        ]
        messages = AutoMemoryCodexStep._codex_entries_to_messages(entries)
        assert len(messages) == 1
        assert messages[0]["content"] == "real message"

    def test_codex_tool_use_rendered(self):
        entry = {
            "type": "response_item",
            "uuid": "abc-002",
            "message": {
                "role": "developer",
                "content": [
                    {"type": "input_text", "input_text": "Let me search"},
                    {"type": "tool_use", "name": "search", "id": "call-1", "input": {"query": "test"}},
                ],
            },
        }
        messages = AutoMemoryCodexStep._codex_entries_to_messages([entry])
        assert len(messages) == 1
        assert "[tool search(" in messages[0]["content"]
        assert "Let me search" in messages[0]["content"]

    def test_unknown_entry_types_dropped(self):
        """Only response_item entries are processed; everything else is skipped."""
        entries = [
            {"type": "system", "uuid": "s1", "message": {"role": "system", "content": "prompt"}},
            {"type": "user", "uuid": "u2", "message": {"role": "user", "content": "cc user"}},
            {"type": "assistant", "uuid": "u3", "message": {"role": "assistant", "content": "cc asst"}},
            _make_codex_entry("user", "only this one", uuid="u4"),
        ]
        messages = AutoMemoryCodexStep._codex_entries_to_messages(entries)
        assert len(messages) == 1
        assert messages[0]["content"] == "only this one"


# ---- _render_codex_content -------------------------------------------------


class TestRenderCodexContent:
    def test_string_content(self):
        assert AutoMemoryCodexStep._render_codex_content("  hello  ") == "hello"

    def test_input_text_blocks(self):
        content = [
            {"type": "input_text", "input_text": "first"},
            {"type": "input_text", "input_text": "second"},
        ]
        result = AutoMemoryCodexStep._render_codex_content(content)
        assert result == "first\nsecond"

    def test_mixed_text_and_input_text(self):
        """Both 'text' (CC compat) and 'input_text' (Codex) blocks are rendered."""
        content = [
            {"type": "text", "text": "cc style"},
            {"type": "input_text", "input_text": "codex style"},
        ]
        result = AutoMemoryCodexStep._render_codex_content(content)
        assert "cc style" in result
        assert "codex style" in result

    def test_tool_result_rendered(self):
        content = [
            {"type": "tool_result", "content": "search output here"},
        ]
        result = AutoMemoryCodexStep._render_codex_content(content)
        assert "[tool_result search output here]" in result


# ---- _resolve_transcript_path ----------------------------------------------


class TestResolveTranscriptPath:
    @pytest.mark.asyncio
    async def test_valid_path_returned_as_is(self, tmp_path):
        """When the given path exists, return it unchanged."""
        transcript = tmp_path / "session-123.jsonl"
        transcript.write_text('{"type": "response_item", "uuid": "u1"}\n', encoding="utf-8")

        step = AutoMemoryCodexStep()
        result = await step._resolve_transcript_path(str(transcript), "session-123")
        assert result == str(transcript)

    @pytest.mark.asyncio
    async def test_empty_path_returns_none(self):
        step = AutoMemoryCodexStep()
        result = await step._resolve_transcript_path("", "sess-1")
        assert result is None

    @pytest.mark.asyncio
    async def test_stale_path_falls_back_to_search(self, tmp_path, monkeypatch):
        """When transcript_path doesn't exist, search CODEX_HOME/sessions for session_id."""
        sessions_dir = tmp_path / "codex_sessions" / "2026" / "07" / "20"
        sessions_dir.mkdir(parents=True)
        real_transcript = sessions_dir / "session-abc.jsonl"
        real_transcript.write_text('{"type": "response_item", "uuid": "u1"}\n', encoding="utf-8")

        monkeypatch.setenv("CODEX_HOME", str(tmp_path / "codex_home"))
        monkeypatch.setattr(
            AutoMemoryCodexStep,
            "_codex_sessions_dir",
            staticmethod(lambda: tmp_path / "codex_sessions"),
        )

        step = AutoMemoryCodexStep()
        # Pass a stale path that doesn't exist
        result = await step._resolve_transcript_path(
            "/nonexistent/stale-session-abc.jsonl",
            "session-abc",
        )
        assert result == str(real_transcript)

    @pytest.mark.asyncio
    async def test_fallback_returns_none_when_nothing_found(self, tmp_path, monkeypatch):
        """When neither the given path nor any fallback exists, return None."""
        monkeypatch.setattr(
            AutoMemoryCodexStep,
            "_codex_sessions_dir",
            staticmethod(lambda: tmp_path / "nonexistent_sessions"),
        )
        step = AutoMemoryCodexStep()
        result = await step._resolve_transcript_path("/stale/path.jsonl", "unknown-session")
        assert result is None


# ---- _load_codex_transcript ------------------------------------------------


@pytest.mark.asyncio
async def test_load_transcript_reads_jsonl(tmp_path: Path):
    """Loading a transcript should return all JSONL entries."""
    transcript = tmp_path / "session-123.jsonl"
    entries = [
        _make_codex_entry("user", "msg1", uuid="u1"),
        _make_codex_entry("developer", "msg2", uuid="u2"),
    ]
    transcript.write_text(
        "\n".join(json.dumps(e, separators=(",", ":")) for e in entries),
        encoding="utf-8",
    )

    step = AutoMemoryCodexStep()
    result = await step._load_codex_transcript(str(transcript))
    assert len(result) == 2
    assert result[0]["uuid"] == "u1"
    assert result[1]["uuid"] == "u2"


@pytest.mark.asyncio
async def test_load_transcript_returns_empty_on_missing_file():
    """Missing transcript files should return an empty list, not crash."""
    step = AutoMemoryCodexStep()
    result = await step._load_codex_transcript("/nonexistent/transcript.jsonl")
    assert result == []


@pytest.mark.asyncio
async def test_load_transcript_returns_empty_on_empty_path():
    step = AutoMemoryCodexStep()
    result = await step._load_codex_transcript("")
    assert result == []


# ---- _save_codex_session ---------------------------------------------------


@pytest.mark.asyncio
async def test_save_session_dedups_by_uuid(tmp_path: Path, monkeypatch):
    """Only entries with new uuids should be returned as the increment."""
    store_root = tmp_path / "codex"
    step = AutoMemoryCodexStep()
    monkeypatch.setattr(step, "_codex_store", lambda: _fake_store(store_root))

    batch1 = [
        _make_codex_entry("user", "m1", uuid="u1"),
        _make_codex_entry("developer", "m2", uuid="u2"),
    ]
    inc = await step._save_codex_session("sess-1", batch1)
    assert len(inc) == 2

    batch2 = [
        _make_codex_entry("user", "m2", uuid="u2"),  # duplicate
        _make_codex_entry("developer", "m3", uuid="u3"),  # new
    ]
    inc = await step._save_codex_session("sess-1", batch2)
    assert len(inc) == 1
    assert inc[0]["uuid"] == "u3"


@pytest.mark.asyncio
async def test_save_session_filters_uuidless_entries(tmp_path, monkeypatch):
    """Entries without a uuid are not copied to the store."""
    store_root = tmp_path / "codex"
    step = AutoMemoryCodexStep()
    monkeypatch.setattr(step, "_codex_store", lambda: _fake_store(store_root))

    entries = [
        _make_codex_entry("user", "has uuid", uuid="u1"),
        {"type": "response_item", "message": {"role": "user", "content": "no uuid"}},
    ]
    inc = await step._save_codex_session("sess-1", entries)
    assert len(inc) == 1
    assert inc[0]["uuid"] == "u1"


def _fake_store(root: Path):
    from reme.components.agent_wrapper import CcFileSessionStore

    return CcFileSessionStore(root)
