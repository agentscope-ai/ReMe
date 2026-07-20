"""Unit tests for the auto_memory_codex step."""

# pylint: disable=missing-class-docstring,missing-function-docstring,protected-access

import json
from pathlib import Path

import pytest

from reme.steps.evolve.auto_memory_codex import AutoMemoryCodexStep

# ---- helpers: real Codex RolloutItem / ResponseItem schema ----------------

# Codex transcript rows follow the RolloutItem envelope:
#   {"type": "response_item",
#    "payload": {"type": "message", "id": "...", "role": "user", "content": [...]}}
#
# Content blocks are:
#   {"type": "input_text",  "text": "..."}
#   {"type": "output_text", "text": "..."}


def _make_rollout(role: str, text: str, payload_id: str = "rollout-001") -> dict:
    """Create a Codex RolloutItem row matching the real persisted schema."""
    return {
        "type": "response_item",
        "payload": {
            "type": "message",
            "id": payload_id,
            "role": role,
            "content": [{"type": "input_text", "text": text}],
        },
    }


def _make_rollout_with_content(role: str, content: list, payload_id: str = "rollout-001") -> dict:
    """Create a RolloutItem with an explicit content block list."""
    return {
        "type": "response_item",
        "payload": {
            "type": "message",
            "id": payload_id,
            "role": role,
            "content": content,
        },
    }


# ---- _codex_entries_to_messages --------------------------------------------


class TestCodexEntriesToMessages:
    def test_user_and_assistant_entries_rendered(self):
        """Payload role 'user' and 'assistant' are rendered directly."""
        entries = [
            _make_rollout("user", "Hello", payload_id="r1"),
            _make_rollout("assistant", "Hi there", payload_id="r2"),
        ]
        messages = AutoMemoryCodexStep._codex_entries_to_messages(entries)
        assert len(messages) == 2
        assert messages[0] == {"role": "user", "name": "user", "content": "Hello"}
        assert messages[1] == {"role": "assistant", "name": "assistant", "content": "Hi there"}

    def test_non_message_payloads_filtered(self):
        """Payloads whose type != 'message' are skipped (tool calls, system events)."""
        entries = [
            {
                "type": "response_item",
                "payload": {"type": "tool_call", "id": "tc1", "name": "search"},
            },
            _make_rollout("user", "real message", payload_id="r1"),
        ]
        messages = AutoMemoryCodexStep._codex_entries_to_messages(entries)
        assert len(messages) == 1
        assert messages[0]["content"] == "real message"

    def test_non_user_assistant_roles_skipped(self):
        """Payloads with role 'system' are not rendered."""
        entries = [
            _make_rollout("system", "prompt", payload_id="r1"),
            _make_rollout("user", "real message", payload_id="r2"),
        ]
        messages = AutoMemoryCodexStep._codex_entries_to_messages(entries)
        assert len(messages) == 1
        assert messages[0]["content"] == "real message"

    def test_tool_use_rendered_in_content(self):
        """Tool use blocks inside message content are rendered as [tool ...]."""
        entry = _make_rollout_with_content(
            "assistant",
            [
                {"type": "input_text", "text": "Let me search"},
                {"type": "tool_use", "name": "search", "id": "call-1", "input": {"query": "test"}},
            ],
            payload_id="r1",
        )
        messages = AutoMemoryCodexStep._codex_entries_to_messages([entry])
        assert len(messages) == 1
        assert "[tool search(" in messages[0]["content"]
        assert "Let me search" in messages[0]["content"]

    def test_non_response_item_rows_dropped(self):
        """Rows without type='response_item' are silently skipped."""
        entries = [
            {"type": "user", "uuid": "u1", "message": {"role": "user", "content": "cc user"}},
            _make_rollout("user", "only this one", payload_id="r1"),
        ]
        messages = AutoMemoryCodexStep._codex_entries_to_messages(entries)
        assert len(messages) == 1
        assert messages[0]["content"] == "only this one"


# ---- _render_codex_content -------------------------------------------------


class TestRenderCodexContent:
    def test_string_content(self):
        assert AutoMemoryCodexStep._render_codex_content("  hello  ") == "hello"

    def test_input_text_blocks(self):
        """Real Codex input_text blocks — text is in the 'text' field."""
        content = [
            {"type": "input_text", "text": "first"},
            {"type": "input_text", "text": "second"},
        ]
        result = AutoMemoryCodexStep._render_codex_content(content)
        assert result == "first\nsecond"

    def test_output_text_blocks(self):
        """Codex output_text blocks also carry text in the 'text' field."""
        content = [
            {"type": "output_text", "text": "assistant output"},
        ]
        result = AutoMemoryCodexStep._render_codex_content(content)
        assert result == "assistant output"

    def test_mixed_input_and_output_text(self):
        content = [
            {"type": "input_text", "text": "user msg"},
            {"type": "output_text", "text": "assistant reply"},
        ]
        result = AutoMemoryCodexStep._render_codex_content(content)
        assert "user msg" in result
        assert "assistant reply" in result

    def test_tool_result_rendered(self):
        content = [
            {"type": "tool_result", "content": "search output here"},
        ]
        result = AutoMemoryCodexStep._render_codex_content(content)
        assert "[tool_result search output here]" in result


# ---- _resolve_transcript_path ----------------------------------------------


class TestResolveTranscriptPath:
    @pytest.mark.asyncio
    async def test_valid_path_returned_as_is(self, tmp_path, monkeypatch):
        """When the given path exists inside sessions dir, return it unchanged."""
        sessions = tmp_path / "sessions"
        sessions.mkdir(parents=True)
        transcript = sessions / "rollout-20260720-abc.jsonl"
        transcript.write_text(
            json.dumps(_make_rollout("user", "hello", payload_id="r1")) + "\n",
            encoding="utf-8",
        )
        monkeypatch.setattr(
            AutoMemoryCodexStep,
            "_codex_sessions_dir",
            staticmethod(lambda: sessions),
        )

        step = AutoMemoryCodexStep()
        result = await step._resolve_transcript_path(str(transcript), "abc")
        assert result == str(transcript)

    @pytest.mark.asyncio
    async def test_path_outside_sessions_rejected(self, tmp_path, monkeypatch):
        """A transcript outside $CODEX_HOME/sessions must be rejected."""
        sessions = tmp_path / "sessions"
        sessions.mkdir(parents=True)
        outside = tmp_path / "outside.jsonl"
        outside.write_text(
            json.dumps(_make_rollout("user", "hello", payload_id="r1")) + "\n",
            encoding="utf-8",
        )
        monkeypatch.setattr(
            AutoMemoryCodexStep,
            "_codex_sessions_dir",
            staticmethod(lambda: sessions),
        )

        step = AutoMemoryCodexStep()
        result = await step._resolve_transcript_path(str(outside), "abc")
        assert result is None

    @pytest.mark.asyncio
    async def test_empty_path_returns_none(self):
        step = AutoMemoryCodexStep()
        result = await step._resolve_transcript_path("", "sess-1")
        assert result is None

    @pytest.mark.asyncio
    async def test_fallback_searches_rollout_pattern(self, tmp_path, monkeypatch):
        """When the given path doesn't exist, search for rollout-*-<id>.jsonl."""
        sessions = tmp_path / "sessions" / "2026" / "07" / "20"
        sessions.mkdir(parents=True)
        real = sessions / "rollout-1721400000-session-abc.jsonl"
        real.write_text(
            json.dumps(_make_rollout("user", "hello", payload_id="r1")) + "\n",
            encoding="utf-8",
        )
        monkeypatch.setattr(
            AutoMemoryCodexStep,
            "_codex_sessions_dir",
            staticmethod(lambda: tmp_path / "sessions"),
        )

        step = AutoMemoryCodexStep()
        result = await step._resolve_transcript_path(
            "/nonexistent/stale.jsonl",
            "session-abc",
        )
        assert result == str(real)

    @pytest.mark.asyncio
    async def test_fallback_returns_none_when_nothing_found(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            AutoMemoryCodexStep,
            "_codex_sessions_dir",
            staticmethod(lambda: tmp_path / "nonexistent"),
        )
        step = AutoMemoryCodexStep()
        result = await step._resolve_transcript_path("/stale/path.jsonl", "unknown")
        assert result is None


# ---- _load_codex_transcript ------------------------------------------------


@pytest.mark.asyncio
async def test_load_transcript_reads_jsonl(tmp_path: Path):
    """Loading a transcript should return all JSONL entries."""
    transcript = tmp_path / "rollout-001.jsonl"
    entries = [
        _make_rollout("user", "msg1", payload_id="r1"),
        _make_rollout("assistant", "msg2", payload_id="r2"),
    ]
    transcript.write_text(
        "\n".join(json.dumps(e, separators=(",", ":")) for e in entries),
        encoding="utf-8",
    )

    step = AutoMemoryCodexStep()
    result = await step._load_codex_transcript(str(transcript))
    assert len(result) == 2
    assert result[0]["payload"]["id"] == "r1"
    assert result[1]["payload"]["id"] == "r2"


@pytest.mark.asyncio
async def test_load_transcript_returns_empty_on_missing_file():
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
async def test_save_session_dedups_by_payload_id(tmp_path: Path, monkeypatch):
    """Only entries with new payload.id values are returned as the increment."""
    store_root = tmp_path / "codex"
    step = AutoMemoryCodexStep()
    monkeypatch.setattr(step, "_codex_store", lambda: _fake_store(store_root))

    batch1 = [
        _make_rollout("user", "m1", payload_id="r1"),
        _make_rollout("assistant", "m2", payload_id="r2"),
    ]
    inc = await step._save_codex_session("sess-1", batch1)
    assert len(inc) == 2

    batch2 = [
        _make_rollout("user", "m2", payload_id="r2"),  # duplicate id
        _make_rollout("assistant", "m3", payload_id="r3"),  # new id
    ]
    inc = await step._save_codex_session("sess-1", batch2)
    assert len(inc) == 1
    assert inc[0]["payload"]["id"] == "r3"


@pytest.mark.asyncio
async def test_save_session_filters_rows_without_payload_id(tmp_path, monkeypatch):
    """Rows without a payload.id are skipped (not conversational entries)."""
    store_root = tmp_path / "codex"
    step = AutoMemoryCodexStep()
    monkeypatch.setattr(step, "_codex_store", lambda: _fake_store(store_root))

    entries = [
        _make_rollout("user", "has id", payload_id="r1"),
        {"type": "response_item", "payload": {"type": "message", "role": "user", "content": "no id"}},
        {"type": "response_item"},
    ]
    inc = await step._save_codex_session("sess-1", entries)
    assert len(inc) == 1
    assert inc[0]["payload"]["id"] == "r1"


def _fake_store(root: Path):
    from reme.components.agent_wrapper import CcFileSessionStore

    return CcFileSessionStore(root)
