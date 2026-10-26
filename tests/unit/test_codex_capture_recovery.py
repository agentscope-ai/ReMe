"""Recovery behavior for Codex Stop hooks when transcript writes settle late."""

# pylint: disable=redefined-outer-name

import importlib.util
import json
from pathlib import Path

import pytest

PLUGIN = Path(__file__).resolve().parents[2] / "integrations/codex/plugins/reme"


@pytest.fixture
def codex_hook(tmp_path, monkeypatch):
    """Load the Codex hook against an isolated host configuration."""
    spec = importlib.util.spec_from_file_location("codex_capture_recovery", PLUGIN / "hooks/auto_memory.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setenv(module.HOME_ENV, str(tmp_path / "host"))
    monkeypatch.setattr(module, "CAPTURE_SETTLE_SECONDS", 0)
    return module


def completed(turn_id, item_type, text):
    """Build one completed Codex transcript item."""
    return {
        "type": "event_msg",
        "timestamp": "2026-10-26T01:00:00Z",
        "payload": {
            "type": "item_completed",
            "turn_id": turn_id,
            "item": {
                "type": item_type,
                "content": [{"type": "text" if item_type == "UserMessage" else "Text", "text": text}],
            },
        },
    }


def test_accepts_unterminated_final_transcript_line(codex_hook, tmp_path):
    """Accept a final row that is complete JSON but has no trailing newline."""
    path = tmp_path / "rollout.jsonl"
    rows = [completed("turn-one", "UserMessage", "question"), completed("turn-one", "AgentMessage", "answer")]
    path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    payload = {"transcript_path": str(path), "last_assistant_message": "answer", "turn_id": "turn-one"}
    pair = codex_hook.completed_turn(payload, "session")
    assert [message["content"][0]["text"] for message in pair] == ["question", "answer"]


def test_uses_last_same_turn_assistant_when_stop_text_differs(codex_hook, tmp_path):
    """Fall back to the last same-turn agent message when Stop text differs."""
    path = tmp_path / "rollout.jsonl"
    rows = [
        completed("turn-one", "UserMessage", "question"),
        completed("turn-one", "AgentMessage", "progress"),
        completed("turn-one", "AgentMessage", "final answer"),
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    payload = {
        "transcript_path": str(path),
        "last_assistant_message": "rendered final answer",
        "turn_id": "turn-one",
    }
    pair = codex_hook.completed_turn(payload, "session")
    assert [message["content"][0]["text"] for message in pair] == ["question", "final answer"]
    payload["last_assistant_message"] = None
    pair = codex_hook.completed_turn(payload, "session")
    assert [message["content"][0]["text"] for message in pair] == ["question", "final answer"]


def test_does_not_cross_turns_when_turn_id_is_unknown(codex_hook, tmp_path):
    """Never capture messages from another turn."""
    path = tmp_path / "rollout.jsonl"
    rows = [completed("turn-one", "UserMessage", "question"), completed("turn-one", "AgentMessage", "answer")]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    payload = {"transcript_path": str(path), "last_assistant_message": "answer", "turn_id": "unknown"}
    assert codex_hook.completed_turn(payload, "session") == []


def test_capture_retries_turn_materialization(codex_hook, monkeypatch):
    """Retry until the current turn is fully materialized."""
    pair = [
        {
            "id": "codex-1",
            "role": "user",
            "content": [{"type": "text", "text": "question"}],
            "created_at": "2026-10-26T01:00:00+00:00",
        },
        {
            "id": "codex-2",
            "role": "assistant",
            "content": [{"type": "text", "text": "answer"}],
            "created_at": "2026-10-26T01:00:01+00:00",
        },
    ]
    responses = iter([[], pair])
    monkeypatch.setattr(codex_hook, "completed_turn", lambda *_args, **_kwargs: next(responses))
    assert codex_hook.completed_turn_with_retry({}, "session", timeout=0.2) == pair


def test_capture_delivers_with_native_session_id(codex_hook, tmp_path):
    """Deliver with the native session ID rather than the queue hash."""
    path = tmp_path / "rollout.jsonl"
    rows = [completed("turn-one", "UserMessage", "question"), completed("turn-one", "AgentMessage", "answer")]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    payload = {
        "hook_event_name": "Stop",
        "session_id": "01a1766c-4b23-7df3-914c-32d0020e7285",
        "transcript_path": str(path),
        "last_assistant_message": "answer",
        "turn_id": "turn-one",
    }
    root = codex_hook.capture(codex_hook.load_config(), payload)
    queued = json.loads(next(root.glob("*.json")).read_text(encoding="utf-8"))
    assert queued["session_id"] == "codex-01a1766c-4b23-7df3-914c-32d0020e7285"
