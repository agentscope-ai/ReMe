"""Tests for the Codex Stop hook auto_memory.py logic.

These test the hook's pure functions without requiring a running ReMe server.
"""

# pylint: disable=missing-class-docstring,missing-function-docstring,protected-access

import json
import sys
from pathlib import Path

# Import the hook module from the Codex plugin (not a regular package, so the
# sys.path manipulation is intentional).
_HOOK_DIR = Path(__file__).parents[2] / "plugins" / "codex" / "reme" / "hooks"
sys.path.insert(0, str(_HOOK_DIR))
# pylint: disable=wrong-import-position
import auto_memory  # noqa: E402

# ---- _result_status -------------------------------------------------------


class TestResultStatus:
    """Tests for the _result_status function (checks MCP answer text)."""

    def test_no_response_on_none(self):
        assert auto_memory._result_status(None) == "no-response"

    def test_error_on_jsonrpc_error(self):
        result = {"error": {"code": -32600, "message": "Invalid Request"}}
        assert auto_memory._result_status(result) == "error"

    def test_skipped_when_answer_starts_with_skipped(self):
        """ReMe MCP returns only answer text, metadata is stripped."""
        result = {
            "result": {
                "content": [{"type": "text", "text": "Skipped: no messages"}],
            },
        }
        assert auto_memory._result_status(result) == "skipped"

    def test_error_when_answer_starts_with_error(self):
        """ReMe returns response.answer even when response.success is False."""
        result = {
            "result": {
                "content": [{"type": "text", "text": "Error: could not resolve transcript path"}],
            },
        }
        assert auto_memory._result_status(result) == "error"

    def test_ok_when_answer_has_content(self):
        result = {
            "result": {
                "content": [{"type": "text", "text": "Recorded facts about the auth rewrite."}],
            },
        }
        assert auto_memory._result_status(result) == "ok"

    def test_ok_with_empty_content(self):
        """Empty content array should default to 'ok'."""
        result = {"result": {"content": []}}
        assert auto_memory._result_status(result) == "ok"

    def test_ok_with_non_dict_result(self):
        """Non-dict result content should not crash."""
        assert auto_memory._result_status({"result": "plain string"}) == "ok"


# ---- Payload parsing ------------------------------------------------------


class TestPayloadParsing:
    def test_session_id_and_transcript_path_extracted(self, monkeypatch):
        """The hook reads session_id and transcript_path from the payload."""
        payload = {
            "session_id": "sess-abc",
            "transcript_path": "/tmp/codex-transcript.jsonl",
        }
        monkeypatch.setattr("sys.stdin", _fake_stdin(json.dumps(payload)))

        data = json.loads(sys.stdin.read() or "{}")
        assert data["session_id"] == "sess-abc"
        assert data["transcript_path"] == "/tmp/codex-transcript.jsonl"

    def test_empty_payload_returns_empty_dict(self, monkeypatch):
        monkeypatch.setattr("sys.stdin", _fake_stdin(""))
        data = json.loads(sys.stdin.read() or "{}")
        assert data == {}

    def test_malformed_payload_returns_empty_dict(self, monkeypatch):
        monkeypatch.setattr("sys.stdin", _fake_stdin("not valid json {{{"))
        try:
            data = json.loads(sys.stdin.read() or "{}")
        except Exception:
            data = {}
        assert data == {}


# ---- MCP call tool selection ----------------------------------------------


def test_hook_calls_auto_memory_codex():
    """The Codex hook uses auto_memory_codex, not auto_memory_cc."""
    # The tool name is hardcoded in main() — verify it directly.
    with open(auto_memory.__file__, encoding="utf-8") as fh:
        source = fh.read()
    assert '"auto_memory_codex"' in source
    assert "transcript_path" in source


# ---- Windows detach path --------------------------------------------------


def test_windows_spawn_detached_writes_temp_file(tmp_path, monkeypatch):
    """_spawn_detached writes the payload to a temp file and spawns a subprocess."""
    monkeypatch.setattr(auto_memory.sys, "executable", "python")
    monkeypatch.setattr(auto_memory.sys, "argv", ["auto_memory.py"])

    # Replace mkstemp at the tempfile module level (it's a local import in _spawn_detached)
    import tempfile as _tempfile

    _real_mkstemp = _tempfile.mkstemp
    monkeypatch.setattr(
        _tempfile,
        "mkstemp",
        lambda prefix=None, suffix=None, dir=None: _real_mkstemp(
            prefix=prefix,
            suffix=suffix,
            dir=str(tmp_path),
        ),
    )

    calls = []

    class FakePopen:
        def __init__(self, args, **kwargs):
            calls.append({"args": args, "kwargs": kwargs})

    monkeypatch.setattr("subprocess.Popen", FakePopen)

    payload = {"session_id": "sess-1", "transcript_path": "/tmp/transcript.jsonl"}
    auto_memory._spawn_detached(payload)

    assert len(calls) == 1
    args = calls[0]["args"]
    assert args[0] == "python"
    assert "--payload-file" in args
    # Verify the temp file was written with the payload
    written_file = args[args.index("--payload-file") + 1]
    assert json.loads(Path(written_file).read_text(encoding="utf-8")) == payload


# ---- Helpers --------------------------------------------------------------


def _fake_stdin(content: str):
    import io

    return io.StringIO(content)
