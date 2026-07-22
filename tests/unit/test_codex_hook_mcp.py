"""Round-trip tests for the hook's MCP HTTP client against a loopback server.

These exercise the full chain: hook → JSON-RPC over HTTP → response parsing,
without requiring a running ReMe service.
"""

# pylint: disable=protected-access,missing-function-docstring

from __future__ import annotations

import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest

# Import the hook module from the Codex plugin.
_HOOK_DIR = Path(__file__).parents[2] / "plugins" / "codex" / "reme" / "hooks"
sys.path.insert(0, str(_HOOK_DIR))
# pylint: disable=wrong-import-position
import auto_memory  # noqa: E402

# ---- mock MCP server -------------------------------------------------------


class _McpHandler(BaseHTTPRequestHandler):
    """Minimal JSON-RPC handler that captures calls and returns canned responses."""

    calls: list[dict[str, Any]]
    session_header: str
    tool_result: dict[str, Any] | None

    def _read_body(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        return json.loads(self.rfile.read(length) or b"{}") if length else {}

    def _send_json(self, status: int, body: dict[str, Any], extra_headers: dict[str, str] | None = None) -> None:
        encoded = json.dumps(body).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        if extra_headers:
            for k, v in extra_headers.items():
                self.send_header(k, v)
        self.end_headers()
        self.wfile.write(encoded)

    def do_POST(self) -> None:  # noqa: N802
        body = self._read_body()
        self.calls.append(body)
        method = body.get("method", "")

        if method == "initialize":
            self._send_json(
                200,
                {"jsonrpc": "2.0", "id": body.get("id"), "result": {}},
                extra_headers={"mcp-session-id": self.session_header},
            )
        elif method == "notifications/initialized":
            self._send_json(202, {})
        elif method == "tools/call":
            if self.tool_result is not None:
                self._send_json(200, self.tool_result)
            else:
                self._send_json(200, {"jsonrpc": "2.0", "id": body.get("id"), "result": {"content": []}})
        else:
            self._send_json(
                400,
                {"jsonrpc": "2.0", "id": body.get("id"), "error": {"code": -32601, "message": "Method not found"}},
            )

    def log_message(self, _format: str, *_args: object) -> None:
        return


class McpServer:
    """Context manager that runs a JSON-RPC MCP simulator on a random port."""

    def __init__(self, tool_result: dict[str, Any] | None = None, session_id: str = "test-mcp-session"):
        self._tool_result = tool_result
        self._session_id = session_id
        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None
        self.calls: list[dict[str, Any]] = []

    @property
    def url(self) -> str:
        assert self._server is not None
        host, port = self._server.server_address
        return f"http://{host}:{port}"

    def start(self) -> McpServer:
        handler = type(
            "Handler",
            (_McpHandler,),
            {
                "calls": self.calls,
                "session_header": self._session_id,
                "tool_result": self._tool_result,
            },
        )
        self._server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
        if self._thread is not None:
            self._thread.join(timeout=2)

    def __enter__(self) -> McpServer:
        return self.start()

    def __exit__(self, *args: object) -> None:
        self.stop()


# ---- helpers ---------------------------------------------------------------


def _tool_result(answer: str) -> dict[str, Any]:
    """Build a JSON-RPC tools/call response with ReMe's answer format."""
    return {
        "jsonrpc": "2.0",
        "id": 2,
        "result": {"content": [{"type": "text", "text": answer}]},
    }


def _tool_error(code: int, message: str) -> dict[str, Any]:
    """Build a JSON-RPC error response."""
    return {
        "jsonrpc": "2.0",
        "id": 2,
        "error": {"code": code, "message": message},
    }


# ---- tests -----------------------------------------------------------------


def test_mcp_call_returns_result_on_success():
    """_mcp_call should complete the full handshake and return the tool result."""
    with McpServer(tool_result=_tool_result("Recorded facts about auth rewrite.")) as server:
        result = auto_memory._mcp_call(
            server.url,
            "auto_memory_codex",
            {"transcript_path": "/tmp/t.jsonl", "session_id": "sess-1"},
        )

    assert result is not None
    assert "error" not in result
    content = result["result"]["content"]
    assert content[0]["text"] == "Recorded facts about auth rewrite."


def test_mcp_call_handshake_sequence():
    """The MCP handshake must send initialize → initialized → tools/call in order."""
    with McpServer(tool_result=_tool_result("ok")) as server:
        auto_memory._mcp_call(
            server.url,
            "auto_memory_codex",
            {"transcript_path": "/tmp/t.jsonl", "session_id": "sess-1"},
        )

    assert len(server.calls) == 3
    assert server.calls[0]["method"] == "initialize"
    assert server.calls[1]["method"] == "notifications/initialized"
    assert server.calls[2]["method"] == "tools/call"
    assert server.calls[2]["params"]["name"] == "auto_memory_codex"
    assert server.calls[2]["params"]["arguments"]["transcript_path"] == "/tmp/t.jsonl"


def test_mcp_call_raises_on_unreachable():
    """When the server is not running, _mcp_call raises URLError (caught by main())."""
    import urllib.error

    try:
        auto_memory._mcp_call("http://127.0.0.1:1", "auto_memory_codex", {"session_id": "x"})
    except urllib.error.URLError:
        pass  # expected — connection refused
    except OSError:
        pass  # also possible on some platforms
    else:
        pytest.fail("expected URLError or OSError for unreachable server")


def test_result_status_skipped_from_real_mcp_response():
    """_result_status must detect 'Skipped' in MCP content text."""
    mcp_result = _tool_result("Skipped: no messages")
    assert auto_memory._result_status(mcp_result) == "skipped"


def test_result_status_error_from_real_mcp_response():
    mcp_result = _tool_error(-32000, "Internal error")
    assert auto_memory._result_status(mcp_result) == "error"


def test_result_status_error_from_answer_text():
    """_result_status must detect 'Error:' in MCP answer text."""
    mcp_result = _tool_result("Error: could not resolve transcript path")
    assert auto_memory._result_status(mcp_result) == "error"


def test_result_status_ok_from_real_mcp_response():
    mcp_result = _tool_result("Recorded 3 facts into daily/2026-07-20/auth.md")
    assert auto_memory._result_status(mcp_result) == "ok"


def test_result_status_no_response_from_none():
    assert auto_memory._result_status(None) == "no-response"
