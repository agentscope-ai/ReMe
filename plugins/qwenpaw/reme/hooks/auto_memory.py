#!/usr/bin/env python3
"""ReMe Stop hook for QwenPaw: fire-and-forget auto-memory for the conversation.

QwenPaw runs this on the ``Stop`` event and feeds the hook payload as JSON on
stdin. We read ``conversation_id`` and ``profile_id`` from it and hand those to
ReMe's server-side ``auto_memory`` tool over the (already-running) MCP server —
the server resolves *this* conversation's transcript on disk and records the
durable facts. No messages are sent from here; the agent never has to record by
hand.

The actual server-side run spins up an inner agent and can be slow, so we detach
(double-fork) and return immediately: stopping is never blocked. Any failure is
logged, never surfaced — recording is best-effort.
"""

from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request
from datetime import datetime

_CALL_TIMEOUT = 600


def _plugin_root() -> str:
    return os.environ.get("QWENPAW_PLUGIN_ROOT") or os.path.dirname(
        os.path.dirname(os.path.abspath(__file__)),
    )


def _server_url() -> str:
    """ReMe MCP endpoint. Prefer the bundled .mcp.json so it stays in sync."""
    mcp_json = os.path.join(_plugin_root(), ".mcp.json")
    try:
        with open(mcp_json, encoding="utf-8") as f:
            url = json.load(f)["mcpServers"]["reme"]["url"]
            if url:
                return url
    except Exception:
        pass
    host = os.environ.get("REME_HOST", "127.0.0.1")
    port = os.environ.get("REME_PORT", "2333")
    return f"http://{host}:{port}/mcp"


def _log(conv_id: str, status: str, detail: str = "") -> None:
    try:
        log_dir = os.path.join(_plugin_root(), "logs")
        os.makedirs(log_dir, exist_ok=True)
        stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        line = f"{stamp} conv={conv_id} {status}"
        if detail:
            line += f" {detail}"
        with open(
            os.path.join(log_dir, "auto_memory_hook.log"),
            "a",
            encoding="utf-8",
        ) as f:
            f.write(line + "\n")
    except Exception:
        pass


def _post(url: str, body: dict, headers: dict):
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    return urllib.request.urlopen(req, timeout=_CALL_TIMEOUT)


def _read_jsonrpc(resp) -> dict | None:
    ctype = resp.headers.get("content-type", "")
    body = resp.read().decode("utf-8", "replace")
    if "text/event-stream" in ctype:
        result = None
        for line in body.splitlines():
            line = line.strip()
            if not line.startswith("data:"):
                continue
            try:
                obj = json.loads(line[len("data:") :].strip())
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict) and ("result" in obj or "error" in obj):
                result = obj
        return result
    try:
        return json.loads(body)
    except json.JSONDecodeError:
        return None


def _mcp_call(url: str, tool: str, arguments: dict) -> dict | None:
    base = {"Content-Type": "application/json", "Accept": "application/json, text/event-stream"}
    init = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {
            "protocolVersion": "2025-06-18",
            "capabilities": {},
            "clientInfo": {"name": "reme-qwenpaw-stop-hook", "version": "1.0"},
        },
    }
    with _post(url, init, base) as resp:
        mcp_session = resp.headers.get("mcp-session-id")
        _read_jsonrpc(resp)
    headers = dict(base)
    if mcp_session:
        headers["mcp-session-id"] = mcp_session
    try:
        with _post(
            url,
            {"jsonrpc": "2.0", "method": "notifications/initialized", "params": {}},
            headers,
        ) as resp:
            resp.read()
    except urllib.error.HTTPError:
        pass
    call = {
        "jsonrpc": "2.0",
        "id": 2,
        "method": "tools/call",
        "params": {"name": tool, "arguments": arguments},
    }
    with _post(url, call, headers) as resp:
        return _read_jsonrpc(resp)


def _daemonize() -> None:
    if os.fork() > 0:
        os._exit(0)
    os.setsid()
    if os.fork() > 0:
        os._exit(0)
    devnull = os.open(os.devnull, os.O_RDWR)
    for fd in (0, 1, 2):
        os.dup2(devnull, fd)


def main() -> None:
    """Entry point: read the hook payload from stdin and record the conversation."""
    try:
        payload = json.loads(sys.stdin.read() or "{}")
    except Exception:
        payload = {}

    conv_id = payload.get("conversation_id") or payload.get("session_id") or ""
    profile_id = payload.get("profile_id") or ""
    if not conv_id:
        return

    if hasattr(os, "fork"):
        _daemonize()

    url = _server_url()
    tool_args = {"session_id": conv_id}
    if profile_id:
        tool_args["profile_id"] = profile_id
    try:
        result = _mcp_call(url, "auto_memory", tool_args)
        if result is None:
            _log(conv_id, "no-response")
        elif "error" in result:
            _log(conv_id, "error", json.dumps(result["error"], ensure_ascii=False)[:500])
        else:
            _log(conv_id, "ok")
    except urllib.error.URLError as exc:
        _log(conv_id, "unreachable", str(exc.reason))
    except Exception as exc:  # noqa: BLE001 - best-effort
        _log(conv_id, "exception", repr(exc)[:500])


if __name__ == "__main__":
    main()
