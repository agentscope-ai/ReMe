#!/usr/bin/env python3
"""ReMe Stop hook: fire-and-forget auto-memory for the current session.

Codex runs this on the ``Stop`` event and feeds the hook payload as JSON
on stdin. We read ``session_id`` and ``transcript_path`` from it and hand those
to ReMe's server-side ``auto_memory_codex`` tool over the (already-running) MCP
server — the server reads the transcript from the given path and records the
durable facts. No messages are sent from here; the agent never has to record by
hand.

The actual run spins up an inner agent and can take a while, so we detach
(double-fork on Unix / CREATE_NEW_PROCESS_GROUP on Windows) and return
immediately: stopping is never blocked. Any failure is logged, never
surfaced — recording is best-effort.
"""

from __future__ import annotations

# Many small with-statements in this script; reusing file-handle names is fine.
# pylint: disable=redefined-outer-name

import json
import os
import sys
import urllib.error
import urllib.request
from datetime import datetime

# auto_memory drives an inner agent; give it room. The foreground process has
# already returned by the time this matters (we are detached), so a long ceiling
# is harmless.
_CALL_TIMEOUT = 600


def _plugin_root() -> str:
    """Return the plugin install root.

    Codex sets ``PLUGIN_ROOT`` (and ``CLAUDE_PLUGIN_ROOT`` as a compat alias).
    """
    return (
        os.environ.get("PLUGIN_ROOT")
        or os.environ.get("CLAUDE_PLUGIN_ROOT")
        or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )


def _result_status(result: dict | None) -> str:
    """Classify a JSON-RPC tool result for logging.

    Returns one of: ``ok``, ``skipped``, ``error``, ``no-response``.
    Because ReMe's MCP transport only returns ``answer`` (metadata is
    dropped), we inspect the answer text directly.
    """
    if result is None:
        return "no-response"
    if "error" in result:
        return "error"
    # Extract the answer text from the MCP content array.
    r = result.get("result", {}) if isinstance(result, dict) else {}
    content = r.get("content", []) if isinstance(r, dict) else []
    answer = ""
    if isinstance(content, list):
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                answer = (block.get("text") or "").strip()
                break
    if answer.startswith("Skipped"):
        return "skipped"
    if answer.startswith("Error:"):
        return "error"
    return "ok"


def _server_url() -> str:
    """ReMe MCP endpoint. Prefer the bundled .mcp.json so it stays in sync."""
    mcp_json = os.path.join(_plugin_root(), ".mcp.json")
    try:
        with open(mcp_json, encoding="utf-8") as fh:
            url = json.load(fh)["mcpServers"]["reme"]["url"]
            if url:
                return url
    except Exception:
        pass
    host = os.environ.get("REME_HOST", "127.0.0.1")
    port = os.environ.get("REME_PORT", "2333")
    return f"http://{host}:{port}/mcp"


def _log(session_id: str, status: str, detail: str = "") -> None:
    try:
        log_dir = os.path.join(_plugin_root(), "logs")
        os.makedirs(log_dir, exist_ok=True)
        stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        line = f"{stamp} session={session_id} {status}"
        if detail:
            line += f" {detail}"
        with open(os.path.join(log_dir, "auto_memory_hook.log"), "a", encoding="utf-8") as fh:
            fh.write(line + "\n")
    except Exception:
        pass


def _post(url: str, body: dict, headers: dict) -> "urllib.request.addinfourl":
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    return urllib.request.urlopen(req, timeout=_CALL_TIMEOUT)


def _read_jsonrpc(resp) -> dict | None:
    """Return the JSON-RPC envelope from a JSON or text/event-stream response."""
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
    """Minimal MCP streamable-http client: initialize -> initialized -> tools/call."""
    base = {"Content-Type": "application/json", "Accept": "application/json, text/event-stream"}

    # 1. initialize (captures the session id header)
    init = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {
            "protocolVersion": "2025-06-18",
            "capabilities": {},
            "clientInfo": {"name": "reme-stop-hook", "version": "1.0"},
        },
    }
    with _post(url, init, base) as resp:
        mcp_session = resp.headers.get("mcp-session-id")
        _read_jsonrpc(resp)

    headers = dict(base)
    if mcp_session:
        headers["mcp-session-id"] = mcp_session

    # 2. notifications/initialized (no id; 202 with empty body)
    try:
        with _post(url, {"jsonrpc": "2.0", "method": "notifications/initialized", "params": {}}, headers) as resp:
            resp.read()
    except urllib.error.HTTPError:
        pass

    # 3. tools/call
    call = {"jsonrpc": "2.0", "id": 2, "method": "tools/call", "params": {"name": tool, "arguments": arguments}}
    with _post(url, call, headers) as resp:
        return _read_jsonrpc(resp)


def _daemonize() -> None:
    """Double-fork + setsid so the (slow) call outlives the hook and is reaped by init."""
    if os.fork() > 0:
        os._exit(0)  # original process returns -> hook completes, Codex stops
    os.setsid()
    if os.fork() > 0:
        os._exit(0)
    devnull = os.open(os.devnull, os.O_RDWR)
    for fd in (0, 1, 2):
        os.dup2(devnull, fd)


def main() -> None:
    """Entry point: read the hook payload from stdin and record the session."""
    try:
        payload = json.loads(sys.stdin.read() or "{}")
    except Exception:
        payload = {}

    session_id = payload.get("session_id") or ""
    transcript_path = payload.get("transcript_path") or ""

    if not session_id and not transcript_path:
        return  # nothing to anchor a recording on

    # Detach before the slow agent run. Without fork() (e.g. Windows) we re-spawn
    # this same script as a fully detached subprocess.
    if hasattr(os, "fork"):
        _daemonize()
    else:
        _spawn_detached(payload)
        return

    url = _server_url()
    tool = "auto_memory_codex"
    arguments = {"transcript_path": transcript_path, "session_id": session_id}
    try:
        result = _mcp_call(url, tool, arguments)
        status = _result_status(result)
        if status == "error":
            _log(session_id, status, json.dumps(result["error"], ensure_ascii=False)[:500])
        elif status == "skipped":
            _log(session_id, status, f"transcript_path={transcript_path}")
        else:
            _log(session_id, status)
    except urllib.error.URLError as exc:
        # Server not running / unreachable — expected when ReMe isn't started.
        _log(session_id, "unreachable", str(exc.reason))
    except Exception as exc:  # noqa: BLE001 - best-effort, never surface
        _log(session_id, "exception", repr(exc)[:500])


def _spawn_detached(payload: dict) -> None:
    """Windows-compatible detachment: re-spawn as a fully detached subprocess."""
    import subprocess
    import tempfile

    # Write payload to a temp file so the child can read it.
    fd, tmp = tempfile.mkstemp(prefix="reme-hook-", suffix=".json")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(payload, fh)
    except Exception:
        try:
            os.close(fd)
        except OSError:
            pass
        return

    # Fire-and-forget: we intentionally don't wait for the child (no `with`).
    # pylint: disable-next=consider-using-with
    subprocess.Popen(
        [sys.executable, __file__, "--payload-file", tmp],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        creationflags=getattr(subprocess, "DETACHED_PROCESS", 0) | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0),
        close_fds=True,
    )


if __name__ == "__main__":
    # --payload-file is the Windows detach path: the parent already wrote the
    # hook payload to a temp file and re-spawned us detached. Parse the payload
    # first (before the with-block closes the file), then invoke main with it
    # set as stdin.
    if len(sys.argv) > 2 and sys.argv[1] == "--payload-file":
        payload_file = sys.argv[2]
        try:
            with open(payload_file, encoding="utf-8") as fh:
                payload_raw = fh.read()
        except Exception:
            payload_raw = None
        if payload_raw:
            sys.stdin = __import__("io").StringIO(payload_raw)
            main()
        try:
            os.unlink(payload_file)
        except OSError:
            pass
    else:
        main()
