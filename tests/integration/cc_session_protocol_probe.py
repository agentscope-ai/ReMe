"""Probe real Claude SDK/CLI sessions, optionally using the repository's LLM.

Run with the Python environment containing claude-agent-sdk::

    python tests/integration/cc_session_protocol_probe.py --output /tmp/cc-session-protocol.json

The default is a deterministic local Messages API with no provider calls.
For a real model, pass --real-model --base-url <Anthropic-compatible URL>.
LLM_API_KEY and LLM_MODEL_NAME come from --env-file; no key is printed.
All session files live in a temporary directory. Reports distinguish
acknowledgement, persistence, model input, and actual model recall.
"""

import argparse
import asyncio
import json
import os
import subprocess
import threading
from contextlib import asynccontextmanager, contextmanager, nullcontext, suppress
from dataclasses import asdict, dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib.metadata import version
from pathlib import Path
from tempfile import TemporaryDirectory
from uuid import uuid4

import claude_agent_sdk
from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    ClaudeSDKClient,
    ResultMessage,
    SystemMessage,
    TextBlock,
)

MODEL = "claude-sonnet-4-6"
REPLY = "PROBE_MODEL_REPLY"


@dataclass(repr=False)
class Backend:
    """Endpoint selection; keep credentials out of generated representations."""

    base_url: str
    model: str = MODEL
    api_key: str = "sk-ant-local-protocol-probe"
    direct: bool = False


@contextmanager
def local_model():
    """Serve real Anthropic HTTP/SSE frames and retain the received messages."""
    requests = []

    class Handler(BaseHTTPRequestHandler):
        """A deterministic Messages API; never calls a real model."""

        def log_message(self, _format, *args):
            """Keep the probe output limited to its findings."""

        def do_POST(self):  # pylint: disable=invalid-name
            """Record model input and return a deterministic text response."""
            payload = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
            request = {"path": self.path, "messages": payload.get("messages", [])}
            requests.append(request)
            if "count_tokens" in self.path:
                body = json.dumps({"input_tokens": 10}).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            message = {
                "id": f"msg_{uuid4().hex}",
                "type": "message",
                "role": "assistant",
                "model": MODEL,
                "content": [],
                "stop_reason": None,
                "stop_sequence": None,
                "usage": {"input_tokens": 10, "output_tokens": 0},
            }
            events = [
                {"type": "message_start", "message": message},
                {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
                {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": REPLY}},
                {"type": "content_block_stop", "index": 0},
                {"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 3}},
                {"type": "message_stop"},
            ]
            if payload.get("stream"):
                body = "".join(f"event: {event['type']}\ndata: {json.dumps(event)}\n\n" for event in events).encode()
                content_type = "text/event-stream"
            else:
                message.update(content=[{"type": "text", "text": REPLY}], stop_reason="end_turn")
                message["usage"]["output_tokens"] = 3
                body = json.dumps(message).encode()
                content_type = "application/json"
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}", requests
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def options(root, backend, cli_path, **kwargs):
    """Use private config/cwd and disable all tools and incidental traffic."""
    workspace = root / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    return ClaudeAgentOptions(
        cli_path=str(cli_path),
        cwd=str(workspace),
        model=backend.model,
        tools=[],
        setting_sources=[],
        system_prompt="Follow the probe instruction. Return only the requested exact marker, with no explanation.",
        max_turns=1,
        thinking={"type": "disabled"},
        env={
            "CLAUDE_CONFIG_DIR": str(root / "config"),
            "ANTHROPIC_BASE_URL": backend.base_url,
            "ANTHROPIC_API_KEY": backend.api_key,
            "ANTHROPIC_AUTH_TOKEN": "",
            "CLAUDE_CODE_OAUTH_TOKEN": "",
            "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
            "CLAUDE_CODE_DISABLE_TERMINAL_TITLE": "1",
            "CLAUDE_CODE_EMIT_SESSION_STATE_EVENTS": "1",
            "CLAUDE_CODE_BG_TASKS_REPORT_RUNNING": "1",
            "CLAUDE_CODE_FORCE_SESSION_PERSISTENCE": "1",
            "CLAUDE_CODE_SKIP_PROMPT_HISTORY": "",
            "CLAUDE_CODE_MAX_OUTPUT_TOKENS": "128",
            "API_TIMEOUT_MS": "30000",
            "DISABLE_TELEMETRY": "1",
            "DISABLE_ERROR_REPORTING": "1",
            "HTTP_PROXY": "",
            "HTTPS_PROXY": "",
            "ALL_PROXY": "",
            "NO_PROXY": "127.0.0.1,localhost",
        },
        **kwargs,
    )


class Session:
    """Read continuously, including status frames emitted after a result."""

    def __init__(self, sdk_options, backend):
        self.client = ClaudeSDKClient(sdk_options)
        self.backend = backend
        self.events = []
        self.queue = asyncio.Queue()
        self.reader = None

    async def read(self):
        """Retain native events and signal transport errors to the caller."""
        try:
            async for message in self.client.receive_messages():
                self.events.append(message)
                await self.queue.put(message)
        except Exception as exc:
            await self.queue.put(exc)

    async def send(self, payload):
        """Send a dict through the public async-input client API."""

        async def frames():
            yield payload

        await self.client.query(frames())

    async def result(self, timeout=15):
        """Wait for an actual CLI acknowledgement, with a bounded deadline."""
        async with asyncio.timeout(timeout):
            while True:
                message = await self.queue.get()
                if isinstance(message, Exception):
                    raise message
                if isinstance(message, ResultMessage):
                    return asdict(message)

    def statuses(self):
        """Return observed status facts, without equating result with idle."""
        return [
            message.data.get("state")
            for message in self.events
            if isinstance(message, SystemMessage) and message.subtype == "session_state_changed"
        ]


@asynccontextmanager
async def connected(sdk_options, backend):
    """Connect and disconnect in the same task; always close the receiver."""
    session = Session(sdk_options, backend)
    await session.client.connect()
    session.reader = asyncio.create_task(session.read())
    try:
        yield session
    finally:
        session.reader.cancel()
        with suppress(asyncio.CancelledError):
            await session.reader
        await session.client.disconnect()


def frame(session_id, text, *, message_id=None, should_query=False, outer_role="user", inner_role="user"):
    """Construct native input without claiming assistant replay is supported."""
    message = {"role": inner_role, "content": [{"type": "text", "text": text}]}
    if outer_role == "assistant":
        message.update(
            id=f"msg_{uuid4().hex}",
            type="message",
            model=MODEL,
            stop_reason="end_turn",
            stop_sequence=None,
            usage={"input_tokens": 0, "output_tokens": 0},
        )
    return {
        "type": outer_role,
        "session_id": session_id,
        "uuid": message_id or str(uuid4()),
        "message": message,
        "parent_tool_use_id": None,
        "shouldQuery": should_query,
        "isSynthetic": True,
    }


def transcript(root, session_id, marker=""):
    """Inspect the CLI's files without rewriting them or changing global env."""
    paths = list((root / "config" / "projects").glob(f"*/{session_id}.jsonl"))
    entries = [json.loads(line) for path in paths for line in path.read_text().split("\n") if line.strip()]
    matches = [entry for entry in entries if marker and marker in json.dumps(entry.get("message", {}))]
    return {
        "exists": bool(paths),
        "entry_types": [entry.get("type") for entry in entries],
        "matches": [
            {"type": entry.get("type"), "uuid": entry.get("uuid"), "message": entry.get("message")} for entry in matches
        ],
    }


async def followup(session, session_id, requests, marker):
    """Inspect the model's received context rather than trusting an SDK echo."""
    start = len(requests)
    prompt = "Reply PROBE_MODEL_REPLY."
    if session.backend.direct:
        prompt = (
            "Find the earlier context marker beginning with PROBE_, excluding PROBE_SEED and PROBE_FLUSH. "
            "Reply with that exact marker only. If no such marker exists, reply PROBE_NO_MARKER."
        )
    await session.send(frame(session_id, prompt, should_query=True))
    result = await session.result()
    await asyncio.sleep(0.1)
    assert result["is_error"] is False, result
    if not session.backend.direct:
        assert result["result"] == REPLY, result
        assert len(requests) > start, "The local model server received no request"
    assistant_text = "\n".join(
        block.text
        for message in session.events
        if isinstance(message, AssistantMessage)
        for block in message.content
        if isinstance(block, TextBlock)
    )
    return {
        "result": result,
        "same_session": result["session_id"] == session_id,
        "states": session.statuses(),
        "model_requests": requests[start:],
        "marker_in_model_context": (
            None if session.backend.direct else any(marker in json.dumps(request) for request in requests[start:])
        ),
        "assistant_text": assistant_text,
        "marker_in_reply": marker in (result.get("result") or assistant_text),
    }


async def resume(root, backend, cli_path, session_id, requests, marker, retry_frame=None):
    """Prove resume, and optionally test whether UUID dedup survives restart."""
    start = len(requests)
    try:
        async with connected(options(root, backend, cli_path, resume=session_id), backend) as session:
            retry = {}
            if retry_frame is not None:
                await session.send(retry_frame)
                try:
                    retry["ack"] = await session.result(timeout=3)
                except Exception as exc:
                    retry["ack_error"] = f"{type(exc).__name__}: {exc}"
                retry["transcript"] = transcript(root, session_id, marker)
            return {**await followup(session, session_id, requests, marker), "retry": retry}
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}", "model_requests": requests[start:]}


async def probe(root, backend, cli_path, requests, case):
    """Exercise creation, no-query append/retry, and both assistant encodings."""
    session_id = str(uuid4())
    marker = f"PROBE_{case.upper()}"
    start = len(requests)
    report = {"session_id": session_id, "marker": marker}
    message = None
    async with connected(options(root, backend, cli_path, session_id=session_id), backend) as session:
        if case == "empty":
            await asyncio.sleep(0.2)
        else:
            if case.startswith("assistant"):
                await session.send(frame(session_id, "PROBE_SEED"))
                report["seed_ack"] = await session.result()
            outer_role = "assistant" if case.startswith("assistant") and case != "assistant_inner_role" else "user"
            inner_role = "assistant" if case.startswith("assistant") else "user"
            message = frame(session_id, marker, outer_role=outer_role, inner_role=inner_role)
            await session.send(message)
            try:
                report["ack"] = await session.result(timeout=3 if case.startswith("assistant") else 15)
            except Exception as exc:
                report["ack_error"] = f"{type(exc).__name__}: {exc}"
            report["after_send"] = transcript(root, session_id, marker)
            if case == "no_query" and "ack" in report:
                report["before_retry"] = transcript(root, session_id, marker)
                await session.send(message)
                try:
                    report["retry_ack"] = await session.result(timeout=3)
                except Exception as exc:
                    report["retry_ack_error"] = f"{type(exc).__name__}: {exc}"
            if case == "assistant_frame":
                report["same_process_followup"] = await followup(session, session_id, requests, marker)
            elif case in {"assistant_flush", "assistant_empty_flush"}:
                flush_text = "PROBE_FLUSH" if case == "assistant_flush" else ""
                await session.send(frame(session_id, flush_text))
                try:
                    report["flush_ack"] = await session.result(timeout=3)
                except Exception as exc:
                    report["flush_ack_error"] = f"{type(exc).__name__}: {exc}"
            await asyncio.sleep(0.2)
        report["states"] = session.statuses()
    report["before_resume"] = transcript(root, session_id, marker)
    report["requests_before_resume"] = requests[start:]
    retry_frame = message if case == "no_query" else None
    report["resume"] = await resume(root, backend, cli_path, session_id, requests, marker, retry_frame)
    report["after_resume"] = transcript(root, session_id, marker)
    return report


async def main(output, env_file=None, base_url=None):
    """Produce a reproducible report, including the exact SDK and CLI versions."""
    cli_path = Path(claude_agent_sdk.__file__).parent / "_bundled" / ("claude.exe" if os.name == "nt" else "claude")
    report = {
        "sdk_version": version("claude-agent-sdk"),
        "cli_version": subprocess.check_output([str(cli_path), "--version"], text=True, timeout=20).strip(),
        "model_backend": "localhost deterministic Messages API; no real model",
        "cases": {},
    }
    backend = None
    if env_file is not None:
        from reme.utils.env_utils import parse_env_file

        config = parse_env_file(env_file)
        backend = Backend(
            base_url=base_url or config["LLM_BASE_URL"],
            api_key=config["LLM_API_KEY"],
            model=config["LLM_MODEL_NAME"],
            direct=True,
        )
        report["model_backend"] = f"real {backend.model}, SDK/CLI directly connected to {backend.base_url}"
    source = nullcontext((backend.base_url, [])) if backend else local_model()
    with source as (endpoint, requests):
        backend = backend or Backend(endpoint)
        with TemporaryDirectory(prefix="reme-cc-protocol-") as directory:
            for case in (
                "empty",
                "no_query",
                "assistant_frame",
                "assistant_frame_no_followup",
                "assistant_inner_role",
                "assistant_flush",
                "assistant_empty_flush",
            ):
                try:
                    result = await probe(Path(directory) / case, backend, cli_path, requests, case)
                except Exception as exc:
                    result = {"error": f"{type(exc).__name__}: {exc}"}
                report["cases"][case] = result
                report["completed_real_model_turns"] = (
                    sum(
                        item.get("result", {}).get("num_turns", 0)
                        for current in report["cases"].values()
                        for item in (current.get("resume", {}), current.get("same_process_followup", {}))
                    )
                    if backend.direct
                    else 0
                )
                summary = {
                    "case": case,
                    "error": result.get("error"),
                    "persisted_matches": len(result.get("before_resume", {}).get("matches", [])),
                    "resume_error": result.get("resume", {}).get("error"),
                    "resumed": result.get("resume", {}).get("same_session"),
                    "marker_in_context": result.get("resume", {}).get("marker_in_model_context"),
                    "marker_in_reply": result.get("resume", {}).get("marker_in_reply"),
                }
                print(json.dumps(summary, ensure_ascii=False), flush=True)
                if output is not None:
                    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="Write the evidence report outside the repository")
    parser.add_argument("--real-model", action="store_true", help="Make real model calls using the env file")
    parser.add_argument("--env-file", type=Path, default=Path(__file__).resolve().parents[2] / ".env")
    parser.add_argument("--base-url", help="Override LLM_BASE_URL with an Anthropic-compatible endpoint")
    arguments = parser.parse_args()
    asyncio.run(main(arguments.output, arguments.env_file if arguments.real_model else None, arguments.base_url))
