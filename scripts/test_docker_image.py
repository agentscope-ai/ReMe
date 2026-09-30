"""Exercise a built Docker image with isolated memory and no model credentials."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import tempfile
import time
from urllib.request import ProxyHandler, Request, build_opener
import uuid

_OFFLINE_JOB_OVERRIDES = [
    f"jobs.{name}.{key}={value}"
    for name in ("dream_cron", "proactive_refresh_cron", "optimize_index_cron")
    for key, value in (("backend", "base"), ("enable_serve", "false"))
]


def docker(*arguments: str) -> str:
    """Run Docker without interpolating shell text."""
    return subprocess.check_output(["docker", *arguments], text=True).strip()


def request(base_url: str, path: str, payload: dict | None = None, headers: dict | None = None) -> tuple[object, dict]:
    """Read JSON, MCP's SSE response, or a static resource from the container."""
    request_headers = {"Content-Type": "application/json", **(headers or {})}
    body = json.dumps(payload).encode() if payload is not None else None
    req = Request(base_url + path, data=body, headers=request_headers)
    with build_opener(ProxyHandler({})).open(req, timeout=10) as response:
        text = response.read().decode()
        response_headers = dict(response.headers)
        if "text/event-stream" in response.headers.get("Content-Type", ""):
            messages = [json.loads(line[5:].strip()) for line in text.splitlines() if line.startswith("data:")]
            return messages[-1], response_headers
        try:
            return json.loads(text), response_headers
        except ValueError:
            return text, response_headers


def wait_ready(container: str, base_url: str, port: int) -> None:
    """Wait for initialization, checking the same health command Docker runs."""
    deadline = time.monotonic() + 180
    while time.monotonic() < deadline:
        try:
            payload, _ = request(base_url, "/health_check", {})
            if payload["success"] and payload["metadata"]["health"]["healthy"]:
                docker("exec", container, "python", "/usr/local/lib/reme_container.py", "--healthcheck")
                assert " - healthy" in docker("exec", container, "reme", "health_check")
                assert f"PORT={port} " in docker("exec", container, "reme", "find_reme")
                return
        except (OSError, ValueError, KeyError, TypeError):
            pass
        if docker("inspect", "--format", "{{.State.Running}}", container) != "true":
            raise RuntimeError("ReMe exited before it became healthy")
        time.sleep(1)
    raise TimeoutError("ReMe did not become healthy within 180 seconds")


def start_container(image: str, container: str, data: Path, port: int = 2432, cli_override: bool = False) -> str:
    """Publish a random loopback port and mount only the disposable workspace."""
    docker(
        "run",
        "--detach",
        "--name",
        container,
        "--user",
        f"{os.getuid()}:{os.getgid()}",
        "--publish",
        f"127.0.0.1::{port}",
        "--mount",
        f"type=bind,source={data},target=/data",
        "--env",
        "LLM_API_KEY=",
        "--env",
        f"REME_PORT={2333 if cli_override else port}",
        image,
        "start",
        *_OFFLINE_JOB_OVERRIDES,
        *([f"service.port={port}"] if cli_override else []),
    )
    mappings = json.loads(docker("inspect", "--format", "{{json .NetworkSettings.Ports}}", container))
    host_port = mappings[f"{port}/tcp"][0]["HostPort"]
    base_url = f"http://127.0.0.1:{host_port}"
    wait_ready(container, base_url, port)
    return base_url


def stop_container(container: str) -> None:
    """Require SIGTERM to close the application instead of timing out into SIGKILL."""
    docker("stop", "--time", "60", container)
    exit_code = docker("inspect", "--format", "{{.State.ExitCode}}", container)
    # Uvicorn re-raises SIGTERM after completing its lifespan shutdown.
    assert exit_code in {"0", "143"}, f"Container did not shut down cleanly: {exit_code}"
    logs = subprocess.check_output(["docker", "logs", container], text=True, stderr=subprocess.STDOUT)
    assert "Application shutdown complete" in logs, logs
    docker("rm", container)


def check_mcp(base_url: str) -> None:
    """Initialize streamable HTTP MCP and verify its tools remain available."""
    headers = {"Accept": "application/json, text/event-stream"}
    result, response_headers = request(
        base_url,
        "/mcp",
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2025-03-26",
                "capabilities": {},
                "clientInfo": {"name": "smoke", "version": "1"},
            },
        },
        headers,
    )
    headers["MCP-Protocol-Version"] = result["result"]["protocolVersion"]
    session_id = next((value for key, value in response_headers.items() if key.lower() == "mcp-session-id"), None)
    if session_id:
        headers["Mcp-Session-Id"] = session_id
    request(base_url, "/mcp", {"jsonrpc": "2.0", "method": "notifications/initialized"}, headers)
    tools, _ = request(base_url, "/mcp", {"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}}, headers)
    names = {tool["name"] for tool in tools["result"]["tools"]}
    assert {"version", "search", "health_check", "read"} <= names


def run_smoke(image: str) -> None:
    """Verify packaged assets, file safety, indexing, restart recovery, and signals."""
    docker("run", "--rm", image, "python", "-c", "import os, faiss, zvec; assert os.getuid() != 0")
    # Keep scheduled model jobs inactive even if a check crosses a cron boundary.
    version = docker("run", "--rm", image, "reme", "start", "job=version", *_OFFLINE_JOB_OVERRIDES)
    with tempfile.TemporaryDirectory(prefix="reme-docker-smoke-") as directory:
        data = Path(directory) / "data"
        data.mkdir()
        container = f"reme-smoke-{uuid.uuid4().hex[:12]}"
        try:
            base_url = start_container(image, container, data)
            page, _ = request(base_url, "/")
            assert isinstance(page, str) and '<div id="root">' in page
            result, _ = request(base_url, "/version", {})
            assert result["success"] and result["answer"] == version
            check_mcp(base_url)
            note = "# Docker persistence\n\nDurable quokka memory survives container replacement.\n"
            result, _ = request(base_url, "/write", {"path": "daily/2026-09-30/smoke.md", "content": note})
            assert result["success"], result
            result, _ = request(base_url, "/write", {"path": "../escape.md", "content": "blocked"})
            assert not result["success"], result
            note_file = data / "daily/2026-09-30/smoke.md"
            assert note_file.read_text() == note
            assert note_file.stat().st_uid == os.getuid()
            stop_container(container)
            # Recreate with a different CLI port to catch probes using stale ENV defaults.
            base_url = start_container(image, container, data, port=2433, cli_override=True)
            result, _ = request(base_url, "/read", {"path": "daily/2026-09-30/smoke.md"})
            assert result["success"] and "Durable quokka" in result["answer"]
            deadline = time.monotonic() + 30
            while True:
                result, _ = request(base_url, "/search", {"query": "quokka"})
                assert result["success"], result
                if "smoke.md" in json.dumps(result):
                    break
                if time.monotonic() >= deadline:
                    raise AssertionError(f"Persisted memory was not indexed: {result}")
                time.sleep(1)
            stop_container(container)
        except Exception:
            subprocess.run(["docker", "logs", container], check=False)
            raise
        finally:
            subprocess.run(["docker", "rm", "--force", container], check=False, capture_output=True)
    print(f"Docker smoke checks passed: {image}")


def main() -> None:
    """Run against an already-built image; never build or publish implicitly."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", default="reme:local")
    args = parser.parse_args()
    run_smoke(args.image)


if __name__ == "__main__":
    main()
