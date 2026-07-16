"""Hermes provider contract tests using a real loopback HTTP server."""

# pylint: disable=redefined-outer-name

from __future__ import annotations

import importlib.util
import json
import sys
import threading
import time
import types

from collections.abc import Iterator
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest


class _MemoryProvider:
    """Minimal Hermes ABC stand-in; the real loader is exercised separately."""


@pytest.fixture
def plugin_module(monkeypatch: pytest.MonkeyPatch):
    """Load the plugin the same way Hermes loads an isolated plugin package."""
    agent = types.ModuleType("agent")
    agent.__path__ = []  # type: ignore[attr-defined]
    memory_provider = types.ModuleType("agent.memory_provider")
    memory_provider.MemoryProvider = _MemoryProvider
    monkeypatch.setitem(sys.modules, "agent", agent)
    monkeypatch.setitem(sys.modules, "agent.memory_provider", memory_provider)

    module_name = "_reme_hermes_test_plugin"
    for name in list(sys.modules):
        if name == module_name or name.startswith(f"{module_name}."):
            monkeypatch.delitem(sys.modules, name, raising=False)
    plugin_dir = Path(__file__).resolve().parents[2] / "plugins" / "hermes_agent"
    spec = importlib.util.spec_from_file_location(
        module_name,
        plugin_dir / "__init__.py",
        submodule_search_locations=[str(plugin_dir)],
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, module_name, module)
    spec.loader.exec_module(module)
    return module


class _ActionHandler(BaseHTTPRequestHandler):
    calls: list[tuple[str, dict[str, Any]]]
    responses: dict[str, tuple[int, Any] | tuple[int, Any, float]]

    def do_POST(self) -> None:  # noqa: N802 - stdlib callback name
        """Serve one ReMe-compatible action request."""
        length = int(self.headers.get("Content-Length", "0"))
        body = json.loads(self.rfile.read(length) or b"{}")
        self.calls.append((self.path, body))
        spec = self.responses.get(self.path, (404, {"detail": "not found"}))
        status, response = spec[:2]
        if len(spec) == 3:
            time.sleep(spec[2])
        encoded = json.dumps(response).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        try:
            self.wfile.write(encoded)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def log_message(self, _format: str, *_args: object) -> None:
        return


@contextmanager
def action_server(
    responses: dict[str, tuple[int, Any] | tuple[int, Any, float]],
) -> Iterator[tuple[str, list[tuple[str, dict[str, Any]]]]]:
    """Run a loopback ReMe action server and expose captured requests."""
    calls: list[tuple[str, dict[str, Any]]] = []
    handler = type("Handler", (_ActionHandler,), {"calls": calls, "responses": responses})
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        host, port = server.server_address
        yield f"http://{host}:{port}", calls
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def _healthy_response(answer: str = "healthy") -> dict[str, Any]:
    return {
        "success": True,
        "answer": answer,
        "metadata": {"health": {"healthy": True, "version": "test"}},
    }


def _wait_for_call(calls: list[tuple[str, dict[str, Any]]], path: str, timeout: float = 2.0) -> None:
    """Wait until the background writer reaches a loopback action."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if any(call_path == path for call_path, _ in calls):
            return
        time.sleep(0.01)
    raise AssertionError(f"timed out waiting for {path}; calls={calls!r}")


def _write_config(plugin_module, home: Path, endpoint: str, **overrides: Any) -> None:
    """Seed runtime config without exercising the separately tested setup path."""
    del plugin_module
    values = {"endpoint": endpoint, "health_retry_seconds": 0.1, **overrides}
    home.mkdir(parents=True, exist_ok=True)
    (home / "reme.json").write_text(json.dumps(values), encoding="utf-8")


def test_setup_is_profile_scoped_and_atomic(plugin_module, tmp_path: Path) -> None:
    """Keep endpoint configuration private to each Hermes profile."""
    first = tmp_path / "profile-a"
    second = tmp_path / "profile-b"
    provider = plugin_module.ReMeMemoryProvider()

    responses = {"/health_check": (200, _healthy_response())}
    with action_server(responses) as (first_endpoint, _), action_server(responses) as (second_endpoint, _):
        provider.save_config({"endpoint": first_endpoint}, str(first))
        provider.save_config({"endpoint": second_endpoint}, str(second))

    first_config = json.loads((first / "reme.json").read_text(encoding="utf-8"))
    second_config = json.loads((second / "reme.json").read_text(encoding="utf-8"))
    assert first_config["endpoint"] == first_endpoint
    assert second_config["endpoint"] == second_endpoint
    assert (first / "reme.json").stat().st_mode & 0o777 == 0o600
    assert not list(first.glob(".reme.json.*"))


def test_setup_rejects_unhealthy_endpoint_without_overwrite(plugin_module, tmp_path: Path) -> None:
    """Preserve the last working profile config when endpoint validation fails."""
    healthy = {"/health_check": (200, _healthy_response())}
    unhealthy = {"/health_check": (503, {"detail": "starting"})}
    provider = plugin_module.ReMeMemoryProvider()

    with action_server(healthy) as (healthy_endpoint, _), action_server(unhealthy) as (unhealthy_endpoint, _):
        provider.save_config({"endpoint": healthy_endpoint}, str(tmp_path))
        with pytest.raises(plugin_module.ReMeServiceError, match="HTTP 503"):
            provider.save_config({"endpoint": unhealthy_endpoint}, str(tmp_path))

    saved = json.loads((tmp_path / "reme.json").read_text(encoding="utf-8"))
    assert saved["endpoint"] == healthy_endpoint


def test_setup_rejects_incomplete_health_envelope(plugin_module, tmp_path: Path) -> None:
    """Do not accept an unrelated endpoint that returns generic JSON."""
    responses = {"/health_check": (200, {"success": True, "answer": "ok"})}
    with action_server(responses) as (endpoint, _):
        with pytest.raises(plugin_module.ReMeServiceError, match="healthy component snapshot"):
            plugin_module.ReMeMemoryProvider().save_config({"endpoint": endpoint}, str(tmp_path))


def test_client_requires_explicit_success_envelope(plugin_module) -> None:
    """Reject action responses that omit ReMe's explicit success signal."""
    responses = {"/search": (200, {"answer": "not a ReMe envelope"})}
    with action_server(responses) as (endpoint, _):
        client = plugin_module.ReMeHttpClient(endpoint, timeout=1.0)
        with pytest.raises(plugin_module.ReMeServiceError, match="not a ReMe envelope"):
            client.call("search", {"query": "test"})


def test_lifecycle_retrieves_records_and_switches_sessions(plugin_module, tmp_path: Path) -> None:
    """Exercise recall, recording, session switching, and shutdown."""
    responses = {
        "/health_check": (200, _healthy_response()),
        "/search": (200, {"success": True, "answer": "remembered project decision", "metadata": {}}),
        "/auto_memory": (200, {"success": True, "answer": "recorded", "metadata": {}}),
    }
    with action_server(responses) as (endpoint, calls):
        _write_config(plugin_module, tmp_path, endpoint)
        provider = plugin_module.ReMeMemoryProvider()
        provider.initialize("conversation-one", hermes_home=str(tmp_path), agent_identity="coder")

        assert provider.prefetch("What did we decide?") == "remembered project decision"
        provider.sync_turn("Use SQLite", "Recorded that decision")
        provider.on_session_switch("conversation-two")
        provider.sync_turn("Use BM25 too", "Recorded the retrieval choice")
        provider.shutdown()

    assert [path for path, _ in calls] == [
        "/health_check",
        "/search",
        "/auto_memory",
        "/auto_memory",
    ]
    first_record = calls[2][1]
    second_record = calls[3][1]
    assert first_record["session_id"].startswith("hermes-coder-conversation-one-")
    assert second_record["session_id"].startswith("hermes-coder-conversation-two-")
    assert first_record["session_id"] != second_record["session_id"]
    assert first_record["messages"] == [
        {"name": "user", "role": "user", "content": "Use SQLite"},
        {"name": "assistant", "role": "assistant", "content": "Recorded that decision"},
    ]


def test_gateway_session_argument_preserves_conversation_boundary(plugin_module, tmp_path: Path) -> None:
    """Use per-request gateway sessions instead of cached provider state."""
    responses = {
        "/health_check": (200, _healthy_response()),
        "/auto_memory": (200, {"success": True, "answer": "recorded", "metadata": {}}),
    }
    with action_server(responses) as (endpoint, calls):
        _write_config(plugin_module, tmp_path, endpoint)
        provider = plugin_module.ReMeMemoryProvider()
        provider.initialize("cached-agent", hermes_home=str(tmp_path), agent_identity="gateway")
        provider.sync_turn("first", "one", session_id="chat-a")
        provider.sync_turn("second", "two", session_id="chat-b")
        provider.shutdown()

    assert calls[1][1]["session_id"].startswith("hermes-gateway-chat-a-")
    assert calls[2][1]["session_id"].startswith("hermes-gateway-chat-b-")


def test_non_primary_context_does_not_write(plugin_module, tmp_path: Path) -> None:
    """Avoid recording internal cron, flush, and subagent turns."""
    responses = {
        "/health_check": (200, _healthy_response()),
        "/auto_memory": (200, {"success": True, "answer": "recorded", "metadata": {}}),
    }
    with action_server(responses) as (endpoint, calls):
        _write_config(plugin_module, tmp_path, endpoint)
        provider = plugin_module.ReMeMemoryProvider()
        provider.initialize(
            "cron-session",
            hermes_home=str(tmp_path),
            agent_identity="default",
            agent_context="cron",
        )
        provider.sync_turn("scheduled system prompt", "scheduled result")

    assert [path for path, _ in calls] == ["/health_check"]


def test_unavailable_service_fails_open_and_reports_dropped_write(plugin_module, tmp_path: Path, caplog) -> None:
    """Keep Hermes usable while making lost persistence explicit."""
    responses = {"/health_check": (503, {"detail": "starting"})}
    with action_server(responses) as (endpoint, calls):
        _write_config(plugin_module, tmp_path, endpoint)
        provider = plugin_module.ReMeMemoryProvider()
        with caplog.at_level("WARNING"):
            provider.initialize("session", hermes_home=str(tmp_path), agent_identity="default")
            assert provider.prefetch("question") == ""
            provider.sync_turn("important fact", "answer")
            provider.shutdown()

    assert [path for path, _ in calls] == ["/health_check"]
    assert "recall is disabled" in caplog.text
    assert "did not record completed turn" in caplog.text


def test_unhealthy_snapshot_is_not_treated_as_available(plugin_module, tmp_path: Path) -> None:
    """Reject a successful HTTP response whose health snapshot is unhealthy."""
    responses = {
        "/health_check": (
            200,
            {
                "success": True,
                "answer": "unhealthy",
                "metadata": {"health": {"healthy": False}},
            },
        ),
    }
    with action_server(responses) as (endpoint, calls):
        _write_config(plugin_module, tmp_path, endpoint)
        provider = plugin_module.ReMeMemoryProvider()
        provider.initialize("session", hermes_home=str(tmp_path), agent_identity="default")
        assert provider.prefetch("question") == ""

    assert [path for path, _ in calls] == ["/health_check"]


def test_recall_timeout_does_not_block_hermes_turn(plugin_module, tmp_path: Path) -> None:
    """Bound inline recall independently from slow automatic-memory writes."""
    responses = {
        "/health_check": (200, _healthy_response()),
        "/search": (200, {"success": True, "answer": "too late", "metadata": {}}, 0.5),
        "/auto_memory": (200, {"success": True, "answer": "recorded", "metadata": {}}),
    }
    with action_server(responses) as (endpoint, calls):
        _write_config(plugin_module, tmp_path, endpoint, recall_timeout=0.1)
        provider = plugin_module.ReMeMemoryProvider()
        provider.initialize("session", hermes_home=str(tmp_path), agent_identity="default")

        started = time.monotonic()
        assert provider.prefetch("question") == ""
        elapsed = time.monotonic() - started
        assert provider.prefetch("cooldown") == ""
        provider.sync_turn("recall timed out", "write still works")
        provider.shutdown()

    assert elapsed < 0.4
    assert [path for path, _ in calls] == ["/health_check", "/search", "/auto_memory"]


def test_recording_failure_does_not_disable_recall(plugin_module, tmp_path: Path) -> None:
    """Keep retrieval healthy when only ReMe's LLM-backed write path fails."""
    responses = {
        "/health_check": (200, _healthy_response()),
        "/auto_memory": (200, {"success": False, "answer": "LLM unavailable", "metadata": {}}),
        "/search": (200, {"success": True, "answer": "recall still works", "metadata": {}}),
    }
    with action_server(responses) as (endpoint, calls):
        _write_config(plugin_module, tmp_path, endpoint)
        provider = plugin_module.ReMeMemoryProvider()
        provider.initialize("session", hermes_home=str(tmp_path), agent_identity="default")
        provider.sync_turn("remember this", "attempted")
        _wait_for_call(calls, "/auto_memory")

        assert provider.prefetch("existing fact") == "recall still works"
        provider.shutdown()

    assert [path for path, _ in calls] == ["/health_check", "/auto_memory", "/search"]


def test_shutdown_is_bounded_when_write_is_slow(plugin_module, tmp_path: Path, caplog) -> None:
    """Do not let an in-flight automatic-memory request wedge Hermes exit."""
    responses = {
        "/health_check": (200, _healthy_response()),
        "/auto_memory": (200, {"success": True, "answer": "recorded", "metadata": {}}, 0.8),
    }
    with action_server(responses) as (endpoint, calls):
        _write_config(plugin_module, tmp_path, endpoint, shutdown_timeout=0.1)
        provider = plugin_module.ReMeMemoryProvider()
        provider.initialize("session", hermes_home=str(tmp_path), agent_identity="default")
        provider.sync_turn("slow", "write")
        _wait_for_call(calls, "/auto_memory")
        provider.sync_turn("queued", "must be abandoned")

        with caplog.at_level("WARNING"):
            started = time.monotonic()
            provider.shutdown()
            elapsed = time.monotonic() - started

    assert elapsed < 0.4
    assert "abandoned 1 queued write(s)" in caplog.text
    assert [path for path, _ in calls] == ["/health_check", "/auto_memory"]


def test_process_exit_fallback_drains_once(plugin_module, tmp_path: Path) -> None:
    """Drain a queued turn when Hermes exits without normal provider cleanup."""
    responses = {
        "/health_check": (200, _healthy_response()),
        "/auto_memory": (200, {"success": True, "answer": "recorded", "metadata": {}}),
    }
    with action_server(responses) as (endpoint, calls):
        _write_config(plugin_module, tmp_path, endpoint)
        provider = plugin_module.ReMeMemoryProvider()
        provider.initialize("session", hermes_home=str(tmp_path), agent_identity="default")
        provider.sync_turn("process", "exit")

        shutdown_at_exit = getattr(provider, "_atexit_shutdown")
        shutdown_at_exit()
        shutdown_at_exit()  # idempotent if normal cleanup also ran

    assert [path for path, _ in calls] == ["/health_check", "/auto_memory"]


def test_service_recovers_after_health_retry_cooldown(plugin_module, tmp_path: Path) -> None:
    """Resume recall after a previously unavailable ReMe service recovers."""
    responses = {"/health_check": (503, {"detail": "starting"})}
    with action_server(responses) as (endpoint, calls):
        _write_config(plugin_module, tmp_path, endpoint)
        provider = plugin_module.ReMeMemoryProvider()
        provider.initialize("session", hermes_home=str(tmp_path), agent_identity="default")

        responses["/health_check"] = (200, _healthy_response())
        responses["/search"] = (200, {"success": True, "answer": "recovered", "metadata": {}})
        time.sleep(0.11)
        assert provider.prefetch("question") == "recovered"

    assert [path for path, _ in calls] == ["/health_check", "/health_check", "/search"]


def test_register_exposes_provider(plugin_module) -> None:
    """Register exactly one provider without model-visible tools."""
    registered: list[Any] = []
    context = types.SimpleNamespace(register_memory_provider=registered.append)
    plugin_module.register(context)
    assert len(registered) == 1
    assert registered[0].name == "reme"
    assert registered[0].get_tool_schemas() == []
