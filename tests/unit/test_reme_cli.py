"""Tests for the ReMe CLI entry helpers."""

import asyncio
from types import SimpleNamespace

import pytest

from reme.components.service import cli_service
from reme.components.service.cli_service import CliService
from reme import reme as reme_module


def test_main_loads_env_before_calling_server(monkeypatch):
    """Client actions can resolve connection settings from the local .env."""
    events = []

    main_globals = reme_module.main.__globals__
    monkeypatch.setitem(main_globals, "load_env", lambda: events.append("load_env"))
    monkeypatch.setitem(main_globals, "parse_args", lambda *_args: ("shell", {"cmd": "pwd"}))

    async def fake_call_server(action, **kwargs):
        events.append(("call_server", action, kwargs))

    monkeypatch.setitem(main_globals, "call_server", fake_call_server)

    reme_module.main()

    assert events == ["load_env", ("call_server", "shell", {"cmd": "pwd"})]


def test_prepare_start_config_moves_unknown_start_args_to_job_args(monkeypatch):
    """``reme start job=...`` is translated into a one-shot cli service config."""

    monkeypatch.setattr(
        cli_service,
        "resolve_app_config",
        lambda **kwargs: {
            **kwargs,
            "service": {"backend": "http", "host": "127.0.0.1"},
        },
    )

    cfg = cli_service.prepare_start_config(
        {
            "config": "jinli_lme",
            "workspace_dir": "/tmp/reme",
            "job": "search",
            "query": "hello",
            "limit": 3,
        },
    )

    assert cfg["config"] == "jinli_lme"
    assert cfg["workspace_dir"] == "/tmp/reme"
    assert cfg["enable_logo"] is False
    assert cfg["log_to_console"] is False
    assert cfg["service"] == {
        "backend": "cli",
        "host": "127.0.0.1",
        "job": "search",
        "job_args": {"query": "hello", "limit": 3},
    }


def test_should_precheck_start_skips_cli_service():
    """CLI service is local execution and should not run port prechecks."""
    assert cli_service.should_precheck_start({"service": {"backend": "cli"}}) is False
    assert cli_service.should_precheck_start({"service": {"backend": "http"}}) is True


def test_cli_service_runs_configured_job_and_closes_app(capsys):
    """CLI service runs one local job through app lifecycle and prints its answer."""
    events = []

    class FakeApp:
        """Minimal app stub for exercising CliService lifecycle."""

        async def start(self):
            """Record app startup."""
            events.append("start")

        async def close(self):
            """Record app shutdown."""
            events.append("close")

        async def run_job(self, name, **kwargs):
            """Record job execution and return a successful response."""
            events.append(("run_job", name, kwargs))
            return SimpleNamespace(answer="found it", success=True, metadata={"hits": 1})

    service = CliService(job="search", job_args={"query": "hello"})

    service.start_service(FakeApp())

    assert events == [
        "start",
        ("run_job", "search", {"query": "hello"}),
        "close",
    ]
    assert capsys.readouterr().out == "found it\n"


def test_cli_service_can_print_metadata_from_service_config(capsys):
    """service.show_metadata controls optional CLI metadata output."""

    class FakeApp:
        """Minimal app stub for exercising metadata output."""

        async def start(self):
            """No-op app startup."""

        async def close(self):
            """No-op app shutdown."""

        async def run_job(self, _name, **_kwargs):
            """Return a successful response with metadata."""
            return SimpleNamespace(answer="found it", success=True, metadata={"hits": 1})

    service = CliService(job="search", show_metadata=True)

    service.start_service(FakeApp())

    assert capsys.readouterr().out == 'found it\n{"hits": 1}\n'


def test_cli_service_exits_nonzero_on_failed_response(capsys):
    """Failed local CLI jobs write to stderr and produce a failing process status."""
    events = []

    class FakeApp:
        """Minimal app stub for exercising failure handling."""

        async def start(self):
            """Record app startup."""
            events.append("start")

        async def close(self):
            """Record app shutdown."""
            events.append("close")

        async def run_job(self, name, **kwargs):
            """Record job execution and return a failed response."""
            events.append(("run_job", name, kwargs))
            return SimpleNamespace(answer="boom", success=False, metadata={})

    service = CliService(job="search", job_args={"query": "hello"})

    with pytest.raises(SystemExit) as exc_info:
        service.start_service(FakeApp())

    assert exc_info.value.code == 1
    assert events == [
        "start",
        ("run_job", "search", {"query": "hello"}),
        "close",
    ]
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == "boom\n"


def test_call_server_passes_client_kwargs_to_client(monkeypatch, capsys):
    """CLI helper forwards connection options to the selected client."""
    seen = {}

    class FakeClient:
        """Async client stub that records call arguments."""

        def __init__(self, **kwargs):
            seen["client_kwargs"] = kwargs

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc_val, exc_tb):
            return None

        async def __call__(self, action: str, **kwargs):
            seen["action"] = action
            seen["payload"] = kwargs
            yield "ok"

    monkeypatch.setattr(reme_module.R, "get", lambda component_type, backend: FakeClient)
    monkeypatch.setattr(reme_module, "running_service_config", lambda: None)

    async def run():
        await reme_module.call_server(
            "search",
            backend="http",
            host="127.0.0.2",
            port=2444,
            timeout=1.5,
            query="hello",
        )

    asyncio.run(run())

    assert seen["client_kwargs"] == {"host": "127.0.0.2", "port": 2444, "timeout": 1.5}
    assert seen["action"] == "search"
    assert seen["payload"] == {"query": "hello"}
    assert capsys.readouterr().out == "ok\n"


def test_call_server_treats_show_metadata_as_client_kwarg(monkeypatch, capsys):
    """show_metadata controls client display and is not sent as a tool argument."""
    seen = {}

    class FakeClient:
        """Async client stub that records call arguments."""

        def __init__(self, **kwargs):
            seen["client_kwargs"] = kwargs

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc_val, exc_tb):
            return None

        async def __call__(self, action: str, **kwargs):
            seen["action"] = action
            seen["payload"] = kwargs
            yield "ok"

    monkeypatch.setattr(reme_module.R, "get", lambda component_type, backend: FakeClient)
    monkeypatch.setattr(reme_module, "running_service_config", lambda: None)

    async def run():
        await reme_module.call_server("version", backend="http", show_metadata=True)

    asyncio.run(run())

    assert seen["client_kwargs"] == {"show_metadata": True}
    assert seen["action"] == "version"
    assert seen["payload"] == {}
    assert capsys.readouterr().out == "ok\n"


def test_call_server_passes_shell_parameters_as_payload(monkeypatch, capsys):
    """Shell-specific parameter names do not collide with client options."""
    seen = {}

    class FakeClient:
        """Async client stub that records shell request arguments."""

        def __init__(self, **kwargs):
            seen["client_kwargs"] = kwargs

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc_val, exc_tb):
            return None

        async def __call__(self, action: str, **kwargs):
            seen["action"] = action
            seen["payload"] = kwargs
            yield "ok"

    monkeypatch.setattr(reme_module.R, "get", lambda component_type, backend: FakeClient)
    monkeypatch.setattr(reme_module, "running_service_config", lambda: None)

    async def run():
        await reme_module.call_server("shell", backend="http", cmd="ls", shell_timeout=5)

    asyncio.run(run())

    assert seen["action"] == "shell"
    assert seen["payload"] == {"cmd": "ls", "shell_timeout": 5}
    assert capsys.readouterr().out == "ok\n"
