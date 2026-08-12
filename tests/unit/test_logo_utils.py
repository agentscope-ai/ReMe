"""Startup logo metadata tests."""

from io import StringIO
from types import SimpleNamespace

from rich.console import Console

from reme import application as application_module
from reme.application import Application
from reme.constants import REME_DEFAULT_HOST, REME_DEFAULT_PORT
from reme.schema import ApplicationConfig, ComponentConfig
from reme.utils import logo_utils


def _render_logo(monkeypatch, config: ApplicationConfig, runtime_service=None) -> str:
    output = StringIO()
    console = Console(file=output, force_terminal=False, width=120)
    monkeypatch.setattr(logo_utils, "Console", lambda: console)

    logo_utils.print_logo(config, runtime_service)

    return output.getvalue()


def test_logo_uses_runtime_http_address(monkeypatch) -> None:
    """Display the address resolved by the instantiated service."""
    config = ApplicationConfig(service=ComponentConfig(backend="http"))
    runtime_service = SimpleNamespace(host="0.0.0.0", port=8123)

    output = _render_logo(monkeypatch, config, runtime_service)

    assert "http://0.0.0.0:8123" in output


def test_logo_fallback_matches_service_defaults(monkeypatch) -> None:
    """Keep direct print_logo callers aligned with service defaults."""
    config = ApplicationConfig(service=ComponentConfig(backend="http"))

    output = _render_logo(monkeypatch, config)

    assert f"http://{REME_DEFAULT_HOST}:{REME_DEFAULT_PORT}" in output


def test_application_passes_instantiated_service_to_logo(monkeypatch, tmp_path) -> None:
    """Render the service address after backend defaults and overrides resolve."""
    captured = {}
    monkeypatch.setattr(Application, "_init_components", lambda self: None)
    monkeypatch.setattr(Application, "_init_jobs", lambda self: None)
    monkeypatch.setattr(
        application_module,
        "print_logo",
        lambda config, runtime_service: captured.update(
            host=runtime_service.host,
            port=runtime_service.port,
        ),
    )

    Application(
        workspace_dir=str(tmp_path),
        service={"backend": "http", "host": "0.0.0.0", "port": 8123},
        log_to_console=False,
        log_to_file=False,
    )

    assert captured == {"host": "0.0.0.0", "port": 8123}
