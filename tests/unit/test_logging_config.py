"""Tests for logging configuration handoff during app startup."""

from reme.application import Application
from reme.config.config_parser import resolve_app_config


class DummyLogger:
    """Minimal logger used to capture initialization without touching sinks."""

    def bind(self, **_kwargs):
        """No-op."""
        return self

    def info(self, *_args, **_kwargs):
        """No-op."""
        return None


def test_resolve_app_config_does_not_create_file_logger(monkeypatch):
    """Config-loading messages should not create empty run log files."""
    calls = []

    def fake_get_logger(**kwargs):
        calls.append(kwargs)
        return DummyLogger()

    monkeypatch.setattr("reme.utils.get_logger", fake_get_logger)

    resolve_app_config(config="demo")

    assert calls[0]["log_to_file"] is False


def test_application_reinitializes_logger_from_final_config(monkeypatch, tmp_path):
    """Application startup should install sinks from the resolved ApplicationConfig."""
    calls = []

    def fake_get_logger(**kwargs):
        calls.append(kwargs)
        return DummyLogger()

    monkeypatch.setattr("reme.application.get_logger", fake_get_logger)
    monkeypatch.setattr("reme.components.base_component.get_logger", lambda **_kwargs: DummyLogger())
    monkeypatch.setattr(Application, "_init_service", lambda self: setattr(self.context, "service", None))
    monkeypatch.setattr(Application, "_init_components", lambda self: None)
    monkeypatch.setattr(Application, "_init_jobs", lambda self: None)

    Application(
        enable_logo=False,
        log_to_console=False,
        log_to_file=True,
        workspace_dir=str(tmp_path / "workspace"),
        service={"backend": "unused"},
    )

    assert calls[0] == {
        "log_to_console": False,
        "log_to_file": True,
        "force_init": True,
    }
