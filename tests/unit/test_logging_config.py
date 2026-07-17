"""Tests for logging configuration handoff during app startup."""

import concurrent.futures
import threading
import time

import pytest

from reme.application import Application
from reme.config.config_parser import resolve_app_config
from reme.utils import logger_utils


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


@pytest.mark.parametrize("use_loguru", [True, False])
def test_concurrent_force_init_is_serialized(monkeypatch, use_loguru):
    """Concurrent application startup must not overlap global logger resets."""
    state_lock = threading.Lock()
    active_initializers = 0
    max_active_initializers = 0
    initialization_count = 0
    sentinel_logger = object()

    def fake_init(*_args, **_kwargs):
        nonlocal active_initializers
        nonlocal max_active_initializers
        nonlocal initialization_count

        with state_lock:
            active_initializers += 1
            initialization_count += 1
            max_active_initializers = max(
                max_active_initializers,
                active_initializers,
            )
        time.sleep(0.01)
        with state_lock:
            active_initializers -= 1
        return sentinel_logger

    monkeypatch.setattr(logger_utils, "_logger", None)
    monkeypatch.setattr(logger_utils, "_enable_loguru", lambda: use_loguru)
    monkeypatch.setattr(logger_utils, "_init_loguru", fake_init)
    monkeypatch.setattr(logger_utils, "_init_stdlib", fake_init)

    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as pool:
        results = list(
            pool.map(
                lambda _index: logger_utils.get_logger(force_init=True),
                range(32),
            ),
        )

    assert results == [sentinel_logger] * 32
    assert initialization_count == 32
    assert max_active_initializers == 1
