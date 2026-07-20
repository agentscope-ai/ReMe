"""Tests for logging configuration handoff during app startup."""

import concurrent.futures
from datetime import datetime
import io
import logging
import threading
import time
from unittest.mock import Mock

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


def test_loguru_filename_includes_start_time_and_process_id(monkeypatch, tmp_path):
    """Independent ReMe processes should write to distinct Loguru files."""
    fixed_datetime = Mock()
    fixed_datetime.now.return_value = datetime(2026, 7, 20, 15, 42, 18)
    monkeypatch.setattr(logger_utils, "datetime", fixed_datetime)
    monkeypatch.setattr(logger_utils.os, "getpid", lambda: 31247)

    logger = logger_utils._init_loguru(str(tmp_path), "INFO", False, True)  # pylint: disable=protected-access
    try:
        assert (tmp_path / "2026-07-20_15-42-18_31247.log").is_file()
    finally:
        logger.remove()


def test_stdlib_formatter_matches_qwenpaw_console_format(monkeypatch, tmp_path, capsys):
    """Stdlib logs should use QwenPaw's level/path/time/message layout."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("REME_DISABLE_LOGURU", "true")
    source_path = tmp_path / "src" / "qwenpaw" / "worker.py"
    record = logging.LogRecord(
        name="reme",
        level=logging.INFO,
        pathname=str(source_path),
        lineno=42,
        msg="Memory index loaded",
        args=(),
        exc_info=None,
    )
    record.created = 0

    logger = logger_utils.get_logger(log_to_file=False, force_init=True)
    logger.handle(record)

    formatted_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(record.created))
    assert capsys.readouterr().out == (f"INFO src/qwenpaw/worker.py:42 | {formatted_time} | Memory index loaded\n")
    logger_utils.get_logger(log_to_console=False, log_to_file=False, force_init=True)


def test_stdlib_forwards_screen_and_file_logs_to_qwenpaw(monkeypatch, tmp_path):
    """Embedded stdlib logging should reuse QwenPaw's active sinks."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("REME_DISABLE_LOGURU", "true")
    monkeypatch.setattr(logger_utils, "_logger", None)

    qwenpaw_logger = logging.getLogger("qwenpaw")
    original_handlers = list(qwenpaw_logger.handlers)
    original_level = qwenpaw_logger.level
    original_propagate = qwenpaw_logger.propagate
    for handler in original_handlers:
        qwenpaw_logger.removeHandler(handler)

    console_stream = io.StringIO()
    console_handler = logging.StreamHandler(console_stream)
    file_path = tmp_path / "qwenpaw.log"
    file_handler = logging.FileHandler(file_path, encoding="utf-8")
    formatter = logging.Formatter("%(levelname)s | %(message)s")
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)
    qwenpaw_logger.addHandler(console_handler)
    qwenpaw_logger.addHandler(file_handler)
    qwenpaw_logger.setLevel(logging.INFO)
    qwenpaw_logger.propagate = False

    reme_logger = logging.getLogger("reme")
    try:
        logger = logger_utils.get_logger(
            log_to_console=True,
            log_to_file=True,
            force_init=True,
        )
        logger.info("Memory index loaded")
        file_handler.flush()

        assert console_stream.getvalue() == "INFO | Memory index loaded\n"
        assert file_path.read_text(encoding="utf-8") == "INFO | Memory index loaded\n"
        assert not (tmp_path / "logs").exists()
        assert len(reme_logger.handlers) == 1
        assert reme_logger.handlers[0].target_name == "qwenpaw"
    finally:
        for handler in list(reme_logger.handlers):
            reme_logger.removeHandler(handler)
            handler.close()
        qwenpaw_logger.removeHandler(console_handler)
        qwenpaw_logger.removeHandler(file_handler)
        console_handler.close()
        file_handler.close()
        for handler in original_handlers:
            qwenpaw_logger.addHandler(handler)
        qwenpaw_logger.setLevel(original_level)
        qwenpaw_logger.propagate = original_propagate


def test_explicit_loguru_enable_keeps_original_backend(monkeypatch):
    """An explicit false value must continue to select Loguru unchanged."""
    sentinel_logger = object()
    calls = []

    def fake_init(*args, **kwargs):
        calls.append((args, kwargs))
        return sentinel_logger

    monkeypatch.setenv("REME_DISABLE_LOGURU", "false")
    monkeypatch.setattr(logger_utils, "_logger", None)
    monkeypatch.setattr(logger_utils, "_init_loguru", fake_init)
    monkeypatch.setattr(
        logger_utils,
        "_init_stdlib",
        lambda *_args, **_kwargs: pytest.fail("stdlib backend selected"),
    )

    result = logger_utils.get_logger(force_init=True)

    assert result is sentinel_logger
    assert len(calls) == 1


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
