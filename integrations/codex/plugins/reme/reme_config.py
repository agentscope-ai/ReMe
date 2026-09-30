"""Persistent Codex plugin settings and owner-only, atomic local storage."""

from __future__ import annotations

import contextlib
import json
import math
import os
import tempfile
from pathlib import Path
from urllib.parse import urlsplit
from zoneinfo import ZoneInfo

HOME_ENV = "CODEX_HOME"
HOME_DEFAULT = "~/.codex"
DEFAULTS = {
    "mcp_url": "http://127.0.0.1:2333/mcp",
    "auto_recall": True,
    "auto_memory": True,
    "recall_limit": 5,
    "recall_min_score": 0.0,
    "recall_timeout": 5.0,
    "request_timeout": 600.0,
    "memory_interval": 5,
    "shutdown_timeout": 2.0,
    "context_max_chars": 8000,
    "timezone": "Asia/Shanghai",
}


def data_dir() -> Path:
    """Keep configuration and retry state outside the versioned plugin cache."""
    return Path(os.environ.get(HOME_ENV) or HOME_DEFAULT).expanduser() / "reme"


def load_config() -> dict:
    """Read the persistent settings shared by the MCP server and lifecycle hooks."""
    path = data_dir() / "config.json"
    values = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    return validate_config(values)


def validate_config(values: dict) -> dict:
    """Validate user overrides before the MCP server or hooks can use them."""
    if not isinstance(values, dict) or values.keys() - (DEFAULTS.keys() | {"api_url"}):
        raise ValueError("Unknown ReMe configuration fields")
    # Accept the former HTTP base only for migration; no request uses it.
    legacy_api = values.get("api_url", "")
    if not isinstance(legacy_api, str):
        raise ValueError("api_url must be a string")
    if legacy_api:
        validate_url(legacy_api)
    config = {**DEFAULTS, **{key: value for key, value in values.items() if key != "api_url"}}
    for key, default in DEFAULTS.items():
        value = config[key]
        if isinstance(default, bool):
            if not isinstance(value, bool):
                raise ValueError(f"{key} must be a boolean")
        elif isinstance(default, str):
            if not isinstance(value, str):
                raise ValueError(f"{key} must be a string")
        elif isinstance(default, (int, float)):
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"{key} must be a finite number")
            if value < 0 or (value == 0 and key != "recall_min_score"):
                raise ValueError(f"{key} must be positive")
            if isinstance(default, int) and not isinstance(value, int):
                raise ValueError(f"{key} must be an integer")
    if config["recall_timeout"] > 10 or config["request_timeout"] > 600 or config["shutdown_timeout"] > 2:
        raise ValueError("Timeout exceeds the lifecycle hook budget")
    try:
        ZoneInfo(config["timezone"])
    except (KeyError, ValueError) as exc:
        raise ValueError("timezone must be an IANA timezone") from exc
    config["mcp_url"] = validate_url(config["mcp_url"])
    # Preserve standard /mcp retry-directory identities from the HTTP adapter.
    # Custom legacy api_url queues remain untouched rather than being sent elsewhere.
    config["endpoint"] = config["mcp_url"][:-4] if config["mcp_url"].endswith("/mcp") else "mcp:" + config["mcp_url"]
    return config


def validate_url(url: str) -> str:
    """Accept HTTP endpoints without embedded credentials or ambiguous URL suffixes."""
    url = url.rstrip("/")
    parsed = urlsplit(url)
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.hostname
        or any((parsed.username is not None, parsed.password is not None, parsed.query, parsed.fragment))
        or any(character.isspace() for character in url)
    ):
        raise ValueError("ReMe URLs must be absolute HTTP(S) URLs without credentials, query, or fragment")
    _ = parsed.port
    return url


@contextlib.contextmanager
def lock(path: Path):
    """Use a non-blocking OS lock, released automatically if a hook is killed."""
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    with path.open("a+b") as handle:
        if os.name == "nt":
            import msvcrt

            handle.write(b"\0")
            handle.flush()
            handle.seek(0)

            def acquire():
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)

            def release():
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)

        else:
            import fcntl

            def acquire():
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)

            def release():
                fcntl.flock(handle, fcntl.LOCK_UN)

        try:
            acquire()
        except OSError:
            yield False
            return
        try:
            yield True
        finally:
            release()


def write_json(path: Path, value: dict) -> None:
    """Atomically persist retry metadata with owner-only file permissions."""
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=".reme-")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def settings_values(config: dict) -> dict:
    """Expose only editable settings, excluding the derived retry-directory identity."""
    return {key: config[key] for key in DEFAULTS}


def update_settings(changes: dict) -> dict:
    """Validate and merge a UI patch under the same cross-process lock as other editors."""
    if not isinstance(changes, dict) or not changes or changes.keys() - DEFAULTS.keys():
        raise ValueError("Provide at least one supported ReMe setting")
    with lock(data_dir() / "config.lock") as acquired:
        if not acquired:
            raise RuntimeError("Another settings update is in progress; please retry")
        current = settings_values(load_config())
        config = validate_config({**current, **changes})
        values = settings_values(config)
        write_json(data_dir() / "config.json", values)
    return values
