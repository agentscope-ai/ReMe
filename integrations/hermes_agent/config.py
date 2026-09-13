"""Profile-local configuration for the Hermes ReMe memory provider."""

from __future__ import annotations

import json
import math
import os
import tempfile

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .client import normalize_http_endpoint

CONFIG_DIRECTORY = "reme"
CONFIG_FILENAME = "config.json"
LEGACY_CONFIG_FILENAME = "reme.json"
VALID_MODES = {"http", "embedded"}


class ReMeConfigError(ValueError):
    """Raised when provider configuration is invalid."""


@dataclass(frozen=True)
class ReMeConfig:
    """Validated settings shared by the provider and both backends."""

    mode: str = "http"
    endpoint: str = "http://127.0.0.1:2333"
    workspace_dir: str = ""
    reme_config: str = "default"
    request_timeout: float = 600.0
    recall_timeout: float = 5.0
    health_timeout: float = 2.0
    health_retry_seconds: float = 30.0
    shutdown_timeout: float = 30.0
    recall_limit: int = 5


def config_path(hermes_home: str | Path) -> Path:
    """Return the path used by Hermes' generic provider configuration UI."""
    return Path(hermes_home).expanduser() / CONFIG_DIRECTORY / CONFIG_FILENAME


def legacy_config_path(hermes_home: str | Path) -> Path:
    """Return the pre-dashboard configuration path retained for compatibility."""
    return Path(hermes_home).expanduser() / LEGACY_CONFIG_FILENAME


def _default_hermes_home() -> Path:
    from hermes_constants import get_hermes_home

    return Path(get_hermes_home())


def _read_json_object(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ReMeConfigError(
            f"Unable to read ReMe provider config {path}: {exc}",
        ) from exc
    if not isinstance(loaded, dict):
        raise ReMeConfigError(
            f"ReMe provider config must contain a JSON object: {path}",
        )
    return loaded


def _read_config_values(home: Path) -> dict[str, Any]:
    """Merge legacy values under the current Dashboard-managed config."""
    values = _read_json_object(legacy_config_path(home))
    values.update(_read_json_object(config_path(home)))
    return values


def _positive_float(value: Any, key: str, default: float) -> float:
    if value in (None, ""):
        return default
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ReMeConfigError(f"'{key}' must be a positive number") from exc
    if not math.isfinite(number) or number <= 0:
        raise ReMeConfigError(f"'{key}' must be a positive number")
    return number


def _positive_int(value: Any, key: str, default: int) -> int:
    if value in (None, ""):
        return default
    if isinstance(value, bool):
        raise ReMeConfigError(f"'{key}' must be a positive integer")
    try:
        number = int(value)
    except (TypeError, ValueError) as exc:
        raise ReMeConfigError(f"'{key}' must be a positive integer") from exc
    if number <= 0 or (isinstance(value, float) and not value.is_integer()):
        raise ReMeConfigError(f"'{key}' must be a positive integer")
    return number


def parse_config(values: dict[str, Any], *, hermes_home: str | Path) -> ReMeConfig:
    """Validate and normalize an already loaded configuration mapping."""
    del hermes_home  # Reserved for future profile-relative settings.
    defaults = ReMeConfig()
    mode = str(values.get("mode", defaults.mode) or defaults.mode).strip().lower()
    if mode not in VALID_MODES:
        raise ReMeConfigError("'mode' must be either 'http' or 'embedded'")

    endpoint = str(values.get("endpoint", defaults.endpoint) or "").strip().rstrip("/")
    if mode == "http":
        try:
            endpoint = normalize_http_endpoint(endpoint)
        except ValueError as exc:
            raise ReMeConfigError(
                "ReMe endpoint must be an absolute http(s) URL",
            ) from exc

    raw_workspace = str(
        values.get("workspace_dir", defaults.workspace_dir) or "",
    ).strip()
    if mode == "embedded" and not raw_workspace:
        raise ReMeConfigError("'workspace_dir' is required in embedded mode")
    workspace_dir = str(Path(raw_workspace).expanduser().absolute()) if raw_workspace else ""

    reme_config = str(values.get("reme_config", defaults.reme_config) or "").strip()
    if mode == "embedded" and not reme_config:
        raise ReMeConfigError("'reme_config' cannot be empty in embedded mode")

    return ReMeConfig(
        mode=mode,
        endpoint=endpoint or defaults.endpoint,
        workspace_dir=workspace_dir,
        reme_config=reme_config or defaults.reme_config,
        request_timeout=_positive_float(
            values.get("request_timeout"),
            "request_timeout",
            defaults.request_timeout,
        ),
        recall_timeout=_positive_float(
            values.get("recall_timeout"),
            "recall_timeout",
            defaults.recall_timeout,
        ),
        health_timeout=_positive_float(
            values.get("health_timeout"),
            "health_timeout",
            defaults.health_timeout,
        ),
        health_retry_seconds=_positive_float(
            values.get("health_retry_seconds"),
            "health_retry_seconds",
            defaults.health_retry_seconds,
        ),
        shutdown_timeout=_positive_float(
            values.get("shutdown_timeout"),
            "shutdown_timeout",
            defaults.shutdown_timeout,
        ),
        recall_limit=_positive_int(
            values.get("recall_limit"),
            "recall_limit",
            defaults.recall_limit,
        ),
    )


def load_config(hermes_home: str | Path | None = None) -> ReMeConfig:
    """Load current config, inheriting omitted fields from the legacy file."""
    home = Path(hermes_home).expanduser() if hermes_home is not None else _default_hermes_home()
    return parse_config(_read_config_values(home), hermes_home=home)


def save_config(values: dict[str, Any], hermes_home: str | Path) -> ReMeConfig:
    """Validate and atomically write config in the current Hermes layout."""
    home = Path(hermes_home).expanduser()
    current = config_path(home)
    merged = _read_config_values(home)
    merged.update(
        {key: value for key, value in dict(values or {}).items() if value is not None},
    )
    validated = parse_config(merged, hermes_home=home)

    current.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{current.name}.", dir=current.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(
                asdict(validated),
                handle,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(tmp_name, 0o600)
        os.replace(tmp_name, current)
    finally:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)
    return validated
