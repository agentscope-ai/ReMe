"""Parser for YAML config with CLI argument overrides."""

import json
from pathlib import Path
from typing import Any

import yaml

# Config files are looked up relative to this module's directory
_CONFIG_DIR = Path(__file__).parent
_SUPPORTED_EXTS = (".yaml", ".yml", ".json")


# Pre-scan config directory: maps basename(without ext) -> Path
def _discover_configs() -> dict[str, Path]:
    discovered: dict[str, Path] = {}
    if _CONFIG_DIR.is_dir():
        for p in _CONFIG_DIR.iterdir():
            if p.is_file() and p.suffix in _SUPPORTED_EXTS:
                discovered.setdefault(p.stem, p)
    return discovered


_CONFIG_REGISTRY = _discover_configs()


def parse_dot_notation(dot_list: list[str]) -> dict:
    """Parse "key.subkey=value" strings into nested dict."""
    result: dict = {}
    for item in dot_list:
        if "=" not in item:
            raise ValueError(f"Invalid dot notation format (missing '='): {item}")
        key_path, value_str = item.split("=", 1)
        keys = key_path.split(".")
        current = result
        for key in keys[:-1]:
            if key in current and not isinstance(current[key], dict):
                raise ValueError(f"Cannot set nested key '{key_path}': '{key}' is already a value")
            current = current.setdefault(key, {})
        current[keys[-1]] = _convert_value(value_str)
    return result


def _convert_value(value_str: str) -> Any:
    """Convert string to appropriate Python type.

    Only converts "true"/"false" (case-insensitive) to boolean.
    Use JSON format (e.g., '"yes"', '"no"') to preserve these as strings.
    """
    s = value_str.strip()
    lower = s.lower()

    # Handle special values (null, bool)
    if lower in ("none", "null"):
        return None
    if lower == "true":
        return True
    if lower == "false":
        return False

    # Try numeric and JSON conversions
    for converter in (int, float, json.loads):
        try:
            return converter(s)
        except (ValueError, json.JSONDecodeError):
            continue

    # Fallback to string
    return s


def _load_yaml(name_or_path: str, encoding: str = "utf-8") -> dict:
    """Load YAML or JSON file.

    First check if name_or_path matches a pre-discovered config (key in _CONFIG_REGISTRY).
    If not, treat as a file path and load directly.
    """
    # 1. Try pre-discovered configs first
    if name_or_path in _CONFIG_REGISTRY:
        return _read_config_file(_CONFIG_REGISTRY[name_or_path], encoding)

    # 2. Treat as file path
    p = Path(name_or_path)
    if p.suffix in _SUPPORTED_EXTS:
        if not p.exists():
            raise FileNotFoundError(f"Config file not found: {p}")
        return _read_config_file(p, encoding)

    known = ", ".join(sorted(_CONFIG_REGISTRY)) if _CONFIG_REGISTRY else "none"
    raise FileNotFoundError(f"Config file not found: {name_or_path}. Available: {known}")


def _read_config_file(path: Path, encoding: str = "utf-8") -> dict:
    """Read YAML or JSON file based on extension."""
    with path.open(encoding=encoding) as f:
        if path.suffix == ".json":
            result = json.load(f)
            return result if result is not None else {}
        else:
            result = yaml.safe_load(f)
            return result if result is not None else {}


def _deep_merge(base: dict, update: dict) -> dict:
    """Recursively merge dicts."""
    result = base.copy()
    for k, v in update.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = v
    return result


def parse_args(*args, **kwargs) -> tuple[str, dict]:
    """Parse CLI args: first arg is action, rest are config overrides.

    Usage: reme app config=paw.yaml service.name=test
    Returns: (action, merged_config_dict)
    """
    if not args:
        raise ValueError("No arguments provided")

    first = args[0].lstrip("-")
    if "=" in first:
        raise ValueError(f"First argument must be action, got: {args[0]}")

    action = first
    configs: list[dict] = []

    for arg in args[1:]:
        arg = arg.lstrip("-")
        if arg.startswith("config="):
            path = arg.split("=", 1)[1].strip()
            if path:
                configs.append(_load_yaml(path))
        elif "=" in arg:
            configs.append(parse_dot_notation([arg]))

    configs.append(kwargs)

    merged: dict = {}
    for cfg in configs:
        merged = _deep_merge(merged, cfg)

    return action, merged
