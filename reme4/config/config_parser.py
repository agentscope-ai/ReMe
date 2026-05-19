"""Parser for YAML config with CLI argument overrides."""

import json
import os
import re
from pathlib import Path
from typing import Any

import yaml

# Config files are looked up relative to this module's directory
_CONFIG_DIR = Path(__file__).parent
# Extensions in priority order: yaml > yml > json when stems collide
_SUPPORTED_EXTS = (".yaml", ".yml", ".json")
_ENV_VAR_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?}")
# Strings like "007" / "00501" must stay as strings, not be coerced to numbers
_LEADING_ZERO_RE = re.compile(r"^-?0\d")


def _repl(m: re.Match) -> str:
    name: str = m.group(1)
    # group(2) is None when the placeholder has no `:-default` part
    default: str | None = m.group(2)
    v = os.environ.get(name)
    if v is None:
        if default is not None:
            return default
        raise ValueError(f"Config references undefined env var: {name}")
    return v


def _expand_env_vars(value: Any) -> Any:
    """Recursively expand `${VAR}` / `${VAR:-default}` placeholders in strings."""
    if isinstance(value, str):
        return _ENV_VAR_RE.sub(_repl, value)
    if isinstance(value, dict):
        return {k: _expand_env_vars(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand_env_vars(v) for v in value]
    return value


def _discover_configs() -> dict[str, Path]:
    """Pre-scan config directory: maps file stem (name without ext) -> Path."""
    discovered: dict[str, Path] = {}
    if _CONFIG_DIR.is_dir():
        # Sort by ext priority so registration order is deterministic across filesystems
        files = sorted(
            (p for p in _CONFIG_DIR.iterdir() if p.is_file() and p.suffix in _SUPPORTED_EXTS),
            key=lambda p: (_SUPPORTED_EXTS.index(p.suffix), p.name),
        )
        for p in files:
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
        # Symmetric to the prefix check above: refuse scalar-over-dict overwrite
        last_key = keys[-1]
        if last_key in current and isinstance(current[last_key], dict):
            raise ValueError(f"Cannot overwrite nested dict at '{key_path}' with scalar value")
        current[last_key] = _convert_value(value_str)
    return result


def _convert_value(value_str: str) -> Any:
    """Convert string to appropriate Python type.

    Only converts "true"/"false" (case-insensitive) to boolean.
    Use JSON format (e.g., '"yes"', '"no"') to preserve these as strings.
    Leading-zero strings (e.g., "007", "00501") are kept as strings.
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

    # Skip int/float for leading-zero strings to keep zip codes / ids intact
    if not _LEADING_ZERO_RE.match(s):
        for converter in (int, float):
            try:
                return converter(s)
            except ValueError:
                continue

    # JSON handles lists, dicts, and explicitly-quoted strings
    try:
        return json.loads(s)
    except (ValueError, json.JSONDecodeError):
        pass

    # Fallback to original string
    return s


def _load_config(name_or_path: str, encoding: str = "utf-8") -> dict:
    """Load a YAML or JSON config file.

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
    """Read YAML or JSON file based on extension. Expands ${ENV_VAR}."""
    with path.open(encoding=encoding) as f:
        if path.suffix == ".json":
            result = json.load(f)
        else:
            result = yaml.safe_load(f)
    if result is None:
        return {}
    return _expand_env_vars(result)


def _deep_merge(base: dict, update: dict) -> dict:
    """Recursively merge dicts."""
    result = base.copy()
    for k, v in update.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = v
    return result


def _strip_arg_dashes(arg: str) -> str:
    """Strip a single leading `--` or `-` prefix (not all leading dashes)."""
    if arg.startswith("--"):
        return arg[2:]
    if arg.startswith("-"):
        return arg[1:]
    return arg


def parse_args(*args) -> tuple[str, dict]:
    """Parse CLI args: first arg is action, rest are key=value pairs.

    Usage: reme app config=paw.yaml service.name=test
    Returns: (action, parsed_kv_dict)
    """
    if not args:
        raise ValueError("No arguments provided")

    first = _strip_arg_dashes(args[0])
    if "=" in first:
        raise ValueError(f"First argument must be action, got: {args[0]}")

    kvs: list[str] = []
    for raw in args[1:]:
        arg = _strip_arg_dashes(raw)
        if "=" in arg:
            kvs.append(arg)

    parsed = parse_dot_notation(kvs) if kvs else {}
    return first, parsed


def resolve_app_config(**kwargs) -> dict:
    """Resolve full app-start config: load `config=path` file, fall back to
    `default`, then deep-merge with the remaining kwargs as overrides.
    """
    from ..utils import get_logger

    logger = get_logger()
    configs: list[dict] = []

    # `config=path` arrives as a string here; `config.foo=bar` arrives as a
    # nested dict and is left in `kwargs` to be merged as a normal override.
    config_value = kwargs.get("config")
    if isinstance(config_value, str):
        kwargs.pop("config")
        logger.info(f"Loading config: {config_value}")
        configs.append(_load_config(config_value))
    elif "default" in _CONFIG_REGISTRY:
        logger.info("No config specified, loading 'default'")
        configs.append(_load_config("default"))

    configs.append(kwargs)

    merged: dict = {}
    for cfg in configs:
        merged = _deep_merge(merged, cfg)

    return merged
