"""Parser for YAML config with CLI argument overrides."""

import json
import os
import re
from importlib import metadata
from pathlib import Path
from typing import Any

import yaml

# Config files are looked up relative to this module's directory
_CONFIG_DIR = Path(__file__).parent
# Extensions in priority order: yaml > yml > json when stems collide
_SUPPORTED_EXTS = (".yaml", ".yml", ".json")
_CONFIG_ENTRY_POINT_GROUP = "reme.configs"
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
        expanded = _ENV_VAR_RE.sub(_repl, value)
        return _convert_value(expanded) if expanded != value else value
    if isinstance(value, dict):
        return {k: _expand_env_vars(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand_env_vars(v) for v in value]
    return value


def expand_env_vars(value: Any) -> Any:
    """Expand environment placeholders in an arbitrary plugin config value."""
    return _expand_env_vars(value)


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
        if not key_path or any(not key for key in keys):
            raise ValueError(f"Invalid dot notation key: {key_path!r}")
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


def _external_config_path(name: str) -> Path | None:
    """Resolve an installed plugin config exposed through ``reme.configs``."""
    discovered = metadata.entry_points()
    if hasattr(discovered, "select"):
        entries = list(discovered.select(group=_CONFIG_ENTRY_POINT_GROUP, name=name))
    else:
        entries = [entry for entry in discovered.get(_CONFIG_ENTRY_POINT_GROUP, ()) if entry.name == name]
    if not entries:
        return None
    if len(entries) > 1:
        providers = ", ".join(sorted(entry.value for entry in entries))
        raise ValueError(f"Config '{name}' has multiple installed providers: {providers}")
    loaded = entries[0].load()
    value = loaded() if callable(loaded) else loaded
    path = Path(value)
    if path.suffix not in _SUPPORTED_EXTS or not path.is_file():
        raise ValueError(f"Config entry point '{name}' did not resolve to a YAML or JSON file")
    return path


def _load_config(name_or_path: str, encoding: str = "utf-8", _stack: tuple[str, ...] = ()) -> dict:
    """Load a built-in, installed-plugin, or direct YAML/JSON config."""
    if name_or_path in _stack:
        chain = " -> ".join((*_stack, name_or_path))
        raise ValueError(f"Circular config inheritance: {chain}")

    if name_or_path in _CONFIG_REGISTRY:
        return _load_config_path(_CONFIG_REGISTRY[name_or_path], name_or_path, encoding, _stack)

    if external := _external_config_path(name_or_path):
        return _load_config_path(external, name_or_path, encoding, _stack)

    p = Path(name_or_path)
    if p.suffix in _SUPPORTED_EXTS:
        candidates = [p]
        if not p.is_absolute():
            candidates.append(_CONFIG_DIR / p)
        for candidate in candidates:
            if candidate.exists():
                identity = str(candidate.resolve())
                return _load_config_path(candidate, identity, encoding, _stack)
        raise FileNotFoundError(f"Config file not found: {p}")

    known = ", ".join(sorted(_CONFIG_REGISTRY)) if _CONFIG_REGISTRY else "none"
    raise FileNotFoundError(f"Config file not found: {name_or_path}. Available: {known}")


def _load_config_path(path: Path, identity: str, encoding: str, stack: tuple[str, ...]) -> dict:
    """Load one config and merge its optional parents before its own values."""
    config = _read_config_file(path, encoding)
    raw_parents = config.pop("extends", ())
    parents = [raw_parents] if isinstance(raw_parents, str) else list(raw_parents or ())
    merged: dict = {}
    for parent in parents:
        if not isinstance(parent, str) or not parent:
            raise ValueError(f"Config 'extends' entries must be non-empty strings: {path}")
        parent_name = parent
        relative = path.parent / parent
        if Path(parent).suffix in _SUPPORTED_EXTS and relative.is_file():
            parent_name = str(relative.resolve())
        merged = _deep_merge(merged, _load_config(parent_name, encoding, (*stack, identity)))
    return _deep_merge(merged, config)


def _read_config_file(path: Path, encoding: str = "utf-8") -> dict:
    """Read YAML or JSON file based on extension. Expands ${ENV_VAR}."""
    with path.open(encoding=encoding) as f:
        if path.suffix == ".json":
            result = json.load(f)
        else:
            result = yaml.safe_load(f)
    if result is None:
        return {}
    if not isinstance(result, dict):
        raise ValueError(f"Config root must be a mapping/object: {path}")
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
        else:
            raise ValueError(f"Invalid argument format (expected key=value): {raw}")

    parsed = parse_dot_notation(kvs) if kvs else {}
    return first, parsed


def resolve_app_config(*, log_config: bool = True, **kwargs) -> dict:
    """Resolve full app-start config: load `config=path` file, fall back to
    `default`, then deep-merge with the remaining kwargs as overrides.

    Set ``log_config=False`` for user-facing client calls that should print only
    the requested job's output.
    """
    from ..utils import get_logger

    logger = get_logger(log_to_file=False)
    configs: list[dict] = []

    # `config=path` arrives as a string here; `config.foo=bar` arrives as a
    # nested dict and is left in `kwargs` to be merged as a normal override.
    config_value = kwargs.get("config")
    if isinstance(config_value, str):
        kwargs.pop("config")
        if log_config:
            logger.info(f"Loading config: {config_value}")
        configs.append(_load_config(config_value))
    elif "default" in _CONFIG_REGISTRY:
        if log_config:
            logger.info("No config specified, loading 'default'")
        configs.append(_load_config("default"))

    configs.append(kwargs)

    merged: dict = {}
    for cfg in configs:
        merged = _deep_merge(merged, cfg)

    return merged
