"""Parser for YAML config with CLI argument overrides."""

import json
import os
import re
from collections.abc import Mapping
from importlib.metadata import EntryPoint
from pathlib import Path
from typing import Any

import yaml
from yaml.representer import SafeRepresenter

from ..entry_point import (
    CONFIG_ENTRY_POINT_GROUP,
    find_entry_points,
    load_entry_point,
    unique_entry_point,
)

# Config files are looked up relative to this module's directory
_CONFIG_DIR = Path(__file__).parent
# Extensions in priority order: yaml > yml > json when stems collide
_SUPPORTED_EXTS = (".yaml", ".yml", ".json")
_ENV_VAR_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?}")
# Strings like "007" / "00501" must stay as strings, not be coerced to numbers
_LEADING_ZERO_RE = re.compile(r"^-?0\d")


class ResolvedConfigSection(dict[str, Any]):
    """A resolved mapping that retains which values were explicitly overridden.

    ``resolve_app_config`` still returns an ordinary dict-compatible value, but
    named resources need field-level provenance so plugins can replace a
    complete definition before CLI dot-notation overrides are applied. Only
    paths are retained: values are always read from the visible mapping, so
    normal dict mutations cannot diverge from hidden configuration snapshots.
    """

    def __init__(
        self,
        value: Mapping[str, Any],
        *,
        explicit_paths: tuple[tuple[str, ...], ...],
    ) -> None:
        super().__init__(value)
        self.explicit_paths = explicit_paths


# PyYAML dispatches representers by exact type, so teach its safe dumper that
# this dict-compatible provenance carrier serializes exactly like a plain dict.
yaml.SafeDumper.add_representer(ResolvedConfigSection, SafeRepresenter.represent_dict)


def _mapping_leaf_paths(
    value: Mapping[str, Any],
    prefix: tuple[str, ...] = (),
) -> tuple[tuple[str, ...], ...]:
    """Return the leaf paths explicitly supplied by one config override."""
    paths: list[tuple[str, ...]] = []
    for key, child in value.items():
        path = (*prefix, key)
        if isinstance(child, Mapping) and child:
            paths.extend(_mapping_leaf_paths(child, path))
        else:
            paths.append(path)
    return tuple(paths)


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


def _external_config_path(name: str, entry: EntryPoint | None) -> Path | None:
    """Resolve an installed plugin config exposed through ``reme.configs``."""
    if entry is None:
        return None
    value = load_entry_point(entry, invoke=True)
    path = Path(value)
    if path.suffix not in _SUPPORTED_EXTS or not path.is_file():
        raise ValueError(f"Config entry point '{name}' did not resolve to a YAML or JSON file")
    return path


def _load_config(name_or_path: str, encoding: str = "utf-8", _stack: tuple[str, ...] = ()) -> dict:
    """Load a built-in, installed-plugin, or direct YAML/JSON config."""
    if name_or_path in _stack:
        chain = " -> ".join((*_stack, name_or_path))
        raise ValueError(f"Circular config inheritance: {chain}")

    built_in = _CONFIG_REGISTRY.get(name_or_path)
    external_entries = find_entry_points(CONFIG_ENTRY_POINT_GROUP, name_or_path)
    if built_in is not None and external_entries:
        raise ValueError(f"Config '{name_or_path}' is provided by both ReMe and an installed distribution")
    if built_in is not None:
        return _load_config_path(built_in, name_or_path, encoding, _stack)

    external_entry = unique_entry_point(external_entries, name_or_path, provider="Config")
    external = _external_config_path(name_or_path, external_entry)
    if external is not None:
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
        merged = deep_merge_config(merged, _load_config(parent_name, encoding, (*stack, identity)))
    return deep_merge_config(merged, config)


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


def deep_merge_config(base: Mapping[str, Any], update: Mapping[str, Any]) -> dict[str, Any]:
    """Recursively merge configuration mappings without mutating either input."""
    result = dict(base)
    for k, v in update.items():
        if k in result and isinstance(result[k], Mapping) and isinstance(v, Mapping):
            result[k] = deep_merge_config(result[k], v)
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


def parse_action(arg: str) -> str:
    """Parse and validate one top-level CLI action."""
    action = _strip_arg_dashes(arg)
    if "=" in action:
        raise ValueError(f"First argument must be action, got: {arg}")
    return action


def parse_kwargs(*args: str) -> dict:
    """Parse application-style ``key=value`` CLI arguments."""
    kvs: list[str] = []
    for raw in args:
        arg = _strip_arg_dashes(raw)
        if "=" in arg:
            kvs.append(arg)
        else:
            raise ValueError(f"Invalid argument format (expected key=value): {raw}")

    return parse_dot_notation(kvs) if kvs else {}


def parse_args(*args: str) -> tuple[str, dict]:
    """Parse an application CLI action followed by ``key=value`` arguments.

    Usage: reme app config=paw.yaml service.name=test
    Returns: (action, parsed_kv_dict)
    """
    if not args:
        raise ValueError("No arguments provided")

    return parse_action(args[0]), parse_kwargs(*args[1:])


def resolve_app_config(*, log_config: bool = True, **kwargs) -> dict:
    """Resolve full app-start config: load `config=path` file, fall back to
    `default`, then deep-merge with the remaining kwargs as overrides.

    Therefore ``reme start plugins=[...]`` layers that plugin selection over
    ``default.yaml`` without requiring an explicit ``config=default``.

    Set ``log_config=False`` for user-facing client calls that should print only
    the requested job's output.
    """
    from ..utils import get_logger

    logger = get_logger(log_to_file=False)
    base_config: dict = {}

    # `config=path` arrives as a string here; `config.foo=bar` arrives as a
    # nested dict and is left in `kwargs` to be merged as a normal override.
    config_value = kwargs.get("config")
    if isinstance(config_value, str):
        kwargs.pop("config")
        if log_config:
            logger.info(f"Loading config: {config_value}")
        base_config = _load_config(config_value)
    elif "default" in _CONFIG_REGISTRY:
        if log_config:
            logger.info("No config specified, loading 'default'")
        base_config = _load_config("default")

    explicit_overrides = kwargs
    merged = deep_merge_config(base_config, explicit_overrides)

    # Jobs and named component instances are complete resource definitions
    # when supplied by config files or plugins. CLI dot-notation values remain
    # field-level overrides, so retain their paths until plugin resolution.
    for section_name in ("jobs", "components"):
        section = merged.get(section_name)
        section_overrides = explicit_overrides.get(section_name)
        if not isinstance(section, Mapping) or (
            section_name in explicit_overrides and not isinstance(section_overrides, Mapping)
        ):
            # Preserve invalid input verbatim so the schema or plugin merger can
            # report it instead of obscuring the source through normalization.
            continue
        merged[section_name] = ResolvedConfigSection(
            section,
            explicit_paths=_mapping_leaf_paths(section_overrides) if isinstance(section_overrides, Mapping) else (),
        )

    return merged
