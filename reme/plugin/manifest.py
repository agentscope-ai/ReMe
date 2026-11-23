"""Parse the small, declarative contract shared by plugin runtime and CLI."""

from dataclasses import dataclass
from importlib import resources
from typing import Any

import yaml

PLUGIN_MANIFEST = "plugin.yaml"


@dataclass(frozen=True)
class PluginManifest:
    """The two contributions an installed plugin may declare."""

    backends: dict[str, str]
    application_defaults: dict[str, Any]


def parse_plugin_manifest(text: str, *, plugin_name: str) -> PluginManifest:
    """Parse and validate plugin.yaml without importing backend modules."""
    try:
        value = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise ValueError(f"Plugin '{plugin_name}' manifest is invalid YAML") from exc
    if not isinstance(value, dict):
        raise ValueError(f"Plugin '{plugin_name}' manifest root must be a mapping")

    unknown = set(value) - {"backends", "application_defaults"}
    if unknown:
        raise ValueError(f"Plugin '{plugin_name}' manifest has unknown keys: {', '.join(sorted(unknown))}")

    backends = value.get("backends")
    application_defaults = value.get("application_defaults")
    backends = {} if backends is None else backends
    application_defaults = {} if application_defaults is None else application_defaults
    if not isinstance(backends, dict):
        raise TypeError(f"Plugin '{plugin_name}' manifest 'backends' must be a mapping")
    if not isinstance(application_defaults, dict):
        raise TypeError(f"Plugin '{plugin_name}' manifest 'application_defaults' must be a mapping")
    for name, target in backends.items():
        if not isinstance(name, str) or not name:
            raise TypeError(f"Plugin '{plugin_name}' backend names must be non-empty strings")
        if not isinstance(target, str) or not target:
            raise TypeError(f"Plugin '{plugin_name}' backend target for '{name}' must be a non-empty string")
    return PluginManifest(backends=dict(backends), application_defaults=dict(application_defaults))


def load_package_manifest(package: str, *, plugin_name: str) -> PluginManifest:
    """Read plugin.yaml from an importable package."""
    try:
        path = resources.files(package).joinpath(PLUGIN_MANIFEST)
        text = path.read_text(encoding="utf-8")
    except (ModuleNotFoundError, FileNotFoundError, TypeError) as exc:
        raise ValueError(f"Plugin '{plugin_name}' does not provide {package}/{PLUGIN_MANIFEST}") from exc
    return parse_plugin_manifest(text, plugin_name=plugin_name)
