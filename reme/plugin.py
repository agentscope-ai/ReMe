"""Installed plugin discovery and application-local registration."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from importlib import metadata
from typing import Any

from .components.base_component import ComponentMixin
from .components.component_registry import ComponentRegistry
from .config import expand_env_vars
from .schema.application_config import PluginConfig

PLUGIN_ENTRY_POINT_GROUP = "reme.plugins"


@dataclass(frozen=True)
class Backend:
    """One named component, step, or job backend contributed by a plugin."""

    name: str
    implementation: type[ComponentMixin]


@dataclass(frozen=True)
class Plugin:
    """Declarative plugin loaded from the ``reme.plugins`` entry-point group."""

    name: str
    backends: tuple[Backend, ...] = ()
    config: Mapping[str, Any] = field(default_factory=dict)


def _entry_points(group: str, name: str) -> list[metadata.EntryPoint]:
    """Return matching entry points across supported importlib.metadata APIs."""
    discovered = metadata.entry_points()
    if hasattr(discovered, "select"):
        return list(discovered.select(group=group, name=name))
    return [entry for entry in discovered.get(group, ()) if entry.name == name]


def _merge(base: Mapping[str, Any], update: Mapping[str, Any]) -> dict[str, Any]:
    """Recursively merge configuration mappings without mutating either input."""
    result = dict(base)
    for key, value in update.items():
        if key in result and isinstance(result[key], Mapping) and isinstance(value, Mapping):
            result[key] = _merge(result[key], value)
        else:
            result[key] = value
    return result


class PluginManager:
    """Resolve enabled plugins and apply their contributions to one application."""

    def __init__(self, plugins: Iterable[Plugin] = ()) -> None:
        self.plugins = tuple(plugins)

    @classmethod
    def discover(cls, specs: Iterable[str | Mapping[str, Any] | PluginConfig]) -> "PluginManager":
        """Load explicitly enabled plugins by entry-point name."""
        plugins: list[Plugin] = []
        seen: set[str] = set()
        for raw in specs:
            if isinstance(raw, PluginConfig):
                raw = raw.model_dump()
            if isinstance(raw, str):
                name, enabled = raw, True
            elif isinstance(raw, Mapping):
                name = str(raw.get("name", ""))
                enabled = bool(raw.get("enabled", True))
            else:
                raise TypeError(f"Invalid plugin specification: {raw!r}")
            if not enabled:
                continue
            if not name:
                raise ValueError("Plugin name cannot be empty")
            if name in seen:
                raise ValueError(f"Plugin '{name}' is enabled more than once")
            entries = _entry_points(PLUGIN_ENTRY_POINT_GROUP, name)
            if not entries:
                raise ValueError(f"Plugin '{name}' is not installed")
            if len(entries) > 1:
                providers = ", ".join(sorted(entry.value for entry in entries))
                raise ValueError(f"Plugin '{name}' has multiple installed providers: {providers}")
            loaded = entries[0].load()
            plugin = loaded() if callable(loaded) and not isinstance(loaded, Plugin) else loaded
            if not isinstance(plugin, Plugin):
                raise TypeError(f"Plugin entry point '{name}' did not return reme.plugin.Plugin")
            if plugin.name != name:
                raise ValueError(f"Plugin entry point '{name}' returned plugin '{plugin.name}'")
            plugins.append(plugin)
            seen.add(name)
        return cls(plugins)

    def merge_config(self, application_config: Mapping[str, Any]) -> dict[str, Any]:
        """Place plugin defaults below the user's resolved application config."""
        merged: dict[str, Any] = {}
        for plugin in self.plugins:
            merged = _merge(merged, expand_env_vars(plugin.config))
        return _merge(merged, application_config)

    def register(self, registry: ComponentRegistry) -> None:
        """Register every backend into an application-local registry."""
        for plugin in self.plugins:
            for backend in plugin.backends:
                registry.add(backend.name, backend.implementation, owner=plugin.name)
