"""Installed plugin discovery and application-local registration."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

from .components.base_component import ComponentMixin
from .components.component_registry import ComponentRegistry
from .config import deep_merge_config, expand_env_vars
from .entry_point import find_entry_points, load_entry_point, unique_entry_point

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


class PluginManager:
    """Resolve enabled plugins and apply their contributions to one application."""

    def __init__(self, plugins: Iterable[Plugin] = ()) -> None:
        self.plugins = tuple(plugins)

    @classmethod
    def discover(cls, specs: Iterable[str]) -> "PluginManager":
        """Load explicitly enabled plugins by entry-point name."""
        plugins: list[Plugin] = []
        seen: set[str] = set()
        for name in specs:
            if not isinstance(name, str):
                raise TypeError(f"Invalid plugin name: {name!r}")
            if not name:
                raise ValueError("Plugin name cannot be empty")
            if name in seen:
                raise ValueError(f"Plugin '{name}' is enabled more than once")
            entries = find_entry_points(PLUGIN_ENTRY_POINT_GROUP, name)
            entry = unique_entry_point(entries, name, provider="Plugin")
            if entry is None:
                raise ValueError(f"Plugin '{name}' is not installed")
            plugin = load_entry_point(entry, invoke=True)
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
            merged = deep_merge_config(merged, expand_env_vars(plugin.config))
        return deep_merge_config(merged, application_config)

    def register(self, registry: ComponentRegistry) -> None:
        """Register every backend into an application-local registry."""
        for plugin in self.plugins:
            for backend in plugin.backends:
                registry.add(backend.name, backend.implementation, owner=plugin.name)
