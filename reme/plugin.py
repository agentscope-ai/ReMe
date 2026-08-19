"""Installed plugin discovery and application-local registration."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from importlib import metadata
from typing import Any

from .components.base_component import ComponentMixin
from .components.component_registry import ComponentRegistry, R
from .config import deep_merge_config, expand_env_vars

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
    """Return matching entry points for one group and name."""
    return list(metadata.entry_points().select(group=group, name=name))


class PluginManager:
    """Resolve enabled plugins and apply their contributions to one application."""

    def __init__(self, plugins: Iterable[Plugin] = ()) -> None:
        self.plugins = tuple(plugins)

    @classmethod
    def discover(cls, specs: Iterable[str]) -> "PluginManager":
        """Load explicitly enabled plugins by entry-point name."""
        # Loading an entry point imports third-party code. Roll back any legacy
        # ``@R.register`` side effects so only declared backends reach an app.
        with R.preserve():
            return cls._discover(specs)

    @classmethod
    def _discover(cls, specs: Iterable[str]) -> "PluginManager":
        """Load plugin descriptors while the global registry is protected."""
        plugins: list[Plugin] = []
        seen: set[str] = set()
        for name in specs:
            if not isinstance(name, str):
                raise TypeError(f"Invalid plugin name: {name!r}")
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
            merged = deep_merge_config(merged, expand_env_vars(plugin.config))
        return deep_merge_config(merged, application_config)

    def register(self, registry: ComponentRegistry) -> None:
        """Register every backend into an application-local registry."""
        for plugin in self.plugins:
            for backend in plugin.backends:
                registry.add(backend.name, backend.implementation, owner=plugin.name)
