"""Load explicitly enabled plugins into one application's config and registry.

Plugins expose a package-only ``reme.plugins`` entry point and declare two
optional mappings in ``plugin.yaml``: ``backends`` and ``application_defaults``.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from importlib import import_module
from importlib.metadata import EntryPoint
from typing import Any

from .components.base_component import ComponentMixin
from .components.component_registry import ComponentRegistry, create_application_registry
from .config import deep_merge_config, expand_env_vars
from .entry_point import PLUGIN_ENTRY_POINT_GROUP, find_entry_points, unique_entry_point
from .plugin_manifest import PluginManifest, load_package_manifest


@dataclass(frozen=True)
class Backend:
    """One named component, step, or job backend contributed by a plugin."""

    name: str
    implementation: type[ComponentMixin]


@dataclass(frozen=True)
class Plugin:
    """Resolved backend classes and defaults from one package manifest."""

    name: str
    backends: tuple[Backend, ...] = ()
    config: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PluginRuntime:
    """Application config and registry after applying enabled plugins."""

    config: dict[str, Any]
    registry: ComponentRegistry


def _load_backend(target: str, *, plugin_name: str) -> type[ComponentMixin]:
    """Import one ``module:class`` backend target from a plugin manifest."""
    module_name, separator, attribute = target.partition(":")
    if not separator or not module_name or not attribute or ":" in attribute:
        raise ValueError(f"Plugin '{plugin_name}' has invalid backend target: {target!r}")
    try:
        value: Any = import_module(module_name)
        for part in attribute.split("."):
            value = getattr(value, part)
    except (AttributeError, ImportError) as exc:
        raise ValueError(f"Plugin '{plugin_name}' cannot load backend '{target}': {exc}") from exc
    if not isinstance(value, type) or not issubclass(value, ComponentMixin):
        raise TypeError(f"Plugin '{plugin_name}' backend '{target}' is not a ComponentMixin class")
    return value


def _plugin_from_manifest(name: str, manifest: PluginManifest) -> Plugin:
    """Convert a parsed manifest into one runtime plugin descriptor."""
    backends = tuple(
        Backend(backend_name, _load_backend(target, plugin_name=name))
        for backend_name, target in manifest.backends.items()
    )
    return Plugin(name=name, backends=backends, config=manifest.application_defaults)


def _load_plugin(name: str, entry: EntryPoint) -> Plugin:
    """Load declared contributions without retaining import-time registrations."""
    from .components.component_registry import R

    if ":" in entry.value:
        raise ValueError(f"Plugin '{name}' requires a package-only entry point and plugin.yaml")
    with R.preserve(allow_mutation=True):
        manifest = load_package_manifest(entry.value, plugin_name=name)
        return _plugin_from_manifest(name, manifest)


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
            plugin = _load_plugin(name, entry)
            plugins.append(plugin)
            seen.add(name)
        return cls(plugins)

    def merge_config(self, application_config: Mapping[str, Any]) -> dict[str, Any]:
        """Place plugin application defaults below the user's resolved config."""
        merged: dict[str, Any] = {}
        for plugin in self.plugins:
            merged = deep_merge_config(merged, expand_env_vars(plugin.config))
        return deep_merge_config(merged, application_config)

    def register(self, registry: ComponentRegistry) -> None:
        """Register every backend into an application-local registry."""
        for plugin in self.plugins:
            for backend in plugin.backends:
                registry.add(backend.name, backend.implementation, owner=plugin.name)


def resolve_plugin_runtime(application_config: Mapping[str, Any]) -> PluginRuntime:
    """Build one local registry, with user config overriding plugin application defaults."""
    manager = PluginManager.discover(application_config.get("plugins") or ())
    registry = create_application_registry()
    manager.register(registry)
    return PluginRuntime(config=manager.merge_config(application_config), registry=registry)
