"""Load explicitly enabled plugins into one application's config and registry.

New plugins expose a package-only ``reme.plugins`` entry point and declare two
optional mappings in ``plugin.yaml``: ``backends`` and ``application_defaults``.
Python ``Plugin`` descriptors remain supported as a compatibility boundary.
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
from .entry_point import PLUGIN_ENTRY_POINT_GROUP, find_entry_points, load_entry_point, unique_entry_point
from .plugin_manifest import PluginManifest, load_package_manifest


@dataclass(frozen=True)
class Backend:
    """One named component, step, or job backend contributed by a plugin."""

    name: str
    implementation: type[ComponentMixin]


@dataclass(frozen=True)
class Plugin:
    """Legacy Python descriptor accepted during the plugin manifest migration."""

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
    """Load a package manifest, falling back to the legacy descriptor form."""
    if ":" not in entry.value:
        # Package-only targets use plugin.yaml; package:object targets are the
        # legacy Python descriptor contract.
        return _load_package_plugin(name, entry.value)
    plugin = load_entry_point(entry, invoke=True)
    if not isinstance(plugin, Plugin):
        raise TypeError(f"Plugin entry point '{name}' did not return reme.plugin.Plugin")
    return plugin


def _load_package_plugin(name: str, package: str) -> Plugin:
    """Load an explicitly selected importable package without entry-point discovery."""
    from .components.component_registry import R

    if not isinstance(package, str) or not package or ":" in package:
        raise ValueError(f"Plugin '{name}' requires an importable package name: {package!r}")
    with R.preserve(allow_mutation=True):
        manifest = load_package_manifest(package, plugin_name=name)
        return _plugin_from_manifest(name, manifest)


class PluginManager:
    """Resolve enabled plugins and apply their contributions to one application."""

    def __init__(self, plugins: Iterable[Plugin] = ()) -> None:
        self.plugins = tuple(plugins)

    @classmethod
    def discover(cls, specs: Iterable[str], *, packages: Mapping[str, str] | None = None) -> "PluginManager":
        """Load enabled plugins, preferring explicitly supplied Python packages."""
        plugins: list[Plugin] = []
        seen: set[str] = set()
        for name in specs:
            if not isinstance(name, str):
                raise TypeError(f"Invalid plugin name: {name!r}")
            if not name:
                raise ValueError("Plugin name cannot be empty")
            if name in seen:
                raise ValueError(f"Plugin '{name}' is enabled more than once")
            if packages is not None and name in packages:
                plugin = _load_package_plugin(name, packages[name])
            else:
                entries = find_entry_points(PLUGIN_ENTRY_POINT_GROUP, name)
                entry = unique_entry_point(entries, name, provider="Plugin")
                if entry is None:
                    raise ValueError(f"Plugin '{name}' is not installed")
                plugin = _load_plugin(name, entry)
            if plugin.name != name:
                raise ValueError(f"Plugin entry point '{name}' returned plugin '{plugin.name}'")
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


def resolve_plugin_runtime(
    application_config: Mapping[str, Any],
    *,
    plugin_packages: Mapping[str, str] | None = None,
) -> PluginRuntime:
    """Build one local registry, with user config overriding plugin application defaults."""
    manager = PluginManager.discover(application_config.get("plugins") or (), packages=plugin_packages)
    registry = create_application_registry()
    manager.register(registry)
    return PluginRuntime(config=manager.merge_config(application_config), registry=registry)
