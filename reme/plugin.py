"""Load explicitly enabled plugins into one application's config and registry.

New plugins expose a package-only ``reme.plugins`` entry point and declare two
optional mappings in ``plugin.yaml``: ``backends`` and ``application_defaults``.
Python ``Plugin`` descriptors remain supported as a compatibility boundary.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from copy import deepcopy
from dataclasses import dataclass, field
from importlib import import_module
from importlib.metadata import EntryPoint
from typing import Any

from .components.base_component import ComponentMixin
from .components.component_registry import ComponentRegistry, create_application_registry
from .config import deep_merge_config, expand_env_vars
from .entry_point import PLUGIN_ENTRY_POINT_GROUP, find_entry_points, load_entry_point, unique_entry_point
from .enumeration import component_type_name
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
    config_warnings: tuple[str, ...] = ()


_MISSING = object()


def _without_named_resources(config: Mapping[str, Any]) -> dict[str, Any]:
    """Return fields that retain the ordinary plugin-default merge behavior."""
    return {key: value for key, value in config.items() if key not in {"jobs", "components"}}


def _overlay_jobs(
    current: Any,
    owners: dict[tuple[Any, ...], str],
    warnings: list[str],
    layer: Mapping[str, Any],
    source: str,
) -> Any:
    """Atomically overlay jobs from one source, warning on cross-source replacement."""
    if "jobs" not in layer:
        return current
    jobs = layer["jobs"]
    if not isinstance(jobs, Mapping):
        owners.clear()
        return deepcopy(jobs)

    merged = dict(current) if isinstance(current, Mapping) else {}
    for name, definition in jobs.items():
        identity = ("jobs", name)
        previous = owners.get(identity)
        if previous is not None:
            warnings.append(
                f"Config collision at jobs.{name}: {source} replaces the complete definition from {previous}",
            )
        merged[name] = deepcopy(definition)
        owners[identity] = source
    return merged


def _overlay_components(
    current: Any,
    owners: dict[tuple[Any, ...], str],
    warnings: list[str],
    layer: Mapping[str, Any],
    source: str,
) -> Any:
    """Atomically overlay named component instances from one source."""
    if "components" not in layer:
        return current
    components = layer["components"]
    if not isinstance(components, Mapping):
        owners.clear()
        return deepcopy(components)

    # Match ApplicationConfig normalization before comparing identities. If
    # one source repeats an equivalent type, its later mapping wins without a
    # new same-source diagnostic, preserving the existing validator behavior.
    normalized = {component_type_name(component_type): group for component_type, group in components.items()}
    merged = dict(current) if isinstance(current, Mapping) else {}
    for component_type, group in normalized.items():
        if not isinstance(group, Mapping):
            merged[component_type] = deepcopy(group)
            for identity in [identity for identity in owners if identity[1] == component_type]:
                del owners[identity]
            continue

        current_group = merged.get(component_type)
        merged_group = dict(current_group) if isinstance(current_group, Mapping) else {}
        for name, definition in group.items():
            identity = ("components", component_type, name)
            previous = owners.get(identity)
            if previous is not None:
                warnings.append(
                    f"Config collision at components.{component_type}.{name}: "
                    f"{source} replaces the complete definition from {previous}",
                )
            merged_group[name] = deepcopy(definition)
            owners[identity] = source
        merged[component_type] = merged_group
    return merged


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
        from .components.component_registry import R

        with R.preserve(allow_mutation=True):
            manifest = load_package_manifest(entry.value, plugin_name=name)
            return _plugin_from_manifest(name, manifest)
    plugin = load_entry_point(entry, invoke=True)
    if not isinstance(plugin, Plugin):
        raise TypeError(f"Plugin entry point '{name}' did not return reme.plugin.Plugin")
    return plugin


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
            if plugin.name != name:
                raise ValueError(f"Plugin entry point '{name}' returned plugin '{plugin.name}'")
            plugins.append(plugin)
            seen.add(name)
        return cls(plugins)

    def _merge_config(self, application_config: Mapping[str, Any]) -> tuple[dict[str, Any], tuple[str, ...]]:
        """Merge config and retain startup diagnostics for named resource replacement."""
        plugin_layers = [expand_env_vars(plugin.config) for plugin in self.plugins]

        # Ordinary fields retain their existing priority: plugin defaults are
        # merged in enablement order, then application config wins.
        merged: dict[str, Any] = {}
        for layer in plugin_layers:
            merged = deep_merge_config(merged, _without_named_resources(layer))
        merged = deep_merge_config(merged, _without_named_resources(application_config))

        # Jobs and named component instances use the explicit plugin priority:
        # application config is lowest and later enabled plugins are highest.
        jobs: Any = _MISSING
        components: Any = _MISSING
        job_owners: dict[tuple[Any, ...], str] = {}
        component_owners: dict[tuple[Any, ...], str] = {}
        warnings: list[str] = []

        jobs = _overlay_jobs(jobs, job_owners, warnings, application_config, "application config")
        components = _overlay_components(
            components,
            component_owners,
            warnings,
            application_config,
            "application config",
        )
        for plugin, layer in zip(self.plugins, plugin_layers):
            source = f"plugin '{plugin.name}'"
            jobs = _overlay_jobs(jobs, job_owners, warnings, layer, source)
            components = _overlay_components(components, component_owners, warnings, layer, source)

        if jobs is not _MISSING:
            merged["jobs"] = jobs
        if components is not _MISSING:
            merged["components"] = components
        return merged, tuple(warnings)

    def merge_config(self, application_config: Mapping[str, Any]) -> dict[str, Any]:
        """Apply plugin config while atomically replacing same-name jobs and components."""
        merged, _ = self._merge_config(application_config)
        return merged

    def register(self, registry: ComponentRegistry) -> None:
        """Register every backend into an application-local registry."""
        for plugin in self.plugins:
            for backend in plugin.backends:
                registry.add(backend.name, backend.implementation, owner=plugin.name)


def resolve_plugin_runtime(application_config: Mapping[str, Any]) -> PluginRuntime:
    """Build one local registry and merge explicitly enabled plugin configuration."""
    manager = PluginManager.discover(application_config.get("plugins") or ())
    registry = create_application_registry()
    manager.register(registry)
    config, config_warnings = manager._merge_config(application_config)  # pylint: disable=protected-access
    return PluginRuntime(config=config, registry=registry, config_warnings=config_warnings)
