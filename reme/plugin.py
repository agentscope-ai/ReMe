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
from .config import ResolvedConfigSection, deep_merge_config, expand_env_vars
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


def _without_named_resources(config: Mapping[str, Any]) -> dict[str, Any]:
    """Return fields that retain the ordinary plugin-default merge behavior."""
    return {key: value for key, value in config.items() if key not in {"jobs", "components"}}


def _require_mapping(value: Any, path: str) -> Mapping[str, Any]:
    """Reject an invalid config fragment at its source."""
    if not isinstance(value, Mapping):
        raise TypeError(f"{path} must be a mapping")
    return value


def _replace_named(
    target: dict[str, Any],
    incoming: Mapping[str, Any],
    owners: dict[tuple[str, ...], str],
    warnings: list[str],
    source: str,
    prefix: tuple[str, ...],
) -> None:
    """Atomically replace resources identified by ``prefix + name``."""
    for name, definition in incoming.items():
        if not isinstance(name, str):
            raise TypeError(f"{'.'.join(prefix)} names must be strings, got {type(name).__name__}")
        identity = (*prefix, name)
        previous = owners.get(identity)
        if previous is not None:
            warnings.append(
                f"Config collision at {'.'.join(identity)}: "
                f"{source} replaces the complete definition from {previous}",
            )
        target[name] = definition
        owners[identity] = source


def _overlay_atomic_resources(
    jobs: dict[str, Any],
    components: dict[str, dict[str, Any]],
    job_owners: dict[tuple[str, ...], str],
    component_owners: dict[tuple[str, ...], str],
    warnings: list[str],
    layer: Mapping[str, Any],
    source: str,
) -> tuple[bool, bool]:
    """Overlay the two resource kinds with atomic identities."""
    has_jobs = "jobs" in layer
    has_components = "components" in layer

    if has_jobs:
        incoming_jobs = _require_mapping(layer["jobs"], f"{source}.jobs")
        _replace_named(jobs, incoming_jobs, job_owners, warnings, source, ("jobs",))

    if not has_components:
        return has_jobs, has_components

    raw_components = _require_mapping(layer["components"], f"{source}.components")

    # Match ApplicationConfig normalization before comparing identities. If
    # one source repeats an equivalent type, its later mapping wins without a
    # new same-source diagnostic, preserving the existing validator behavior.
    normalized = {
        component_type_name(component_type): _require_mapping(
            group,
            f"{source}.components.{component_type}",
        )
        for component_type, group in raw_components.items()
    }
    for component_type, group in normalized.items():
        target_group = components.setdefault(component_type, {})
        _replace_named(
            target_group,
            group,
            component_owners,
            warnings,
            source,
            ("components", component_type),
        )
    return has_jobs, has_components


def _split_application_resource_layers(
    application_config: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Separate atomic definitions from explicit field-level overrides."""
    application_layer: dict[str, Any] = {}
    explicit_overrides: dict[str, Any] = {}
    for section_name in ("jobs", "components"):
        if section_name not in application_config:
            continue
        section = application_config[section_name]
        application_layer[section_name] = section
        if not isinstance(section, ResolvedConfigSection):
            continue

        projected: dict[str, Any] = {}
        for path in section.explicit_paths:
            current: Any = section
            for key in path:
                if not isinstance(current, Mapping) or key not in current:
                    break
                current = current[key]
            else:
                target = projected
                for key in path[:-1]:
                    target = target.setdefault(key, {})
                target[path[-1]] = current
        if projected:
            explicit_overrides[section_name] = projected
    return application_layer, explicit_overrides


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
        plugin_layers = [
            (
                f"plugin '{plugin.name}'",
                _require_mapping(expand_env_vars(plugin.config), f"plugin '{plugin.name}'"),
            )
            for plugin in self.plugins
        ]

        # Ordinary fields retain their existing priority: plugin defaults are
        # merged in enablement order, then application config wins.
        merged: dict[str, Any] = {}
        for _, layer in plugin_layers:
            merged = deep_merge_config(merged, _without_named_resources(layer))
        merged = deep_merge_config(merged, _without_named_resources(application_config))

        # Complete jobs and named component instances are atomic resources:
        # loaded application config is lowest and later plugins are highest.
        # Explicit CLI dot-notation values are field-level overrides applied
        # after the winning complete resource has been selected.
        application_resources, explicit_resource_overrides = _split_application_resource_layers(application_config)
        jobs: dict[str, Any] = {}
        components: dict[str, dict[str, Any]] = {}
        job_owners: dict[tuple[str, ...], str] = {}
        component_owners: dict[tuple[str, ...], str] = {}
        warnings: list[str] = []
        has_jobs = False
        has_components = False

        atomic_layers = [("application config", application_resources), *plugin_layers]
        for source, layer in atomic_layers:
            layer_has_jobs, layer_has_components = _overlay_atomic_resources(
                jobs,
                components,
                job_owners,
                component_owners,
                warnings,
                layer,
                source,
            )
            has_jobs |= layer_has_jobs
            has_components |= layer_has_components

        if has_jobs:
            merged["jobs"] = jobs
        if has_components:
            merged["components"] = components

        # CLI/config kwargs use the established deep-merge contract. They are
        # patches to the selected complete resources, not additional providers.
        merged = deep_merge_config(merged, explicit_resource_overrides)
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
