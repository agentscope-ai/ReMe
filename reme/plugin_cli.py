"""Local CLI for inspecting and managing installed ReMe plugin packages."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path
import subprocess
import sys
import tomllib
from typing import Sequence

from .entry_point import PLUGIN_ENTRY_POINT_GROUP
from .plugin_manifest import PLUGIN_MANIFEST, PluginManifest, load_package_manifest, parse_plugin_manifest


@dataclass(frozen=True)
class InstalledPlugin:
    """Package metadata for one installed ``reme.plugins`` entry point."""

    name: str
    target: str
    distribution: str
    version: str
    entry: metadata.EntryPoint

    @property
    def format(self) -> str:
        """Return the declarative or compatibility contract used by the entry point."""
        return "manifest" if ":" not in self.target else "legacy"


def _installed_plugins() -> list[InstalledPlugin]:
    """Discover installed plugins without importing their packages."""
    entries = metadata.entry_points().select(group=PLUGIN_ENTRY_POINT_GROUP)
    plugins: list[InstalledPlugin] = []
    for entry in entries:
        distribution = getattr(entry, "dist", None)
        dist_name = distribution.metadata.get("Name", "") if distribution is not None else ""
        version = distribution.version if distribution is not None else ""
        plugins.append(
            InstalledPlugin(
                name=entry.name,
                target=entry.value,
                distribution=dist_name or "unknown",
                version=version or "unknown",
                entry=entry,
            ),
        )
    return sorted(plugins, key=lambda item: (item.name, item.distribution, item.target))


def _select_plugin(name: str) -> InstalledPlugin:
    matches = [plugin for plugin in _installed_plugins() if plugin.name == name]
    if not matches:
        raise ValueError(f"Plugin '{name}' is not installed")
    if len(matches) > 1:
        providers = ", ".join(f"{plugin.distribution} ({plugin.target})" for plugin in matches)
        raise ValueError(f"Plugin '{name}' has multiple installed providers: {providers}")
    return matches[0]


def _print_table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> None:
    widths = [max(len(header), *(len(row[index]) for row in rows)) for index, header in enumerate(headers)]
    print("  ".join(header.ljust(widths[index]) for index, header in enumerate(headers)))
    print("  ".join("-" * width for width in widths))
    for row in rows:
        print("  ".join(value.ljust(widths[index]) for index, value in enumerate(row)))


def _enabled_plugins(config: str | None) -> set[str] | None:
    if config is None:
        return None
    from .config.config_parser import resolve_app_config

    value = resolve_app_config(config=config, log_config=False).get("plugins") or []
    if not isinstance(value, list) or not all(isinstance(name, str) for name in value):
        raise TypeError("Application config 'plugins' must be a list of strings")
    return set(value)


def _list_plugins(args: argparse.Namespace) -> int:
    plugins = _installed_plugins()
    enabled = _enabled_plugins(args.config)
    records = [
        {
            "name": plugin.name,
            "distribution": plugin.distribution,
            "version": plugin.version,
            "format": plugin.format,
            "target": plugin.target,
            **({"enabled": plugin.name in enabled} if enabled is not None else {}),
        }
        for plugin in plugins
    ]
    if args.json:
        print(json.dumps(records, ensure_ascii=False, indent=2))
        return 0
    if not records:
        print("No ReMe plugins installed.")
        return 0
    headers = ["PLUGIN", "DISTRIBUTION", "VERSION", "FORMAT"]
    if enabled is not None:
        headers.append("ENABLED")
    rows = [
        [
            record["name"],
            record["distribution"],
            record["version"],
            record["format"],
            *(["yes" if record["enabled"] else "no"] if enabled is not None else []),
        ]
        for record in records
    ]
    _print_table(headers, rows)
    return 0


def _installed_manifest(plugin: InstalledPlugin) -> PluginManifest:
    if plugin.format != "manifest":
        raise ValueError(f"Plugin '{plugin.name}' uses the legacy Python descriptor format")
    distribution = getattr(plugin.entry, "dist", None)
    if distribution is not None:
        relative = Path(*plugin.target.split(".")).joinpath(PLUGIN_MANIFEST)
        path = Path(distribution.locate_file(relative))
        if path.is_file():
            return parse_plugin_manifest(path.read_text(encoding="utf-8"), plugin_name=plugin.name)
    # Editable installs may not expose package data through ``locate_file``.
    # Importlib resources then has to import the package, whose ``__init__``
    # may still contain legacy registration side effects.
    from .components.component_registry import R

    with R.preserve(allow_mutation=True):
        return load_package_manifest(plugin.target, plugin_name=plugin.name)


def _plugin_details(plugin: InstalledPlugin) -> dict:
    details = {
        "name": plugin.name,
        "distribution": plugin.distribution,
        "version": plugin.version,
        "format": plugin.format,
        "target": plugin.target,
        "backends": [],
        "default_jobs": [],
    }
    if plugin.format == "manifest":
        manifest = _installed_manifest(plugin)
        details["backends"] = list(manifest.backends)
        jobs = manifest.application_defaults.get("jobs") or {}
        details["default_jobs"] = list(jobs) if isinstance(jobs, dict) else []
    return details


def _show_plugin(args: argparse.Namespace) -> int:
    details = _plugin_details(_select_plugin(args.plugin))
    if args.json:
        print(json.dumps(details, ensure_ascii=False, indent=2))
        return 0
    for label, key in (
        ("Plugin", "name"),
        ("Distribution", "distribution"),
        ("Version", "version"),
        ("Format", "format"),
        ("Entry point", "target"),
    ):
        print(f"{label}: {details[key]}")
    for label, key in (("Backends", "backends"), ("Default jobs", "default_jobs")):
        values = details[key]
        print(f"{label}:" if values else f"{label}: none")
        for value in values:
            print(f"  {value}")
    return 0


def _run_pip(arguments: list[str]) -> int:
    command = [sys.executable, "-m", "pip", *arguments]
    try:
        return subprocess.run(command, check=False).returncode
    except OSError as exc:
        raise RuntimeError(f"Unable to run pip with {sys.executable}: {exc}") from exc


def _install_plugin(args: argparse.Namespace) -> int:
    command = ["install"]
    if args.editable:
        command.append("--editable")
    if args.upgrade:
        command.append("--upgrade")
    command.append(args.target)
    result = _run_pip(command)
    if result == 0:
        print("Package installed. Run 'reme plugins list' to verify it, then enable its plugin name in app config.")
    return result


def _uninstall_plugin(args: argparse.Namespace) -> int:
    plugin = _select_plugin(args.plugin)
    if plugin.distribution == "unknown":
        raise ValueError(f"Cannot determine the distribution that provides plugin '{plugin.name}'")
    siblings = [
        candidate.name
        for candidate in _installed_plugins()
        if candidate.distribution == plugin.distribution and candidate.name != plugin.name
    ]
    if siblings:
        print(f"Distribution '{plugin.distribution}' also provides: {', '.join(siblings)}")
    command = ["uninstall"]
    if args.yes:
        command.append("--yes")
    command.append(plugin.distribution)
    result = _run_pip(command)
    if result == 0:
        print(f"Plugin package '{plugin.distribution}' uninstalled. Remove '{plugin.name}' from application configs.")
    return result


def _validate_plugins(plugins) -> None:
    """Validate imports, registry ownership, and merged application schema."""
    from .components.component_registry import create_application_registry
    from .config.config_parser import resolve_app_config
    from .plugin import PluginManager
    from .schema.application_config import ApplicationConfig

    manager = PluginManager(plugins)
    registry = create_application_registry()
    manager.register(registry)
    ApplicationConfig(**manager.merge_config(resolve_app_config(log_config=False)))


def _validate_installed(name: str) -> list[str]:
    from .plugin import PluginManager

    manager = PluginManager.discover([name])
    _validate_plugins(manager.plugins)
    return [name]


def _validate_local(path: Path) -> list[str]:
    from .components.component_registry import R
    from .plugin import Backend, Plugin, _load_backend

    project_file = path if path.name == "pyproject.toml" else path / "pyproject.toml"
    if not project_file.is_file():
        raise FileNotFoundError(f"pyproject.toml not found: {project_file}")
    project = tomllib.loads(project_file.read_text(encoding="utf-8"))
    entry_points = project.get("project", {}).get("entry-points", {}).get(PLUGIN_ENTRY_POINT_GROUP)
    if not isinstance(entry_points, dict) or not entry_points:
        raise ValueError(f"No {PLUGIN_ENTRY_POINT_GROUP} entry points found in {project_file}")

    setuptools = project.get("tool", {}).get("setuptools", {})
    package_dir = setuptools.get("package-dir", {}) if isinstance(setuptools, dict) else {}
    source_root = project_file.parent / (package_dir.get("") or ".")
    plugins = []
    sys.path.insert(0, str(source_root.resolve()))
    try:
        # Match installed-plugin loading: imports may execute compatibility
        # decorators, but they must not mutate the frozen built-in template.
        with R.preserve(allow_mutation=True):
            for name, package in entry_points.items():
                if not isinstance(name, str) or not isinstance(package, str) or ":" in package:
                    raise ValueError("Local validation requires package-only manifest entry points")
                manifest_path = source_root.joinpath(*package.split(".")).joinpath(PLUGIN_MANIFEST)
                if not manifest_path.is_file():
                    raise FileNotFoundError(f"Plugin manifest not found: {manifest_path}")
                manifest = parse_plugin_manifest(manifest_path.read_text(encoding="utf-8"), plugin_name=name)
                backends = tuple(
                    Backend(backend_name, _load_backend(target, plugin_name=name))
                    for backend_name, target in manifest.backends.items()
                )
                plugins.append(Plugin(name=name, backends=backends, config=manifest.application_defaults))
        _validate_plugins(plugins)
    finally:
        sys.path.remove(str(source_root.resolve()))
    return [plugin.name for plugin in plugins]


def _validate_plugin(args: argparse.Namespace) -> int:
    path = Path(args.target).expanduser()
    names = _validate_local(path.resolve()) if path.exists() else _validate_installed(args.target)
    print(f"Valid ReMe plugin: {', '.join(names)}")
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="reme plugins", description="Manage ReMe plugin packages.")
    commands = parser.add_subparsers(dest="command", required=True)

    list_parser = commands.add_parser("list", help="List installed ReMe plugins.")
    list_parser.add_argument("--config", help="Show whether plugins are enabled by this config.")
    list_parser.add_argument("--json", action="store_true", help="Print JSON output.")
    list_parser.set_defaults(handler=_list_plugins)

    show_parser = commands.add_parser("show", help="Show one installed plugin.")
    show_parser.add_argument("plugin")
    show_parser.add_argument("--json", action="store_true", help="Print JSON output.")
    show_parser.set_defaults(handler=_show_plugin)

    install_parser = commands.add_parser("install", help="Install a plugin package with this Python interpreter.")
    install_parser.add_argument("target", help="Distribution specifier, wheel, VCS URL, or local path.")
    install_parser.add_argument("--editable", action="store_true", help="Install a local project in editable mode.")
    install_parser.add_argument("--upgrade", action="store_true", help="Upgrade an existing installation.")
    install_parser.set_defaults(handler=_install_plugin)

    uninstall_parser = commands.add_parser("uninstall", help="Uninstall the distribution providing a plugin.")
    uninstall_parser.add_argument("plugin", help="Plugin entry-point name, such as auto-fin.")
    uninstall_parser.add_argument("--yes", action="store_true", help="Do not ask pip for confirmation.")
    uninstall_parser.set_defaults(handler=_uninstall_plugin)

    validate_parser = commands.add_parser("validate", help="Validate an installed plugin or local plugin project.")
    validate_parser.add_argument("target", help="Installed plugin name, project directory, or pyproject.toml.")
    validate_parser.set_defaults(handler=_validate_plugin)
    return parser


def plugin_cli(argv: Sequence[str]) -> int:
    """Run the local-only plugin command group and return a process status."""
    args = _parser().parse_args(list(argv))
    try:
        return args.handler(args)
    except (FileNotFoundError, KeyError, ModuleNotFoundError, RuntimeError, TypeError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
