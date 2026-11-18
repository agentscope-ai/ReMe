#!/usr/bin/env python3
"""Prepare a user-owned Codex or Claude Code plugin with HTTP or stdio MCP."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import shlex
import shutil
import tempfile

ROOT = Path(__file__).resolve().parent
HOSTS = {
    "codex": ("codex", "plugins/reme", ".agents/plugins/marketplace.json", "reme_config.py", "mcpUrl"),
    "claude-code": (
        "claude_code",
        "reme",
        ".claude-plugin/marketplace.json",
        "hooks/auto_memory.py",
        "mcp_url",
    ),
}


def configure(host: str, transport: str = "http", mcp_url: str | None = None, api_url: str | None = None) -> Path:
    """Snapshot validated settings into a local marketplace without editing an installed cache."""
    if host not in HOSTS or transport not in {"http", "stdio"}:
        raise ValueError("Choose codex or claude-code and an http or stdio transport")
    if api_url is not None and host != "claude-code":
        raise ValueError("--api-url is only used by Claude Code's HTTP Hooks")
    directory, plugin_path, marketplace_path, config_path, url_key = HOSTS[host]
    source = ROOT / directory
    plugin = source / plugin_path
    spec = importlib.util.spec_from_file_location("reme_connection_config", plugin / config_path)
    adapter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(adapter)

    destination = adapter.data_dir() / "plugin-marketplace"
    with adapter.lock(adapter.data_dir() / "config.lock") as acquired:
        if not acquired:
            raise RuntimeError("Another ReMe configuration update is in progress; please retry")
        settings_path = adapter.data_dir() / "config.json"
        values = json.loads(settings_path.read_text(encoding="utf-8")) if settings_path.exists() else {}
        if not isinstance(values, dict):
            raise ValueError("ReMe configuration must be a JSON object")
        if mcp_url is not None and values.get("api_url") and api_url is None:
            raise ValueError("Also pass --api-url when updating an address with an explicit HTTP API base")
        changes = {url_key: mcp_url} if mcp_url is not None else {}
        if api_url is not None:
            changes["api_url"] = api_url
        config = adapter.validate_config({**values, **changes})
        if api_url:
            config["api_url"] = config["endpoint"]
        if destination.is_symlink():
            raise ValueError("The generated marketplace must not be a symlink")
        marker = destination / ".reme-generated"
        if destination.exists() and (not marker.is_file() or marker.read_text(encoding="utf-8") != host):
            raise ValueError(f"Refusing to replace an unmanaged directory: {destination}")

        with tempfile.TemporaryDirectory(prefix=".mcp-", dir=adapter.data_dir()) as temporary:
            stage = Path(temporary) / "marketplace"
            target = stage / plugin_path
            shutil.copytree(plugin, target, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
            marketplace = stage / marketplace_path
            marketplace.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source / marketplace_path, marketplace)
            if transport == "stdio":
                connection = json.loads((plugin / ".mcp.stdio.json").read_text(encoding="utf-8"))
            else:
                connection = json.loads((plugin / ".mcp.json").read_text(encoding="utf-8"))
                connection["mcpServers"]["reme"]["url"] = config[url_key]
            adapter.write_json(target / ".mcp.json", connection)
            (stage / ".reme-generated").write_text(host, encoding="utf-8")
            # The retry queue, settings, and logs are siblings and are never copied or replaced.
            backup = Path(temporary) / "previous"
            if destination.exists():
                os.replace(destination, backup)
            try:
                os.replace(stage, destination)
                if changes:
                    adapter.write_json(settings_path, {**values, **{key: config[key] for key in changes}})
            except BaseException:
                shutil.rmtree(destination, ignore_errors=True)
                if backup.exists():
                    os.replace(backup, destination)
                raise
    return destination.resolve()


def main() -> None:
    """Print native installation commands after preparing the selected connection."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("host", choices=HOSTS)
    parser.add_argument("--transport", choices=("http", "stdio"), default="http")
    parser.add_argument(
        "--mcp-url",
        help="Update the existing user JSON settings and the generated MCP address together.",
    )
    parser.add_argument("--api-url", help="Claude Code HTTP Job API base for custom MCP paths.")
    args = parser.parse_args()
    try:
        destination = configure(args.host, args.transport, args.mcp_url, args.api_url)
    except (OSError, ValueError, RuntimeError) as exc:
        parser.exit(1, f"ReMe configuration failed: {exc}\n")
    print(f"Prepared {args.transport} MCP in {destination}")
    if args.host == "codex":
        print(f"codex plugin marketplace add {shlex.quote(str(destination))}")
        print("codex plugin add reme@reme-codex")
    else:
        print(f"claude plugin marketplace add {shlex.quote(str(destination))}")
        print("claude plugin install reme@reme-marketplace")
    print(
        "For an existing installation, reinstall the plugin from this source, restart the host, and review Hooks.",
    )
    print("When replacing a marketplace registered from another path, remove that registration before adding this one.")
    if args.host == "codex" and args.transport == "http":
        print("HTTP uses ReMe's tools and service-side Dream schedule; the native settings/status panel uses stdio.")


if __name__ == "__main__":
    main()
