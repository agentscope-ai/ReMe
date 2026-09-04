"""ReMe memory management application entry point."""

import asyncio
from collections.abc import Sequence
from dataclasses import dataclass
import sys

from .application import Application
from .components.service.cli_service import prepare_start_config, should_precheck_start
from .config import ResolvedAppConfig, parse_action, parse_kwargs, resolve_app_config
from .enumeration import ComponentEnum
from .plugin import resolve_plugin_runtime
from .utils import cli_find_reme, load_env, precheck_start, running_app_config

_CLIENT_KWARGS = {"host", "port", "timeout", "transport", "command", "args", "show_metadata"}


class ReMe(Application):
    """ReMe memory management application."""


@dataclass(frozen=True)
class CliInvocation:
    """A top-level CLI action with arguments in that action's own syntax."""

    action: str
    arguments: tuple[str, ...]


def parse_cli_invocation(argv: Sequence[str]) -> CliInvocation:
    """Parse only the grammar shared by every CLI command family."""
    if not argv:
        raise ValueError("No arguments provided")
    return CliInvocation(action=parse_action(argv[0]), arguments=tuple(argv[1:]))


async def call_server(action: str, **kwargs):
    """Call the running server with a client matching its *actual* service config.

    The client backend and its transport/host/port are taken from the running
    ``reme start`` process — its start args replayed through the same
    ``resolve_app_config`` the server used — so a bare ``reme <action>`` reaches
    the server however it was actually started (``http`` REST or ``mcp``
    streamable-http / sse / stdio), including ``service.*`` overrides that never
    touched the on-disk config file. Falls back to local config resolution when
    no server is detected. Explicit ``backend=`` / ``transport=`` / ``host=`` /
    ``port=`` kwargs still win and never leak into the tool payload.
    """
    # config-selecting keys steer client construction; they are not tool args.
    resolve_kwargs = {}
    if isinstance(kwargs.get("config"), str):
        resolve_kwargs["config"] = kwargs.pop("config")
    if isinstance(kwargs.get("service"), dict):
        resolve_kwargs["service"] = kwargs.pop("service")

    # Prefer the running server's complete config so its enabled client plugins
    # are available even when the caller does not repeat config=<name>.
    app_config = running_app_config()
    if app_config is None:
        app_config = resolve_app_config(log_config=False, **resolve_kwargs)
    runtime = resolve_plugin_runtime(app_config)
    app_config = runtime.config
    service = app_config.get("service")
    service = service if isinstance(service, dict) else {}

    backend: str = kwargs.pop("backend", None) or service.get("backend", "http")
    # Seed client kwargs from the service config only when we are actually using
    # that service's backend — transport/host/port are backend-specific, so an
    # explicit backend override must not inherit the other backend's settings.
    seed = service if backend == service.get("backend") else {}
    client_kwargs = {k: seed[k] for k in _CLIENT_KWARGS if k in seed}
    client_kwargs.update({key: kwargs.pop(key) for key in list(kwargs) if key in _CLIENT_KWARGS})

    client_cls = runtime.registry.get(ComponentEnum.CLIENT, backend)
    if client_cls is None:
        raise ValueError(f"Unknown client backend: {backend!r}")
    async with client_cls(**client_kwargs) as client:
        async for chunk in client(action=action, **kwargs):
            print(chunk, end="", flush=True)
        print()


def _run_plugin_command(argv: Sequence[str]) -> None:
    """Run local package management without initializing the application."""
    from .plugin_cli import plugin_cli

    status = plugin_cli(argv)
    if status:
        raise SystemExit(status)


def _start_application(kwargs: dict, environment: dict) -> None:
    """Resolve startup configuration and run the application."""
    prepared = prepare_start_config(kwargs)
    config = prepared if isinstance(prepared, ResolvedAppConfig) else ResolvedAppConfig(overrides=prepared)
    config = config.with_overrides({"environment": environment})
    visible = config.materialize()
    if should_precheck_start(config) and not precheck_start(visible.get("service")):
        return
    ReMe(resolved_config=config).run_app()


def main() -> None:
    """Parse CLI arguments and launch the appropriate mode."""
    invocation = parse_cli_invocation(sys.argv[1:])
    action = invocation.action

    if action == "plugins":
        # Package management is local-only and must not load application config,
        # environment files, or a running service.
        _run_plugin_command(invocation.arguments)
        return

    environment = load_env()
    kwargs = parse_kwargs(*invocation.arguments)
    if action == "start":
        _start_application(kwargs, environment)
    elif action == "find_reme":
        cli_find_reme()
    else:
        asyncio.run(call_server(action, **kwargs))


if __name__ == "__main__":
    main()
