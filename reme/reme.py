"""ReMe memory management application entry point."""

import asyncio
import sys

from .application import Application
from .components import R
from .components.service.cli_service import prepare_start_config, should_precheck_start
from .config import parse_args, resolve_app_config
from .enumeration import ComponentEnum
from .utils import cli_find_reme, load_env, precheck_start, running_service_config

_CLIENT_KWARGS = {"host", "port", "timeout", "transport", "command", "args", "show_metadata"}


class ReMe(Application):
    """ReMe memory management application."""


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

    # Prefer the running server's real config; fall back to the local config file.
    service = running_service_config()
    if service is None:
        service = resolve_app_config(**resolve_kwargs).get("service")
    service = service if isinstance(service, dict) else {}

    backend: str = kwargs.pop("backend", None) or service.get("backend", "http")
    # Seed client kwargs from the service config only when we are actually using
    # that service's backend — transport/host/port are backend-specific, so an
    # explicit backend override must not inherit the other backend's settings.
    seed = service if backend == service.get("backend") else {}
    client_kwargs = {k: seed[k] for k in _CLIENT_KWARGS if k in seed}
    client_kwargs.update({key: kwargs.pop(key) for key in list(kwargs) if key in _CLIENT_KWARGS})

    client_cls = R.get(ComponentEnum.CLIENT, backend)
    if client_cls is None:
        raise ValueError(f"Unknown client backend: {backend!r}")
    async with client_cls(**client_kwargs) as client:
        async for chunk in client(action=action, **kwargs):
            print(chunk, end="", flush=True)
        print()


def main():
    """Parse CLI arguments and launch the appropriate mode."""
    action, kwargs = parse_args(*sys.argv[1:])
    if action == "start":
        load_env()
        kwargs = prepare_start_config(kwargs)
        if should_precheck_start(kwargs) and not precheck_start(kwargs.get("service")):
            return
        ReMe(**kwargs).run_app()
    elif action == "find_reme":
        cli_find_reme()
    else:
        asyncio.run(call_server(action, **kwargs))


if __name__ == "__main__":
    main()
