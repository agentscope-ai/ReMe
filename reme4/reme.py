"""ReMe memory management application entry point."""

import asyncio
import sys

from .application import Application
from .components import R
from .config import parse_args, resolve_app_config
from .enumeration import ComponentEnum
from .utils import cli_find_reme, load_env, precheck_start


class ReMe(Application):
    """ReMe memory management application."""


async def call_server(action: str, **kwargs):
    """Call the appropriate server component."""
    backend: str = kwargs.pop("backend", "http")
    client_cls = R.get(ComponentEnum.CLIENT, backend)
    if client_cls is None:
        raise ValueError(f"Unknown client backend: {backend!r}")
    async with client_cls() as client:
        async for chunk in client(action=action, **kwargs):
            print(chunk, end="", flush=True)
        print()


def main():
    """Parse CLI arguments and launch the appropriate mode."""
    action, kwargs = parse_args(*sys.argv[1:])
    if action == "start":
        load_env()
        kwargs = resolve_app_config(**kwargs)
        if not precheck_start(kwargs.get("service")):
            return
        ReMe(**kwargs).run_app()
    elif action == "find_reme":
        cli_find_reme()
    else:
        asyncio.run(call_server(action, **kwargs))


if __name__ == "__main__":
    main()
