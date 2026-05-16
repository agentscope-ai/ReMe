"""ReMe memory management application entry point."""

import asyncio
import sys

from .application import Application
from .components import R
from .config import parse_args, resolve_app_config
from .enumeration import ComponentEnum
from .utils import load_env


class ReMe(Application):
    """ReMe memory management application."""


async def call_server(action: str, **kwargs):
    """Call the appropriate server component."""
    backend: str = kwargs.pop("backend", "http")
    client_cls = R.get(ComponentEnum.CLIENT, backend)
    async with client_cls(action=action, **kwargs) as client:
        await client()


def main():
    """Parse CLI arguments and launch the appropriate mode."""
    action, kwargs = parse_args(*sys.argv[1:])
    if action == "start":
        load_env()
        kwargs = resolve_app_config(**kwargs)
        ReMe(**kwargs).run_app()
    else:
        asyncio.run(call_server(action, **kwargs))


if __name__ == "__main__":
    main()
