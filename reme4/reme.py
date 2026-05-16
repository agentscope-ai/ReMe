"""ReMe memory management application entry point."""

import asyncio
import sys

from .application import Application
from .components import R
from .config import parse_args
from .enumeration import ComponentEnum
from .utils import load_env

load_env()


class ReMe(Application):
    """ReMe memory management application."""


def main():
    """Parse CLI arguments and launch the appropriate mode."""
    action, config = parse_args(*sys.argv[1:])
    if action == "start":
        reme = ReMe(**config)
        reme.run_app()
    else:
        backend: str = config.pop("backend", "http")
        client_cls = R.get(ComponentEnum.CLIENT, backend)
        client = client_cls(action=action, **config)
        asyncio.run(client())


if __name__ == "__main__":
    main()
