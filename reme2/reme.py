"""ReMe CLI application entry point."""

import sys

from .application import Application
from .component import R
from .config import parse_args
from .enumeration import ComponentEnum
from .utils import run_coro_safely


class ReMe(Application):
    """ReMe memory management application."""


def main():
    """Entry point for ReMe CLI."""
    action, config = parse_args(sys.argv[1:])
    if action == "start":
        reme = ReMe(**config)
        reme.run_app()

    else:
        backend: str = config.pop("backend", "http")
        client_cls = R.get(ComponentEnum.CLIENT, backend)
        client = client_cls(action=action, **config)
        run_coro_safely(client())


if __name__ == "__main__":
    main()
