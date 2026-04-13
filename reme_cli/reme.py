import sys

from .application import Application
from .config import parse_args


class ReMe(Application):
    ...


class ReMeClient:
    ...


def main():
    action, config = parse_args(sys.argv[1:])
    if action == "app":
        reme = ReMe(**config)
        reme.run_app()

    else:
        ...
        """Main entry point for running ReMe from command line."""


if __name__ == "__main__":
    main()
