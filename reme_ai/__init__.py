# pylint: disable=wrong-import-position
"""ReMe AI - A memory management framework for AI agents."""

import os

os.environ["FLOW_APP_NAME"] = "ReMe"

from . import agent  # noqa: E402
from . import config  # noqa: E402
from . import constants  # noqa: E402
from . import context  # noqa: E402
from . import enumeration  # noqa: E402
from . import retrieve  # noqa: E402
from . import schema  # noqa: E402
from . import service  # noqa: E402
from . import summary  # noqa: E402
from . import utils  # noqa: E402
from . import vector_store  # noqa: E402
from .main import ReMeApp  # noqa: E402 F401

__all__ = [
    "agent",
    "config",
    "constants",
    "context",
    "enumeration",
    "retrieve",
    "schema",
    "service",
    "summary",
    "utils",
    "vector_store",
]

__version__ = "0.2.0.0"
