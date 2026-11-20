"""ReMe CLI package."""

from ._version import __version__
from . import config
from . import constants
from . import enumeration
from . import schema
from . import steps
from . import utils
from .components import BaseComponent, R
from .core.application import Application, ReMe
from .plugin import Backend, Plugin

# Component and Step packages above have completed their decorator-driven
# bootstrap. Runtime code receives mutable copies of this immutable template.
R.freeze()

__all__ = [
    "__version__",
    "Application",
    "BaseComponent",
    "Backend",
    "Plugin",
    "ReMe",
    # submodules
    "config",
    "constants",
    "enumeration",
    "schema",
    "steps",
    "utils",
]
