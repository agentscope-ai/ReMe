"""ReMe CLI package."""

__version__ = "0.4.1.8"

from . import config
from . import constants
from . import enumeration
from . import schema
from . import steps
from . import utils
from .components import BaseComponent, R
from .application import Application
from .plugin import Backend, Plugin
from .reme import ReMe

# Component and Step packages above have completed their decorator-driven
# bootstrap. Runtime code receives mutable copies of this immutable template.
R.freeze()

__all__ = [
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
