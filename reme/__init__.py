"""ReMe CLI package."""

from ._version import __version__
from . import config
from . import constants
from . import enumeration
from . import schema
from . import steps
from . import utils
from .core.application import Application
from .core.registry import R as _R
from .plugin import Backend, Plugin

# Component and Step packages above have completed their decorator-driven
# bootstrap. Runtime code receives mutable copies of this immutable template.
_R.freeze()

__all__ = [
    "__version__",
    "Application",
    "Backend",
    "Plugin",
    # submodules
    "config",
    "constants",
    "enumeration",
    "schema",
    "steps",
    "utils",
]
