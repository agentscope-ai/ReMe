"""ReMe"""

from . import config
from . import core
from . import extension
from . import memory
from .reme import ReMe

__version__ = "0.3.1.8"

__all__ = [
    "config",
    "core",
    "extension",
    "memory",
    "ReMe",
]

"""
conda create -n fl_test2 python=3.10
conda activate fl_test2
conda env remove -n fl_test2
"""
