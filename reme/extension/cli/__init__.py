"""File-based memory operations."""

from .fb_cli import FbCli
from .fb_compactor import FbCompactor
from .fb_context_checker import FbContextChecker
from .fb_summarizer import FbSummarizer
from ...core.registry_factory import R

__all__ = [
    "FbCli",
    "FbCompactor",
    "FbContextChecker",
    "FbSummarizer",
]

for name in __all__:
    op_class = globals()[name]
    R.ops.register(op_class)
