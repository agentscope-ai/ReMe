"""Extension operations and tools."""

from . import procedural_memory
from .simple_chat import SimpleChat
from .stream_chat import StreamChat
from .test_op import TestOp
from .translate_ts import TranslateTs
from ..core.registry_factory import R

__all__ = [
    "procedural_memory",
    "SimpleChat",
    "StreamChat",
    "TestOp",
    "TranslateTs",
]

for name in __all__:
    op_class = globals()[name]
    R.ops.register(op_class)
