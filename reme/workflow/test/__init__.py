from .test_op import TestOp
from ...core import R

__all__ = [
    "TestOp",
]

R.op.register()(TestOp)
