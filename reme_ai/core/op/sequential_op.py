from typing import Dict, List

from .base_op import BaseOp


class SequentialOp(BaseOp):

    def execute_sync(self):
        for op in self.sub_ops.values():
            assert not op.async_mode
            op.call_sync(context=self.context)

    async def execute(self):
        for op in self.sub_ops.values():
            assert op.async_mode
            await op.call(context=self.context)

    def __lshift__(self, op: Dict[str, BaseOp] | List[BaseOp] | BaseOp):
        raise RuntimeError(f"`<<` is not supported in `{self.name}`")

    def __rshift__(self, op: BaseOp):
        if isinstance(op, SequentialOp) and op.sub_ops:
            self.add_sub_ops(op.sub_ops)
        else:
            self.add_sub_op(op)
        return self
