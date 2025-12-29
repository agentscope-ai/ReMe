from typing import Dict, List

from .base_op import BaseOp


class ParallelOp(BaseOp):

    def execute_sync(self):
        for op in self.sub_ops.values():
            assert not op.async_mode
            self.submit_sync_task(op.call_sync, context=self.context)
        self.join_sync_tasks()

    async def execute(self):
        for op in self.sub_ops.values():
            assert op.async_mode
            self.submit_async_task(op.call, context=self.context)
        await self.join_async_tasks()

    def __lshift__(self, op: Dict[str, BaseOp] | List[BaseOp] | BaseOp):
        raise RuntimeError(f"`<<` is not supported in `{self.name}`")

    def __or__(self, op: BaseOp):
        if isinstance(op, ParallelOp) and op.sub_ops:
            self.add_sub_ops(op.sub_ops)
        else:
            self.add_sub_op(op)
        return self
