"""Module providing the ParallelOp class for concurrent operation execution."""

from .base_op import BaseOp


class ParallelOp(BaseOp):
    """Operation class that executes multiple sub-operations in parallel."""

    async def execute(self):
        """Executes all sub-operations concurrently using asynchronous tasks."""
        for op in self.sub_ops:
            assert op.async_mode
            self.submit_async_task(op.call, context=self.context)
        await self.join_async_tasks()

    def execute_sync(self):
        """Executes all sub-operations concurrently using synchronous task management."""
        for op in self.sub_ops:
            assert not op.async_mode
            self.submit_sync_task(op.call_sync, context=self.context)
        self.join_sync_tasks()

    def __lshift__(self, op: dict[str, BaseOp] | list[BaseOp] | BaseOp):
        """Raises RuntimeError as the shift operator is not supported for parallel operations."""
        raise RuntimeError(f"`<<` is not supported in `{self.name}`")

    def __or__(self, op: BaseOp):
        """Adds sub-operations to the current parallel group using the bitwise OR operator."""
        if isinstance(op, ParallelOp) and op.sub_ops:
            self.add_sub_ops(op.sub_ops)
        else:
            self.add_sub_op(op)
        return self
