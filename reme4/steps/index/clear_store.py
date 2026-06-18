"""Clear the file store before reusing the standard init change producer."""

from ..base_step import BaseStep
from ...components import R


@R.register("clear_store_step")
class ClearStoreStep(BaseStep):
    """Wipe ``file_store`` so ``init_changes_step(store=file_store)`` sees all files as added."""

    async def execute(self):
        assert self.context is not None
        await self.file_store.clear()
        self.context.response.metadata["cleared_store"] = True
        self.logger.info(f"[{self.name}] cleared file_store")
        return self.context.response
