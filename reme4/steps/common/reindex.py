"""Wipe the file store and rebuild it from the watcher's tracked files."""

from ..base_step import BaseStep
from ...components import R


@R.register("reindex_step")
class ReindexStep(BaseStep):
    """Full re-index: stop watcher, clear store, sync from disk, then restart."""

    async def execute(self):
        assert self.context is not None

        await self.file_watcher.close()
        try:
            await self.file_store.clear()
            counts = await self.file_watcher.update_store()
        finally:
            await self.file_watcher.start()

        self.logger.info(f"[{self.name}] reindexed {counts}")
        self.context.response.answer = f"🔄 Reindexed {counts['added']} file(s)"
        self.context.response.metadata["counts"] = counts
        return self.context.response
