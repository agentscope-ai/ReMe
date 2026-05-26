"""Wipe the file store and rebuild it by scanning the vault from disk."""

from ..base_step import BaseStep
from ...components import R


@R.register("reindex_step")
class ReindexStep(BaseStep):
    """Full re-index: clear the store, walk the vault, hand the file list to ``index_changes``."""

    async def execute(self):
        assert self.context is not None
        suffixes = tuple("." + s.strip(".") for s in self.context.get("suffix_filters", ["md"]))

        await self.file_store.clear()
        paths = [
            str(p.absolute())
            for p in self.vault_path.rglob("*")
            if p.is_file() and (not suffixes or str(p).endswith(suffixes))
        ]

        if paths:
            await self.run_job("index_changes", changes=[{"change": "added", "path": p} for p in paths])
            await self.file_store.dump()

        counts = {"added": len(paths), "modified": 0, "deleted": 0}
        self.logger.info(f"[{self.name}] reindexed {counts}")
        self.context.response.answer = f"🔄 Reindexed {counts['added']} file(s)"
        self.context.response.metadata["counts"] = counts
        return self.context.response
