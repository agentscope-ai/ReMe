"""Wipe the file store and rebuild it by scanning the vault from disk."""

from ..base_step import BaseStep
from ...components import R


@R.register("reindex_step")
class ReindexStep(BaseStep):
    """Full re-index: clear store, walk vault, hand the file list to index_changes."""

    async def execute(self):
        assert self.context is not None

        suffix_filters: list[str] = self.context.get("suffix_filters", ["md"])
        suffixes = tuple("." + s.strip(".") for s in suffix_filters) if suffix_filters else None

        await self.file_store.clear()

        paths: list[str] = []
        for p in self.vault_path.rglob("*"):
            if not p.is_file():
                continue
            if suffixes and not str(p).endswith(suffixes):
                continue
            paths.append(str(p.absolute()))

        if paths:
            await self.run_job("index_changes", changes=[{"change": "added", "path": p} for p in paths])
            await self.file_store.dump()

        counts = {"added": len(paths), "modified": 0, "deleted": 0}
        self.logger.info(f"[{self.name}] reindexed {counts}")
        self.context.response.answer = f"🔄 Reindexed {counts['added']} file(s)"
        self.context.response.metadata["counts"] = counts
        return self.context.response
