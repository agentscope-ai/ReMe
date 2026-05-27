"""Wipe the file store and emit every vault file as an ``added`` change.

Designed to be chained before ``update_index_step`` so that the second step
performs the actual re-indexing and persistence.
"""

from ..base_step import BaseStep
from ...components import R


@R.register("clear_and_scan_step")
class ClearAndScanStep(BaseStep):
    """Clear the file store, walk the vault, and write changes into the context."""

    async def execute(self):
        assert self.context is not None
        suffixes = tuple("." + s.strip(".") for s in self.context.get("suffix_filters", ["md"]))

        await self.file_store.clear()
        paths = [
            str(p.absolute())
            for p in self.vault_path.rglob("*")
            if p.is_file() and (not suffixes or str(p).endswith(suffixes))
        ]

        self.context["changes"] = [{"change": "added", "path": p} for p in paths]
        counts = {"added": len(paths), "modified": 0, "deleted": 0}
        self.context.response.metadata["counts"] = counts
        self.logger.info(f"[{self.name}] cleared store and scanned {len(paths)} file(s)")
        return self.context.response
