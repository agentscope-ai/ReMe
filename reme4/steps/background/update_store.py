"""Initial sync: diff watch_paths vs file_store, then index the diff."""

from pathlib import Path

from ..base_step import BaseStep
from ...components import R


@R.register("update_store_step")
class UpdateStoreStep(BaseStep):
    """One-shot sync: compute added/modified/deleted vs file_store and index."""

    def __init__(self, recursive: bool = True, dump: bool = True, **kwargs):
        super().__init__(**kwargs)
        self.recursive: bool = recursive
        self.dump: bool = dump

    async def execute(self):
        assert self.context is not None
        if self.file_store is None:
            raise RuntimeError("file_store is not initialized!")

        raw: list[str] = self.context.get("watch_paths", [])
        suffixes: list[str] = self.context.get("suffix_filters", ["md"])
        working_path = self.working_path

        paths = [raw] if isinstance(raw, str) else raw
        watch_paths = [working_path / x for x in paths if (working_path / x).exists()]

        existing: dict[str, float] = {}
        for path in watch_paths:
            candidates = [path] if path.is_file() else (path.rglob("*") if self.recursive else path.iterdir())
            for p in candidates:
                if not p.is_file():
                    continue
                if suffixes and not any(str(p).endswith("." + s.strip(".")) for s in suffixes):
                    continue
                abs_p = p.absolute()
                existing[str(abs_p)] = abs_p.stat().st_mtime

        indexed: dict[str, float] = {
            str(Path(n.path) if Path(n.path).is_absolute() else working_path / n.path): n.st_mtime
            for n in await self.file_store.get_nodes()
        }

        to_delete = list(indexed.keys() - existing.keys())
        to_add = list(existing.keys() - indexed.keys())
        to_modify = [p for p in existing.keys() & indexed.keys() if existing[p] != indexed[p]]

        changes: list[dict] = (
            [{"change": "added", "path": p} for p in to_add]
            + [{"change": "modified", "path": p} for p in to_modify]
            + [{"change": "deleted", "path": p} for p in to_delete]
        )
        counts = {"added": len(to_add), "modified": len(to_modify), "deleted": len(to_delete)}

        if changes:
            self.logger.info(f"[{self.name}] initial sync: {counts}")
            await self.run_job("index_changes", changes=changes)
            if self.dump:
                await self.file_store.dump()
        else:
            self.logger.info(f"[{self.name}] store is up to date")

        self.context.response.metadata["counts"] = counts
        return self.context.response
