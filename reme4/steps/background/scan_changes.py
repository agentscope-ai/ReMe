"""One-shot scan: diff watch_paths vs file_store and emit changes for indexing."""

from pathlib import Path

from ..base_step import BaseStep
from ...components import R


@R.register("scan_changes_step")
class ScanChangesStep(BaseStep):
    """One-shot scan: compute added/modified/deleted vs file_store and dispatch."""

    def __init__(
        self,
        recursive: bool = True,
        persist: bool = False,
        dispatch_job: str = "",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.recursive: bool = recursive
        self.persist: bool = persist
        self.dispatch_job: str = dispatch_job

    async def execute(self):
        assert self.context is not None
        if self.file_store is None:
            raise RuntimeError("file_store is not initialized!")

        raw: list[str] = self.context.get("watch_paths", [])
        suffixes: list[str] = self.context.get("suffix_filters", ["md"])
        vault_path = self.vault_path

        paths = [raw] if isinstance(raw, str) else raw
        watch_paths = [vault_path / x for x in paths if (vault_path / x).exists()]

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
            str(Path(n.path) if Path(n.path).is_absolute() else vault_path / n.path): n.st_mtime
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
            self.logger.info(f"[{self.name}] scan: {counts}")
            if self.dispatch_job:
                await self.run_job(
                    self.dispatch_job,
                    changes=changes,
                    persist=self.persist,
                )
        else:
            self.logger.info(f"[{self.name}] store is up to date")

        self.context.response.metadata["counts"] = counts
        return self.context.response
