"""One-shot scan: diff watch_paths vs a state source, write changes into context.

Source defaults to ``file_store`` (used by ``update_store_index_loop``) but can
be set to ``file_catalog`` so a sibling loop (e.g. ``auto_dream_loop``) can
maintain its own independent state without contending with the index writer.
"""

from pathlib import Path

from ..base_step import BaseStep, Ref
from ...components import R
from ...components.file_catalog import BaseFileCatalog
from ...enumeration import ComponentEnum


@R.register("scan_changes_step")
class ScanChangesStep(BaseStep):
    """One-shot scan: compute added/modified/deleted vs the chosen source."""

    file_catalog: BaseFileCatalog = Ref(BaseFileCatalog, ComponentEnum.FILE_CATALOG, optional=True)

    def __init__(self, recursive: bool = True, source: str = "file_store", **kwargs):
        super().__init__(**kwargs)
        self.recursive: bool = recursive
        if source not in ("file_store", "file_catalog"):
            raise ValueError(f"source must be 'file_store' or 'file_catalog', got {source!r}")
        self.source: str = source

    async def execute(self):
        assert self.context is not None

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

        if self.source == "file_catalog":
            if self.file_catalog is None:
                raise RuntimeError("file_catalog is not configured but source='file_catalog'")
            nodes = await self.file_catalog.get_nodes()
        else:
            if self.file_store is None:
                raise RuntimeError("file_store is not initialized!")
            nodes = await self.file_store.get_nodes()

        indexed: dict[str, float] = {
            str(Path(n.path) if Path(n.path).is_absolute() else vault_path / n.path): n.st_mtime
            for n in nodes
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

        self.context["changes"] = changes
        if changes:
            self.logger.info(f"[{self.name}] scan: {counts}")
        else:
            self.logger.info(f"[{self.name}] store is up to date")

        self.context.response.metadata["counts"] = counts
        return self.context.response
