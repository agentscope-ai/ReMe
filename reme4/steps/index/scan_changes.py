"""One-shot scan: diff watch_paths vs an indexed-state source, write changes into context.

Two symmetric variants — pick the one whose state source matches the loop's
write target so the index loop and the dream loop never contend on the
same component:

* :class:`ScanStoreChangesStep` (``scan_store_changes_step``) — diffs
  against ``file_store``; used by ``update_store_index_loop`` (sole
  writer of ``file_store``).
* :class:`ScanCatalogChangesStep` (``scan_catalog_changes_step``) — diffs
  against ``file_catalog``; used by ``auto_dream_loop`` (sole writer of
  ``file_catalog``).

Both share the same diff vocabulary (``added`` / ``modified`` / ``deleted``)
and write into ``context["changes"]`` in the same shape, so downstream
steps don't care which variant produced the batch.
"""

from pathlib import Path
from typing import Iterable

from ..base_step import BaseStep, Ref
from ...components import R
from ...components.file_catalog import BaseFileCatalog
from ...enumeration import ComponentEnum
from ...schema import FileNode


def _collect_existing(
    raw: list[str] | str,
    suffixes: list[str],
    vault_path: Path,
    recursive: bool,
) -> dict[str, float]:
    """Walk watch_paths under ``vault_path`` and return ``{abs_path: st_mtime}``."""
    paths = [raw] if isinstance(raw, str) else raw
    watch_paths = [vault_path / x for x in paths if (vault_path / x).exists()]

    existing: dict[str, float] = {}
    for path in watch_paths:
        candidates = [path] if path.is_file() else (path.rglob("*") if recursive else path.iterdir())
        for p in candidates:
            if not p.is_file():
                continue
            if suffixes and not any(str(p).endswith("." + s.strip(".")) for s in suffixes):
                continue
            abs_p = p.absolute()
            existing[str(abs_p)] = abs_p.stat().st_mtime
    return existing


def _diff(
    existing: dict[str, float],
    nodes: Iterable[FileNode],
    vault_path: Path,
) -> tuple[list[dict], dict[str, int]]:
    """Compute added/modified/deleted vs ``nodes`` and return (changes, counts)."""
    indexed: dict[str, float] = {
        str(Path(n.path) if Path(n.path).is_absolute() else vault_path / n.path): n.st_mtime for n in nodes
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
    return changes, counts


class _ScanChangesBase(BaseStep):
    """Shared scaffolding: collect on-disk state, defer node-loading to the subclass."""

    def __init__(self, recursive: bool = True, **kwargs):
        super().__init__(**kwargs)
        self.recursive: bool = recursive

    async def _load_indexed_nodes(self) -> Iterable[FileNode]:
        raise NotImplementedError

    async def execute(self):
        assert self.context is not None
        vault_path = self.vault_path
        raw: list[str] = self.context.get("watch_paths", []) or []
        suffixes: list[str] = self.context.get("suffix_filters", ["md"]) or ["md"]

        existing = _collect_existing(
            raw=raw,
            suffixes=suffixes,
            vault_path=vault_path,
            recursive=self.recursive,
        )

        nodes = await self._load_indexed_nodes()
        changes, counts = _diff(existing, nodes, vault_path)

        self.context["changes"] = changes
        if changes:
            self.logger.info(f"[{self.name}] scan: {counts}")
        else:
            self.logger.info(f"[{self.name}] store is up to date")

        self.context.response.metadata["counts"] = counts
        return self.context.response


@R.register("scan_store_changes_step")
class ScanStoreChangesStep(_ScanChangesBase):
    """Diff vault against ``file_store``; used by the index loop."""

    async def _load_indexed_nodes(self) -> Iterable[FileNode]:
        if self.file_store is None:
            raise RuntimeError("file_store is not initialized!")
        return await self.file_store.get_nodes()


@R.register("scan_catalog_changes_step")
class ScanCatalogChangesStep(_ScanChangesBase):
    """Diff vault against ``file_catalog``; used by the dream loop."""

    file_catalog: BaseFileCatalog = Ref(BaseFileCatalog, ComponentEnum.FILE_CATALOG)

    async def _load_indexed_nodes(self) -> Iterable[FileNode]:
        return await self.file_catalog.get_nodes()
