"""One-shot scan: diff watch_paths vs an indexed-state source, write changes into context.

Two symmetric variants — pick the one whose state source matches the loop's
write target so the index loop and the dream loop never contend on the
same component:

* :class:`ScanStoreChangesStep` (``scan_store_changes_step``) — diffs
  against ``file_store``; used by ``index_update_loop`` (sole writer of
  ``file_store``).
* :class:`ScanCatalogChangesStep` (``scan_catalog_changes_step``) — diffs
  against ``file_catalog``; used by ``resource_watch_loop`` and
  ``digest_watch_loop`` (sole writers of ``file_catalog``).

Both share the same diff vocabulary (``added`` / ``modified`` / ``deleted``)
and write into ``context["changes"]`` in the same shape, so downstream
steps don't care which variant produced the batch.
"""

from pathlib import Path
from typing import Iterable

from ._watch_rules import WatchRule, build_watch_rules, collect_existing
from ..base_step import BaseStep, Ref
from ...components import R
from ...components.file_catalog import BaseFileCatalog
from ...enumeration import ComponentEnum
from ...schema import FileNode


def _diff(existing: dict[str, float], nodes: Iterable[FileNode], vault_path: Path) -> tuple[list[dict], dict[str, int]]:
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

    def _get_watch_rules(self) -> list[WatchRule]:
        """Build watch rules from context-level watch_dirs/watch_suffixes."""
        assert self.context is not None
        app_config = self.app_context.app_config if self.app_context else None
        if app_config is None:
            return []
        watch_dirs: list[str] = self.context.get("watch_dirs", [])
        watch_suffixes: list[str] = self.context.get("watch_suffixes", [])
        if not watch_dirs:
            return []
        return build_watch_rules(app_config, self.vault_path, watch_dirs=watch_dirs, watch_suffixes=watch_suffixes)

    async def _load_indexed_nodes(self) -> Iterable[FileNode]:
        raise NotImplementedError

    async def execute(self):
        assert self.context is not None
        rules = self._get_watch_rules()
        existing = collect_existing(rules, recursive=self.recursive)
        nodes = await self._load_indexed_nodes()
        changes, counts = _diff(existing, nodes, self.vault_path)
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
    """Diff vault against ``file_catalog``; used by resource/digest loops."""

    async def _load_indexed_nodes(self) -> Iterable[FileNode]:
        return await self.file_catalog.get_nodes()
