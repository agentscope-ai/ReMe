"""DreamStep — dream a batch of file changes; track state in file_catalog.

Dream-loop is the *write side* of digest authoring; index-loop is the *read
side* of the vault. They share neither state nor write target:

* ``update_store_index_loop`` (sole writer of ``file_store``) parses every
  vault file and persists chunks/links so search and traversal work.
* ``auto_dream_loop`` (sole writer of ``file_catalog``) runs ``dream_one``
  per change, then upserts the path's ``(rel, st_mtime)`` into the catalog
  so the next ``scan_changes_step`` (with ``source: file_catalog``) only
  re-dreams files whose mtime changed since their last successful dream.
  Failures leave the catalog untouched — the change is re-reported on the
  next scan and we'll retry until success.

The two loops cannot trample each other because they touch disjoint
storage: ``file_store`` writes only happen inside the index loop;
``file_catalog`` writes only happen here. The previous design coupled the
dream loop to ``UpdateIndexStep`` and got the predictable race on
``local_file_store.dump()``'s tmp→jsonl rename — that is now structurally
impossible.

Inputs (RuntimeContext):
    changes (list[dict]): each ``{"change": "added"|"modified"|"deleted",
        "path": <abs_path>}``. Both ``ScanChangesStep`` (string change names)
        and ``WatchChangesStep`` (``watchfiles.Change`` enum) are tolerated;
        any path outside the vault root is silently dropped.
"""

from pathlib import Path

from watchfiles import Change

from .auto_dream import Dreamer
from ..base_step import Ref
from ...components import R
from ...components.file_catalog import BaseFileCatalog
from ...enumeration import ComponentEnum
from ...schema import FileNode


@R.register("dream_step")
class DreamStep(Dreamer):
    """Process ``context.changes`` — dream + record state in file_catalog.

    Step kwargs (from yaml ``backend: dream_step``):
        persist (bool, default True): when True, ``file_catalog.dump()`` is
            called after the batch so progress survives a restart.
        hint (str, default ""): forwarded to every ``dream_one`` call.
    """

    file_catalog: BaseFileCatalog = Ref(BaseFileCatalog, ComponentEnum.FILE_CATALOG)

    def __init__(self, persist: bool = True, hint: str = "", **kwargs):
        super().__init__(**kwargs)
        self.persist: bool = persist
        self.hint: str = hint

    async def execute(self):
        assert self.context is not None
        changes: list[dict] = self.context.get("changes") or []

        added_modified: list[tuple[str, str]] = []  # (abs_path, rel_path)
        deleted: list[str] = []  # rel_paths

        for item in changes:
            c = item.get("change")
            if isinstance(c, Change):
                c = c.name
            abs_path = item.get("path")
            if not abs_path:
                continue
            try:
                rel = str(Path(abs_path).absolute().relative_to(self.vault_path))
            except ValueError:
                continue  # outside vault — defensive guard
            if c in ("added", "modified"):
                added_modified.append((abs_path, rel))
            elif c == "deleted":
                deleted.append(rel)

        # Drop catalog entries for deleted files first — cheap, no LLM, and
        # keeps the catalog consistent even if the dream pass below errors.
        if deleted:
            try:
                await self.file_catalog.delete(deleted)
            except Exception as e:  # noqa: BLE001
                self.logger.exception(
                    f"[{self.name}] file_catalog.delete failed: {type(e).__name__}: {e}",
                )

        # Dream then upsert the catalog entry per file. Single-file granularity
        # means an LLM failure on file N doesn't block files N+1..K from
        # advancing their catalog mtime.
        results: list[dict] = []
        upsert_nodes: list[FileNode] = []
        for abs_path, rel in added_modified:
            try:
                outcome = await self.dream_one(rel, self.hint)
            except Exception as e:  # noqa: BLE001
                self.logger.exception(
                    f"[{self.name}] dream_one failed on {rel}: {type(e).__name__}: {e}",
                )
                results.append({"path": rel, "success": False, "error": str(e)})
                continue
            if outcome.error:
                self.logger.error(f"[{self.name}] {rel}: {outcome.error}")
                results.append({"path": rel, "success": False, "error": outcome.error})
                continue

            # Success (including ``skipped``: Phase 1 said "nothing to extract",
            # but we still mark the file as seen so we don't redo Phase 1 on
            # every restart).
            try:
                stat = Path(abs_path).stat()
            except OSError as e:
                self.logger.error(f"[{self.name}] stat failed on {abs_path}: {e}")
                results.append({"path": rel, "success": False, "error": str(e)})
                continue
            upsert_nodes.append(FileNode(path=rel, st_mtime=stat.st_mtime))

            if outcome.skipped:
                self.logger.info(f"[{self.name}] {rel}: skipped (Phase 1 empty), catalogued")
            else:
                self.logger.info(
                    f"[{self.name}] {rel}: +{len(outcome.nodes_created)} created, "
                    f"~{len(outcome.nodes_updated)} updated, catalogued",
                )
            results.append({"path": rel, "success": True})

        if upsert_nodes:
            try:
                await self.file_catalog.upsert(upsert_nodes)
            except Exception as e:  # noqa: BLE001
                self.logger.exception(
                    f"[{self.name}] file_catalog.upsert failed: {type(e).__name__}: {e}",
                )

        if self.persist and (upsert_nodes or deleted):
            try:
                await self.file_catalog.dump()
            except Exception as e:  # noqa: BLE001
                self.logger.exception(
                    f"[{self.name}] file_catalog.dump failed: {type(e).__name__}: {e}",
                )

        self.context.response.answer = results
        self.context.response.metadata.update(
            {
                "deleted": len(deleted),
                "dreamed": sum(1 for r in results if r["success"]),
                "failed": sum(1 for r in results if not r["success"]),
            },
        )
        self.context.response.success = all(r["success"] for r in results) if results else True
        return self.context.response
