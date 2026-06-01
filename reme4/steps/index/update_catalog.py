"""Update file catalog with a batch of file changes."""

from pathlib import Path

from watchfiles import Change

from ..base_step import BaseStep, Ref
from ...components import R
from ...components.file_catalog import BaseFileCatalog
from ...enumeration import ComponentEnum
from ...schema import FileNode


@R.register("update_catalog_step")
class UpdateCatalogStep(BaseStep):
    """Classify raw watcher changes and update the file_catalog."""

    file_catalog: BaseFileCatalog = Ref(BaseFileCatalog, ComponentEnum.FILE_CATALOG)

    async def execute(self):
        assert self.context is not None
        # Each item: {"change": Change | "added"|"modified"|"deleted", "path": absolute path}
        changes: list[dict] = self.context.get("changes") or []
        persist: bool = bool(self.context.get("persist", False))

        buckets: dict[Change, list[str]] = {Change.added: [], Change.modified: [], Change.deleted: []}
        for item in changes:
            c = item["change"]
            if isinstance(c, str):
                c = Change.__members__.get(c)
            if isinstance(c, Change) and c in buckets:
                buckets[c].append(item["path"])

        results: list[dict] = []

        for change, action in ((Change.added, "Adding"), (Change.modified, "Updating")):
            paths = buckets[change]
            if not paths:
                continue
            self.logger.info(f"Detected {len(paths)} {change.name} file(s)")
            nodes: list[FileNode] = []
            ok_paths: list[str] = []
            for path in paths:
                abs_path = Path(path)
                if not abs_path.is_file():
                    results.append({"change": change.name, "path": path, "success": False, "error": "not a file"})
                    continue
                self.logger.info(f"{action} file: {path}")
                try:
                    stat = abs_path.stat()
                    nodes.append(FileNode(path=self.to_vault_relative(abs_path), st_mtime=stat.st_mtime))
                    ok_paths.append(path)
                except Exception as e:
                    self.logger.exception(f"Failed to stat {path}")
                    results.append({"change": change.name, "path": path, "success": False, "error": str(e)})
            if nodes:
                try:
                    await self.file_catalog.delete([n.path for n in nodes])
                    await self.file_catalog.upsert(nodes)
                    results.extend({"change": change.name, "path": p, "success": True} for p in ok_paths)
                except Exception as e:
                    self.logger.exception(f"Failed to upsert {len(nodes)} {change.name} file(s)")
                    results.extend(
                        {"change": change.name, "path": p, "success": False, "error": str(e)} for p in ok_paths
                    )

        if deleted := buckets[Change.deleted]:
            if self.file_catalog is None:
                raise RuntimeError("file_catalog is not initialized!")
            self.logger.info(f"Detected {len(deleted)} deleted file(s)")
            rel_deleted = [self.to_vault_relative(p) for p in deleted]
            try:
                await self.file_catalog.delete(rel_deleted)
                results.extend({"change": "deleted", "path": p, "success": True} for p in deleted)
            except Exception as e:
                self.logger.exception(f"Failed to delete {len(deleted)} file(s)")
                results.extend({"change": "deleted", "path": p, "success": False, "error": str(e)} for p in deleted)

        if persist and results:
            await self.file_catalog.dump()

        self.context.response.answer = results
        self.context.response.success = all(r["success"] for r in results) if results else True
        return self.context.response
