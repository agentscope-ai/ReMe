"""Update index with a batch of file changes."""

from pathlib import Path

from watchfiles import Change

from ..base_step import BaseStep
from ...components import R
from ...schema import FileChunk, FileNode


@R.register("update_index_step")
class UpdateIndexStep(BaseStep):
    """Classify raw watcher changes and update the file_store index."""

    def __init__(self, persist: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.persist: bool = persist

    async def execute(self):
        assert self.context is not None
        # Each item: {"change": Change | "added"|"modified"|"deleted", "path": absolute path}
        changes: list[dict] = self.context.get("changes") or []

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
            parsed: list[tuple[FileNode, list[FileChunk]]] = []
            ok_paths: list[str] = []
            for path in paths:
                abs_path = Path(path)
                if not abs_path.is_file():
                    results.append({"change": change.name, "path": path, "success": False, "error": "not a file"})
                    continue
                self.logger.info(f"{action} file: {path}")
                try:
                    parsed.append(await self.parse_file(abs_path))
                    ok_paths.append(path)
                except Exception as e:
                    self.logger.exception(f"Failed to parse {path}")
                    results.append({"change": change.name, "path": path, "success": False, "error": str(e)})
            if parsed:
                try:
                    await self.file_store.delete([n.path for n, _ in parsed])
                    await self.file_store.upsert(parsed)
                    results.extend({"change": change.name, "path": p, "success": True} for p in ok_paths)
                except Exception as e:
                    self.logger.exception(f"Failed to persist {len(parsed)} {change.name} file(s)")
                    results.extend(
                        {"change": change.name, "path": p, "success": False, "error": str(e)} for p in ok_paths
                    )

        if deleted := buckets[Change.deleted]:
            if self.file_store is None:
                raise RuntimeError("file_store is not initialized!")
            self.logger.info(f"Detected {len(deleted)} deleted file(s)")
            rel_deleted: list[str] = []
            for path in deleted:
                p = Path(path).absolute()
                try:
                    rel_deleted.append(str(p.relative_to(self.vault_path)))
                except ValueError:
                    rel_deleted.append(str(p))
            try:
                await self.file_store.delete(rel_deleted)
                results.extend({"change": "deleted", "path": p, "success": True} for p in deleted)
            except Exception as e:
                self.logger.exception(f"Failed to delete {len(deleted)} file(s)")
                results.extend({"change": "deleted", "path": p, "success": False, "error": str(e)} for p in deleted)

        if self.persist and results:
            await self.file_store.dump()

        self.context.response.answer = results
        self.context.response.success = all(r["success"] for r in results) if results else True
        return self.context.response
