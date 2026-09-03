"""Apply filesystem change batches to the file-level tag index."""

import asyncio
from pathlib import Path

from watchfiles import Change

from ._change_batch import bucket_changes
from ._tag_index_io import read_frontmatter, relative_deleted_path, relative_existing_path, validated_watch_rules
from ..base_step import BaseStep, Ref
from ...components import R
from ...components.tag_index import BaseTagIndex
from ...enumeration import ComponentEnum
from ...schema import TagSourceRecord


@R.register("update_tag_index_step")
class UpdateTagIndexStep(BaseStep):
    """Read bounded frontmatter and update tag relationships for one batch."""

    tag_index: BaseTagIndex = Ref(BaseTagIndex, ComponentEnum.TAG_INDEX)

    def __init__(self, persist: bool | None = None, **kwargs):
        super().__init__(**kwargs)
        self.persist = persist

    async def execute(self):
        if self.context is None:
            raise RuntimeError("update_tag_index_step requires context")
        app_config = self.app_context.app_config if self.app_context else None
        rules = validated_watch_rules(app_config, self.workspace_path, self.context)
        changes: list[dict] = self.context.get("changes") or []
        buckets = bucket_changes(changes, path_exists=self._exists)
        records: list[TagSourceRecord] = []
        deleted: list[str] = []
        results: list[dict] = []

        for change in (Change.added, Change.modified):
            for raw_path in buckets[change]:
                try:
                    path, relative = relative_existing_path(raw_path, self.workspace_path, rules)
                except (OSError, ValueError) as exc:
                    results.append(self._skipped(change.name, str(raw_path), type(exc).__name__))
                    continue
                read = await asyncio.to_thread(read_frontmatter, path, self.tag_index.max_frontmatter_bytes)
                if read.status in ("io_failed", "changed_during_read"):
                    results.append(self._skipped(change.name, relative, read.status))
                    continue
                metadata = read.metadata or {}
                records.append(
                    TagSourceRecord(
                        path=relative,
                        mtime_ns=read.mtime_ns if read.mtime_ns is not None else path.stat().st_mtime_ns,
                        tags=self.tag_index.normalize_tags(metadata.get("tags")),
                    ),
                )
                results.append({"change": change.name, "path": relative, "success": True, "skipped": False})

        for raw_path in buckets[Change.deleted]:
            try:
                relative = relative_deleted_path(raw_path, self.workspace_path, rules)
            except ValueError as exc:
                results.append(self._skipped("deleted", str(raw_path), type(exc).__name__))
                continue
            deleted.append(relative)
            results.append({"change": "deleted", "path": relative, "success": True, "skipped": False})

        persist = bool(self.context.get("persist", True)) if self.persist is None else self.persist
        if records or deleted:
            await self.tag_index.reconcile(records, deleted, persist=persist)
        counts = {
            "added": sum(item["change"] == "added" and not item["skipped"] for item in results),
            "modified": sum(item["change"] == "modified" and not item["skipped"] for item in results),
            "deleted": sum(item["change"] == "deleted" and not item["skipped"] for item in results),
            "audited": len(records),
            "skipped": sum(item["skipped"] for item in results),
        }
        self.context.response.answer = results
        self.context.response.metadata["counts"] = counts
        self.context.response.success = True
        return self.context.response

    def _exists(self, path: str) -> bool:
        candidate = Path(path)
        if not candidate.is_absolute():
            candidate = self.workspace_path / candidate
        return candidate.is_file()

    @staticmethod
    def _skipped(change: str, path: str, reason: str) -> dict:
        return {"change": change, "path": path, "success": True, "skipped": True, "reason": reason}
