"""Coordinated tag-index startup audit and live watcher."""

import asyncio
import contextlib

from watchfiles import Change, awatch

from ._change_batch import coalesce_changes
from ._tag_index_io import (
    read_frontmatter,
    scan_watch_scope,
    validated_watch_rules,
)
from ._watch_rules import match_file
from .watch_changes import DEFAULT_LOW_POWER_POLL_MS, DEFAULT_WATCH_DEBOUNCE_MS, DEFAULT_WATCH_STEP_MS
from ..base_step import BaseStep, Ref
from ...components import R
from ...components.tag_index import BaseTagIndex
from ...enumeration import ComponentEnum
from ...schema import TagSourceRecord


@R.register("tag_index_watch_step")
class TagIndexWatchStep(BaseStep):
    """Audit/rebuild, close the watcher startup window, then consume events."""

    tag_index: BaseTagIndex = Ref(BaseTagIndex, ComponentEnum.TAG_INDEX)

    def __init__(
        self,
        recursive: bool = True,
        force_polling: bool = True,
        debounce: int = DEFAULT_WATCH_DEBOUNCE_MS,
        step: int = DEFAULT_WATCH_STEP_MS,
        poll_delay_ms: int = DEFAULT_LOW_POWER_POLL_MS,
        rust_timeout: int = 5_000,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.recursive = recursive
        self.force_polling = force_polling
        self.debounce = debounce
        self.step = step
        self.poll_delay_ms = poll_delay_ms
        self.rust_timeout = rust_timeout
        self._rules = []

    async def execute(self):
        if self.context is None or self.context.stop_event is None:
            raise RuntimeError("tag_index_watch_step requires context with stop_event")
        app_config = self.app_context.app_config if self.app_context else None
        self._rules = validated_watch_rules(app_config, self.workspace_path, self.context)
        snapshot_a = scan_watch_scope(self.workspace_path, self._rules, self.recursive)
        startup_results = await self._initial_sync(snapshot_a)

        queue: asyncio.Queue[set[tuple[Change, str]]] = asyncio.Queue()
        ready = asyncio.Event()
        collector = asyncio.create_task(self._collect(queue, ready))
        try:
            ready_wait = asyncio.create_task(ready.wait())
            stop_wait = asyncio.create_task(self.context.stop_event.wait())
            done, pending = await asyncio.wait(
                {ready_wait, stop_wait, collector},
                return_when=asyncio.FIRST_COMPLETED,
            )
            for task in pending:
                if task is not collector:
                    task.cancel()
            if collector in done:
                await collector
                raise RuntimeError("tag-index watcher stopped before becoming ready")
            if stop_wait in done:
                return self.context.response

            snapshot_b = scan_watch_scope(self.workspace_path, self._rules, self.recursive)
            delta = self._snapshot_delta(snapshot_a, snapshot_b)
            if delta:
                await self.dispatch_steps(self.dispatch_step_specs, changes=delta)
            buffered = self._drain(queue)
            if buffered:
                await self.dispatch_steps(self.dispatch_step_specs, changes=buffered)

            self.context.response.metadata["startup_counts"] = self._counts(startup_results)
            while not self.context.stop_event.is_set():
                get_task = asyncio.create_task(queue.get())
                stop_task = asyncio.create_task(self.context.stop_event.wait())
                done, pending = await asyncio.wait(
                    {get_task, stop_task, collector},
                    return_when=asyncio.FIRST_COMPLETED,
                )
                for task in pending:
                    if task is not collector:
                        task.cancel()
                if stop_task in done:
                    break
                if collector in done:
                    await collector
                    raise RuntimeError("tag-index watcher stopped unexpectedly")
                raw_batches = [get_task.result()]
                while not queue.empty():
                    raw_batches.append(queue.get_nowait())
                changes = self._raw_changes(raw_batches)
                if changes:
                    await self.dispatch_steps(self.dispatch_step_specs, changes=changes)
            return self.context.response
        finally:
            collector.cancel()
            with contextlib.suppress(BaseException):
                await collector

    async def _initial_sync(self, current: dict[str, tuple[object, int]]) -> list[dict]:
        indexed = {record.path: record for record in await self.tag_index.get_records()}
        rebuild = not bool(getattr(self.tag_index, "loaded", False))
        records: list[TagSourceRecord] = []
        results: list[dict] = []
        transient_failures: list[str] = []
        for relative, (path, _mtime_ns) in sorted(current.items()):
            read = await asyncio.to_thread(read_frontmatter, path, self.tag_index.max_frontmatter_bytes)
            if read.status in ("io_failed", "changed_during_read"):
                transient_failures.append(relative)
                results.append(
                    {
                        "change": "modified",
                        "path": relative,
                        "success": not rebuild,
                        "skipped": True,
                        "reason": read.status,
                    },
                )
                continue
            normalized_tags = self.tag_index.normalize_tags((read.metadata or {}).get("tags"))
            records.append(
                TagSourceRecord(
                    path=relative,
                    mtime_ns=read.mtime_ns,
                    tags=normalized_tags,
                ),
            )
            previous = indexed.get(relative)
            if previous is None:
                change = "added"
            elif previous.mtime_ns != read.mtime_ns or previous.tags != normalized_tags:
                change = "modified"
            else:
                change = "audited"
            results.append({"change": change, "path": relative, "success": True, "skipped": False})
        if rebuild and transient_failures:
            self.context.response.answer = results
            self.context.response.metadata["counts"] = self._counts(results)
            self.context.response.success = False
            paths = ", ".join(transient_failures[:3])
            if len(transient_failures) > 3:
                paths += f", ... ({len(transient_failures)} total)"
            raise RuntimeError(f"tag-index cold rebuild deferred after transient read failures: {paths}")
        deleted = sorted(set(indexed) - set(current))
        results.extend({"change": "deleted", "path": path, "success": True, "skipped": False} for path in deleted)
        await self.tag_index.reconcile(records, deleted, rebuild=rebuild, persist=True)
        self.context.response.answer = results
        self.context.response.metadata["counts"] = self._counts(results)
        self.context.response.success = True
        return results

    async def _collect(self, queue: asyncio.Queue, ready: asyncio.Event) -> None:
        paths = list(dict.fromkeys(rule.path for rule in self._rules))
        async for raw in awatch(
            *paths,
            watch_filter=lambda change, path: match_file(path, self._rules),
            recursive=self.recursive,
            force_polling=self.force_polling,
            debounce=self.debounce,
            step=self.step,
            poll_delay_ms=self.poll_delay_ms,
            rust_timeout=self.rust_timeout,
            yield_on_timeout=True,
            stop_event=self.context.stop_event,
        ):
            if raw:
                await queue.put(raw)
            ready.set()
            if self.context.stop_event.is_set():
                break

    def _snapshot_delta(self, before, after) -> list[dict]:
        changes = []
        for path in sorted(set(after) - set(before)):
            changes.append({"change": "added", "path": str(after[path][0])})
        for path in sorted(set(before) - set(after)):
            changes.append({"change": "deleted", "path": str(before[path][0])})
        for path in sorted(set(before) & set(after)):
            if before[path][1] != after[path][1]:
                changes.append({"change": "modified", "path": str(after[path][0])})
        return changes

    def _drain(self, queue: asyncio.Queue) -> list[dict]:
        batches = []
        while not queue.empty():
            batches.append(queue.get_nowait())
        return self._raw_changes(batches)

    def _raw_changes(self, batches) -> list[dict]:
        changes = [
            {"change": change.name, "path": path}
            for batch in batches
            for change, path in batch
            if change in (Change.added, Change.modified, Change.deleted)
        ]
        return coalesce_changes(changes)

    @staticmethod
    def _counts(results: list[dict]) -> dict[str, int]:
        return {
            "added": sum(item["change"] == "added" and not item["skipped"] for item in results),
            "modified": sum(item["change"] == "modified" and not item["skipped"] for item in results),
            "deleted": sum(item["change"] == "deleted" and not item["skipped"] for item in results),
            "audited": sum(item["change"] != "deleted" and not item["skipped"] for item in results),
            "skipped": sum(item["skipped"] for item in results),
        }
