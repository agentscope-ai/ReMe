"""Long-running awatch loop: convert raw changes into index_changes calls."""

import asyncio

from watchfiles import Change, awatch

from ..base_step import BaseStep
from ...components import R


@R.register("watch_changes_step")
class WatchChangesStep(BaseStep):
    """Watch files and forward each batch of raw changes to the index_changes job."""

    def __init__(
        self,
        recursive: bool = True,
        force_polling: bool = True,
        debounce: int = 2000,
        poll_delay_ms: int = 2000,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.recursive: bool = recursive
        self.force_polling: bool = force_polling
        self.debounce: int = debounce
        self.poll_delay_ms: int = poll_delay_ms

    def _filter(self, _change: Change, path: str) -> bool:
        suffixes = (self.context.get("suffix_filters") if self.context else None) or ["md"]
        return not suffixes or any(path.endswith("." + s.strip(".")) for s in suffixes)

    async def execute(self):
        if self.context is None:
            raise RuntimeError("watch_changes_step requires 'context'")
        if self.context.stop_event is None:
            raise RuntimeError("watch_changes_step requires 'stop_event' on context")
        stop_event: asyncio.Event = self.context.stop_event

        raw = self.context.get("watch_paths", [])
        paths = [raw] if isinstance(raw, str) else raw
        valid_paths = [self.vault_path / x for x in paths if (self.vault_path / x).exists()]
        if not valid_paths:
            raise RuntimeError(f"No valid watch paths under {self.vault_path}: {paths}")

        self.logger.info(f"Watching: {[str(p) for p in valid_paths]}")
        async for raw_changes in awatch(
            *valid_paths,
            watch_filter=self._filter,
            recursive=self.recursive,
            force_polling=self.force_polling,
            debounce=self.debounce,
            poll_delay_ms=self.poll_delay_ms,
            stop_event=stop_event,
        ):
            if stop_event.is_set():
                break
            changes = [
                {"change": c.name, "path": p}
                for c, p in raw_changes
                if c in (Change.added, Change.modified, Change.deleted)
            ]
            if changes:
                self.logger.info(f"Detected {len(changes)} change(s)")
                await self.run_job("index_changes", changes=changes)

        return self.context.response
