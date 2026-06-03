"""Long-running awatch loop: convert raw changes into dispatch_step calls.

Two relevant awatch parameters are exposed verbatim:

* ``step`` (default ``50ms``) — awatch yields when the entire watcher
  has gone this long without new changes (and at least one change is
  pending). Raise to ``5 minutes``-ish for ``auto_dream_loop`` so
  half-written sync output isn't dreamed mid-write; keep at default
  for ``update_store_index_loop`` where every fs change should hit
  the index promptly.

* ``debounce`` (default ``2000ms``) — per-batch ceiling, regardless
  of whether activity is still arriving. Set ``debounce > step`` so
  ``step`` is the operative limit; otherwise the watcher pre-empts
  long-quiet-window setups under bursty writes.

Both are global to the watcher (not per-path). The two reme watchers
have disjoint ``watch_paths`` (digest vs daily/resource), so global
quiet windows are good enough — no per-path bookkeeping needed.

awatch internally deduplicates same-path same-change tuples within
the yielded batch, so a file ``modified`` ten times during the
quiet window arrives as one ``(modified, path)`` entry.
"""

import asyncio

from watchfiles import Change, awatch

from ..base_step import BaseStep
from ...components import R, BaseComponent
from ...enumeration import ComponentEnum


@R.register("watch_changes_step")
class WatchChangesStep(BaseStep):
    """Watch files and forward each yielded batch to a downstream step."""

    def __init__(
        self,
        recursive: bool = True,
        force_polling: bool = True,
        debounce: int = 2000,
        step: int = 50,
        poll_delay_ms: int = 2000,
        dispatch_step: str = "",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.recursive: bool = recursive
        self.force_polling: bool = force_polling
        self.debounce: int = debounce
        self.step: int = step
        self.poll_delay_ms: int = poll_delay_ms
        self.dispatch_step: str = dispatch_step

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

        dispatch_step_cls: type[BaseComponent] | None = None
        if self.dispatch_step:
            dispatch_step_cls = R.get(ComponentEnum.STEP, self.dispatch_step)
            if dispatch_step_cls is None:
                raise RuntimeError(f"Unregistered step '{self.dispatch_step}'")

        self.logger.info(
            f"Watching: {[str(p) for p in valid_paths]} step={self.step}ms debounce={self.debounce}ms",
        )

        async for raw_changes in awatch(
            *valid_paths,
            watch_filter=self._filter,
            recursive=self.recursive,
            force_polling=self.force_polling,
            debounce=self.debounce,
            step=self.step,
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
                if dispatch_step_cls is not None:
                    step = dispatch_step_cls(app_context=self.app_context)
                    await step(changes=changes)

        return self.context.response
