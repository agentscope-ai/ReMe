"""Long-running file watcher: convert raw changes into dispatch step calls.

``watch_mode`` defaults to ``awatch``. Use ``watch`` to run watchfiles.watch
on a private thread instead of sharing AnyIO's request-thread limiter.

Relevant awatch parameters are exposed verbatim:

* ``step`` — awatch yields when the entire watcher
  has gone this long without new changes (and at least one change is
  pending). Raise to ``5 minutes``-ish for ``auto_dream_loop`` so
  half-written sync output isn't dreamed mid-write; keep at default
  for ``index_update_loop`` where every fs change should hit
  the index promptly.

* ``debounce`` — per-batch ceiling, regardless
  of whether activity is still arriving. Set ``debounce > step`` so
  ``step`` is the operative limit; otherwise the watcher pre-empts
  long-quiet-window setups under bursty writes.

* ``poll_delay_ms`` — delay between polling scans when
  ``force_polling=True``. This has the most direct effect on idle CPU
  use for the default forced-polling watcher.

These settings are global to the watcher (not per-path). The reme watchers
have disjoint ``watch_dirs`` (configured per job), so global
quiet windows are good enough — no per-path bookkeeping needed.

awatch internally deduplicates same-path same-change tuples within
the yielded batch, so a file ``modified`` ten times during the
quiet window arrives as one ``(modified, path)`` entry.
"""

import asyncio
import threading
from contextlib import aclosing
from typing import Literal

from watchfiles import Change, awatch

from ._change_batch import coalesce_changes
from ._threaded_watch import watch_in_thread
from ._watch_rules import WatchRule, build_context_watch_rules, match_file
from ..base_step import BaseStep
from ...components import R

DEFAULT_WATCH_DEBOUNCE_MS = 5_000
DEFAULT_WATCH_STEP_MS = 1_000
DEFAULT_LOW_POWER_POLL_MS = 5_000


@R.register("watch_changes_step")
class WatchChangesStep(BaseStep):
    """Watch files and forward each yielded batch to downstream steps."""

    def __init__(
        self,
        recursive: bool = True,
        force_polling: bool = True,
        debounce: int = DEFAULT_WATCH_DEBOUNCE_MS,
        step: int = DEFAULT_WATCH_STEP_MS,
        poll_delay_ms: int = DEFAULT_LOW_POWER_POLL_MS,
        watch_mode: Literal["awatch", "watch"] = "awatch",
        monitor_type: Literal["file_store", "file_catalog"] | None = None,
        monitor_name: str = "default",
        **kwargs,
    ):
        if watch_mode not in ("awatch", "watch"):
            raise ValueError("watch_mode must be 'awatch' or 'watch'")
        super().__init__(**kwargs)
        self.watch_mode = watch_mode
        self.recursive: bool = recursive
        self.force_polling: bool = force_polling
        self.debounce: int = debounce
        self.step: int = step
        self.poll_delay_ms: int = poll_delay_ms
        self.monitor_type = monitor_type
        self.monitor_name = monitor_name
        self._rules: list[WatchRule] = []

    def _get_watch_rules(self) -> list[WatchRule]:
        assert self.context is not None
        app_config = self.app_context.app_config if self.app_context else None
        return build_context_watch_rules(app_config, self.workspace_path, self.context)

    def _filter(self, _change: Change, path: str) -> bool:
        return match_file(path, self._rules)

    async def execute(self):
        if self.context is None:
            raise RuntimeError("watch_changes_step requires 'context'")
        if self.context.stop_event is None:
            raise RuntimeError("watch_changes_step requires 'stop_event' on context")
        stop_event: asyncio.Event | threading.Event = self.context.stop_event

        self._rules = self._get_watch_rules()
        if not self._rules:
            raise RuntimeError("No watch rules configured (watch_dirs empty or app_config missing?)")

        valid_paths = list(dict.fromkeys(r.path for r in self._rules if r.path.exists()))
        if not valid_paths:
            raise RuntimeError(f"No valid watch paths exist: {[str(r.path) for r in self._rules]}")

        self.logger.info(
            f"Watching: {[str(p) for p in valid_paths]} "
            f"mode={self.watch_mode} step={self.step}ms debounce={self.debounce}ms "
            f"poll_delay={self.poll_delay_ms}ms",
        )

        watch_fn = awatch if self.watch_mode == "awatch" else watch_in_thread
        watcher = watch_fn(
            *valid_paths,
            watch_filter=self._filter,
            recursive=self.recursive,
            force_polling=self.force_polling,
            debounce=self.debounce,
            step=self.step,
            poll_delay_ms=self.poll_delay_ms,
            stop_event=stop_event,
            yield_on_timeout=self.monitor_type is not None,
        )
        needs_initial_scan = self.monitor_type is not None
        async with aclosing(watcher):
            async for raw_changes in watcher:
                if stop_event.is_set():
                    break
                changes = [
                    {"change": c.name, "path": p}
                    for c, p in raw_changes
                    if c in (Change.added, Change.modified, Change.deleted)
                ]
                if needs_initial_scan:
                    # Subscribe first, then recover writes made after the
                    # preceding scan but before watcher initialization.
                    await self.dispatch_steps(
                        [
                            {
                                "backend": "init_changes_step",
                                "monitor_type": self.monitor_type,
                                "monitor_name": self.monitor_name,
                                "recursive": self.recursive,
                            },
                        ],
                    )
                    # The scan only produces changes. Merge overlapping events
                    # before running handlers, which may have model side effects.
                    changes.extend(self.context.get("changes") or [])
                    needs_initial_scan = False
                changes = coalesce_changes(changes)
                if changes:
                    self.logger.info(f"Detected {len(changes)} change(s)")
                    await self.dispatch_steps(self.dispatch_step_specs, changes=changes)

        return self.context.response
