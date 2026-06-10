"""Long-running awatch loop: convert raw changes into dispatch_step calls.

Two relevant awatch parameters are exposed verbatim:

* ``step`` (default ``50ms``) — awatch yields when the entire watcher
  has gone this long without new changes (and at least one change is
  pending). Raise to ``5 minutes``-ish for ``auto_dream_loop`` so
  half-written sync output isn't dreamed mid-write; keep at default
  for ``index_update_loop`` where every fs change should hit
  the index promptly.

* ``debounce`` (default ``2000ms``) — per-batch ceiling, regardless
  of whether activity is still arriving. Set ``debounce > step`` so
  ``step`` is the operative limit; otherwise the watcher pre-empts
  long-quiet-window setups under bursty writes.

Both are global to the watcher (not per-path). The reme watchers
have disjoint ``watch_dirs`` (configured per job), so global
quiet windows are good enough — no per-path bookkeeping needed.

awatch internally deduplicates same-path same-change tuples within
the yielded batch, so a file ``modified`` ten times during the
quiet window arrives as one ``(modified, path)`` entry.
"""

import asyncio

from watchfiles import Change, awatch

from ._watch_rules import WatchRule, build_watch_rules, match_file
from ..base_step import BaseStep
from ...components import R, BaseComponent
from ...enumeration import ComponentEnum


@R.register("watch_changes_step")
class WatchChangesStep(BaseStep):
    """Watch files and forward each yielded batch to downstream steps."""

    def __init__(
        self,
        recursive: bool = True,
        force_polling: bool = True,
        debounce: int = 2000,
        step: int = 50,
        poll_delay_ms: int = 2000,
        dispatch_step: str = "",
        dispatch_steps: list[str] | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.recursive: bool = recursive
        self.force_polling: bool = force_polling
        self.debounce: int = debounce
        self.step: int = step
        self.poll_delay_ms: int = poll_delay_ms
        self.dispatch_steps: list[str] = dispatch_steps or ([dispatch_step] if dispatch_step else [])
        self._rules: list[WatchRule] = []

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

    def _filter(self, _change: Change, path: str) -> bool:
        return match_file(path, self._rules)

    async def execute(self):
        if self.context is None:
            raise RuntimeError("watch_changes_step requires 'context'")
        if self.context.stop_event is None:
            raise RuntimeError("watch_changes_step requires 'stop_event' on context")
        stop_event: asyncio.Event = self.context.stop_event

        self._rules = self._get_watch_rules()
        if not self._rules:
            raise RuntimeError("No watch rules configured (watch_dirs empty or app_config missing?)")

        valid_paths = list(dict.fromkeys(r.path for r in self._rules if r.path.exists()))
        if not valid_paths:
            raise RuntimeError(f"No valid watch paths exist: {[str(r.path) for r in self._rules]}")

        dispatch_classes: list[type[BaseComponent]] = []
        for name in self.dispatch_steps:
            cls = R.get(ComponentEnum.STEP, name)
            if cls is None:
                raise RuntimeError(f"Unregistered step '{name}'")
            dispatch_classes.append(cls)

        self.logger.info(f"Watching: {[str(p) for p in valid_paths]} step={self.step}ms debounce={self.debounce}ms")

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
                # TODO @jinli
                extra = {k: v for k, v in self.context.data.items() if k not in ("stop_event", "changes")}
                for cls in dispatch_classes:
                    s = cls(app_context=self.app_context)
                    await s(changes=changes, **extra)

        return self.context.response
