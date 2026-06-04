"""``channel_notify_step`` — push a debounced batch of vault changes as one channel event.

Designed to be slotted into ``watch_changes_step.dispatch_step`` next to
``update_index_step``: when the watcher emits a batch of changes, this
step forwards a single human-readable summary to
``ApplicationContext.metadata["channel_sink"]``. The Claude Code main
session then sees a ``<channel source="reme" kind="vault_change" ...>``
tag and reacts per the server's ``instructions``.

One event per batch (not per file) — the watcher already de-bounces and
de-duplicates, so a batch is a meaningful "things that changed together"
unit. Putting N events on the wire per batch would multiply session
turns without adding signal.

No-op (silently) when:

* ``channel_sink`` is absent from the application context metadata
  (e.g. service wasn't an ``MCPService``), so this step is safe in
  any pipeline.
* ``context['changes']`` is missing or empty.
* The sink itself has no bound session (no client called ``claim_channel``).
"""

from pathlib import Path

from ..base_step import BaseStep
from ...components import R


@R.register("channel_notify_step")
class ChannelNotifyStep(BaseStep):
    """Forward a batch of vault changes to the Claude Code channel."""

    async def execute(self):
        sink = self.app_context.metadata.get("channel_sink") if self.app_context is not None else None
        if sink is None:
            return

        changes = (self.context.get("changes", []) if self.context is not None else []) or []
        if not changes:
            return

        # Render paths vault-relative so the agent can pass them directly to
        # slash commands like /dream <path>. Absolute paths that fall outside
        # the vault are left as-is rather than erroring — they shouldn't occur,
        # but a stray entry shouldn't kill the event.
        vault = self.vault_path
        lines: list[str] = []
        for change in changes:
            try:
                raw = Path(change["path"])
            except (KeyError, TypeError):
                continue
            try:
                shown = str(raw.resolve().relative_to(vault))
            except ValueError:
                shown = str(raw)
            lines.append(f"{change.get('change', '?')}: {shown}")

        if not lines:
            return

        self.logger.info(f"[channel_notify] emit batch count={len(lines)}")
        await sink.emit(
            content="Vault 变更:\n" + "\n".join(lines),
            meta={"kind": "vault_change", "count": str(len(lines))},
        )
