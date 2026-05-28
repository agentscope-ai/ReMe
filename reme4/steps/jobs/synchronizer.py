"""Synchronizer — daily-note sync ReAct agent.

Watches the agent's recent conversation and persists in-progress
tasks as a daily note inside reme's vault_dir, so future agent
invocations can pick the work back up. Pure sync mechanism — does
**not** compress the agent's context (compression is the agent's
own concern).

Note layout: a single markdown file ``daily/<YYYY-MM-DD>/<slug>.md``.
Everything worth preserving (verbatim user prompt, key tool output,
intermediate data) goes inline inside this file — there are no
sibling materials. References to it use the full path relative to
the vault (``[[daily/<date>/<slug>.md]]``); short or no-extension
forms do not resolve. Frontmatter carries ``name`` / ``description``
plus an optional ``inherits`` wikilink for cross-day continuation.
Body splits into ``Objective`` / ``Plan`` / ``Progress`` /
``Findings`` / ``Decisions`` / ``Next`` / ``References`` sections
(the last is a list of ``[[resource/<date>/<name>]]`` wikilinks for
inbound assets landed via ``ingest`` — non-markdown
artefacts the task itself produced are summarized inline).

Inputs (from RuntimeContext):
    messages (list[Msg], required): conversation slice to inspect.
    note     (str, optional): caller-supplied note hint (task name
        or ``daily/<date>/<slug>.md`` path) to bias slug selection
        and disambiguate same-day tasks.

Output (written to context.response.answer):
    {
      "skipped":   True if the agent reported [SKIP],
      "actions":   one-line action statement from the agent,
      "note":      note file path relative to the vault, or None,
      "summary":   full markdown content of the synced note,
                   for the calling agent to reload into a
                   compacted context. None when SKIP / failed.
    }

The agent's toolkit is assembled by ``add_as_tool`` — each entry in
``_NOTE_TOOLS`` is a job name (registered in the active config);
the wrapper turns ``job(**kwargs)`` into a ``ToolResponse``. The job
indirection means the agent sees exactly the same tool surface
(name / description / parameter schema) as the L2 MCP layer.

Override interface — note shape (single-file layout, frontmatter
fields, section discipline) is opinionated convention, not a core
invariant, so callers can fully replace it without touching reme4:

* ``prompt_dict`` (inherited from ``BaseStep``) overrides the
  ``system_prompt`` / ``user_message`` templates wholesale — this is
  how a service layer swaps in its own note schema (e.g. a different
  section list, different frontmatter fields).
* ``toolkit`` replaces the tool surface ``_NOTE_TOOLS`` builds.

What ships here (``synchronizer.yaml``) is just one viable convention;
the service layer (plugin configs, custom callers) is the right place
to pin down the *deployment-specific* shape.
"""

import datetime
import re
import zoneinfo
from pathlib import Path

from agentscope.agent import ReActAgent
from agentscope.message import Msg
from agentscope.tool import Toolkit
from pydantic import BaseModel, Field

from ..base_step import BaseStep

from ...components import R


_NOTE_PATH_RE = re.compile(r"daily/\d{4}-\d{2}-\d{2}/[^/\s]+\.md")


_NOTE_TOOLS: tuple[str, ...] = (
    "file_list",
    "file_read",
    "file_write",
    "file_append",
    "file_edit",
    "file_stat",
    "frontmatter_read",
    "frontmatter_update",
    "frontmatter_delete",
    "daily_read",
    "daily_write",
    "daily_reindex",
)


def _coerce_messages(raw) -> list[Msg]:
    """Normalize incoming messages to ``Msg`` instances.

    The Python caller hands in ``list[Msg]`` directly; the MCP layer
    delivers ``list[dict]`` (each dict shaped roughly ``{name?, role?,
    content?}``). Both shapes land here.
    """
    if not raw:
        return []
    out: list[Msg] = []
    for item in raw:
        if isinstance(item, Msg):
            out.append(item)
            continue
        if isinstance(item, dict):
            out.append(
                Msg(
                    name=item.get("name") or item.get("role") or "user",
                    role=item.get("role") or "user",
                    content=item.get("content", ""),
                ),
            )
    return out


def _format_history(messages: list[Msg]) -> str:
    """Render the conversation as a speaker-tagged transcript.

    Skips messages whose text content is empty (tool-only frames
    don't help the LLM judge task state).
    """
    if not messages:
        return "(empty)"
    lines: list[str] = []
    for msg in messages:
        speaker = msg.name or msg.role or "?"
        text = (msg.get_text_content() or "").strip()
        if not text:
            continue
        lines.append(f"[{speaker}]\n{text}")
    return "\n\n".join(lines) or "(no text)"


class SynchronizerResult(BaseModel):
    """Outcome of a single note-sync call.

    Without per-tool audit (the agent's toolkit is the job surface,
    which doesn't expose per-call records back to the orchestrator),
    the structured outcome is just what the agent reports plus what
    we re-read from disk after it returns.
    """

    used_llm: bool = Field(default=False)
    skipped: bool = Field(default=False)
    actions: str = Field(
        default="",
        description="One-line action statement from the agent (e.g. "
        "'updated daily/2026-05-15/auth-refactor.md' or '[SKIP]').",
    )
    note: str | None = Field(
        default=None,
        description="Note file path relative to the vault, "
        "e.g. 'daily/2026-05-15/auth-refactor.md'. None when SKIP / failed.",
    )
    summary: str | None = Field(
        default=None,
        description="Full markdown content of the note file. Lets the calling "
        "agent reload the warm summary into a freshly compacted context without an extra read.",
    )


@R.register("synchronizer")
class Synchronizer(BaseStep):
    """Drive daily-note sync via a ReAct agent."""

    def __init__(
        self,
        toolkit: Toolkit | None = None,
        console_enabled: bool = False,
        timezone: str | None = None,
        inherit_window_days: int = 7,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.toolkit = toolkit
        self.console_enabled = console_enabled
        self.timezone = timezone
        self.inherit_window_days = inherit_window_days

    def _now(self) -> datetime.datetime:
        if self.timezone:
            try:
                return datetime.datetime.now(zoneinfo.ZoneInfo(self.timezone))
            except Exception as e:
                self.logger.error(
                    f"Invalid timezone {self.timezone!r}, falling back to local time: {e}",
                )
        return datetime.datetime.now()

    def _vault_dir(self) -> Path:
        wd = getattr(self.file_store, "vault_path", None)
        return Path(wd).resolve() if wd else Path.cwd().resolve()

    def _build_toolkit(self) -> Toolkit:
        """Bind every note-relevant job as a tool function.

        Each entry in ``_NOTE_TOOLS`` is a job name registered in the
        active config; ``add_as_tool`` wraps ``job(**kwargs)`` into a
        ``ToolResponse``. The job indirection means the agent sees exactly
        the same tool surface (name / description / parameter schema) as
        the L2 MCP layer.
        """
        toolkit = self.toolkit or Toolkit()
        for job_name in _NOTE_TOOLS:
            self.add_as_tool(toolkit, job_name)
        return toolkit

    async def execute(self):
        assert self.context is not None
        messages: list[Msg] = _coerce_messages(self.context.get("messages"))
        note_hint: str = self.context.get("note", "") or ""

        if not messages:
            result = SynchronizerResult(used_llm=False, skipped=True)
            self.context.response.success = True
            self.context.response.answer = "Skipped: no messages supplied"
            self.context.response.metadata.update(result.model_dump())
            return

        toolkit = self._build_toolkit()

        agent = ReActAgent(
            name="reme_synchronizer",
            model=self.as_llm,
            sys_prompt=self.prompt_format("system_prompt"),
            formatter=self.as_llm_formatter,
            toolkit=toolkit,
        )
        agent.set_console_output_enabled(self.console_enabled)

        user_message: str = self.prompt_format(
            "user_message",
            today=self._now().strftime("%Y-%m-%d"),
            vault_dir=str(self._vault_dir()),
            inherit_window_days=self.inherit_window_days,
            note=note_hint or "(none)",
            history=_format_history(messages),
        )

        final_msg: Msg = await agent.reply(
            Msg(name="reme", role="user", content=user_message),
        )
        actions = (final_msg.get_text_content() or "").strip()

        result = SynchronizerResult(used_llm=True, actions=actions)
        if "[SKIP]" in actions.upper():
            result.skipped = True

        # Reload the freshly written note so the calling agent can drop
        # it back into a compacted context without an extra read trip.
        if not result.skipped:
            self._reload_note(result, actions)

        self.context.response.success = True
        self.context.response.answer = actions or "Synchronization completed"
        self.context.response.metadata.update(result.model_dump())

    def _reload_note(self, result: SynchronizerResult, actions: str) -> None:
        """Parse the agent's action line for the note path and read the
        full file back into ``result.summary``. Best-effort: a parse miss
        leaves the context-management fields as None but does not fail
        the step (persistence already succeeded)."""
        match = _NOTE_PATH_RE.search(actions)
        if not match:
            return
        note_path = match.group(0)
        try:
            absolute = (Path(self.file_store.vault_path or ".") / note_path).resolve()
            text = absolute.read_text(encoding="utf-8")
        except Exception as e:
            self.logger.warning(f"synchronizer: could not reload note {note_path!r}: {e}")
            return
        result.note = note_path
        result.summary = text
