"""Summarizer — workspace-sync ReAct agent.

Watches the agent's recent conversation and persists in-progress
tasks as a workspace **folder** inside reme's vault, so future
agent invocations can pick the work back up. Pure sync mechanism —
does **not** compress the agent's context (compression is the
agent's own concern).

Workspace layout: ``daily/<YYYY-MM-DD>/<slug>/<slug>.md`` plus
sibling materials of any file type (md / pdf / doc / csv / ...).
The note filename equals its parent folder name, which makes it a
folder note in reme L1 — short wikilinks like ``[[<slug>]]`` from
sibling files resolve back to it. Frontmatter carries
``title`` / ``description`` / ``status`` / ``created`` / ``updated``
plus an optional ``inherits`` wikilink for cross-day continuation;
body splits into ``Objective`` / ``Plan`` / ``Progress`` /
``Findings`` / ``Decisions`` / ``Next`` / ``Materials`` sections.

Inputs (from RuntimeContext):
    messages   (list[Msg], required): conversation slice to inspect.
    workspace  (str, optional): caller-supplied workspace hint
        (task title or vault-relative ``daily/.../`` path) to bias
        slug selection and disambiguate same-day tasks.

Output (written to context.response.answer):
    {
      "skipped":   True if the agent reported [SKIP],
      "actions":   one-line action statement from the agent,
      "workspace": vault-relative folder path, or None,
      "summary":   full markdown content of the summary note,
                   for the calling agent to reload into a
                   compacted context. None when SKIP / failed.
    }

The toolkit bound for the agent is built in-place from registered
crud / property steps (``list``, ``read``, ``write``, ``stat``,
``property:read``, ``property:update``); each call accumulates into
the audit list returned in ``applied`` / ``failed``.
"""

from __future__ import annotations

import datetime
import re
import zoneinfo
from pathlib import Path

from agentscope.agent import ReActAgent
from agentscope.message import Msg
from agentscope.tool import Toolkit
from pydantic import BaseModel, Field

from ..base_step import BaseStep
from ..runtime_response import _set_answer

from ...component import R
from ...enumeration import ComponentEnum
from ...utils import path_resolver


_NOTE_PATH_RE = re.compile(r"daily/\d{4}-\d{2}-\d{2}/[^/\s]+/[^/\s]+\.md")


_WORKSPACE_TOOLS: tuple[tuple[str, str], ...] = (
    ("list", "file_list"),
    ("read", "file_read"),
    ("write", "file_write"),
    ("stat", "file_stat"),
    ("property:read", "property_read"),
    ("property:update", "property_update"),
)


def _build_toolkit(
    app_context,
    audit: list[dict],
    toolkit: Toolkit | None = None,
) -> Toolkit:
    """Bind workspace-relevant step tool methods into a single Toolkit.

    Each bound instance shares the same ``audit`` list so every tool
    call's outcome lands in one trail.
    """
    toolkit = toolkit or Toolkit()
    for step_name, method_name in _WORKSPACE_TOOLS:
        cls = R.get(ComponentEnum.STEP, step_name)
        if cls is None:
            continue
        instance = cls(app_context=app_context)
        instance.audit = audit  # type: ignore[attr-defined]
        toolkit.register_tool_function(
            getattr(instance, method_name),
            namesake_strategy="override",
        )
    return toolkit


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


class SummarizerResult(BaseModel):
    """Audit trail + context-management payload for a single workspace-sync call."""

    used_llm: bool = Field(default=False)
    applied: list[dict] = Field(default_factory=list)
    failed: list[dict] = Field(default_factory=list)
    skipped: bool = Field(default=False)
    actions: str = Field(
        default="",
        description="One-line action statement from the agent (e.g. "
        "'updated daily/2026-05-15/auth-refactor/auth-refactor.md (+2 materials)' or '[SKIP]').",
    )

    workspace: str | None = Field(
        default=None,
        description="Vault-relative folder path of the synced workspace, "
        "e.g. 'daily/2026-05-15/auth-refactor/'. None when SKIP / failed.",
    )
    summary: str | None = Field(
        default=None,
        description="Full markdown content of the workspace summary note. Lets the calling "
        "agent reload the warm summary into a freshly compacted context without an extra read.",
    )


@R.register("summarizer")
class Summarizer(BaseStep):
    """Drive workspace sync via a ReAct agent."""

    component_type = ComponentEnum.STEP

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

    def _working_dir(self) -> Path:
        wd = getattr(self.file_store, "working_dir", None)
        return Path(wd).resolve() if wd else Path.cwd().resolve()

    async def execute(self):
        assert self.context is not None
        messages: list[Msg] = self.context.get("messages") or []
        workspace_hint: str = self.context.get("workspace", "") or ""

        if not messages:
            result = SummarizerResult(used_llm=False, skipped=True)
            self.context.response.success = True
            _set_answer(self.context, result.model_dump())
            return

        audit: list[dict] = []
        toolkit = _build_toolkit(self.app_context, audit=audit, toolkit=self.toolkit)

        agent = ReActAgent(
            name="reme_summarizer",
            model=self.as_llm,
            sys_prompt=self.prompt_format("system_prompt"),
            formatter=self.as_llm_formatter,
            toolkit=toolkit,
        )
        agent.set_console_output_enabled(self.console_enabled)

        user_message: str = self.prompt_format(
            "user_message",
            today=self._now().strftime("%Y-%m-%d"),
            working_dir=str(self._working_dir()),
            inherit_window_days=self.inherit_window_days,
            workspace=workspace_hint or "(none)",
            history=_format_history(messages),
        )

        final_msg: Msg = await agent.reply(
            Msg(name="reme", role="user", content=user_message),
        )
        actions = (final_msg.get_text_content() or "").strip()

        result = SummarizerResult(used_llm=True, actions=actions)
        for entry in audit:
            (result.applied if entry.get("ok") else result.failed).append(entry)
        if not audit and "[SKIP]" in actions.upper():
            result.skipped = True

        # Reload the freshly written note so the calling agent can drop
        # it back into a compacted context without an extra read trip.
        if not result.skipped and not result.failed:
            self._reload_note(result, actions)

        self.context.response.success = len(result.failed) == 0
        _set_answer(self.context, result.model_dump())

    def _reload_note(self, result: SummarizerResult, actions: str) -> None:
        """Parse the agent's action line for the note path and read the
        full file back into ``result.summary``. Best-effort: a parse miss
        leaves the context-management fields as None but does not fail
        the step (persistence already succeeded)."""
        match = _NOTE_PATH_RE.search(actions)
        if not match:
            return
        note_path = match.group(0)
        try:
            absolute = path_resolver.to_absolute(self.file_store, note_path)
            text = absolute.read_text(encoding="utf-8")
        except Exception as e:
            self.logger.warning(f"summarizer: could not reload note {note_path!r}: {e}")
            return
        result.workspace = str(Path(note_path).parent) + "/"
        result.summary = text
