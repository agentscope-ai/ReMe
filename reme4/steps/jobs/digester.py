"""Smart Digester — knowledge distillation from daily notes to digest/.

The Digester is the **cold-write** counterpart to AutoMemory (hot-write).
It reads completed work in ``daily/<date>/<slug>.md`` note files,
identifies entities / concepts / claims / methods worth preserving
long-term, and sinks them into ``digest/`` as canonical-entry nodes so
the main agent can retrieve them later via search and graph traversal.

``digest/`` is the cold-tier root. Per ``protocol.md``, scope folders
under it may nest arbitrarily; each folder's canonical entry is
``<folder>/<folder>.md``, and slug (folder name) is globally unique across
the whole tree. Pending detection and graph machinery treat nodes at any
depth uniformly.

Drives a ReAct agent with a read/lookup/graph/write toolkit; the
agent follows the protocol in ``protocol.md`` (the opinionated
default schema + R-M-W decision tree). The schema is convention-driven
— reme core only reserves ``name`` / ``description``, so the agent
owns its own discipline rather than relying on a post-write linter.

Distillation state lives in the daily note's ``status`` frontmatter
— a **daily-tier convention owned by this digester** (reme core
reserves only ``name`` / ``description``; ``status`` is just an
extra). After processing each daily, the agent must call
``frontmatter_update`` with ``metadata={"status": "completed"}``
(or ``metadata={"status": "skipped"}`` when intentionally bypassed). Convention: absent
≡ ``pending``, so the next pass finds residual work via
``file_list path=daily recursive=true`` + per-item ``frontmatter_read`` to filter for absent ``status``.
Only the digester writes ``status``; AutoMemory / hand-edits must
leave it alone.

No degraded path — distillation strictly requires an LLM. When ``as_llm``
is unavailable, the step short-circuits with ``skipped=True`` and an
error message.

Override interface — schema is a service-consumption concern, not a
core invariant, so this step ships an **opinionated default** that any
caller can fully replace without touching reme4:

* ``protocol`` / ``protocol_path`` constructor args replace the
  ``protocol.md`` injected as ``{protocol}`` in the system prompt
  (use when keeping the default prompt template but swapping schema).
* ``prompt_dict`` (inherited from ``BaseStep``) replaces the
  ``system_prompt`` / ``user_message`` templates wholesale (use when
  the prompt structure itself needs to change).
* ``toolkit`` replaces the tool surface ``_DIGESTER_TOOLS`` builds.

Service layers (e.g. plugin-side configs) wire these in via component
config; ``digester.py`` / ``protocol.md`` shipped here are just a
reference implementation of one viable convention.

Toolkit. Each entry in ``_DIGESTER_TOOLS`` is a job name registered
in the active config; ``add_as_tool`` wraps ``job(**kwargs)`` into a
``ToolResponse``. The job indirection means the agent sees the same
tool surface (rich descriptions + JSON schema) as the L2 MCP layer.
"""

import datetime
import zoneinfo
from pathlib import Path

from agentscope.agent import ReActAgent
from agentscope.message import Msg
from agentscope.tool import Toolkit
from pydantic import BaseModel, Field

from ..base_step import BaseStep

from ...components import R


_DIGESTER_TOOLS: tuple[str, ...] = (
    "file_list",
    "file_read",
    "file_stat",
    "frontmatter_read",
    "traverse",
    "file_write",
    "file_append",
    "file_move",
    "frontmatter_update",
    "frontmatter_delete",
)


def _pack_daily(file_store, daily_path: str) -> str:
    """Render one daily note file's body into a prompt-friendly block.

    A daily note is a single self-contained markdown file at
    ``daily/<date>/<slug>.md`` — everything the originating task wanted
    the digester to see is inline (no sibling materials). External
    assets land in ``resource/<date>/`` and are linked from the note's
    ``## References`` section; the LLM opens those on demand via
    ``file_read``.
    """
    try:
        absolute = (Path(file_store.vault_path or ".") / daily_path).resolve()
    except Exception as e:
        return f"### {daily_path}\n(error resolving path: {type(e).__name__}: {e})\n"

    if not absolute.is_file():
        return f"### {daily_path}\n(note file not found)\n"

    parts: list[str] = [f"### {daily_path}"]
    try:
        parts.append(absolute.read_text(encoding="utf-8"))
    except Exception as e:
        parts.append(f"(error reading note: {type(e).__name__}: {e})")
    return "\n".join(parts) + "\n"


class DistillResult(BaseModel):
    """Outcome of a single distillation call.

    Without per-tool audit (the agent's toolkit is the job surface,
    which doesn't expose per-call records back to the orchestrator),
    the structured outcome is just the inputs the call was asked to
    process plus the LLM's free-form summary. Per-file write outcomes
    can be verified by re-reading vault_dir afterwards if needed.

    Field semantics:
      * ``daily_read``        — daily paths actually processed (input
        order, deduped)
      * ``summary``           — LLM's free-form one-paragraph summary
      * ``skipped``           — True when no LLM available, no daily
        paths provided, or the LLM reported ``SKIP``
      * ``error``             — short error string when skipped due
        to misconfiguration (e.g. no LLM)
    """

    used_llm: bool = False
    skipped: bool = False
    daily_read: list[str] = Field(default_factory=list)
    summary: str = ""
    error: str = ""


@R.register("digester")
class Digester(BaseStep):
    """Knowledge digester: daily/ → digest/ via a ReAct agent.

    Inputs (from RuntimeContext):
        daily_paths  (list[str], required): vault-relative paths to
            daily note files (``daily/<date>/<slug>.md``) to distill.
            Pass ``[]`` to no-op.
        hint         (str, optional): caller guidance to the LLM
            (e.g. "focus on the auth-related decisions").

    Output (written to context.response.answer):
        DistillResult JSON — see model docstring.
    """

    def __init__(
        self,
        toolkit: Toolkit | None = None,
        console_enabled: bool = False,
        timezone: str | None = None,
        protocol: str | None = None,
        protocol_path: str | None = None,
        **kwargs,
    ):
        """Constructor overrides (service layer customization points):

        * ``toolkit`` — replace the agent's tool surface; default builds
          one from ``_DIGESTER_TOOLS``.
        * ``protocol`` — inline protocol document (highest precedence);
          overrides whatever the agent sees under ``{protocol}`` in the
          system prompt.
        * ``protocol_path`` — path (relative to the vault or absolute) to a protocol
          markdown file; used when ``protocol`` is not given.
        * ``prompt_dict`` (inherited via ``BaseStep``) — override the
          ``system_prompt`` / ``user_message`` templates wholesale, e.g.
          to swap in a service-layer prompt that hardcodes a different
          schema entirely.

        With none of the above, falls back to the opinionated default
        (the ``protocol.md`` and ``digester.yaml`` shipped alongside
        this module).
        """
        super().__init__(**kwargs)
        self.toolkit = toolkit
        self.console_enabled = console_enabled
        self.timezone = timezone
        self._protocol = self._load_protocol(protocol, protocol_path)

    @staticmethod
    def _load_protocol(protocol: str | None, protocol_path: str | None) -> str:
        """Resolve the protocol document; explicit string > path > default."""
        if protocol is not None:
            return protocol
        if protocol_path:
            path = Path(protocol_path)
            if path.exists():
                return path.read_text(encoding="utf-8")
        default_path = Path(__file__).parent / "protocol.md"
        return default_path.read_text(encoding="utf-8") if default_path.exists() else ""

    def _now(self) -> datetime.datetime:
        if self.timezone:
            try:
                return datetime.datetime.now(zoneinfo.ZoneInfo(self.timezone))
            except Exception as e:
                self.logger.error(f"Invalid timezone: {self.timezone}, error={e}")
        return datetime.datetime.now()

    def _vault_dir(self) -> Path:
        vr = getattr(self.file_store, "vault_path", None)
        return Path(vr).resolve() if vr else Path.cwd().resolve()

    def _llm_available(self) -> bool:
        """Pre-flight check: can ``self.as_llm`` resolve without raising?

        ``BaseStep.as_llm`` asserts when no model is registered, so we
        wrap the access here to avoid hard-failing at the call site."""
        try:
            return self.as_llm is not None
        except Exception:
            return False

    def _build_toolkit(self) -> Toolkit:
        """Bind every digester-relevant job as a tool function."""
        toolkit = self.toolkit or Toolkit()
        for job_name in _DIGESTER_TOOLS:
            self.add_as_tool(toolkit, job_name)
        return toolkit

    async def execute(self):
        assert self.context is not None
        daily_paths: list[str] = list(self.context.get("daily_paths") or [])
        hint: str = (self.context.get("hint", "") or "").strip()

        # No work to do: no daily paths supplied.
        if not daily_paths:
            result = DistillResult(used_llm=False, skipped=True)
            self.context.response.success = True
            self.context.response.answer = "Skipped: no daily paths supplied"
            self.context.response.metadata.update(result.model_dump())
            return

        # No LLM available: distillation strictly requires one.
        if not self._llm_available():
            result = DistillResult(
                used_llm=False,
                skipped=True,
                error="no as_llm configured; distillation requires an LLM",
            )
            self.context.response.success = False
            self.context.response.answer = f"Error: {result.error}"
            self.context.response.metadata.update(result.model_dump())
            return

        # Dedupe daily_paths while preserving order.
        seen: set[str] = set()
        deduped: list[str] = []
        for p in daily_paths:
            if p and p not in seen:
                seen.add(p)
                deduped.append(p)
        daily_paths = deduped

        # Build the per-daily blob the agent will see.
        daily_blob = "\n\n".join(_pack_daily(self.file_store, p) for p in daily_paths)

        vault_dir = self._vault_dir()
        toolkit = self._build_toolkit()

        agent = ReActAgent(
            name="reme_digester",
            model=self.as_llm,
            sys_prompt=self.prompt_format(
                "system_prompt",
                vault_dir=str(vault_dir),
                protocol=self._protocol,
            ),
            formatter=self.as_llm_formatter,
            toolkit=toolkit,
        )
        agent.set_console_output_enabled(self.console_enabled)

        user_message: str = self.prompt_format(
            "user_message",
            today=self._now().strftime("%Y-%m-%d"),
            hint=hint or "(none)",
            daily_blob=daily_blob or "(none)",
        )

        final_msg: Msg = await agent.reply(
            Msg(name="reme", role="user", content=user_message),
        )
        summary = (final_msg.get_text_content() or "").strip()

        result = DistillResult(
            used_llm=True,
            daily_read=list(daily_paths),
            summary=summary,
            skipped=summary.upper().startswith("SKIP"),
        )
        self.context.response.success = True
        self.context.response.answer = summary or "Distillation completed"
        self.context.response.metadata.update(result.model_dump())
