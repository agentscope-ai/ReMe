"""Dreamer — auto-dream's create_or_update step.

Reads one daily-event note or resource file at the given vault-relative
``path``, identifies the ABSTRACTIONS the material teaches in Phase 1,
then in Phase 2 makes ONE cognitive write decision (CREATE or one of
the three UPDATE flavors: CORROBORATE / REFINE / CORRECT) per
abstraction. See ``docs4/auto_dream_design.md`` for the model
contract (buckets / nodes / edges / evolution) and ``§4.2`` for the
pipeline.

**Digest is the abstract memory layer** — analogous to a prefrontal
cortex aggregating cognition. Raw details (timestamps, full
procedures, who-said-what, numbers) stay in the material; digest
holds the principle, pattern, or precedent that should survive
once the details fade. Provenance wikilinks (``derived_from::``)
let readers drill back down to the source on demand.

Pipeline (external loop in Python, two distinct ReAct agent invocations,
**light Phase 1 / heavy Phase 2**):

    execute():
        _extract(material_blob)         # 1× ReAct: identify abstractions
                                        #   agent emits ExtractedUnits structured output
                                        #   ({units: [{name, summary}, ...]})
        for unit in self._units:        # Python loop, K iterations (K = num abstractions)
            _integrate_unit(unit)       # 1× ReAct per abstraction: agent sees full material +
                                        #   the sub-unit's name/summary, recalls, decides
                                        #   bucket, makes ONE write decision (CREATE or
                                        #   one of the UPDATE flavors). Sub-unit ↔ digest
                                        #   node is 1:1.

* **Phase 1 (extract / abstract)** uses a read-only toolkit
  and emits an :class:`ExtractedUnits` Pydantic model as its
  final structured answer (no tool call needed for the unit
  list — agentscope's ``structured_model`` enforces the shape).
  The agent identifies the abstractions the material teaches —
  principles, patterns, precedents worth carrying forward once
  specifics fade. Multiple raw facts that illustrate the same
  abstraction collapse into ONE sub-unit. Prompt biases toward
  fewer / coarser sub-units; filing detail under a digest
  sub-unit is the wrong layer. No event-level umbrella node is
  manufactured — the material itself plays that role via
  ``derived_from`` provenance edges.

* **Phase 2 (integrate per abstraction)** runs once per declared
  sub-unit with a fresh ReAct session (clean context) and the full
  read + write toolkit (``search``, ``traverse``, ``read``,
  ``frontmatter_read``, ``digest_write``, ``digest_edit``). Three
  UPDATE shapes are surfaced explicitly
  in the prompt:

    - **corroborate** (most common): the abstraction already
      exists; the material is one more instance → append a
      ``derived_from::`` provenance wikilink so confidence
      accumulates; body unchanged in substance.
    - **refine**: the material reveals nuance / scope / edge
      cases the abstraction under-specified → tighten the
      relevant span + add the new provenance link.
    - **correct**: the material contradicts the abstraction →
      tighten to the narrower form both old and new support,
      or annotate the contradiction inline + add provenance.

  CREATE is reserved for genuinely new abstractions not yet in
  the vault — even thin first-encounter seeds, which grow via
  CORROBORATE / REFINE on later passes. There is no SKIP outcome:
  Phase 1 is the gate for "not worth memorizing"; anything that
  reaches Phase 2 warrants a write.

The trade-off vs heavy Phase 1: full material is sent to LLM K
times in Phase 2 (one per abstraction). The advantages: no
information loss in summary, focused reasoning per call, and
granularity tuned at a single prompt (Phase 1) rather than two.

Mechanical guardrails at the write boundary:

* ``digest_write`` (subclass of WriteStep) rejects paths outside
  ``<digest_dir>/<bucket>/<slug>.md`` (where ``digest_dir`` comes
  from app config and ``bucket`` is in the fixed bucket set), and
  refuses if the path already exists.
* ``digest_edit`` (subclass of EditStep) is a body-only find-and-replace
  on an existing digest node, gated by E-1 strong-conservation: the
  outbound link set BEFORE the replacement must be a subset of the link
  set AFTER. If any edge would be dropped the tool returns
  ``REJECT_CONSERVATION`` and the agent must adjust ``new`` to keep
  the missing links before retrying.

Invocation form (CLI / MCP):
    reme dream path=daily/2026-05-28/auth-refactor/auth-refactor.md
    reme dream path=resource/2026-05-28/spec.pdf hint="focus on auth"
"""

import datetime
import zoneinfo
from pathlib import Path
from typing import Literal

from agentscope.message import Msg, TextBlock
from agentscope.tool import Toolkit, ToolResponse
from pydantic import BaseModel, Field

from .digest_edit import DigestEditStep
from .digest_write import DigestWriteStep, bucket_names, normalize_buckets
from .._evolve import FlexReActAgent
from ...base_step import BaseStep
from ....components import R


_EXTRACT_READ_TOOLS: tuple[str, ...] = ("read",)

_INTEGRATE_READ_TOOLS: tuple[str, ...] = (
    "search",
    "traverse",
    "read",
    "frontmatter_read",
)


def _pack_material(file_store, path: str) -> str:
    """Render one daily-event note or resource file into a prompt block."""
    try:
        absolute = (Path(file_store.vault_path or ".") / path).resolve()
    except Exception as e:
        return f"### {path}\n(error resolving path: {type(e).__name__}: {e})\n"

    if not absolute.is_file():
        return f"### {path}\n(file not found)\n"

    try:
        return f"### {path}\n{absolute.read_text(encoding='utf-8')}\n"
    except Exception as e:
        return f"### {path}\n(error reading: {type(e).__name__}: {e})\n"


class MemoryUnit(BaseModel):
    """One memory sub-unit identified by Phase 1's structured output."""

    name: str = Field(
        description=(
            "Short kebab-case identifier for the abstraction "
            "(e.g. 'jwt-rotation-decision', 'pr-size-pref'). "
            "Agent-internal handle — NOT the eventual digest slug; "
            "Phase 2 picks the actual filing path + bucket."
        ),
    )
    summary: str = Field(
        description=(
            "1-2 sentences naming the abstraction AND pointing at where "
            "in the material the supporting evidence lives "
            "(e.g. 'short-credential compliance drives auth cadence; "
            "illustrated by the 30→24h decision in the 'Decision' section "
            "+ the SOC2 CC6.1 criticism in the 'Observation' section')."
        ),
    )


class ExtractedUnits(BaseModel):
    """Structured output emitted by Phase 1's extract agent."""

    units: list[MemoryUnit] = Field(
        default_factory=list,
        description=(
            "Memory sub-units identified in the material — orthogonal "
            "abstractions (principles / patterns / precedents) worth "
            "lifting into long-term memory. Empty list = nothing worth "
            "lifting (Phase 2 is skipped)."
        ),
    )


def _render_outcome_line(unit_name: str, o: "IntegrateOutcome") -> str:
    """Format one IntegrateOutcome as a one-line summary entry."""
    if o.action == "CREATE":
        body = f"CREATE {o.target_path}"
        if o.note:
            body += f" — {o.note}"
    else:  # CORROBORATE / REFINE / CORRECT (all UPDATE-flavored)
        recovered = " (recovered from REJECT_CONSERVATION)" if o.recovered_from_conservation else ""
        body = f"{o.action} {o.target_path}{recovered}"
        if o.note:
            body += f" — {o.note}"
    return f"[{unit_name}] {body}"


class IntegrateOutcome(BaseModel):
    """Structured outcome reported by Phase 2 for one sub-unit."""

    action: Literal["CREATE", "CORROBORATE", "REFINE", "CORRECT"] = Field(
        description=(
            "Outcome of the write decision for this sub-unit. Phase 1 already "
            "filtered out non-abstractions, so every sub-unit reaching you "
            "warrants a write — pick the matching fine-grained action: "
            "`CREATE` — brand-new digest node (recall returned no node "
            "covering this abstraction); even thin first-encounter seeds go "
            "here, they grow via CORROBORATE / REFINE on later passes. "
            "`CORROBORATE` (most common when a covering node exists) — "
            "provenance append + optional wording strengthening; the "
            "abstraction already covers this material. `REFINE` — covering "
            "node exists but the material reveals nuance, scope, or edge "
            "cases the abstraction under-specified. `CORRECT` — covering "
            "node exists but the material contradicts it; tighten the "
            "abstraction or annotate the contradiction inline."
        ),
    )
    target_path: str = Field(
        description=(
            "The digest path you wrote to — must match what your " "`digest_write` / `digest_edit` call(s) targeted."
        ),
    )
    note: str = Field(
        default="",
        description=(
            "Optional ONE short line, ≤ 200 chars, no newlines, summarizing "
            "what landed (e.g. 'extended scope to also cover X'). Do NOT "
            "dump recall summaries, search results, internal reasoning, or "
            "transcripts here — those belong in the ReAct trace, not the "
            "outcome note."
        ),
    )
    recovered_from_conservation: bool = Field(
        default=False,
        description=(
            "Set to true if `digest_edit` initially returned "
            "REJECT_CONSERVATION and you re-composed `new` to preserve the "
            "missing links. Only meaningful for CORROBORATE / REFINE / CORRECT."
        ),
    )


class DreamResult(BaseModel):
    """Outcome of one dreamer invocation.

    Per-tool audit lives in the toolkit layer (not exposed back to the
    orchestrator). Structured outcome here is the input path the call
    processed, the memory sub-units the agent declared in Phase 1,
    what got created / updated in Phase 2, and any conservation
    rejections that occurred along the way.
    """

    used_llm: bool = False
    skipped: bool = False
    path: str = ""
    units: list[dict] = Field(default_factory=list)
    nodes_created: list[str] = Field(default_factory=list)
    nodes_updated: list[str] = Field(default_factory=list)
    conservation_violations: list[dict] = Field(default_factory=list)
    summary: str = ""
    error: str = ""


@R.register("dreamer_step")
class Dreamer(BaseStep):
    """auto-dream create_or_update step.

    Inputs (from RuntimeContext):
        path       (str, required): vault-relative path of one
            daily-event note or resource file to dream over. Pass
            empty string to no-op.
        hint       (str, optional): caller guidance to the LLM
            (e.g. "focus on the auth-related decisions").
        buckets    (list[str], optional): override the fixed bucket
            set; default ``DEFAULT_BUCKETS``.

    Output (written to context.response.answer):
        ``DreamResult`` JSON in ``metadata``; LLM summary in ``answer``.

    CLI / MCP form:
        reme dream path=daily/2026-05-28/auth-refactor/auth-refactor.md
    """

    def __init__(
        self,
        toolkit: Toolkit | None = None,
        console_enabled: bool = False,
        timezone: str | None = None,
        buckets: list[str] | tuple[str, ...] | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.toolkit = toolkit
        self.console_enabled = console_enabled
        self.timezone = timezone
        self.buckets = normalize_buckets(buckets)
        assert "unknown" in bucket_names(self.buckets), "bucket set must include 'unknown' as the unclassified fallback"
        # Per-invocation outcome trackers, populated by tool callbacks.
        self._units: list[dict] = []
        self._created: list[str] = []
        self._updated: list[str] = []
        self._violations: list[dict] = []

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
        try:
            return self.as_llm is not None
        except Exception:
            return False

    def _make_digest_write_tool(self):
        """Tool closure: wraps :class:`DigestWriteStep` and tracks creates."""

        async def digest_write(path: str, name: str, description: str, content: str) -> ToolResponse:
            step = DigestWriteStep(
                file_store=self.file_store,
                buckets=self.buckets,
                app_context=self.app_context,
            )
            await step(path=path, name=name, description=description, content=content)
            assert step.context is not None
            resp = step.context.response
            if not resp.success:
                return ToolResponse(content=[TextBlock(type="text", text=resp.answer)])
            self._created.append(path)
            return ToolResponse(content=[TextBlock(type="text", text=f"OK: created {path}")])

        return digest_write

    def _make_digest_edit_tool(self):
        """Tool closure: wraps :class:`DigestEditStep` and tracks updates / conservation violations."""

        async def digest_edit(path: str, old: str, new: str) -> ToolResponse:
            step = DigestEditStep(
                file_store=self.file_store,
                buckets=self.buckets,
                app_context=self.app_context,
            )
            await step(path=path, old=old, new=new)
            assert step.context is not None
            resp = step.context.response
            if not resp.success:
                violation = (resp.metadata or {}).get("conservation_violation")
                if violation:
                    self._violations.append(violation)
                return ToolResponse(content=[TextBlock(type="text", text=resp.answer)])
            self._updated.append(path)
            return ToolResponse(content=[TextBlock(type="text", text=f"OK: updated {path}")])

        return digest_edit

    def _build_extract_toolkit(self) -> Toolkit:
        """Read-only toolkit for the extract agent. Sub-units come back via
        :class:`ExtractedUnits` structured output, not via a tool call."""
        toolkit = Toolkit()
        for job_name in _EXTRACT_READ_TOOLS:
            self.add_as_tool(toolkit, job_name)
        return toolkit

    def _build_integrate_toolkit(self) -> Toolkit:
        """Full read + conservation-aware write toolkit for the integrate agent."""
        toolkit = self.toolkit or Toolkit()
        for job_name in _INTEGRATE_READ_TOOLS:
            self.add_as_tool(toolkit, job_name)
        digest_dir = getattr(self.app_context.app_config, "digest_dir", "")
        path_shape = f"'{digest_dir}/<bucket>/<slug>.md'"
        digest_write_desc = (
            "Create a NEW digest node — same shape as the canonical `write` job, plus "
            f"path-shape validation: `path` must be {path_shape} where "
            f"bucket is one of {list(bucket_names(self.buckets))} (use 'unknown' when "
            "no specialized bucket fits — it is a first-class bucket, not a failure "
            "state). `name` and `description` go into the YAML frontmatter; `content` "
            "is the body. Fails if the path already exists; use `digest_edit` then."
        )
        toolkit.register_tool_function(
            tool_func=self._make_digest_write_tool(),
            func_name="digest_write",
            func_description=digest_write_desc,
            json_schema={
                "type": "function",
                "function": {
                    "name": "digest_write",
                    "description": digest_write_desc,
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {
                                "type": "string",
                                "description": f"vault-relative path; must match {path_shape}",
                            },
                            "name": {
                                "type": "string",
                                "description": "frontmatter name (usually the slug)",
                            },
                            "description": {
                                "type": "string",
                                "description": "frontmatter description — one-line summary of the abstraction",
                            },
                            "content": {
                                "type": "string",
                                "description": "body (markdown; no frontmatter "
                                "— name/description go in the fields above)",
                            },
                        },
                        "required": ["path", "name", "description", "content"],
                    },
                },
            },
        )
        digest_edit_desc = (
            "Find-and-replace inside an existing digest node's body — same shape as "
            "the canonical `edit` job, plus path-shape validation (`path` must be "
            f"{path_shape}, file must exist) and E-1 strong edge conservation: every "
            "outbound wikilink present BEFORE the replacement must still be present "
            "AFTER. If you drop any edge the tool returns REJECT_CONSERVATION and you "
            "must adjust `new` to keep the missing links. Operates on body only; "
            "frontmatter is untouched. Prefer narrow `old` spans."
        )
        toolkit.register_tool_function(
            tool_func=self._make_digest_edit_tool(),
            func_name="digest_edit",
            func_description=digest_edit_desc,
            json_schema={
                "type": "function",
                "function": {
                    "name": "digest_edit",
                    "description": digest_edit_desc,
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {
                                "type": "string",
                                "description": f"vault-relative path; must match {path_shape} and exist",
                            },
                            "old": {
                                "type": "string",
                                "description": (
                                    "Substring to locate in the EXISTING body (frontmatter excluded). "
                                    "Must match verbatim. Pick a span large enough to be unique."
                                ),
                            },
                            "new": {
                                "type": "string",
                                "description": (
                                    "Replacement text. Should weave new material into the existing "
                                    "wording without dropping any wikilinks the `old` span contained."
                                ),
                            },
                        },
                        "required": ["path", "old", "new"],
                    },
                },
            },
        )
        return toolkit

    async def _extract(self, material_blob: str, hint: str, vault_dir: Path) -> str:
        """Phase 1: one ReAct invocation — read material + emit ExtractedUnits. Returns LLM summary."""
        toolkit = self._build_extract_toolkit()
        agent = FlexReActAgent(
            name="reme_dreamer_extract",
            model=self.as_llm,
            sys_prompt=self.prompt_format(
                "extract_system_prompt",
                vault_dir=str(vault_dir),
                buckets=", ".join(bucket_names(self.buckets)),
            ),
            formatter=self.as_llm_formatter,
            toolkit=toolkit,
        )
        agent.set_console_output_enabled(self.console_enabled)
        user_message = self.prompt_format(
            "extract_user_message",
            today=self._now().strftime("%Y-%m-%d"),
            hint=hint or "(none)",
            material_blob=material_blob,
        )
        msg = await agent.reply(
            Msg(name="reme", role="user", content=user_message),
            structured_model=ExtractedUnits,
        )

        # Structured output lands in msg.metadata as a dict matching ExtractedUnits.
        # Empty / missing → no sub-units (Phase 2 will skip).
        meta = msg.metadata if isinstance(msg.metadata, dict) else {}
        cleaned: list[dict] = []
        for raw in meta.get("units") or []:
            if not isinstance(raw, dict):
                continue
            name = str(raw.get("name") or "").strip()
            summary = str(raw.get("summary") or "").strip()
            if name and summary:
                cleaned.append({"name": name, "summary": summary})
        self._units = cleaned
        return (msg.get_text_content() or "").strip()

    async def _integrate_unit(self, unit: dict, material_blob: str, hint: str, vault_dir: Path) -> IntegrateOutcome:
        """One ReAct invocation per memory sub-unit. Returns the parsed
        :class:`IntegrateOutcome` reported by the agent.

        File writes happen as side effects via the ``digest_write`` /
        ``digest_edit`` tool calls during the ReAct loop (which populate
        ``self._created`` / ``self._updated`` / ``self._violations``); the
        structured outcome here is the agent's own summary of what it
        decided — useful for rendering and for catching hallucinations
        (action=CREATE without the matching write call landing in trackers).
        """
        toolkit = self._build_integrate_toolkit()
        digest_dir = getattr(self.app_context.app_config, "digest_dir", "")
        buckets_block = "\n".join(
            f"  - `{digest_dir}/{b['name']}/`" + (f"  — {b['description']}" if b.get("description") else "")
            for b in self.buckets
        )
        agent = FlexReActAgent(
            name=f"reme_dreamer_integrate_{unit.get('name', 'unit')}",
            model=self.as_llm,
            sys_prompt=self.prompt_format(
                "integrate_system_prompt",
                vault_dir=str(vault_dir),
                digest_dir=digest_dir,
                buckets=buckets_block,
            ),
            formatter=self.as_llm_formatter,
            toolkit=toolkit,
        )
        agent.set_console_output_enabled(self.console_enabled)
        user_message = self.prompt_format(
            "integrate_user_message",
            hint=hint or "(none)",
            unit_name=unit.get("name", ""),
            unit_summary=unit.get("summary", ""),
            material_blob=material_blob,
        )
        # Snapshot trackers so we can reconstruct the outcome from the
        # filesystem side effects if the agent's structured emission slips.
        created_before = len(self._created)
        updated_before = len(self._updated)
        msg = await agent.reply(
            Msg(name="reme", role="user", content=user_message),
            structured_model=IntegrateOutcome,
        )
        meta = msg.metadata if isinstance(msg.metadata, dict) else {}
        try:
            return IntegrateOutcome.model_validate(meta)
        except Exception:
            # The LLM occasionally drops the final structured emission even
            # after a successful tool call. The trackers are the source of
            # truth — reconstruct the outcome from the new entries this
            # session added.
            new_created = self._created[created_before:]
            new_updated = self._updated[updated_before:]
            if new_created:
                return IntegrateOutcome(action="CREATE", target_path=new_created[-1])
            if new_updated:
                return IntegrateOutcome(action="CORROBORATE", target_path=new_updated[-1])
            raise

    async def dream_one(self, path: str, hint: str = "") -> DreamResult:
        """Run the full extract + integrate pipeline on one vault-relative
        material path. Returns a structured :class:`DreamResult`. Safe to
        call repeatedly on the same instance — per-invocation trackers are
        reset at the start of each call. Used both by :meth:`execute`
        (single file from context) and by :class:`CronDreamer` (loop over
        today's materials).
        """
        path = (path or "").strip()
        hint = (hint or "").strip()

        if not path:
            return DreamResult(used_llm=False, skipped=True)

        if not self._llm_available():
            return DreamResult(
                used_llm=False,
                skipped=True,
                path=path,
                error="no as_llm configured; dreaming requires an LLM",
            )

        material_blob = _pack_material(self.file_store, path)

        # Reset per-invocation trackers.
        self._units.clear()
        self._created.clear()
        self._updated.clear()
        self._violations.clear()

        vault_dir = self._vault_dir()

        # Phase 1 — extract (light). Agent emits ExtractedUnits structured output to commit the
        # memory sub-units worth lifting.
        self.logger.info(f"[{self.name}] extract phase: path={path!r}")
        extract_summary = await self._extract(material_blob, hint, vault_dir)

        if not self._units:
            return DreamResult(
                used_llm=True,
                path=path,
                summary=extract_summary or "no memory sub-units declared",
                skipped=True,
            )

        self.logger.info(
            f"[{self.name}] integrate phase: {len(self._units)} sub-unit(s): "
            f"{', '.join(u['name'] for u in self._units)}",
        )

        # Phase 2 — integrate, one fresh ReAct per sub-unit. Python-level
        # loop, not agent loop. Each session emits a structured
        # IntegrateOutcome; file writes happen as side effects via
        # digest_write / digest_edit tool calls.
        per_unit_lines: list[str] = []
        for i, unit in enumerate(self._units, start=1):
            name = unit.get("name", "?")
            try:
                outcome = await self._integrate_unit(unit, material_blob, hint, vault_dir)
            except Exception as e:
                self.logger.error(
                    f"[{self.name}] integrate {i}/{len(self._units)} (unit={name}) " f"failed: {type(e).__name__}: {e}",
                )
                per_unit_lines.append(f"[{name}] FAILED: {type(e).__name__}: {e}")
                continue
            per_unit_lines.append(_render_outcome_line(name, outcome))

        summary = (
            f"Declared {len(self._units)} sub-unit(s) "
            f"({', '.join(u['name'] for u in self._units)}); "
            f"created {len(self._created)}, updated {len(self._updated)}, "
            f"conservation violations {len(self._violations)}.\n" + "\n".join(per_unit_lines)
        )

        return DreamResult(
            used_llm=True,
            path=path,
            units=list(self._units),
            nodes_created=list(self._created),
            nodes_updated=list(self._updated),
            conservation_violations=list(self._violations),
            summary=summary,
            skipped=False,
        )

    async def execute(self):
        assert self.context is not None
        path: str = (self.context.get("path", "") or "").strip()
        hint: str = (self.context.get("hint", "") or "").strip()

        result = await self.dream_one(path, hint)

        if not path:
            self.context.response.success = True
            self.context.response.answer = "Skipped: no path supplied"
        elif result.error:
            self.context.response.success = False
            self.context.response.answer = f"Error: {result.error}"
        elif result.skipped:
            self.context.response.success = True
            self.context.response.answer = result.summary or "Skipped: no memory sub-units declared"
        else:
            self.context.response.success = True
            self.context.response.answer = result.summary
        self.context.response.metadata.update(result.model_dump())
