"""Dreamer — auto-dream's create_or_update step.

Reads one daily-event note or resource file at the given vault-relative
``path``, identifies the ABSTRACTIONS the material teaches in Phase 1
(each tagged with one of the three buckets), then in Phase 2 makes
ONE cognitive write decision (CREATE or one of the three UPDATE
flavors: CORROBORATE / REFINE / CORRECT) per abstraction using a
**bucket-specific** integrate prompt.

**Digest is the abstract memory layer** — raw details stay in the
material; digest holds the principle, pattern, or precedent worth
recalling once the specifics fade. Provenance wikilinks
(``derived_from::``) let readers drill back down to the source.

Pipeline (external loop in Python, two distinct ReAct agent invocations,
**light Phase 1 / heavy Phase 2**):

    execute():
        _extract(material_blob)         # 1× ReAct: identify abstractions
                                        #   agent emits ExtractedUnits
                                        #   ({units: [{name, bucket, summary}, ...]})
        for unit in self._units:        # Python loop, K iterations
            _integrate_unit(unit)       # 1× ReAct per abstraction, dispatched
                                        #   to integrate_system_prompt_<bucket>;
                                        #   recalls cross-bucket, decides write,
                                        #   uses canonical write/edit tools.

The bucket vocabulary is hard-coded (:data:`BUCKETS`) — three buckets,
each with a dedicated Phase 2 prompt:

* ``procedure`` — how-to-do-X: steps, methods, recipes, workflows.
* ``personal``  — user/team specific: identity, preferences,
  conventions, things they avoid.
* ``wiki``      — general knowledge: definitions, principles,
  observations, decisions-as-precedent. Default catch-all.

There is no SKIP outcome in Phase 2: Phase 1 is the gate for "not
worth memorizing"; anything reaching Phase 2 warrants a write.

Phase 2 uses the **canonical** ``write`` / ``edit`` jobs (no
constrained variants). Bucket placement and edge conservation are
prompt-level discipline; the tools themselves perform no path-shape
or conservation validation.

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

from .._evolve import FlexReActAgent
from ...base_step import BaseStep
from ....components import R


# Hard-coded bucket vocabulary. Phase 1 classifies each sub-unit into
# one of these; Phase 2 dispatches to the bucket-specific prompt.
# Order matters for prompt rendering — keep procedure/personal/wiki.
BUCKETS: tuple[str, ...] = ("procedure", "personal", "wiki")

# Bucket = Literal of BUCKETS. Pydantic Literal must be a static type;
# update both BUCKETS and Bucket together if the vocabulary changes.
Bucket = Literal["procedure", "personal", "wiki"]


_EXTRACT_READ_TOOLS: tuple[str, ...] = ("read",)

_INTEGRATE_READ_TOOLS: tuple[str, ...] = (
    "search",
    "traverse",
    "read",
    "frontmatter_read",
)

_INTEGRATE_WRITE_TOOLS: tuple[str, ...] = (
    "write",
    "edit",
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
            "Phase 2 picks the actual filing path."
        ),
    )
    bucket: Bucket = Field(
        description=(
            "Which bucket this abstraction belongs in — Phase 2 dispatches "
            "to a bucket-specific prompt based on this. Pick exactly one: "
            "`procedure` (how-to-do-X — steps, methods, recipes, workflows), "
            "`personal` (user/team-specific — identity, preferences, "
            "conventions, things they avoid), `wiki` (general knowledge — "
            "definitions, principles, observations, decisions-as-precedent; "
            "default catch-all when nothing else fits)."
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
            "lifting into long-term memory. Each is tagged with its "
            "bucket. Empty list = nothing worth lifting (Phase 2 is skipped)."
        ),
    )


def _render_outcome_line(unit_name: str, bucket: str, o: "IntegrateOutcome") -> str:
    """Format one IntegrateOutcome as a one-line summary entry."""
    body = f"{o.action} {o.target_path}"
    if o.note:
        body += f" — {o.note}"
    return f"[{unit_name}/{bucket}] {body}"


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
            "The digest path you wrote to — must match what your `write` / "
            "`edit` call(s) targeted."
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


class DreamResult(BaseModel):
    """Outcome of one dreamer invocation.

    Per-tool audit lives in the toolkit layer (not exposed back to the
    orchestrator). Structured outcome here is the input path the call
    processed, the memory sub-units the agent declared in Phase 1, and
    what got created / updated in Phase 2.
    """

    used_llm: bool = False
    skipped: bool = False
    path: str = ""
    units: list[dict] = Field(default_factory=list)
    nodes_created: list[str] = Field(default_factory=list)
    nodes_updated: list[str] = Field(default_factory=list)
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
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.toolkit = toolkit
        self.console_enabled = console_enabled
        self.timezone = timezone
        # Per-invocation outcome trackers, populated by tool callbacks.
        self._units: list[dict] = []
        self._created: list[str] = []
        self._updated: list[str] = []

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

    def _make_write_tool(self):
        """Wrap the canonical ``write`` job with create-tracking.

        Same shape as the underlying job; the wrapper just records
        successful paths into ``self._created`` so the dreamer can
        reconstruct outcomes when the LLM drops its structured emission.
        """
        job = self.get_job("write")
        if job is None:
            raise RuntimeError("write job not registered")

        async def write(path: str, name: str, description: str, content: str) -> ToolResponse:
            resp = await job(
                path=path,
                name=name,
                description=description,
                content=content,
            )
            if resp.success:
                self._created.append(path)
            return ToolResponse(content=[TextBlock(type="text", text=resp.answer)])

        return write, job

    def _make_edit_tool(self):
        """Wrap the canonical ``edit`` job with update-tracking."""
        job = self.get_job("edit")
        if job is None:
            raise RuntimeError("edit job not registered")

        async def edit(path: str, old: str, new: str) -> ToolResponse:
            resp = await job(path=path, old=old, new=new)
            if resp.success:
                self._updated.append(path)
            return ToolResponse(content=[TextBlock(type="text", text=resp.answer)])

        return edit, job

    def _build_extract_toolkit(self) -> Toolkit:
        """Read-only toolkit for the extract agent. Sub-units come back via
        :class:`ExtractedUnits` structured output, not via a tool call."""
        toolkit = Toolkit()
        for job_name in _EXTRACT_READ_TOOLS:
            self.add_as_tool(toolkit, job_name)
        return toolkit

    def _build_integrate_toolkit(self) -> Toolkit:
        """Full read + canonical write/edit toolkit for the integrate agent.

        write/edit are wrapped in tracker closures (created/updated paths)
        so the outer loop can reconstruct outcomes when an LLM call drops
        its structured emission. Read-only tools go through ``add_as_tool``
        unchanged.
        """
        toolkit = self.toolkit or Toolkit()
        for job_name in _INTEGRATE_READ_TOOLS:
            self.add_as_tool(toolkit, job_name)

        write_tool, write_job = self._make_write_tool()
        toolkit.register_tool_function(
            tool_func=write_tool,
            func_name="write",
            func_description=write_job.description,
            json_schema={
                "type": "function",
                "function": {
                    "name": "write",
                    "description": write_job.description,
                    "parameters": write_job.parameters,
                },
            },
        )

        edit_tool, edit_job = self._make_edit_tool()
        toolkit.register_tool_function(
            tool_func=edit_tool,
            func_name="edit",
            func_description=edit_job.description,
            json_schema={
                "type": "function",
                "function": {
                    "name": "edit",
                    "description": edit_job.description,
                    "parameters": edit_job.parameters,
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
                buckets=", ".join(BUCKETS),
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
            bucket = str(raw.get("bucket") or "").strip()
            if not name or not summary:
                continue
            if bucket not in BUCKETS:
                # Defensive: structured_model should already reject this,
                # but if it slips through we route to wiki (the catch-all).
                self.logger.warning(
                    f"[{self.name}] unit {name!r} emitted bucket {bucket!r} "
                    f"not in {list(BUCKETS)}; routing to 'wiki'",
                )
                bucket = "wiki"
            cleaned.append({"name": name, "summary": summary, "bucket": bucket})
        self._units = cleaned
        return (msg.get_text_content() or "").strip()

    async def _integrate_unit(self, unit: dict, material_blob: str, hint: str, vault_dir: Path) -> IntegrateOutcome:
        """One ReAct invocation per memory sub-unit, dispatched to the
        bucket-specific system prompt. Returns the parsed
        :class:`IntegrateOutcome` reported by the agent.

        File writes happen as side effects via the canonical ``write`` /
        ``edit`` tool calls (which populate ``self._created`` /
        ``self._updated`` via the tracker closures); the structured
        outcome here is the agent's own summary of what it decided —
        useful for rendering and for catching hallucinations (action=
        CREATE without the matching write call landing in trackers).
        """
        bucket = unit.get("bucket") or "wiki"
        toolkit = self._build_integrate_toolkit()
        digest_dir = getattr(self.app_context.app_config, "digest_dir", "")
        agent = FlexReActAgent(
            name=f"reme_dreamer_integrate_{unit.get('name', 'unit')}",
            model=self.as_llm,
            sys_prompt=self.prompt_format(
                f"integrate_system_prompt_{bucket}",
                vault_dir=str(vault_dir),
                digest_dir=digest_dir,
                bucket=bucket,
            ),
            formatter=self.as_llm_formatter,
            toolkit=toolkit,
        )
        agent.set_console_output_enabled(self.console_enabled)
        user_message = self.prompt_format(
            "integrate_user_message",
            hint=hint or "(none)",
            unit_name=unit.get("name", ""),
            unit_bucket=bucket,
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

        vault_dir = self._vault_dir()

        # Phase 1 — extract (light). Agent emits ExtractedUnits structured output to commit the
        # memory sub-units worth lifting. Each unit carries its own bucket.
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
            + ", ".join(f"{u['name']}/{u['bucket']}" for u in self._units),
        )

        # Phase 2 — integrate, one fresh ReAct per sub-unit, dispatched to
        # the bucket-specific system prompt. Python-level loop, not agent
        # loop. Each session emits a structured IntegrateOutcome; file
        # writes happen as side effects via the canonical write / edit
        # tool calls.
        per_unit_lines: list[str] = []
        for i, unit in enumerate(self._units, start=1):
            name = unit.get("name", "?")
            bucket = unit.get("bucket", "?")
            try:
                outcome = await self._integrate_unit(unit, material_blob, hint, vault_dir)
            except Exception as e:
                self.logger.error(
                    f"[{self.name}] integrate {i}/{len(self._units)} "
                    f"(unit={name}, bucket={bucket}) failed: {type(e).__name__}: {e}",
                )
                per_unit_lines.append(f"[{name}/{bucket}] FAILED: {type(e).__name__}: {e}")
                continue
            per_unit_lines.append(_render_outcome_line(name, bucket, outcome))

        summary = (
            f"Declared {len(self._units)} sub-unit(s) "
            + "("
            + ", ".join(f"{u['name']}/{u['bucket']}" for u in self._units)
            + "); "
            f"created {len(self._created)}, updated {len(self._updated)}.\n"
            + "\n".join(per_unit_lines)
        )

        return DreamResult(
            used_llm=True,
            path=path,
            units=list(self._units),
            nodes_created=list(self._created),
            nodes_updated=list(self._updated),
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
