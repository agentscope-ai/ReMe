"""DreamStep — single-file digest step (auto-dream's create_or_update primitive).

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
        units, _ = _extract(material_blob)  # 1× ReAct: identify abstractions
                                            #   agent emits ExtractedUnits
                                            #   ({units: [{name, bucket, summary}, ...]})
        for unit in units:                  # Python loop, K iterations
            _integrate_unit(unit)           # 1× ReAct per abstraction, dispatched
                                            #   to integrate_system_prompt_<bucket>;
                                            #   recalls cross-bucket, decides write,
                                            #   uses canonical write/edit/frontmatter_update tools.

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

from agentscope.agent import Agent
from agentscope.message import Msg, TextBlock
from agentscope.permission import PermissionContext, PermissionMode
from agentscope.state import AgentState
from agentscope.tool import Toolkit
from pydantic import BaseModel, Field

from ..base_step import BaseStep
from ...components import R


# Hard-coded bucket vocabulary. Phase 1 classifies each sub-unit into
# one of these; Phase 2 dispatches to the bucket-specific prompt.
# Order matters for prompt rendering — keep procedure/personal/wiki.
BUCKETS: tuple[str, ...] = ("procedure", "personal", "wiki")

# Bucket = Literal of BUCKETS. Pydantic Literal must be a static type;
# update both BUCKETS and Bucket together if the vocabulary changes.
Bucket = Literal["procedure", "personal", "wiki"]


_EXTRACT_TOOLS: tuple[str, ...] = ("read",)

_INTEGRATE_TOOLS: tuple[str, ...] = (
    # read — dream uses its own node-level digest search (NOT the
    # general chunk-level `search`), specialized for dedup + synapse
    # recall. See reme4/steps/index/node_search.py for the rationale.
    # NO traverse here: traverse is a retrieve-time subgraph mining
    # tool (used by external retrieval agents); dream is a write-time
    # candidate recall operation, structurally a different problem.
    "node_search",
    "read",
    "frontmatter_read",
    # write
    "write",
    "edit",
    "frontmatter_update",
)


# ============================================================
# Schema (Pydantic models for structured ReAct output) + helpers
# ============================================================


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
        description=("The digest path you wrote to — must match what your `write` / " "`edit` call(s) targeted."),
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
    """Outcome of one DreamStep invocation.

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


# ============================================================
# DreamStep — the per-file create_or_update step.
# ============================================================


@R.register("dream_step")
class DreamStep(BaseStep):
    """auto-dream create_or_update step — one file per call.

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
        timezone: str | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.toolkit = toolkit
        self.timezone = timezone

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

    def _build_extract_toolkit(self) -> Toolkit:
        """Read-only toolkit for the extract agent. Sub-units come back via
        :class:`ExtractedUnits` structured output, not via a tool call."""
        toolkit = Toolkit()
        for job_name in _EXTRACT_TOOLS:
            self.add_as_tool(toolkit, job_name)
        return toolkit

    def _build_integrate_toolkit(self) -> Toolkit:
        """Full read + canonical write/edit/frontmatter_update toolkit for
        the integrate agent. All tools are registered via :meth:`add_as_tool`
        — same as every other step in this codebase. Outcome tracking is
        driven by the agent's :class:`IntegrateOutcome` structured emission,
        not by per-tool callbacks."""
        toolkit = self.toolkit or Toolkit()
        for job_name in _INTEGRATE_TOOLS:
            self.add_as_tool(toolkit, job_name)
        return toolkit

    async def _extract(self, material_blob: str, hint: str, vault_dir: Path) -> tuple[list[dict], str]:
        """Phase 1: one ReAct invocation — read material + emit ExtractedUnits.

        Returns ``(units, llm_summary)`` where ``units`` is the cleaned
        sub-unit list (each entry has ``name`` / ``bucket`` / ``summary``)
        and ``llm_summary`` is whatever free-form text the agent produced
        alongside its structured emission.
        """
        toolkit = self._build_extract_toolkit()
        agent = Agent(
            name="reme_dreamer_extract",
            model=self.as_llm,
            system_prompt=self.prompt_format(
                "extract_system_prompt",
                vault_dir=str(vault_dir),
                buckets=", ".join(BUCKETS),
            ),
            toolkit=toolkit,
            state=AgentState(
                permission_context=PermissionContext(
                    mode=PermissionMode.BYPASS,
                ),
            ),
        )
        user_message = self.prompt_format(
            "extract_user_message",
            today=self._now().strftime("%Y-%m-%d"),
            hint=hint or "(none)",
            material_blob=material_blob,
        )
        msg = await agent.reply(
            Msg(name="reme", role="user", content=[TextBlock(text=user_message)]),
        )

        structured_resp = await self.as_llm.generate_structured_output(
            agent.state.context,
            structured_model=ExtractedUnits,
        )
        meta = structured_resp.content if isinstance(structured_resp.content, dict) else {}
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
        return cleaned, (msg.get_text_content() or "").strip()

    async def _integrate_unit(self, unit: dict, material_blob: str, hint: str, vault_dir: Path) -> IntegrateOutcome:
        """One ReAct invocation per memory sub-unit, dispatched to the
        bucket-specific system prompt. Returns the parsed
        :class:`IntegrateOutcome` reported by the agent — that's the
        single source of truth for what got written (action +
        target_path)."""
        bucket = unit.get("bucket") or "wiki"
        toolkit = self._build_integrate_toolkit()
        digest_dir = getattr(self.app_context.app_config, "digest_dir", "")
        agent = Agent(
            name=f"reme_dreamer_integrate_{unit.get('name', 'unit')}",
            model=self.as_llm,
            system_prompt=self.prompt_format(
                f"integrate_system_prompt_{bucket}",
                vault_dir=str(vault_dir),
                digest_dir=digest_dir,
                bucket=bucket,
            ),
            toolkit=toolkit,
            state=AgentState(
                permission_context=PermissionContext(
                    mode=PermissionMode.BYPASS,
                ),
            ),
        )
        user_message = self.prompt_format(
            "integrate_user_message",
            hint=hint or "(none)",
            unit_name=unit.get("name", ""),
            unit_bucket=bucket,
            unit_summary=unit.get("summary", ""),
            material_blob=material_blob,
        )
        await agent.reply(
            Msg(name="reme", role="user", content=[TextBlock(text=user_message)]),
        )
        structured_resp = await self.as_llm.generate_structured_output(
            agent.state.context,
            structured_model=IntegrateOutcome,
        )
        return IntegrateOutcome.model_validate(structured_resp.content)

    async def dream_one(self, path: str, hint: str = "") -> DreamResult:
        """Run the full extract + integrate pipeline on one vault-relative
        material path. Returns a structured :class:`DreamResult`. Safe to
        call repeatedly on the same instance — per-invocation trackers are
        reset at the start of each call. Used by :meth:`execute` (single
        file from context), :class:`AutoDreamStep` (loop over today's
        materials), and :class:`WatchDreamStep` (loop over a change batch).
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
                error="no llm configured; dreaming requires an LLM",
            )

        material_blob = _pack_material(self.file_store, path)

        vault_dir = self._vault_dir()

        # Phase 1 — extract (light). Agent emits ExtractedUnits structured output to commit the
        # memory sub-units worth lifting. Each unit carries its own bucket.
        self.logger.info(f"[{self.name}] extract phase: path={path!r}")
        units, extract_summary = await self._extract(material_blob, hint, vault_dir)

        if not units:
            return DreamResult(
                used_llm=True,
                path=path,
                summary=extract_summary or "no memory sub-units declared",
                skipped=True,
            )

        unit_handles = ", ".join(f"{u['name']}/{u['bucket']}" for u in units)
        self.logger.info(f"[{self.name}] integrate phase: {len(units)} sub-unit(s): {unit_handles}")

        # Phase 2 — integrate, one fresh ReAct per sub-unit, dispatched to
        # the bucket-specific system prompt. Python-level loop, not agent
        # loop. Each session emits a structured IntegrateOutcome whose
        # action + target_path are the source of truth for what landed.
        nodes_created: list[str] = []
        nodes_updated: list[str] = []
        per_unit_lines: list[str] = []
        for i, unit in enumerate(units, start=1):
            name = unit.get("name", "?")
            bucket = unit.get("bucket", "?")
            try:
                outcome = await self._integrate_unit(unit, material_blob, hint, vault_dir)
            except Exception as e:
                self.logger.error(
                    f"[{self.name}] integrate {i}/{len(units)} "
                    f"(unit={name}, bucket={bucket}) failed: {type(e).__name__}: {e}",
                )
                per_unit_lines.append(f"[{name}/{bucket}] FAILED: {type(e).__name__}: {e}")
                continue
            if outcome.action == "CREATE":
                nodes_created.append(outcome.target_path)
            else:
                nodes_updated.append(outcome.target_path)
            per_unit_lines.append(_render_outcome_line(name, bucket, outcome))

        per_unit_block = "\n".join(per_unit_lines)
        summary = (
            f"Declared {len(units)} sub-unit(s) ({unit_handles}); "
            f"created {len(nodes_created)}, updated {len(nodes_updated)}.\n"
            f"{per_unit_block}"
        )

        return DreamResult(
            used_llm=True,
            path=path,
            units=units,
            nodes_created=nodes_created,
            nodes_updated=nodes_updated,
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
