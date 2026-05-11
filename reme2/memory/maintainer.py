"""Maintainer — treatment service. One pass, four signals, one plan.

Per the architecture blueprint, the Maintainer is the third memory
service alongside the Retriever (read) and the Ingestor (cold-write).
It's woken by cron or thresholds, not by per-turn agent calls.

DESIGN: single `Maintainer` Step, NOT four. Vault hygiene is one
operation with multiple signal sources that contend for the same files —
splitting them into independent Steps would let merge & split reverse
each other, let decay archive a file that merge wanted to absorb, and
force every cron tick to scan the vault N times. So the Maintainer
follows a plan-then-apply pipeline:

    scan_signals()         # one walk, all signals shared
        ↓
    propose_*()            # each signal source emits Op records
        ↓
    resolve_conflicts()    # data-driven matrix dedupes / orders ops
        ↓
    apply()                # lint → enrich → discover → decay → merge → split
        ↓
    audit                  # unified trail returned to caller

OPERATIONS

    LintFinding   read-only diagnostic; never conflicts.
    EnrichOp      upgrade a bare `[[X]]` in body to `[predicate:: [[X]]]`
                  via memory_update with a unique context snippet.
    DiscoverOp    append newly discovered `predicate:: [[Y]]` lines under
                  the file's `## Relations` section (creating it if needed).
    DecayOp       move a stale event under <vault>/<archive_dir>/.
    MergeOp       absorb sources[] into canonical; rewrite incoming
                  wikilinks; archive sources.
    SplitOp       extract sections of source[] into new files; replace
                  in original with [[…]] stubs.

CONFLICT MATRIX (resolved before apply)

    MergeOp(source=P)   ⊕ SplitOp(source=P)    → drop split
    MergeOp(source=P)   ⊕ DecayOp(path=P)      → drop decay
    SplitOp(source=P)   ⊕ DecayOp(path=P)      → drop split
    EnrichOp(path=P)    ⊕ DecayOp(path=P)      → drop enrich
    EnrichOp(path=P)    ⊕ MergeOp(touches P)   → drop enrich
    DiscoverOp(path=P)  ⊕ DecayOp(path=P)      → drop discover
    DiscoverOp(path=P)  ⊕ MergeOp(touches P)   → drop discover
    MergeOp(canonical=A,…) × N (same A)        → union sources
    SplitOp(source=P) × N                      → keep highest confidence
    LintFinding ⊕ anything                     → coexist

APPLY ORDER is fixed: lint → enrich → discover → decay → merge → split.
Edits to bodies (enrich, discover) run before file-level rearrangements
(decay/merge/split) so the parser's next pass sees the canonical edge
form before the file moves or merges.

LLM-DRIVEN PROPOSERS (enrich, discover, merge, split) are scaffolded —
the structure + op shape + conflict path are wired and verified, but
the LLM-backed heuristics are pending implementation and currently
return empty proposal lists. Lint and decay are fully implemented.
"""

from __future__ import annotations

import datetime
from collections import defaultdict
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from ..component import R
from ..component.base_step import BaseStep
from ..component.runtime_response import _set_answer
from ..enumeration import ComponentEnum
from . import memory_io
from .schema import parse_frontmatter


# ---------------------------------------------------------------------------
# Op records
# ---------------------------------------------------------------------------


class LintFinding(BaseModel):
    """Read-only diagnostic. Never conflicts; never mutates."""

    op: Literal["lint"] = "lint"
    path: str
    kind: Literal["broken_wikilink", "schema_violation", "stem_collision"]
    detail: str


class DecayOp(BaseModel):
    """Archive a stale event. Moves the file; doesn't change body."""

    op: Literal["decay"] = "decay"
    path: str
    age_days: int
    reason: str = "past freshness window"


class EnrichOp(BaseModel):
    """Wrap a bare `[[X]]` body occurrence as `[predicate:: [[X]]]`.

    Realised at apply-time as `memory_update(path, old_string, new_string)`
    where `old_string` is a unique context window around the bare wikilink
    and `new_string` is the same window with the wikilink wrapped.
    """

    op: Literal["enrich"] = "enrich"
    path: str
    target: str
    predicate: str
    old_string: str
    new_string: str
    confidence: float = 0.0
    reason: str = ""


class DiscoverOp(BaseModel):
    """Append newly proposed `predicate:: [[Y]]` lines under `## Relations`.

    Realised at apply-time by ensuring the file ends with a `## Relations`
    section (creating it if absent) and adding one Dataview line-level
    field per (target, predicate) pair. Idempotency is enforced against
    the file's existing edges before emit.
    """

    op: Literal["discover"] = "discover"
    path: str
    edges: list[dict] = Field(default_factory=list)  # [{target, predicate}, ...]
    confidence: float = 0.0
    reason: str = ""


class MergeOp(BaseModel):
    """Consolidate `sources` into `canonical`. Sources get archived."""

    op: Literal["merge"] = "merge"
    canonical: str
    sources: list[str] = Field(default_factory=list)
    confidence: float = 0.0
    reason: str = ""


class SplitOp(BaseModel):
    """Extract sections of `source` into new sibling files."""

    op: Literal["split"] = "split"
    source: str
    sections: list[dict] = Field(default_factory=list)  # [{title, body, target_path}]
    confidence: float = 0.0
    reason: str = ""


# ---------------------------------------------------------------------------
# Signals: one scan, all observers consume the same dict
# ---------------------------------------------------------------------------


class FileSignal(BaseModel):
    """Per-file derived signals shared by every proposer.

    Built once by `_scan_signals`. Holds only what's cheap to compute
    from `file_store.nodes` + edge index — no body reads, no LLM calls.
    Heavier signals (token counts, embeddings) are pulled lazily inside
    the proposers that actually need them.

    The 4 schema axes (`lifecycle / scope / source / role`) are the
    primary drivers of decay / merge / split heuristics; legacy
    `category` is preserved for back-compat reads only.
    """

    path: str
    relpath: str = ""           # path relative to working_dir, "" if outside
    lifecycle: str = ""         # streaming / evolving / frozen
    scope: str = ""             # instance / class
    source: str = ""            # auto / curated / derived
    role: str = ""              # observation / claim / question / ...
    category: str = ""          # legacy field, kept for migration windows
    status: str = ""
    age_days: int = 0
    metadata: dict = Field(default_factory=dict)
    declared_topics: list[str] = Field(default_factory=list)  # raw [[…]] strings


# ---------------------------------------------------------------------------
# Maintainer
# ---------------------------------------------------------------------------


@R.register("maintainer")
class Maintainer(BaseStep):
    """Unified vault hygiene. Scan → propose → resolve → apply.

    Reads from RuntimeContext (all optional):
        ops          (list[str], default ["lint","decay"]):
                     subset of {"lint","enrich","discover","decay","merge","split"}
                     to run. enrich/discover/merge/split require an LLM
                     and are off by default.
        dry_run      (bool, default True): if True, returns the plan
                     but doesn't mutate the vault.
        decay_days   (int, default constructor `decay_days`): freshness
                     window for the decay proposer.
        target_prefix (str, default ""): restrict scan to relpaths
                     starting with this prefix (e.g. "events/").
        token_threshold (int, default constructor): split threshold.
        merge_threshold (float, default constructor): cluster cutoff.

    Writes to ctx.response.answer:
        {
          "ops_run":    [...],     # which proposers ran
          "scanned":    int,
          "proposed":   [op,...],  # raw, before conflict resolution
          "plan":       [op,...],  # after conflict resolution
          "applied":    [op,...],  # successfully mutated
          "skipped":    [op,...],  # dropped by conflict resolution
          "failed":     [op,...],  # apply errored
          "dry_run":    bool,
          "ran_at":     iso,
        }
    """

    # Fixed apply order — see module docstring CONFLICT MATRIX section.
    _APPLY_ORDER = ("lint", "enrich", "discover", "decay", "merge", "split")

    def __init__(
        self,
        decay_days: int = 90,
        archive_dir: str = "Archive",
        target_status: str = "distilled",
        token_threshold: int = 4000,
        merge_threshold: float = 0.85,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.decay_days = decay_days
        self.archive_dir = archive_dir
        self.target_status = target_status
        self.token_threshold = token_threshold
        self.merge_threshold = merge_threshold

    # -- entry point --------------------------------------------------------

    async def execute(self):
        assert self.context is not None
        params = self._load_params()

        signals = self._scan_signals(target_prefix=params["target_prefix"])

        proposed: list[BaseModel] = []
        if "lint" in params["ops"]:
            proposed.extend(self._propose_lint(signals))
        if "enrich" in params["ops"]:
            proposed.extend(await self._propose_enrich(signals))
        if "discover" in params["ops"]:
            proposed.extend(await self._propose_discover(signals))
        if "decay" in params["ops"]:
            proposed.extend(self._propose_decay(signals, params["decay_days"]))
        if "merge" in params["ops"]:
            proposed.extend(await self._propose_merge(signals, params["merge_threshold"]))
        if "split" in params["ops"]:
            proposed.extend(await self._propose_split(signals, params["token_threshold"]))

        plan, dropped = self._resolve_conflicts(proposed)

        applied: list[dict] = []
        failed: list[dict] = []
        if not params["dry_run"]:
            applied, failed = await self._apply(plan)

        audit = {
            "ops_run": params["ops"],
            "scanned": len(signals),
            "proposed": [_dump(o) for o in proposed],
            "plan": [_dump(o) for o in plan],
            "applied": applied,
            "skipped": [_dump(o) for o in dropped],
            "failed": failed,
            "dry_run": params["dry_run"],
            "ran_at": datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds"),
        }
        self.context.response.success = len(failed) == 0
        _set_answer(self.context, audit)

    # -- params -------------------------------------------------------------

    def _load_params(self) -> dict:
        """Pull RuntimeContext kwargs with proper defaults.

        Centralizing this kills the `int(ctx.get(...))` / `float(ctx.get(...))`
        pattern that bites when `.get` returns None — every conversion
        below has a guaranteed-non-None operand.
        """
        ctx = self.context
        assert ctx is not None
        ops = ctx.get("ops") or ["lint", "decay"]
        if not isinstance(ops, list):
            raise ValueError(f"ops must be a list of strings, got {type(ops).__name__}")
        bad = [o for o in ops if o not in self._APPLY_ORDER]
        if bad:
            raise ValueError(f"unknown ops: {bad}; allowed: {list(self._APPLY_ORDER)}")
        return {
            "ops": ops,
            "dry_run": bool(ctx.get("dry_run", True)),
            "decay_days": int(ctx.get("decay_days") or self.decay_days),
            "target_prefix": str(ctx.get("target_prefix") or ""),
            "token_threshold": int(ctx.get("token_threshold") or self.token_threshold),
            "merge_threshold": float(ctx.get("merge_threshold") or self.merge_threshold),
        }

    # -- scan ---------------------------------------------------------------

    def _working_dir(self) -> Path | None:
        vr = getattr(self.file_store, "working_dir", None)
        return Path(vr).resolve() if vr else None

    def _scan_signals(self, *, target_prefix: str = "") -> list[FileSignal]:
        """One walk over the indexed files. Cheap signals only.

        Each frontmatter is run through `parse_frontmatter` so the legacy
        `category` field is auto-translated to the 4 axes (see
        `LEGACY_AXES_FROM_CATEGORY` in `memory.schema.memory`). Files that
        fail to parse still get a signal — proposers can use the empty
        axes to decide whether to ignore or surface them.
        """
        working_dir = self._working_dir()
        now = datetime.datetime.now().timestamp()
        signals: list[FileSignal] = []
        for path, meta in memory_io.iter_files(self.file_store):
            relpath = ""
            if working_dir is not None:
                try:
                    relpath = str(Path(path).resolve().relative_to(working_dir))
                except ValueError:
                    pass
            if target_prefix and not relpath.startswith(target_prefix):
                continue
            fm = meta.metadata or {}
            age_seconds = max(0.0, now - (meta.st_mtime or now))
            parsed, _ = parse_frontmatter(fm)
            signals.append(FileSignal(
                path=path,
                relpath=relpath,
                lifecycle=str(parsed.lifecycle.value) if parsed else "",
                scope=str(parsed.scope.value) if parsed else "",
                source=str(parsed.source.value) if parsed else "",
                role=str(parsed.role.value) if parsed else "",
                category=str(fm.get("category") or ""),
                status=str(fm.get("status") or ""),
                age_days=int(age_seconds // 86400),
                metadata=fm,
                declared_topics=list(fm.get("topics") or []),
            ))
        return signals

    # -- proposers ----------------------------------------------------------

    def _propose_lint(self, signals: list[FileSignal]) -> list[LintFinding]:
        """Broken wikilinks + frontmatter schema violations + stem collisions."""
        out: list[LintFinding] = []
        for sig in signals:
            for link in sig.declared_topics:
                if not memory_io.resolve_wikilink(self.file_store, link)["exists"]:
                    out.append(LintFinding(
                        path=sig.path, kind="broken_wikilink",
                        detail=f"unresolved wikilink {link!r}",
                    ))
            # Memory schema check: tolerant parse, surface every error
            # the parser collected. Empty `errors` ↔ valid frontmatter.
            _, errors = parse_frontmatter(sig.metadata)
            for err in errors:
                out.append(LintFinding(
                    path=sig.path, kind="schema_violation",
                    detail=f"Memory: {err}"[:240],
                ))
        # Stem collisions: the engine API exposes the ambiguous-stem map.
        ambig = memory_io.find_collisions(self.file_store)
        for stem, paths in ambig.items():
            for p in paths:
                out.append(LintFinding(
                    path=p, kind="stem_collision",
                    detail=f"stem {stem!r} also claimed by {[x for x in paths if x != p]}",
                ))
        return out

    def _propose_decay(
        self, signals: list[FileSignal], decay_days: int,
    ) -> list[DecayOp]:
        """Distilled streaming memories past the freshness window.

        Schema-driven: a memory decays when its `lifecycle` is `streaming`
        (write-once, decays after a freshness window) AND its `status` has
        already been flipped to the configured terminal target (default:
        `distilled`). Evolving / frozen memories never decay.
        """
        out: list[DecayOp] = []
        for sig in signals:
            if sig.lifecycle != "streaming":
                continue
            if sig.status != self.target_status:
                continue
            if sig.age_days < decay_days:
                continue
            out.append(DecayOp(
                path=sig.path, age_days=sig.age_days,
                reason=f"streaming {sig.status!r} for {sig.age_days}d (≥{decay_days}d window)",
            ))
        return out

    async def _propose_enrich(
        self, signals: list[FileSignal],
    ) -> list[EnrichOp]:
        """Bare wikilinks → typed via inline-bracketed Dataview wrap.

        SCAFFOLD: returns []. When implemented:
            1. For each FileSignal, fetch existing edges via
               `file_store.get_edges(path)` and filter to `predicate is None`.
            2. Read body, ask the LLM (one call per file) to assign one
               of `ALLOWED_PREDICATES` to each bare target — or 'skip'.
            3. For each accepted (target, predicate), locate each bare
               occurrence span via `parse_wikilinks` and build a unique
               context window snippet.
            4. Emit EnrichOp(path, target, predicate, old_string, new_string,
               confidence, reason). The apply step calls memory_update.

        Idempotency: re-running won't re-enrich already-typed edges
        (filter is `predicate is None`). The OOV-tolerant FileEdge
        validator means a malformed LLM output collapses to None at the
        next parse, surfacing it again next sweep — bounded retries
        avoid infinite re-enrichment loops.
        """
        return []

    async def _propose_discover(
        self, signals: list[FileSignal],
    ) -> list[DiscoverOp]:
        """Discover edges absent from body — append to `## Relations`.

        SCAFFOLD: returns []. When implemented:
            1. For each FileSignal, gather body + existing edges +
               `memory_search` neighborhood candidates (top-k by query =
               file's title/description).
            2. Ask the LLM to propose new (target, predicate) pairs that
               aren't already present in the file's edges, drawing only
               from candidate paths it can resolve.
            3. Filter against the file's existing `(target, predicate)`
               set so re-runs don't re-emit the same edge.
            4. Emit DiscoverOp(path, edges=[{target,predicate}, ...],
               confidence, reason). Apply appends them as Dataview
               line-level fields under `## Relations` (creating the
               heading if missing).
        """
        return []

    async def _propose_merge(
        self, signals: list[FileSignal], threshold: float,
    ) -> list[MergeOp]:
        """Cluster near-duplicate topics → MergeOps (LLM-assisted).

        SCAFFOLD: the structure is wired but the clustering implementation
        is pending. Returns []. When implemented:
            1. Embed titles + descriptions; cluster on cosine ≥ threshold.
            2. For each cluster, ask the LLM to pick the canonical and
               summarize what the merged body should preserve.
            3. Emit MergeOp(canonical, sources=[non-canonical], confidence,
               reason=LLM justification).
        """
        return []

    async def _propose_split(
        self, signals: list[FileSignal], token_threshold: int,
    ) -> list[SplitOp]:
        """Topics over the token threshold → SplitOps (LLM-assisted).

        SCAFFOLD: returns []. When implemented:
            1. Filter signals to topics whose body exceeds token_threshold.
            2. Read body, ask the LLM for a section partition + filenames.
            3. Emit SplitOp(source, sections=[{title,body,target_path}],
               confidence, reason).
        """
        return []

    # -- conflict resolution ------------------------------------------------

    def _resolve_conflicts(
        self, proposed: list[BaseModel],
    ) -> tuple[list[BaseModel], list[BaseModel]]:
        """Apply the conflict matrix; return (kept_plan, dropped).

        Steps:
            1. Union MergeOps that share a canonical.
            2. Dedupe SplitOps by source (highest confidence wins).
            3. For each path, resolve {merge-source, split-source, decay,
               enrich, discover} contention per the matrix in the module
               docstring.
            4. Lint findings are pass-through.
        """
        lints = [o for o in proposed if isinstance(o, LintFinding)]
        enriches = [o for o in proposed if isinstance(o, EnrichOp)]
        discovers = [o for o in proposed if isinstance(o, DiscoverOp)]
        merges = [o for o in proposed if isinstance(o, MergeOp)]
        splits = [o for o in proposed if isinstance(o, SplitOp)]
        decays = [o for o in proposed if isinstance(o, DecayOp)]

        dropped: list[BaseModel] = []

        # (1) Union merges by canonical.
        merged_by_canon: dict[str, MergeOp] = {}
        for m in merges:
            existing = merged_by_canon.get(m.canonical)
            if existing is None:
                merged_by_canon[m.canonical] = m
            else:
                merged_sources = list(dict.fromkeys(existing.sources + m.sources))
                existing.sources = merged_sources
                existing.confidence = max(existing.confidence, m.confidence)
                existing.reason = (existing.reason + "; " + m.reason).strip("; ")
        merges = list(merged_by_canon.values())

        # (2) Dedupe splits by source.
        split_by_source: dict[str, SplitOp] = {}
        for s in splits:
            current = split_by_source.get(s.source)
            if current is None or s.confidence > current.confidence:
                if current is not None:
                    dropped.append(current)
                split_by_source[s.source] = s
            else:
                dropped.append(s)
        splits = list(split_by_source.values())

        # (3) Build path → owning op index.
        merge_paths: set[str] = set()
        for m in merges:
            merge_paths.update(m.sources)
            merge_paths.add(m.canonical)

        kept_splits: list[SplitOp] = []
        for s in splits:
            if s.source in merge_paths:
                dropped.append(s)  # MergeOp ⊕ SplitOp(source=P) → drop split
            else:
                kept_splits.append(s)

        split_sources = {s.source for s in kept_splits}

        kept_decays: list[DecayOp] = []
        for d in decays:
            if d.path in merge_paths:
                dropped.append(d)  # MergeOp ⊕ DecayOp(P) → drop decay
            elif d.path in split_sources:
                # SplitOp(source=P) ⊕ DecayOp(P) → matrix says drop split,
                # but split was kept above (no merge contention) so the
                # decay wins here: archive trumps refining what's about
                # to leave the active set.
                # → drop the split, keep decay.
                for s in list(kept_splits):
                    if s.source == d.path:
                        kept_splits.remove(s)
                        dropped.append(s)
                kept_decays.append(d)
            else:
                kept_decays.append(d)

        plan: list[BaseModel] = []
        plan.extend(lints)
        # (4) Body-edits drop if file is being decayed/merged.
        decay_paths = {d.path for d in kept_decays}
        kept_enriches: list[EnrichOp] = []
        for e in enriches:
            if e.path in merge_paths or e.path in decay_paths:
                dropped.append(e)
            else:
                kept_enriches.append(e)
        kept_discovers: list[DiscoverOp] = []
        for d in discovers:
            if d.path in merge_paths or d.path in decay_paths:
                dropped.append(d)
            else:
                kept_discovers.append(d)
        plan.extend(kept_enriches)
        plan.extend(kept_discovers)
        plan.extend(kept_decays)
        plan.extend(merges)
        plan.extend(kept_splits)
        return plan, dropped

    # -- apply --------------------------------------------------------------

    async def _apply(
        self, plan: list[BaseModel],
    ) -> tuple[list[dict], list[dict]]:
        """Execute the plan in fixed order. Each phase yields audit dicts.

        Lint never mutates → recorded as applied no-op so the audit shows
        which findings the cron run surfaced.

        Decay / Merge / Split call into the existing memory_io primitives
        once the heuristics land; for now they record `pending_apply` so
        the dry_run=False path still produces a stable audit shape.
        """
        applied: list[dict] = []
        failed: list[dict] = []

        # Group ops by type, apply in fixed order.
        by_type: dict[str, list] = defaultdict(list)
        for o in plan:
            by_type[o.op].append(o)  # type: ignore[attr-defined]

        for kind in self._APPLY_ORDER:
            for op in by_type.get(kind, []):
                try:
                    result = await self._apply_one(op)
                    applied.append({"op": op.op, **result})  # type: ignore[attr-defined]
                except Exception as e:
                    failed.append({
                        "op": op.op,  # type: ignore[attr-defined]
                        "payload": _dump(op),
                        "error": f"{type(e).__name__}: {e}",
                    })
        return applied, failed

    async def _apply_one(self, op: BaseModel) -> dict:
        if isinstance(op, LintFinding):
            # Diagnostic-only: surfacing the finding IS the work.
            return {"path": op.path, "kind": op.kind, "noop": True}
        if isinstance(op, EnrichOp):
            # TODO: wire to memory_io.update_body once apply-side is enabled.
            return {"path": op.path, "target": op.target, "predicate": op.predicate,
                    "status": "pending_apply",
                    "would": "memory_update wraps bare [[X]] as [predicate:: [[X]]]"}
        if isinstance(op, DiscoverOp):
            # TODO: append Dataview lines under ## Relations (create heading
            # if absent), then call file_store invalidation.
            return {"path": op.path, "edges_added": len(op.edges),
                    "status": "pending_apply",
                    "would": "append predicate:: [[Y]] under ## Relations heading"}
        if isinstance(op, DecayOp):
            # TODO: wire to MemoryArchive once we're ready to actually
            # move files from a cron context. For now, surface intent.
            return {"path": op.path, "status": "pending_apply",
                    "would": "flip status=archived + move to archive_dir"}
        if isinstance(op, MergeOp):
            return {"canonical": op.canonical, "sources": op.sources,
                    "status": "pending_apply",
                    "would": "merge bodies + rewrite incoming wikilinks + archive sources"}
        if isinstance(op, SplitOp):
            return {"source": op.source, "sections": len(op.sections),
                    "status": "pending_apply",
                    "would": "extract sections + replace with [[…]] stubs"}
        raise TypeError(f"unknown op type: {type(op).__name__}")


def _dump(op: BaseModel) -> dict:
    """JSON-friendly snapshot of an op record."""
    return op.model_dump()
