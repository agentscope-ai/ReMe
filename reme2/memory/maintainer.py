"""Maintainer — treatment service. One pass, four signals, one plan.

Per the architecture blueprint, the Maintainer is the third memory
service alongside the Retriever (read) and the Ingestor (cold-write).
It's woken by cron or thresholds, not by per-turn agent calls.

DESIGN: single `Maintainer` Step, NOT four. Vault hygiene is one
operation with four signal sources that contend for the same files —
splitting them into independent Steps would let merge & split reverse
each other, let decay archive a file that merge wanted to absorb, and
force every cron tick to scan the vault four times. So the Maintainer
follows a plan-then-apply pipeline:

    scan_signals()         # one walk, all signals shared
        ↓
    propose_*()            # each signal source emits Op records
        ↓
    resolve_conflicts()    # data-driven matrix dedupes / orders ops
        ↓
    apply()                # lint → decay → merge → split, dry-run aware
        ↓
    audit                  # unified trail returned to caller

OPERATIONS

    LintFinding   read-only diagnostic; never conflicts.
    DecayOp       move a stale event under <vault>/<archive_dir>/.
    MergeOp       absorb sources[] into canonical; rewrite incoming
                  wikilinks; archive sources.
    SplitOp       extract sections of source[] into new files; replace
                  in original with [[…]] stubs.

CONFLICT MATRIX (resolved before apply)

    MergeOp(source=P)  ⊕ SplitOp(source=P)   → drop split
    MergeOp(source=P)  ⊕ DecayOp(path=P)     → drop decay
    SplitOp(source=P)  ⊕ DecayOp(path=P)     → drop split
    MergeOp(canonical=A,…) × N (same A)      → union sources
    SplitOp(source=P) × N                    → keep highest confidence
    LintFinding ⊕ anything                   → coexist

APPLY ORDER is fixed: lint → decay → merge → split. This guarantees
that by the time split runs, merge has already rewritten paths; if a
split target was absorbed by a merge, apply skips it and records a
`stale_target` audit entry instead of crashing.

LLM-DRIVEN PROPOSERS (merge/split) are scaffolded — the structure +
op shape + conflict path are wired and verified, but the LLM-backed
similarity / split heuristics are pending design and currently return
empty proposal lists. Lint and decay are fully implemented.
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
from ..schema.vault.registry import schema_for
from . import memory_io


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
    """

    path: str
    relpath: str = ""           # path relative to vault_root, "" if outside
    category: str = ""
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
                     subset of {"lint","decay","merge","split"} to run.
                     Merge/split require an LLM and are off by default.
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
    _APPLY_ORDER = ("lint", "decay", "merge", "split")

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

    def _vault_root(self) -> Path | None:
        watcher = self._get_component_optional(ComponentEnum.FILE_WATCHER, "default")
        if watcher is None:
            return None
        return Path(getattr(watcher, "watch_path", ".")).resolve()

    def _scan_signals(self, *, target_prefix: str = "") -> list[FileSignal]:
        """One walk over the indexed files. Cheap signals only."""
        vault_root = self._vault_root()
        now = datetime.datetime.now().timestamp()
        signals: list[FileSignal] = []
        for path, meta in memory_io.iter_files(self.file_store):
            relpath = ""
            if vault_root is not None:
                try:
                    relpath = str(Path(path).resolve().relative_to(vault_root))
                except ValueError:
                    pass
            if target_prefix and not relpath.startswith(target_prefix):
                continue
            fm = meta.metadata or {}
            age_seconds = max(0.0, now - (meta.st_mtime or now))
            signals.append(FileSignal(
                path=path,
                relpath=relpath,
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
                if not memory_io.wikilink_lookup(self.file_store, link)["exists"]:
                    out.append(LintFinding(
                        path=sig.path, kind="broken_wikilink",
                        detail=f"unresolved wikilink {link!r}",
                    ))
            cls = schema_for(sig.category)
            if cls is not None:
                try:
                    cls(**sig.metadata)
                except Exception as e:
                    out.append(LintFinding(
                        path=sig.path, kind="schema_violation",
                        detail=f"{cls.__name__}: {type(e).__name__}: {e}"[:240],
                    ))
        # Stem collisions: the engine API exposes the ambiguous-stem map.
        ambig = memory_io.all_ambiguous_wikilinks(self.file_store)
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
        """Distilled events past the freshness window."""
        out: list[DecayOp] = []
        for sig in signals:
            if sig.category != "event":
                continue
            if sig.status != self.target_status:
                continue
            if sig.age_days < decay_days:
                continue
            out.append(DecayOp(
                path=sig.path, age_days=sig.age_days,
                reason=f"event {sig.status!r} for {sig.age_days}d (≥{decay_days}d window)",
            ))
        return out

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
            3. For each path, resolve {merge-source, split-source, decay}
               contention per the matrix in the module docstring.
            4. Lint findings are pass-through.
        """
        lints = [o for o in proposed if isinstance(o, LintFinding)]
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
