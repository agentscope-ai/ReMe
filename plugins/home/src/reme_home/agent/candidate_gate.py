"""Early candidate admission using read-only, commit-pinned source exports."""

from contextlib import asynccontextmanager
from pathlib import Path
import shutil
from typing import Literal, NamedTuple
from uuid import uuid4

from pydantic import Field

from reme.utils.async_utils import complete_in_thread as blocking

from ..code.repository import current_branch, validate_candidate, validate_repository
from ..code.snapshot import export_blob, tree
from ..evaluation.artifacts import contained, read_json
from ..evaluation.history import read_state
from ..evaluation.repetition import previous_candidate_result
from ..preparation.code import git
from ..preparation.spec import SpecModel
from .workflow_common import review_report


class CandidateGateReport(SpecModel):
    """A rejection must be an explicit, schema-validated model decision."""

    decision: Literal["accept", "reject"]
    reason: str = Field(min_length=1, pattern=r"\S")


class AdmissionCheckout(NamedTuple):
    """Pin the checkout and recovery result even when a review is skipped."""

    commit: str
    branch: str
    previous_result: Path | None


def _admission_base(home: Path, repo: Path, commit: str, initial: str) -> str:
    """Find the nearest ancestor with a budget charge and a usable formal result."""
    state = read_state(home)
    started = state["started_validations"]
    evaluated: dict[str, int] = {}
    # Sampling history records completed formal results without infrastructure errors.
    for key in state["processed_validations"]:
        if key.startswith("init/") or key not in started:
            continue
        manifest = read_json(contained(home, f"evaluations/{key}/manifest.json"))
        sha = manifest["commit_sha"]
        evaluated[sha] = max(evaluated.get(sha, 0), started[key])
    if not evaluated:
        return initial

    revisions = git(repo, "rev-list", "--parents", commit).splitlines()
    parents = {sha: lineage for sha, *lineage in (line.split() for line in revisions)}
    visited = {commit}
    frontier = set(parents[commit])
    while frontier:
        matches = frontier & evaluated.keys()
        if matches:
            # Start rounds order formal evaluations even when timestamps coincide.
            return max(matches, key=evaluated.__getitem__)
        visited.update(frontier)
        frontier = {parent for sha in frontier for parent in parents[sha]} - visited
    return initial


def admission_context(home: Path) -> tuple[AdmissionCheckout, dict[str, str] | None]:
    """Capture the checkout and optional review context under the workspace lock."""
    repo, record, commit = validate_candidate(home)
    initial = record["code"]["initial_commit"]
    if commit == initial:
        return AdmissionCheckout(commit, "init", None), None
    checkout = AdmissionCheckout(commit, current_branch(repo), previous_candidate_result(home, commit))
    if checkout.previous_result is not None:
        return checkout, None
    base = _admission_base(home, repo, commit, initial)
    return checkout, {"base_commit": base, "candidate_commit": commit, "branch_name": checkout.branch}


@asynccontextmanager
async def review_export(home, category):
    """Own only this review's export, including partial preparation and cancellation."""
    root = contained(home, f"tmp/{category}-{uuid4().hex}", directory=True)
    root.mkdir()
    try:
        yield root
    finally:
        await blocking(shutil.rmtree, root)


def prepare_review_scope(home, context, *, root):
    """Export regular Git blobs only; expose neither live code nor Git metadata."""
    repo, _ = validate_repository(home)
    for label, commit in (("base", context["base_commit"]), ("candidate", context["candidate_commit"])):
        target = root / label
        target.mkdir()
        for name, oid in tree(repo, commit):
            path = target / name
            path.parent.mkdir(parents=True, exist_ok=True)
            export_blob(repo, oid, path)
    # Keep the entire diff on disk; Read/Grep provide bounded model-visible output.
    diff = git(
        repo,
        "diff",
        "--no-ext-diff",
        "--no-textconv",
        "--no-renames",
        "--ignore-submodules=all",
        context["base_commit"],
        context["candidate_commit"],
        "--",
        strip=False,
    )
    (root / "diff.patch").write_text(diff, encoding="utf-8")
    return {"scratch": str(root), "read_only_paths": [str(root)], "writable_paths": [], "clean_environment": True}


async def review_candidate(wrapper, home, context, budget, *, memory_limit_mib=None):
    """Fail closed on sandbox setup; model/report failures allow validation as in dev_meta."""
    async with review_export(home, "candidate-gate") as root:
        scope = await blocking(prepare_review_scope, home, context, root=root)
        return await review_report(
            wrapper,
            home,
            scope,
            f"Review admission from {context['base_commit']} to {context['candidate_commit']}.\n"
            f"Source exports and complete diff.patch: {scope['scratch']}",
            {**context, "validation_budget": budget, "source_path": scope["scratch"]},
            role="candidate_gate",
            sandbox_label="Candidate gate",
            category="candidate-gates",
            report_model=CandidateGateReport,
            failure_decision="accept",
            failure_reason="Candidate gate failed; validation is allowed",
            memory_limit_mib=memory_limit_mib,
        )
