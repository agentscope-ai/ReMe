"""Early candidate admission using read-only, commit-pinned source exports."""

from contextlib import asynccontextmanager
import shutil
from typing import Literal
from uuid import uuid4

from pydantic import Field

from reme.utils.async_utils import complete_in_thread as blocking

from ..code.repository import current_branch, validate_candidate, validate_repository
from ..code.snapshot import export_blob, tree
from ..evaluation.artifacts import contained, read_json
from ..evaluation.repetition import previous_candidate_result
from ..preparation.code import git
from ..preparation.spec import SpecModel
from .workflow_common import review_report


class CandidateGateReport(SpecModel):
    """A rejection must be an explicit, schema-validated model decision."""

    decision: Literal["accept", "reject"]
    reason: str = Field(min_length=1, pattern=r"\S")


def admission_context(home):
    """Called under the workspace lock; retries and the baseline bypass admission."""
    repo, record, commit = validate_candidate(home)
    if commit == record["code"]["initial_commit"] or previous_candidate_result(home, commit) is not None:
        return None
    ancestors = set(git(repo, "rev-list", commit).splitlines())
    candidates = []
    for path in sorted((home / "evaluations").glob("*/*/*/manifest.json")):
        manifest = read_json(contained(home, path))
        previous = manifest.get("commit_sha")
        if previous != commit and previous in ancestors:
            distance = int(git(repo, "rev-list", "--count", f"{previous}..{commit}"))
            candidates.append((distance, previous))
    base = min(candidates)[1] if candidates else record["code"]["initial_commit"]
    return {"base_commit": base, "candidate_commit": commit, "branch_name": current_branch(repo)}


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
