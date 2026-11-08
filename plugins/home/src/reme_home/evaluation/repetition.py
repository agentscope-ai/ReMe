"""Reject unchanged scored candidates while retaining recoverable attempts."""

from pathlib import Path

from ..preparation.code import git
from .artifacts import contained, read_json


class UnchangedCandidateError(ValueError):
    """A scored candidate needs a code change before another evaluation."""

    def __init__(self, path: Path):
        self.path = path
        super().__init__(
            "Current code has already been evaluated. Inspect the previous result, "
            f"make a targeted code change, and commit before evaluating again: {path}",
        )


def _read(run: Path, relative: str) -> dict:
    path = contained(run, relative)
    if not path.is_file():
        return {}
    value = read_json(path)
    if not isinstance(value, dict):
        raise ValueError(f"Invalid evaluation record: {path}")
    return value


def retryable_result(run: Path) -> bool:
    """Infrastructure errors and interruptions allow retry; candidate failures do not."""
    summary = _read(run, "summary.json")
    if summary:
        cases = summary.get("cases", {})
        cases = list(cases.values()) if isinstance(cases, dict) else cases
        counts = summary.get("case_status_counts", {})
        if counts:
            if counts.get("infra_error", 0):
                return True
            terminal_count = counts.get("completed", 0) + counts.get("candidate_failure", 0)
            return not cases or terminal_count != len(cases)
        # Older summaries can carry statuses on individual cases.
        if any(case.get("status") == "infra_error" for case in cases):
            return True
        return not cases or any(case.get("status") not in {"completed", "candidate_failure"} for case in cases)
    failures = [_read(run, path.relative_to(run).as_posix()) for path in run.glob("cases/**/failure.json")]
    for path in run.glob("cases/*/queries/result.json"):
        failures.extend(_read(run, path.relative_to(run).as_posix()).get("queries", []))
    if any(
        item.get("status") == "infra_error"
        or item.get("failure_kind") == "infra_error"
        or item.get("infrastructure_error")
        for item in failures
    ):
        return True
    return not any(
        item.get("status") == "candidate_failure" or item.get("failure_kind") == "candidate_failure"
        for item in failures
    )


def previous_candidate_result(home: Path, commit: str) -> Path | None:
    """Check every scored training attempt under the formal validation lock.

    Match Git trees across branches and empty commits. Only the exact failed
    commit can retry; an identical tree under another commit remains unchanged.
    Read source artifacts directly, including attempts excluded from history.
    """
    repository = contained(home, "code/repo", directory=True)
    current_tree = git(repository, "rev-parse", "--verify", "--end-of-options", f"{commit}^{{tree}}")
    recoverable = None
    evaluations = contained(home, "evaluations", directory=True)
    for path in sorted(evaluations.glob("*/*/*/manifest.json")):
        run = contained(home, path.parent, directory=True)
        manifest = _read(run, "manifest.json")
        if manifest.get("dataset_kind", "training-set") != "training-set" or manifest.get("mode") == "debug":
            continue
        previous_commit = manifest.get("commit_sha")
        if not previous_commit:
            continue
        previous_tree = git(repository, "rev-parse", "--verify", "--end-of-options", f"{previous_commit}^{{tree}}")
        if previous_tree != current_tree:
            continue
        if previous_commit != commit or not retryable_result(run):
            raise UnchangedCandidateError(run)
        recoverable = run
    return recoverable
