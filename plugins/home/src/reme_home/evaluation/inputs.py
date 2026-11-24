"""Validate prepared inputs and snapshot committed candidates before execution."""

from pathlib import Path
from typing import Literal

import shutil
import tempfile
import tarfile
import zlib

from pydantic import Field, field_validator, model_validator

from ..preparation.code import export_git_tree, git, git_tree, runtime_roots, tree_fingerprint
from ..preparation.data.basic import artifact_name_key, validate_artifact_ids
from ..preparation.data.models import CaseSpec, DatasetManifest, fingerprint
from .artifacts import contained, file_sha256, inspect_archive, read_json
from .docker import SourceSnapshot
from ..preparation.spec import SpecModel
from ..code.repository import validate_repository
from ..sampling import SamplingPolicy


class EvaluationSpec(SpecModel):
    """Selections are injected by callers; omitted cases mean the whole dataset."""

    case_ids: list[str] | None = None
    query_ids: dict[str, list[str]] | None = None
    target_queries: dict[str, list[str] | None] | None = None
    reuse_construction_from: str | None = Field(default=None, min_length=1)
    debug: bool = False
    sampling: bool = False
    expected_commit: str | None = Field(default=None, min_length=1)
    expected_branch: str | None = Field(default=None, min_length=1)
    record_history: bool = True
    max_evaluations: int | None = Field(default=None, ge=1)
    sampling_policy: SamplingPolicy | None = None
    concurrency: int = Field(default=4, ge=1)
    timeout_seconds: float = Field(default=7200.0, gt=0, allow_inf_nan=False)
    base_image: str = Field(default="reme-benchmark-base:agentscope-2.0.11", min_length=1)
    fail_fast: bool = False
    max_construction_failure_rate: float = Field(default=0.03, ge=0, le=1, allow_inf_nan=False)
    max_artifact_bytes: int = Field(default=2_000_000_000, ge=1)

    @property
    def dataset_kind(self) -> Literal["training-set", "debug-cases"]:
        """Resolve the internal dataset location from the requested mode."""
        return "debug-cases" if self.debug else "training-set"

    @field_validator("case_ids")
    @classmethod
    def validate_cases(cls, value):
        """An explicit selection must never silently expand or shrink."""
        if value is not None and (
            not value or any(not item.strip() for item in value) or len(set(value)) != len(value)
        ):
            raise ValueError("case_ids must be nonempty, unique strings, or null for all cases")
        return value

    @field_validator("query_ids")
    @classmethod
    def validate_queries(cls, value):
        """Validate each explicit per-case query selection."""
        if value is not None:
            if not value:
                raise ValueError("query_ids must not be empty")
            for case_id, ids in value.items():
                if not case_id.strip() or not ids or any(not item.strip() for item in ids) or len(set(ids)) != len(ids):
                    raise ValueError("query_ids must contain nonempty unique selections")
        return value

    @model_validator(mode="after")
    def resolve_mode(self):
        """Debug stops on failure by default and never reuses scored evidence."""
        if (self.expected_commit is None) != (self.expected_branch is None):
            raise ValueError("expected_commit and expected_branch must be supplied together")
        if self.max_evaluations is not None and not self.debug and not self.record_history:
            raise ValueError("Budgeted formal evaluations require record_history=true")
        if self.debug and self.reuse_construction_from:
            raise ValueError("Construction reuse requires training-set")
        if self.debug or not self.target_queries:
            self.target_queries = None
        if self.target_queries is not None and (
            not self.sampling or self.case_ids is not None or self.query_ids is not None
        ):
            raise ValueError("target_queries requires sampled formal evaluation without exact case/query selections")
        if self.sampling and not self.debug and (self.case_ids is not None or self.query_ids is not None):
            raise ValueError("sampling=true selects case/query IDs automatically; omit explicit selections")
        if "fail_fast" not in self.model_fields_set:
            self.fail_fast = self.debug
        return self


def entity_id(value: str) -> str:
    """Keep case, query and branch IDs within a single artifact directory."""
    if not value or value in {".", ".."} or "/" in value or "\\" in value or "\0" in value:
        raise ValueError(f"Unsafe evaluation ID: {value!r}")
    return value


def load_inputs(home: Path, spec: EvaluationSpec):
    """Validate selection and immutable dataset fingerprints before using Docker."""
    record = read_json(contained(home, "metadata/preparation.json"))
    directory = contained(home, f"datasets/{spec.dataset_kind}", directory=True)
    manifest = DatasetManifest.model_validate(read_json(directory / "manifest.json"))
    cases = {}
    case_names = set()
    hashes = []
    for path in sorted((directory / "cases").glob("*.json")):
        case = CaseSpec.model_validate(read_json(path))
        validate_artifact_ids(case)
        name = artifact_name_key(case.case_id)
        if name in case_names or not case.queries:
            raise ValueError("Dataset contains duplicate case IDs or a case without queries")
        case_names.add(name)
        hashes.append(fingerprint(case))
        cases[case.case_id] = case
    if list(cases) != manifest.case_ids or fingerprint(hashes) != manifest.normalized_fingerprint:
        raise ValueError("Prepared dataset no longer matches its manifest")
    if manifest.case_count != len(cases) or manifest.query_count != sum(len(c.queries) for c in cases.values()):
        raise ValueError("Prepared dataset counts do not match its manifest")
    if spec.dataset_kind == "training-set" and manifest != DatasetManifest.model_validate(record["dataset"]):
        raise ValueError("Prepared training manifest has changed")
    return record, manifest, select_cases(cases, spec.case_ids, spec.query_ids)


def select_cases(cases: dict[str, CaseSpec], case_ids, query_ids):
    """Select cases and queries in caller order from an already validated dataset."""
    selected = case_ids if case_ids is not None else list(cases)
    if not selected or set(selected) - cases.keys():
        raise ValueError(f"Unknown or empty case selection: {sorted(set(selected) - cases.keys())}")
    if query_ids is not None and set(query_ids) - set(selected):
        raise ValueError("query_ids contains cases outside case_ids")
    result = []
    for case_id in selected:
        case = cases[case_id]
        if query_ids is not None and case_id in query_ids:
            queries = {query.query_id: query for query in case.queries}
            ids = query_ids[case_id]
            if set(ids) - queries.keys():
                raise ValueError(f"Unknown query selection for {case_id}")
            case = case.model_copy(update={"queries": [queries[key] for key in ids]})
        result.append(case)
    return result


def evaluation_head(repository: Path, *, expected_commit=None, expected_branch=None):
    """Read the actual checkout and reject a stale host sampling decision."""
    commit = git(repository, "rev-parse", "--verify", "HEAD^{commit}")
    branch = git(repository, "branch", "--show-current")
    if expected_commit is not None and (commit, branch) != (expected_commit, expected_branch):
        raise ValueError("Checkout changed before evaluation; retry on the intended commit and branch")
    return commit, branch


def snapshot_code(home: Path, record: dict, *, expected_commit=None, expected_branch=None):
    """Combine fixed runtime with clean current HEAD; reject all uncommitted changes."""
    repository, _ = validate_repository(home, record)
    runtime = contained(home, "code/runtime", directory=True)
    if tree_fingerprint(runtime) != record["code"]["runtime_fingerprint"]:
        raise ValueError("Fixed runtime has changed since preparation")
    if git(repository, "status", "--porcelain=v2", "--untracked-files=all", "--ignore-submodules=none"):
        raise ValueError("Commit candidate changes before evaluation")
    commit, checkout_branch = evaluation_head(
        repository,
        expected_commit=expected_commit,
        expected_branch=expected_branch,
    )
    branch = "init" if commit == record["code"]["initial_commit"] else checkout_branch
    entity_id(branch)
    entries = git_tree(repository, commit)
    roots = runtime_roots(record["code"]["plugins"])
    for relative, _, _ in entries:
        if not any(relative.is_relative_to(path) for path in roots):
            raise ValueError(f"Candidate path is outside the prepared runtime projects: {relative}")
    with tempfile.TemporaryDirectory(
        prefix="evaluation-source-",
        dir=contained(home, "tmp", directory=True),
    ) as temporary:
        combined = Path(temporary) / "source"
        shutil.copytree(runtime, combined)
        # Core candidate roots replace their baseline roots, including deletions.
        # Plugin projects exist only in repo and need no baseline overlay cleanup.
        for value in record["code"]["candidate_paths"]:
            if Path(value).parts[0] == "plugins":
                continue
            target = contained(combined, value)
            if target.is_dir():
                shutil.rmtree(target)
            elif target.exists():
                target.unlink()
        export_git_tree(repository, entries, lambda path: combined / path)
        for project in record["code"]["install_projects"]:
            if not (combined / project / "pyproject.toml").is_file():
                raise ValueError(f"Candidate is missing install project {project}")
        snapshot = SourceSnapshot.from_directory(combined)
    return snapshot, commit, branch, checkout_branch


def load_reuse_source(home: Path, value: str, fingerprints: dict, *, sampling: bool = False):
    """Read current source artifacts within the managed evaluation tree."""
    evaluations = contained(home, "evaluations", directory=True)
    source = contained(home, value, directory=True)
    if not source.is_relative_to(evaluations) or source == evaluations:
        raise ValueError("reuse_construction_from must be inside home/evaluations")
    manifest = read_json(contained(source, "manifest.json"))
    summary = read_json(contained(source, "summary.json"))
    if not isinstance(manifest, dict) or not isinstance(summary, dict):
        raise ValueError("Expected construction source manifest and summary objects")
    if (
        manifest.get("dataset_kind") != "training-set"
        or manifest.get("status") != "completed"
        or (sampling and manifest.get("mode") == "debug")
    ):
        if sampling:
            raise ValueError("Replay sampling requires a completed scored training-set run")
        raise ValueError("Reuse requires a completed training-set evaluation")
    validate_reuse_fingerprints(manifest, fingerprints)
    return source, manifest


def validate_reuse_fingerprints(manifest: dict, fingerprints: dict):
    """Apply the same compatibility checks to original and retained snapshots."""
    source_fingerprints = manifest.get("fingerprints")
    for key in ("dataset", "construction_config"):
        if not isinstance(source_fingerprints, dict) or source_fingerprints.get(key) != fingerprints[key]:
            raise ValueError(f"Construction source has incompatible {key}")


def validate_snapshot(source: Path, case_id: str, max_artifact_bytes: int):
    """Resolve and check one successful construction archive."""
    folder = contained(source, f"cases/{case_id}/memory_construction", directory=True)
    result = read_json(contained(folder, "result.json"))
    archive = contained(folder, "reme_workspace.tar.gz")
    if not isinstance(result, dict) or result.get("success") is not True:
        raise ValueError("construction was not successful")
    if not result.get("workspace_sha256") or file_sha256(archive) != result["workspace_sha256"]:
        raise ValueError("construction archive checksum mismatch")
    try:
        inspect_archive(archive, max_artifact_bytes)
    except (EOFError, zlib.error) as error:
        raise ValueError(f"Unreadable construction archive: {archive}") from error
    return archive, result["workspace_sha256"]


def reuse_inputs(home: Path, spec: EvaluationSpec, cases: list, fingerprints: dict):
    """Preflight the entire requested selection; never silently skip or rebuild cases."""
    if spec.reuse_construction_from is None:
        return None, {}
    source, manifest = load_reuse_source(home, spec.reuse_construction_from, fingerprints)
    archives, errors = {}, []
    for case in cases:
        try:
            archives[case.case_id] = validate_snapshot(source, case.case_id, spec.max_artifact_bytes)
        except (ValueError, OSError, KeyError, tarfile.TarError) as error:
            errors.append(f"{case.case_id}: {error}")
    if errors:
        raise ValueError("Invalid construction snapshots: " + "; ".join(errors))
    return {"result_dir": str(source), "commit_sha": manifest["commit_sha"]}, archives


def replay_sampling_inputs(home: Path, spec: EvaluationSpec, cases: list[CaseSpec], fingerprints: dict):
    """Limit explicit replay sampling to preflighted snapshots with successful answers."""
    source, manifest = load_reuse_source(home, spec.reuse_construction_from, fingerprints, sampling=True)
    case_ids = manifest.get("case_ids")
    if not isinstance(case_ids, list) or not case_ids or len(set(case_ids)) != len(case_ids):
        raise ValueError("Replay source must select unique nonempty cases")
    archives = {}
    for case_id in case_ids:
        entity_id(case_id)
        try:
            archives[case_id] = validate_snapshot(source, case_id, spec.max_artifact_bytes)
        except (OSError, ValueError, tarfile.TarError):
            continue
    if (len(case_ids) - len(archives)) / len(case_ids) > 0.05:
        raise ValueError("Replay source construction failure ratio exceeds 5%")
    for case_id in archives:
        validate_replay_answers(source, case_id)
    # A construction snapshot can answer new query subsets from the same case;
    # dev_meta intentionally considers every installed query of a reusable case.
    eligibility = {
        case.case_id: [query.query_id for query in case.queries] for case in cases if case.case_id in archives
    }
    return {"result_dir": str(source), "commit_sha": manifest["commit_sha"]}, archives, eligibility


def validate_replay_answers(source: Path, case_id: str):
    """Require a nonempty set of successful answers from the source run."""
    result_path = contained(source, f"cases/{case_id}/queries/result.json")
    try:
        payload = read_json(result_path)
    except (OSError, ValueError) as error:
        raise ValueError(f"Replay source cannot read answers for case {case_id}: {error}") from error
    queries = payload.get("queries") if isinstance(payload, dict) else None
    if (
        not isinstance(queries, list)
        or not queries
        or not all(
            isinstance(query, dict)
            and query.get("error") is None
            and isinstance(query.get("answer"), str)
            and query["answer"].strip()
            for query in queries
        )
    ):
        raise ValueError(f"Replay source has no completed successful answers for {case_id}")
