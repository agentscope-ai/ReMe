"""Validate prepared inputs and snapshot committed candidates before execution."""

from pathlib import Path
from typing import Literal

import shutil
import subprocess
import tempfile
import tarfile

from pydantic import Field, field_validator, model_validator

from deploy.benchmark_docker import SourceSnapshot

from ..preparation.code import git, tree_fingerprint
from ..preparation.data.models import CaseSpec, DatasetManifest, fingerprint
from .artifacts import contained, file_sha256, inspect_archive, read_json
from ..preparation.spec import SpecModel


class EvaluationSpec(SpecModel):
    """Selections are injected by callers; omitted cases mean the whole dataset."""

    case_ids: list[str] | None = None
    query_ids: dict[str, list[str]] | None = None
    reuse_construction_from: str | None = Field(default=None, min_length=1)
    debug: bool = False
    concurrency: int = Field(default=4, ge=1)
    timeout_seconds: float = Field(default=7200.0, gt=0, allow_inf_nan=False)
    base_image: str = Field(default="reme-benchmark-base:agentscope-2.0.10", min_length=1)
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
        if self.debug and self.reuse_construction_from:
            raise ValueError("Construction reuse requires training-set")
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
    hashes = []
    for path in sorted((directory / "cases").glob("*.json")):
        case = CaseSpec.model_validate(read_json(path))
        entity_id(case.case_id)
        if case.case_id in cases or not case.queries:
            raise ValueError("Dataset contains duplicate case IDs or a case without queries")
        ids = [entity_id(query.query_id) for query in case.queries]
        if len(ids) != len(set(ids)):
            raise ValueError(f"Duplicate query IDs in case {case.case_id}")
        hashes.append(fingerprint(case))
        cases[case.case_id] = case
    if list(cases) != manifest.case_ids or fingerprint(hashes) != manifest.normalized_fingerprint:
        raise ValueError("Prepared dataset no longer matches its manifest")
    if manifest.case_count != len(cases) or manifest.query_count != sum(len(c.queries) for c in cases.values()):
        raise ValueError("Prepared dataset counts do not match its manifest")
    if spec.dataset_kind == "training-set" and manifest.model_dump(mode="json") != record["dataset"]:
        raise ValueError("Prepared training manifest has changed")
    selected = spec.case_ids if spec.case_ids is not None else list(cases)
    if not selected or set(selected) - cases.keys():
        raise ValueError(f"Unknown or empty case selection: {sorted(set(selected) - cases.keys())}")
    if spec.query_ids is not None and set(spec.query_ids) - set(selected):
        raise ValueError("query_ids contains cases outside case_ids")
    result = []
    for case_id in selected:
        case = cases[case_id]
        if spec.query_ids is not None and case_id in spec.query_ids:
            queries = {query.query_id: query for query in case.queries}
            ids = spec.query_ids[case_id]
            if set(ids) - queries.keys():
                raise ValueError(f"Unknown query selection for {case_id}")
            case = case.model_copy(update={"queries": [queries[key] for key in ids]})
        result.append(case)
    return record, manifest, result


def snapshot_code(home: Path, record: dict):
    """Combine fixed runtime with clean current HEAD; reject all uncommitted changes."""
    repository = contained(home, "code/repo", directory=True)
    runtime = contained(home, "code/runtime", directory=True)
    if tree_fingerprint(runtime) != record["code"]["runtime_fingerprint"]:
        raise ValueError("Fixed runtime has changed since preparation")
    if git(repository, "status", "--porcelain=v2", "--untracked-files=all", "--ignore-submodules=none"):
        raise ValueError("Commit candidate changes before evaluation")
    commit = git(repository, "rev-parse", "--verify", "HEAD^{commit}")
    branch = "init" if commit == record["code"]["initial_commit"] else git(repository, "branch", "--show-current")
    entity_id(branch)
    # NUL output validates exact paths without interpreting Git quoting rules.
    entries = subprocess.check_output(
        ["git", "-C", str(repository), "ls-tree", "-r", "-z", commit],
        timeout=120,
    ).split(b"\0")
    parsed = []
    for entry in entries:
        if not entry:
            continue
        header, name = entry.split(b"\t", 1)
        mode, kind, oid = header.decode().split()
        relative = Path(name.decode())
        if (
            relative.is_absolute()
            or ".." in relative.parts
            or ".git" in relative.parts
            or kind != "blob"
            or mode not in {"100644", "100755"}
        ):
            raise ValueError(f"Candidate must contain contained regular files: {relative}")
        if not any(
            relative.is_relative_to(path)
            for path in [
                Path("reme"),
                Path("pyproject.toml"),
                Path("README.md"),
                Path("LICENSE"),
                *[Path("plugins") / p for p in record["code"]["plugins"]],
            ]
        ):
            raise ValueError(f"Candidate path is outside the prepared runtime projects: {relative}")
        parsed.append((relative, mode, oid))
    contents = subprocess.run(
        ["git", "-C", str(repository), "cat-file", "--batch"],
        input="".join(oid + "\n" for _, _, oid in parsed).encode(),
        capture_output=True,
        check=True,
        timeout=120,
    ).stdout
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
        offset = 0
        for relative, mode, oid in parsed:
            end = contents.index(b"\n", offset)
            found, kind, size = contents[offset:end].decode().split()
            if found != oid or kind != "blob":
                raise ValueError("Invalid candidate Git object")
            offset = end + 1
            target = combined / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(contents[offset : offset + int(size)])
            target.chmod(0o755 if mode == "100755" else 0o644)
            offset += int(size) + 1
        for project in record["code"]["install_projects"]:
            if not (combined / project / "pyproject.toml").is_file():
                raise ValueError(f"Candidate is missing install project {project}")
        snapshot = SourceSnapshot.from_directory(combined)
    return snapshot, commit, branch


def reuse_inputs(home: Path, spec: EvaluationSpec, cases: list, fingerprints: dict):
    """Preflight the entire requested selection; never silently skip or rebuild cases."""
    if spec.reuse_construction_from is None:
        return None, {}
    evaluations = contained(home, "evaluations", directory=True)
    value = spec.reuse_construction_from
    source = contained(home, value, directory=True)
    if not source.is_relative_to(evaluations) or source == evaluations:
        raise ValueError("reuse_construction_from must be inside home/evaluations")
    manifest = read_json(source / "manifest.json")
    summary = read_json(source / "summary.json")
    if manifest.get("dataset_kind") != "training-set" or manifest.get("status", summary.get("status")) != "completed":
        raise ValueError("Reuse requires a completed training-set evaluation")
    for key in ("dataset", "construction_config"):
        if manifest.get("fingerprints", {}).get(key) != fingerprints[key]:
            raise ValueError(f"Construction source has incompatible {key}")
    archives, errors = {}, []
    for case in cases:
        try:
            folder = contained(source, f"cases/{case.case_id}/memory_construction", directory=True)
            result = read_json(folder / "result.json")
            archive = contained(folder, "reme_workspace.tar.gz")
            if result.get("success") is not True:
                raise ValueError("construction was not successful")
            if not result.get("workspace_sha256") or file_sha256(archive) != result["workspace_sha256"]:
                raise ValueError("construction archive checksum mismatch")
            inspect_archive(archive, spec.max_artifact_bytes)
            archives[case.case_id] = (archive, result["workspace_sha256"])
        except (ValueError, OSError, KeyError, tarfile.TarError) as error:
            errors.append(f"{case.case_id}: {error}")
    if errors:
        raise ValueError("Invalid construction snapshots: " + "; ".join(errors))
    return {"result_dir": str(source), "commit_sha": manifest["commit_sha"]}, archives
