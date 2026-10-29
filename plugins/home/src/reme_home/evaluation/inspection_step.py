"""Read-only evaluation navigation Jobs, independent of optimization history."""

import asyncio
import json

from pydantic import Field, field_validator, model_validator

from reme.steps.base_step import BaseStep, Ref

from ..code.repository import current_branch, resolve_revision, validate_branch, validate_repository
from ..preparation.code import git
from ..preparation.spec import SpecModel
from ..workspace import HomeWorkspace, _initialization_lock
from .artifacts import contained, read_json
from .inputs import entity_id

TOTAL_FIELDS = (
    "schema_version",
    "scoring_policy",
    "case_count",
    "case_status_counts",
    "construction_status_counts",
    "failure_stage_case_counts",
    "expected_query_count",
    "scored_query_count",
    "query_status_counts",
    "mean_query_score",
    "observed_mean_query_score",
)


class ListValidationResultsInput(SpecModel):
    """Filters affect catalog navigation, never which evaluations enter history."""

    branch_name: str | None = None
    commit_sha: str | None = Field(default=None, pattern=r"^[0-9a-fA-F]+$")
    offset: int = Field(default=0, ge=0)
    limit: int = Field(default=20, ge=1, le=100)

    @field_validator("branch_name")
    @classmethod
    def validate_branch_name(cls, value):
        """Keep historical branch filters within one directory segment."""
        return entity_id(value) if value is not None else None


class InspectValidationInput(SpecModel):
    """A query belongs to a case and every inspection belongs to one run."""

    validation_path: str = Field(min_length=1)
    case_id: str | None = None
    query_id: str | None = None
    offset: int = Field(default=0, ge=0)
    limit: int = Field(default=20, ge=1, le=100)

    @field_validator("case_id", "query_id")
    @classmethod
    def validate_entity(cls, value):
        """Reject path traversal before resolving any artifact."""
        return entity_id(value) if value is not None else None

    @model_validator(mode="after")
    def validate_selection(self):
        """Require a case when selecting one query."""
        if self.query_id is not None and self.case_id is None:
            raise ValueError("query_id requires case_id")
        return self


def _preview(value, limit=600):
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
    return value if len(text) <= limit else text[:limit] + "… [truncated; read source field]"


def _pointer(value):
    return str(value).replace("~", "~0").replace("/", "~1")


def _page(rows, spec):
    return {
        "row_count": len(rows),
        "rows": rows[spec.offset : spec.offset + spec.limit],
        "next_offset": spec.offset + spec.limit if spec.offset + spec.limit < len(rows) else None,
    }


def _object(root, relative, *, required=False):
    path = contained(root, relative)
    if not path.exists() and not required:
        return None
    payload = read_json(path)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def _summary(run):
    for name in ("summary.json", "partial_summary.json"):
        value = _object(run, name)
        if value is not None:
            return value, str(run / name)
    return None, None


def _run_path(home, value):
    run = contained(home, value, directory=True)
    relative = run.relative_to(home / "evaluations")
    if len(relative.parts) != 3 or not run.is_dir():
        raise ValueError("validation_path must identify one directory inside home/evaluations")
    return run


def _commit_filter(home, spec):
    if spec.commit_sha is None:
        return spec.branch_name, None
    # Resolve a local branch first, then match ALL of its reachable commits.
    # Git's global abbreviation rules could otherwise reject a branch-unique prefix.
    with _initialization_lock(contained(home, "metadata/workspace.lock")):
        repo, _ = validate_repository(home)
        branch = spec.branch_name if spec.branch_name is not None else current_branch(repo)
        validate_branch(repo, branch)
        tip = resolve_revision(repo, "refs/heads/" + branch)
        prefix = spec.commit_sha.lower()
        matches = [sha for sha in git(repo, "rev-list", tip, "--").splitlines() if sha.startswith(prefix)]
    if len(matches) != 1:
        raise ValueError(f"commit_sha prefix {prefix!r} has {len(matches)} matches on branch {branch!r}; require one")
    return branch, matches[0]


def _catalog_row(run, manifest):
    summary, summary_path = _summary(run)
    options = manifest.get("options", {})
    row = {
        "validation_path": str(run),
        "manifest_path": str(run / "manifest.json"),
        **{
            key: manifest.get(key)
            for key in (
                "branch_name",
                "commit_sha",
                "validation_id",
                "status",
                "success",
                "started_at",
                "finished_at",
                "dataset",
                "dataset_kind",
                "mode",
                "execution_mode",
                "construction_source",
                "fingerprints",
            )
        },
        "sampling": options.get("sampling", False),
        "record_history": options.get("record_history", True) and manifest.get("mode") != "debug",
        "selected_case_count": len(manifest.get("case_ids", [])),
        "selected_query_count": sum(len(ids) for ids in manifest.get("query_ids", {}).values()),
        "summary_path": summary_path,
        "summary_missing": summary is None,
        "totals": {key: summary[key] for key in TOTAL_FIELDS if key in summary} if summary else {},
    }
    failure = _object(run, "failure.json")
    row["failure"] = _preview(failure)
    row["failure_path"] = str(run / "failure.json") if failure is not None else None
    return row


def _list_results(home, spec):
    branch, commit = _commit_filter(home, spec)
    evaluations = contained(home, "evaluations", directory=True)
    rows = []
    for path in evaluations.glob("*/*/*/manifest.json"):
        if branch is not None and path.parent.parent.parent.name != branch:
            continue
        row = {"validation_path": str(path.parent), "manifest_path": str(path)}
        try:
            run = _run_path(home, path.parent)
            manifest = _object(run, "manifest.json", required=True)
            row.update({key: manifest.get(key) for key in ("branch_name", "commit_sha", "started_at")})
            if commit is not None and row["commit_sha"] != commit:
                continue
            row = _catalog_row(run, manifest)
        except (ValueError, OSError, TypeError, KeyError, AttributeError) as error:
            row["read_error"] = _preview(str(error))
        if commit is None or row.get("commit_sha") == commit:
            rows.append(row)
    rows.sort(key=lambda row: (str(row.get("started_at") or ""), row["validation_path"]), reverse=True)
    return {"branch_name": branch, "commit_sha": commit, **_page(rows, spec)}


def _case_rows(manifest, summary):
    cases = (summary or {}).get("cases", {})
    indexed = (
        {key: (value, f"/cases/{_pointer(key)}") for key, value in cases.items()}
        if isinstance(cases, dict)
        else {value["case_id"]: (value, f"/cases/{index}") for index, value in enumerate(cases)}
    )
    rows = []
    for case_id in manifest.get("case_ids", []):
        case, pointer = indexed.get(case_id, ({}, None))
        rows.append(
            {
                "case_id": case_id,
                "mean_score": case.get("mean_score"),
                "construction": case.get("construction"),
                "construction_error_count": case.get("construction_error_count", 0),
                "query_error_count": case.get("query_error_count", 0),
                "query_tokens": case.get("average_query_total_token_usage"),
                "missing": pointer is None,
                "pointer": pointer,
            },
        )
    rows.sort(
        key=lambda row: (
            row["mean_score"] is not None,
            row["mean_score"] if row["mean_score"] is not None else -1,
            -row["construction_error_count"],
            row["case_id"],
        ),
    )
    return rows


def _case_inspection(run, manifest, spec):
    folder = contained(run, f"cases/{spec.case_id}", directory=True)
    build_path, query_path = "memory_construction/result.json", "queries/result.json"
    build, payload = _object(folder, build_path), _object(folder, query_path)
    failures = []
    jobs = (build or {}).get("jobs", [])
    for index, job in enumerate(jobs):
        outcome = job.get("result") or {}
        if outcome.get("success") is False:
            failures.append(
                {
                    "job": job.get("job"),
                    "error": _preview(outcome.get("error")),
                    "pointer": f"/jobs/{index}/result",
                },
            )
    queries = (payload or {}).get("queries", [])
    by_id = {query["query_id"]: (query, index) for index, query in enumerate(queries)}
    selected = manifest["query_ids"][spec.case_id]
    if spec.query_id is not None and spec.query_id not in selected:
        raise ValueError("query_id is not selected by this evaluation manifest")
    rows = []
    for query_id in ([spec.query_id] if spec.query_id is not None else selected):
        query, index = by_id.get(query_id, ({}, None))
        row = {
            "query_id": query_id,
            "score": query.get("score"),
            "question_type": query.get("question_type"),
            "failure_kind": query.get("failure_kind"),
            "error": _preview(query.get("error")),
            "answer_success": (query.get("answer_result") or {}).get("success"),
            "judge_success": (query.get("judge_result") or {}).get("success"),
            "missing": index is None,
            "pointer": f"/queries/{index}" if index is not None else None,
        }
        if spec.query_id is not None:
            row["fields"] = {
                key: _preview(query[key], 2000)
                for key in (
                    "question",
                    "golden_answer",
                    "answer",
                    "score",
                    "question_type",
                    "answer_result",
                    "judge_result",
                    "error",
                    "exception",
                    "failure_kind",
                    "infrastructure_error",
                )
                if key in query
            }
        rows.append(row)
    result = {
        "case_id": spec.case_id,
        "missing_files": [name for name, value in ((build_path, build), (query_path, payload)) if value is None],
        "case_failure": _preview(_object(folder, "failure.json")),
        "construction": {
            "source": str(folder / build_path),
            "success": (build or {}).get("success"),
            "status": (build or {}).get("status"),
            "job_count": len(jobs),
            "failed_jobs": _page(failures, spec),
        },
        "build_log": str(contained(folder, "memory_construction/build.log")),
        "memory_workspace": str(contained(folder, "memory_construction/reme_workspace", directory=True)),
        "rows_source": str(folder / query_path),
        **_page(rows, spec),
    }
    if spec.query_id is not None:
        query_folder = contained(folder, f"queries/{spec.query_id}", directory=True)
        result.update(
            query_missing=rows[0]["missing"],
            answer_log=str(contained(query_folder, "answer.log")),
            query_failure=_preview(_object(query_folder, "failure.json")),
        )
    return result


def _inspect_result(home, spec):
    run = _run_path(home, spec.validation_path)
    manifest = _object(run, "manifest.json", required=True)
    if spec.case_id is not None and spec.case_id not in manifest.get("case_ids", []):
        raise ValueError("case_id is not selected by this evaluation manifest")
    result = {"validation_path": str(run), "manifest_path": str(run / "manifest.json")}
    if spec.case_id is not None:
        return {**result, **_case_inspection(run, manifest, spec)}
    summary, summary_path = _summary(run)
    return {
        **_catalog_row(run, manifest),
        "rows_source": summary_path,
        **_page(_case_rows(manifest, summary), spec),
    }


async def _execute(step, model, function):
    if step.context.get("_allowed_paths") is not None:
        raise ValueError("Home inspection Jobs require unrestricted access to their managed evaluation artifacts")
    spec = model.model_validate({name: step.context.get(name) for name in model.model_fields if name in step.context})
    if not step.workspace.is_started:
        raise RuntimeError("Start the Home workspace before inspecting evaluations")
    task = asyncio.create_task(asyncio.to_thread(function, step.workspace.home_path, spec))
    cancelled = False
    while True:
        try:
            result = await asyncio.shield(task)
            if cancelled:
                raise asyncio.CancelledError
            step.context.response.answer = result
            return step.context.response
        except asyncio.CancelledError:
            if task.done():
                raise
            cancelled = True


class HomeListValidationResultsStep(BaseStep):
    """List all evaluation artifacts, including failures, debug and standalone runs."""

    workspace: HomeWorkspace = Ref(HomeWorkspace, "home.workspace")

    async def execute(self):
        """Navigate runs without reading or rebuilding optimization history."""
        return await _execute(self, ListValidationResultsInput, _list_results)


class HomeInspectValidationStep(BaseStep):
    """Inspect a run, a case or one query with bounded result previews."""

    workspace: HomeWorkspace = Ref(HomeWorkspace, "home.workspace")

    async def execute(self):
        """Read only the requested result scope and return artifact coordinates."""
        return await _execute(self, InspectValidationInput, _inspect_result)
