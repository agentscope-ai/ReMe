"""Compact, read-only query evidence coordinates from saved evaluation artifacts."""

from typing import Annotated

from pydantic import Field, TypeAdapter, field_validator

from reme.steps.base_step import BaseStep, Ref

from ..preparation.spec import SpecModel
from ..workspace import HomeWorkspace
from .artifacts import contained, read_json
from .inputs import entity_id
from .inspection_step import _execute, _object, _preview
from .memory_inspection_step import _head, _repository, git

SessionIds = list[Annotated[str, Field(min_length=1)]]
MAX_SESSIONS = 50
MAX_CHANGED_FILES = 50


class TraceQueryEvidenceInput(SpecModel):
    """Locate one selected query; inferred evidence is optional and never overrides annotations."""

    validation_path: str = Field(min_length=1)
    case_id: str = Field(min_length=1)
    query_id: str = Field(min_length=1)
    evidence_session_ids: SessionIds | None = None

    @field_validator("case_id", "query_id")
    @classmethod
    def validate_entity(cls, value):
        """Keep artifact selectors within one directory segment."""
        return entity_id(value)


def _source_case(home, manifest, case_id):
    kind = manifest.get("dataset_kind")
    if kind not in {"training-set", "debug-cases"}:
        raise ValueError(f"Unsupported evaluation dataset_kind: {kind!r}")
    root = contained(home, f"datasets/{kind}/cases", directory=True)
    for candidate in sorted(root.glob("*.json")):
        path = contained(root, candidate)
        case = read_json(path)
        if isinstance(case, dict) and case.get("case_id") == case_id:
            return path, case
    raise ValueError(f"Normalized case is missing: {case_id}")


def _session_checkpoints(repository, head, selected):
    if head is None or not selected:
        return {}
    # Separate commit subjects and records without relying on line delimiters.
    raw = git(repository, "log", "--no-show-signature", "--reverse", "--format=%H%x00%s", "-z", head, "--")
    fields = raw.split("\0")
    checkpoints = {}
    for sha, subject in zip(fields[::2], fields[1::2]):
        if subject.startswith("session: ") and subject[9:] in selected:
            checkpoints[subject[9:]] = sha
    return checkpoints


def trace_query_evidence(home, spec):
    """Join source annotations, committed checkpoint coordinates and query result previews."""
    run, manifest, repository = _repository(home, spec.validation_path, spec.case_id)
    if spec.query_id not in manifest.get("query_ids", {}).get(spec.case_id, []):
        raise ValueError("query_id is not selected by this evaluation manifest")
    source_path, case = _source_case(home, manifest, spec.case_id)
    query_index = next(
        (i for i, item in enumerate(case.get("queries", [])) if item.get("query_id") == spec.query_id),
        None,
    )
    if query_index is None:
        raise ValueError(f"Normalized case has no query: {spec.query_id}")
    query = case["queries"][query_index]
    metadata = query.get("metadata") or {}
    annotated = metadata.get("evidence_session_ids")
    selected = TypeAdapter(SessionIds).validate_python(
        annotated if annotated is not None else spec.evidence_session_ids or [],
        strict=True,
    )
    selected = list(dict.fromkeys(selected))
    source = "independent_annotation" if annotated is not None else "diagnostician_inference"
    if annotated is None and spec.evidence_session_ids is None:
        source = "unavailable"
    known = {session["session_id"]: i for i, session in enumerate(case.get("sessions", []))}
    if unknown := set(selected) - known.keys():
        raise ValueError(f"Evidence sessions are not present in case: {sorted(unknown)}")

    status, head = _head(repository)
    checkpoints = _session_checkpoints(repository, head, set(selected))
    sessions = []
    for session_id in selected[:MAX_SESSIONS]:
        checkpoint = checkpoints.get(session_id)
        files = (
            git(repository, "diff-tree", "--root", "--no-commit-id", "--name-only", "-r", "-z", checkpoint, "--")
            .rstrip("\0")
            .split("\0")
            if checkpoint
            else []
        )
        files = [name for name in files if name]
        row = {
            "id": session_id,
            "source_pointer": f"/sessions/{known[session_id]}",
            "checkpoint": checkpoint,
            "changed_files": files[:MAX_CHANGED_FILES],
        }
        if len(files) > MAX_CHANGED_FILES:
            row["omitted_changed_file_count"] = len(files) - MAX_CHANGED_FILES
        sessions.append(row)

    results_path = contained(run, f"cases/{spec.case_id}/queries/result.json")
    payload = _object(run, str(results_path.relative_to(run)))
    result_index = next(
        (i for i, item in enumerate((payload or {}).get("queries", [])) if item.get("query_id") == spec.query_id),
        None,
    )
    outcome = payload["queries"][result_index] if result_index is not None else {}
    rubric = metadata.get("rubric")
    expected_is_rubric = manifest.get("dataset") == "beam" and isinstance(query.get("golden_answer"), list)
    if rubric is None and expected_is_rubric:
        rubric = query["golden_answer"]
    response = {
        "query": {
            "question": _preview(outcome.get("question", query.get("question")), 2000),
            "expected": _preview(outcome.get("golden_answer", query.get("golden_answer")), 2000),
            "expected_kind": "rubric" if expected_is_rubric else "reference_answer",
            "actual": _preview(outcome.get("answer"), 2000),
            "score": outcome.get("score"),
            "error": _preview(outcome.get("error")),
            "result_missing": result_index is None,
        },
        "locations": {
            "source_query": f"/queries/{query_index}",
            "saved_query": f"/queries/{result_index}" if result_index is not None else None,
        },
        "evidence_source": source,
        "evidence_is_certain": metadata.get("evidence_is_certain"),
        "memory_status": status,
        "sessions": sessions,
        "paths": {
            "source_case": str(source_path),
            "memory_repository": str(repository),
            "query_result": str(results_path),
            "query_artifacts": str(contained(run, f"cases/{spec.case_id}/queries/{spec.query_id}", directory=True)),
        },
    }
    judged = ((outcome.get("judge_result") or {}).get("metadata") or {}).get("llm_judge_responses")
    if isinstance(rubric, list):
        field = "metadata/rubric" if metadata.get("rubric") is not None else "golden_answer"
        response["locations"]["rubric"] = f"/queries/{query_index}/{field}"
    if isinstance(rubric, list) and isinstance(judged, list) and len(rubric) == len(judged):
        response["criteria"] = [
            {
                "requirement": _preview(requirement),
                "score": item.get("score"),
                "reason": _preview(item.get("reason")),
                "pointer": f"/queries/{result_index}/judge_result/metadata/llm_judge_responses/{i}",
            }
            for i, (requirement, item) in enumerate(zip(rubric[:20], judged[:20]))
            if isinstance(item, dict)
        ]
        if len(rubric) > 20:
            response["omitted_criterion_count"] = len(rubric) - 20
    if len(selected) > MAX_SESSIONS:
        response["omitted_session_count"] = len(selected) - MAX_SESSIONS
    return response


class HomeTraceQueryEvidenceStep(BaseStep):
    """Expose compact evidence navigation through the normal Home Job lifecycle."""

    workspace: HomeWorkspace = Ref(HomeWorkspace, "home.workspace")

    async def execute(self):
        """Read saved artifacts without model calls or workspace mutations."""
        return await _execute(self, TraceQueryEvidenceInput, trace_query_evidence)
