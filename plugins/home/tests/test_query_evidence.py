"""Compact evidence tracing against real, isolated evaluation and Git artifacts."""

# pylint: disable=missing-function-docstring,redefined-outer-name,unused-argument,protected-access

import json
import shutil

import pytest
from pydantic import ValidationError
from reme_home.evaluation import evidence_step as evidence_tools
from reme_home.evaluation.artifacts import write_json
from reme_home.preparation.code import git

from test_memory_inspection import (  # noqa: F401  # pylint: disable=unused-import
    FakeFactory,
    commit,
    evidence,
    home_application,
    preparation_inputs,
    prepared_home,
)
from test_evaluation_inspection import call, snapshot, write_run


@pytest.fixture
def query_evidence(evidence):  # noqa: F811
    home, run, repo, first, second = evidence
    manifest = json.loads((run / "manifest.json").read_text())
    manifest["query_ids"] = {"case1": ["q1"]}
    write_json(run / "manifest.json", manifest)
    case = {
        "case_id": "case1",
        "sessions": [{"session_id": "session1", "messages": [{"content": "source transcript"}]}],
        "queries": [
            {
                "query_id": "q1",
                "question": "What is the fact?",
                "golden_answer": "old fact",
                "metadata": {"evidence_session_ids": ["session1", "session1"], "evidence_is_certain": False},
            },
        ],
    }
    source = home / "datasets/training-set/cases/000000.json"
    write_json(source, case)
    write_json(
        run / "cases/case1/queries/result.json",
        {"queries": [{"query_id": "q1", "answer": "new fact", "score": 0, "judge_result": {"large": "hidden"}}]},
    )
    return home, run, repo, first, second, source, case


def trace(home, run, **kwargs):
    spec = evidence_tools.TraceQueryEvidenceInput(validation_path=str(run), case_id="case1", query_id="q1", **kwargs)
    return evidence_tools.trace_query_evidence(home, spec)


def test_compact_annotation_root_checkpoint_and_read_only(query_evidence):
    home, run, repo, first, _second, source, _case = query_evidence
    (repo / "memory/note.md").write_text("dirty ignored")
    before = snapshot(home)
    # Independent annotations take precedence, even over nonexistent inferred IDs.
    result = trace(home, run, evidence_session_ids=["unknown"])
    assert result["query"] == {
        "question": "What is the fact?",
        "expected": "old fact",
        "actual": "new fact",
        "score": 0,
        "error": None,
        "result_missing": False,
    }
    assert result["evidence_source"] == "independent_annotation"
    assert result["evidence_is_certain"] is False and result["memory_status"] == "available"
    assert result["sessions"] == [
        {"id": "session1", "checkpoint": first, "changed_files": ["memory/note.md", "removed.txt"]},
    ]
    assert result["paths"]["source_case"] == str(source)
    assert result["paths"]["memory_repository"] == str(repo)
    assert "source transcript" not in json.dumps(result) and "judge_result" not in json.dumps(result)
    assert snapshot(home) == before


def test_inference_missing_checkpoint_and_debug_dataset(query_evidence):
    home, run, _repo, _first, second, source, case = query_evidence
    case["sessions"] += [{"session_id": "session2"}, {"session_id": "not-constructed"}]
    case["queries"][0]["metadata"] = {}
    debug_source = home / "datasets/debug-cases/cases/000001.json"
    write_json(debug_source, case)
    source.unlink()
    manifest = json.loads((run / "manifest.json").read_text())
    manifest["dataset_kind"] = "debug-cases"
    write_json(run / "manifest.json", manifest)
    result = trace(home, run, evidence_session_ids=["session2", "not-constructed", "session2"])
    assert result["evidence_source"] == "diagnostician_inference"
    assert result["paths"]["source_case"] == str(debug_source)
    assert result["sessions"][0]["checkpoint"] == second
    assert result["sessions"][0]["changed_files"] == ["added.bin", "memory/note.md", "removed.txt"]
    assert result["sessions"][1] == {"id": "not-constructed", "checkpoint": None, "changed_files": []}
    assert trace(home, run)["evidence_source"] == "unavailable"
    with pytest.raises(ValueError, match="not present"):
        trace(home, run, evidence_session_ids=["unknown"])


@pytest.mark.parametrize("annotation", [[], [""], "session1", [123]])
def test_empty_and_invalid_annotations(query_evidence, annotation):
    home, run, _repo, _first, _second, source, case = query_evidence
    case["queries"][0]["metadata"]["evidence_session_ids"] = annotation
    write_json(source, case)
    if not annotation:
        result = trace(home, run, evidence_session_ids=["session1"])
        assert not result["sessions"] and result["evidence_source"] == "independent_annotation"
    else:
        with pytest.raises(ValidationError):
            trace(home, run)


@pytest.mark.parametrize("state", ["workspace_missing", "history_missing", "history_empty"])
def test_missing_history_and_results(query_evidence, state):
    home, run, repo, _first, _second, _source, _case = query_evidence
    shutil.rmtree(repo if state == "workspace_missing" else repo / ".git")
    if state == "history_empty":
        git(repo, "init")
    (run / "cases/case1/queries/result.json").unlink()
    result = trace(home, run)
    assert result["memory_status"] == state and result["sessions"][0]["checkpoint"] is None
    assert result["query"]["result_missing"] and result["query"]["score"] is None
    assert result["query"]["actual"] is None and result["query"]["expected"] == "old fact"


def test_large_fields_and_lists_are_bounded(query_evidence):
    home, run, repo, _first, _second, source, case = query_evidence
    for index in range(55):
        (repo / f"file-{index:03}").write_text("fact")
    checkpoint = commit(repo, "session: session1")
    case["sessions"] = [{"session_id": f"s{index}"} for index in range(55)] + case["sessions"]
    case["queries"][0]["metadata"]["evidence_session_ids"] = ["session1"] + [f"s{i}" for i in range(55)]
    case["queries"][0]["question"] = "x" * 100_000
    write_json(source, case)
    result = trace(home, run)
    assert len(result["sessions"]) == 50 and result["omitted_session_count"] == 6
    assert result["sessions"][0]["checkpoint"] == checkpoint
    assert len(result["sessions"][0]["changed_files"]) == 50
    assert result["sessions"][0]["omitted_changed_file_count"] == 5
    assert len(result["query"]["question"]) < 2100
    assert len(json.dumps(result)) < 15_000


def test_reject_unselected_queries_and_external_sources(query_evidence, tmp_path):
    home, run, _repo, _first, _second, source, _case = query_evidence
    manifest = json.loads((run / "manifest.json").read_text())
    manifest["query_ids"] = {"case1": []}
    write_json(run / "manifest.json", manifest)
    with pytest.raises(ValueError, match="not selected"):
        trace(home, run)
    manifest["query_ids"] = {"case1": ["q1"]}
    write_json(run / "manifest.json", manifest)
    source.unlink()
    source.symlink_to(tmp_path / "external.json")
    with pytest.raises(ValueError, match="Symbolic link"):
        trace(home, run)
    source.unlink()
    with pytest.raises(ValueError, match="case is missing"):
        trace(home, run)


def test_missing_query_invalid_dataset_and_corrupt_history(query_evidence):
    home, run, repo, _first, _second, source, case = query_evidence
    case["queries"] = []
    write_json(source, case)
    with pytest.raises(ValueError, match="has no query"):
        trace(home, run)
    manifest = json.loads((run / "manifest.json").read_text())
    manifest["dataset_kind"] = "../outside"
    write_json(run / "manifest.json", manifest)
    with pytest.raises(ValueError, match="dataset_kind"):
        trace(home, run)
    manifest["dataset_kind"] = "training-set"
    write_json(run / "manifest.json", manifest)
    case["queries"] = [{"query_id": "q1"}]
    write_json(source, case)
    (repo / ".git/objects/info/alternates").write_text("/external/objects")
    with pytest.raises(ValueError, match="indirect"):
        trace(home, run)


def test_run_and_case_scope(query_evidence):
    home, run, _repo, _first, _second, _source, _case = query_evidence
    with pytest.raises(ValueError, match="case_id"):
        evidence_tools.trace_query_evidence(
            home,
            evidence_tools.TraceQueryEvidenceInput(validation_path=str(run), case_id="unknown", query_id="q1"),
        )
    with pytest.raises(ValueError, match="escapes"):
        trace(home, home.parent / "outside")


@pytest.mark.parametrize("kwargs", [{"query_id": "../q"}, {"case_id": "../c"}, {"evidence_session_ids": [""]}])
def test_invalid_inputs(kwargs):
    with pytest.raises(ValidationError):
        evidence_tools.TraceQueryEvidenceInput.model_validate(
            {"validation_path": "run", "case_id": "case1", "query_id": "q1", **kwargs},
        )


@pytest.mark.asyncio
async def test_registered_job_and_restricted_scope(app, prepared, query_evidence):
    home, source_run, _repo, _first, _second, _source, _case = query_evidence
    run = write_run(prepared, case_ids=["case1"], query_ids={"case1": ["q1"]})
    shutil.copytree(home / "datasets/training-set", prepared / "datasets/training-set", dirs_exist_ok=True)
    shutil.copytree(source_run / "cases", run / "cases")
    before = snapshot(prepared)
    result = await call(app, "home_trace_query_evidence", validation_path=str(run), case_id="case1", query_id="q1")
    assert result["query"]["score"] == 0 and result["sessions"][0]["checkpoint"]
    response = await app.run_job(
        "home_trace_query_evidence",
        validation_path=str(run),
        case_id="case1",
        query_id="q1",
        _allowed_paths=[str(run)],
    )
    assert not response.success and "unrestricted" in str(response.answer)
    assert snapshot(prepared) == before
