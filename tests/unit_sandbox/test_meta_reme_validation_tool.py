"""Focused tests for the Meta-ReMe AgentScope validation tool."""

# pylint: disable=missing-function-docstring

from __future__ import annotations

import importlib
import json
from pathlib import Path
import sys

META_REME = Path(__file__).resolve().parents[2] / "meta-reme"
if str(META_REME) not in sys.path:
    sys.path.insert(0, str(META_REME))

runtime = importlib.import_module("runtime")
tool_module = importlib.import_module("as.tools.validation_tool")


def _workspace(tmp_path: Path) -> Path:
    workspace = tmp_path / "workspace"
    cases = workspace / "dataset/cases"
    cases.mkdir(parents=True)
    for index, case_id in enumerate(("case-b", "case-a")):
        (cases / f"{index}.json").write_text(json.dumps({"case_id": case_id}), encoding="utf-8")
    return workspace


def test_validation_tool_uses_runtime_defaults_and_returns_summary(tmp_path: Path, monkeypatch) -> None:
    workspace = _workspace(tmp_path)
    runtime.TOOL_RUNTIME.configure(workspace, 3)
    calls = []

    def fake_validation(root, case_ids, concurrency, *, validation_id, fail_fast):
        calls.append((root, case_ids, concurrency, validation_id, fail_fast))
        result_dir = Path(root) / "evaluations/candidate/abc123" / validation_id
        result_dir.mkdir(parents=True)
        (result_dir / "summary.json").write_text(
            json.dumps(
                {
                    "status": "completed",
                    "case_count": 2,
                    "mean_query_score": 0.75,
                    "cases": [{"status": "completed"}, {"status": "candidate_failure"}],
                },
            ),
            encoding="utf-8",
        )
        return result_dir

    monkeypatch.setattr(tool_module, "run_validation", fake_validation)

    result = tool_module.validation_tool(fail_fast=True)

    assert calls[0][0:3] == (workspace.resolve(), ["case-b", "case-a"], 3)
    assert calls[0][-1] is True
    assert result == {
        "status": "completed",
        "requested_cases": 2,
        "run_cases": 2,
        "successful_cases": 1,
        "failed_cases": 1,
        "mean_query_score": 0.75,
        "result_dir": str(workspace / "evaluations/candidate/abc123" / calls[0][3]),
        "details_path": str(workspace / "evaluations/candidate/abc123" / calls[0][3]),
        "error": None,
    }


def test_validation_tool_summarizes_persisted_results_after_fail_fast(tmp_path: Path, monkeypatch) -> None:
    workspace = _workspace(tmp_path)
    runtime.TOOL_RUNTIME.configure(workspace, 1)

    def fake_validation(root, case_ids, concurrency, *, validation_id, fail_fast):
        del case_ids, concurrency, fail_fast
        case_root = Path(root) / "evaluations/candidate/abc123" / validation_id / "cases/case-a"
        case_root.mkdir(parents=True)
        (case_root / "case_result.json").write_text(
            json.dumps({"status": "candidate_failure", "queries": []}),
            encoding="utf-8",
        )
        raise RuntimeError("construction failed")

    monkeypatch.setattr(tool_module, "run_validation", fake_validation)

    result = tool_module.validation_tool(["case-a"], fail_fast=True)

    assert result["status"] == "failed"
    assert result["run_cases"] == 1
    assert result["successful_cases"] == 0
    assert result["failed_cases"] == 1
    assert result["mean_query_score"] is None
    assert result["error"] == "RuntimeError: construction failed"


def test_create_validation_tool_is_not_concurrency_safe() -> None:
    tool = tool_module.create_validation_tool()

    assert tool.name == "validation_tool"
    assert tool.is_concurrency_safe is False
