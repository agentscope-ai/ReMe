"""AgentScope tool for synchronously validating the current Meta-ReMe code."""

# Import roots must be installed before importing the hyphenated meta-reme tree.
# pylint: disable=wrong-import-position

from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any
from uuid import uuid4

from agentscope.tool import FunctionTool

from runtime import TOOL_RUNTIME

# The meta-reme tree is intentionally not a regular Python package.
META_REME_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = META_REME_ROOT.parent
for import_root in (META_REME_ROOT, PROJECT_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from validation import run_validation  # noqa: E402


def validation_tool(case_ids: list[str] | None = None, fail_fast: bool = False) -> dict[str, Any]:
    """Validate selected workspace cases and return only a run-level summary.

    Args:
        case_ids: Case IDs installed in the active workspace. Omit or pass an
            empty list to validate every installed case.
        fail_fast: Stop the validation at the first error when true.

    Returns:
        Counts of cases that ran without errors or failed, the mean query
        score, and the result directory containing full artifacts.
    """

    if not isinstance(fail_fast, bool):
        raise ValueError("fail_fast must be a boolean")
    workspace, concurrency = TOOL_RUNTIME.require_configured()
    return _run_validation_tool(workspace, concurrency, case_ids, fail_fast)


def _run_validation_tool(
    workspace: Path,
    concurrency: int,
    case_ids: list[str] | None,
    fail_fast: bool,
) -> dict[str, Any]:
    """Run the tool against explicit runtime values."""

    selected_case_ids = _case_ids(workspace, case_ids)
    validation_id = uuid4().hex
    result_dir: Path | None = None

    error: str | None = None
    try:
        result_dir = run_validation(
            workspace,
            selected_case_ids,
            concurrency,
            validation_id=validation_id,
            fail_fast=fail_fast,
        )
    except Exception as exc:  # Validation persists diagnostics before raising.
        error = f"{type(exc).__name__}: {exc}"
        result_dir = _find_result_dir(workspace, validation_id)

    return _read_summary(result_dir, len(selected_case_ids), error)


def create_validation_tool(
    workspace: str | Path | None = None,
    concurrency: int | None = None,
) -> FunctionTool:
    """Create a validation tool using global or explicitly bound runtime values."""

    function = validation_tool
    if workspace is not None:
        bound_workspace = Path(workspace).resolve()
        bound_concurrency = 1 if concurrency is None else concurrency
        if not isinstance(bound_concurrency, int) or isinstance(bound_concurrency, bool) or bound_concurrency < 1:
            raise ValueError("concurrency must be a positive integer")

        def bound_validation_tool(
            case_ids: list[str] | None = None,
            fail_fast: bool = False,
        ) -> dict[str, Any]:
            """Validate selected cases in the workspace bound by the host."""

            if not isinstance(fail_fast, bool):
                raise ValueError("fail_fast must be a boolean")
            return _run_validation_tool(bound_workspace, bound_concurrency, case_ids, fail_fast)

        function = bound_validation_tool
    elif concurrency is not None:
        raise ValueError("concurrency requires an explicitly bound workspace")

    return FunctionTool(
        function,
        name="validation_tool",
        description=(
            "Validate selected cases against the current clean Meta-ReMe code commit. "
            "Omit case_ids to run all workspace cases; fail_fast stops at the first error. "
            "The result includes details_path, which points to the complete validation artifacts."
        ),
        is_concurrency_safe=False,
    )


def _case_ids(workspace: Path, requested: list[str] | None) -> list[str]:
    if requested is not None and not isinstance(requested, list):
        raise ValueError("case_ids must be a list of strings or null")
    if requested:
        if any(not isinstance(case_id, str) or not case_id for case_id in requested):
            raise ValueError("case_ids must contain non-empty strings")
        if len(requested) != len(set(requested)):
            raise ValueError("case_ids must be unique")
        return requested

    installed: list[str] = []
    for path in sorted((workspace / "dataset/cases").glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        case_id = payload.get("case_id")
        if not isinstance(case_id, str) or not case_id:
            raise ValueError(f"prepared case has no valid case_id: {path}")
        installed.append(case_id)
    if not installed:
        raise ValueError("active workspace has no prepared cases")
    if len(installed) != len(set(installed)):
        raise ValueError("active workspace contains duplicate case IDs")
    return installed


def _find_result_dir(workspace: Path, validation_id: str) -> Path | None:
    matches = list((workspace / "evaluations").glob(f"*/*/{validation_id}"))
    return matches[0] if len(matches) == 1 else None


def _read_summary(result_dir: Path | None, requested_count: int, error: str | None) -> dict[str, Any]:
    if result_dir is None:
        return {
            "status": "failed",
            "requested_cases": requested_count,
            "run_cases": 0,
            "successful_cases": 0,
            "failed_cases": 0,
            "mean_query_score": None,
            "result_dir": None,
            "details_path": None,
            "error": error or "validation did not create a result directory",
        }
    summary_path = result_dir / "summary.json"
    if summary_path.is_file():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        successful = sum(case.get("status") == "completed" for case in summary.get("cases", []))
        failed = int(summary.get("case_count", 0)) - successful
        return {
            "status": summary.get("status", "completed"),
            "requested_cases": requested_count,
            "run_cases": int(summary.get("case_count", 0)),
            "successful_cases": successful,
            "failed_cases": failed,
            "mean_query_score": summary.get("mean_query_score"),
            "result_dir": str(result_dir),
            "details_path": str(result_dir),
            "error": error,
        }

    case_results = []
    for path in sorted((result_dir / "cases").glob("*/case_result.json")):
        case_results.append(json.loads(path.read_text(encoding="utf-8")))
    query_results = [query for case in case_results for query in case.get("queries", [])]
    scores = [float(query["score"]) if query.get("score") is not None else 0.0 for query in query_results]
    successful = sum(case.get("status") == "completed" for case in case_results)
    return {
        "status": "failed" if error else "partial",
        "requested_cases": requested_count,
        "run_cases": len(case_results),
        "successful_cases": successful,
        "failed_cases": len(case_results) - successful,
        "mean_query_score": sum(scores) / len(scores) if scores else None,
        "result_dir": str(result_dir),
        "details_path": str(result_dir),
        "error": error or "validation did not publish summary.json",
    }
