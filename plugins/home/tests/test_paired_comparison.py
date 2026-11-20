"""Paired evidence and immutable selection using isolated artifacts and Git repos."""

# pylint: disable=missing-function-docstring,redefined-outer-name,unused-argument,protected-access

import asyncio
from pathlib import Path
from unittest.mock import Mock

import pytest

from reme_home.agent.optimizer import OptimizationResult, verify_final
from reme_home.code import repository
from reme_home.evaluation import comparison
from reme_home.evaluation.artifacts import read_json, write_json
from reme_home.evaluation.comparison import compact_comparison, compare_validations, publish_comparisons

from test_evaluation import (  # noqa: F401  # pylint: disable=unused-import
    home_application,
    preparation_inputs,
    prepared_home,
)
from test_preparation import command


def save_run(home, name, questions, *, snapshot="memory", judge="judge"):
    run = home / "evaluations" / name / "abcdef0" / "1"
    write_json(
        run / "manifest.json",
        {
            "mode": "scored",
            "status": "completed",
            "success": True,
            "dataset_kind": "training-set",
            "commit_sha": name,
            "fingerprints": {"dataset": "dataset"},
            "judge": {"fingerprint": judge},
            "case_ids": ["case"],
            "query_ids": {"case": list(questions)},
        },
    )
    write_json(
        run / "summary.json",
        {
            "scoring_enabled": True,
            "case_status_counts": {"infra_error": 0},
            "query_status_counts": {"infra_error": 0},
        },
    )
    write_json(run / "cases/case/memory_construction/result.json", {"workspace_sha256": snapshot})
    write_json(
        run / "cases/case/queries/result.json",
        {"queries": [{"query_id": key, **row} for key, row in questions.items()]},
    )
    return run


def query(score, tokens=None, **kwargs):
    return {
        "score": score,
        "question_type": "fact",
        "answer_result": {"token_usage": {"model": {"total_tokens": tokens}}},
        **kwargs,
    }


def test_comparison_pairs_coverage_regressions_and_unknown_tokens(tmp_path):
    base = save_run(tmp_path, "base", {"a": query(1, 100), "b": query(0, 50), "old": query(1, 10)})
    candidate = save_run(
        tmp_path,
        "candidate",
        {"a": query(0.5, 60), "b": query(1), "new": query(1, 500)},
        snapshot="different",
    )
    result = compare_validations(tmp_path, base, candidate)
    assert result["coverage"] == {
        "base_query_count": 3,
        "candidate_query_count": 3,
        "common_case_count": 1,
        "base_only_query_count": 1,
        "candidate_only_query_count": 1,
    }
    totals = result["totals"]
    assert totals["mean_score_delta"] == 0.25
    assert totals["regressed_query_count"] == totals["improved_query_count"] == 1
    assert totals["same_snapshot_query_count"] == 0
    token = totals["query_tokens_by_model"]["model"]
    assert token["mean_delta"] == -40
    assert token["paired_query_count"] == token["unknown_query_count"] == 1
    table = publish_comparisons(tmp_path, candidate)
    assert table["comparison_count"] == 1
    compact = read_json(Path(table["path"]))["rows"][0]
    assert compact["score"]["delta"] == 0.25 and compact["score"]["regressed"] == 1
    assert "by_case" not in compact and "rows" not in compact
    assert "candidate_validation_path" not in compact


def test_failures_are_zero_but_missing_scores_are_unknown(tmp_path):
    base = save_run(tmp_path, "base", {"a": query(1), "b": query(1), "c": query(1)})
    candidate = save_run(
        tmp_path,
        "candidate",
        {"a": query(None, failure_kind="candidate_failure"), "b": query(None), "c": query(1)},
    )
    result = compare_validations(tmp_path, base, candidate)
    assert result["totals"]["scored_pair_count"] == 2
    assert result["totals"]["unknown_score_pair_count"] == 1
    assert result["totals"]["mean_score_delta"] == -0.5
    assert result["totals"]["same_snapshot_query_count"] == 3
    assert result["totals"]["query_tokens_by_model"]["model"]["mean_delta"] is None
    # Missing rows after a known construction failure still count as zero.
    write_json(candidate / "cases/case/queries/result.json", {"queries": []})
    write_json(candidate / "cases/case/failure.json", {"failure_kind": "candidate_failure"})
    assert compare_validations(tmp_path, base, candidate)["totals"]["mean_score_delta"] == -1


def test_contract_mismatch_no_overlap_and_infra_rejection(tmp_path):
    base = save_run(tmp_path, "base", {"a": query(1)})
    candidate = save_run(tmp_path, "candidate", {"b": query(1)}, judge="other")
    with pytest.raises(ValueError, match="scoring policies"):
        compare_validations(tmp_path, base, candidate)
    candidate = save_run(tmp_path, "candidate", {"b": query(1)})
    totals = compare_validations(tmp_path, base, candidate)["totals"]
    assert totals["common_query_count"] == 0 and totals["mean_score_delta"] is None
    summary = read_json(candidate / "summary.json")
    summary["query_status_counts"]["infra_error"] = 1
    write_json(candidate / "summary.json", summary)
    with pytest.raises(ValueError, match="infrastructure"):
        compare_validations(tmp_path, base, candidate)
    with pytest.raises(ValueError):
        compare_validations(tmp_path, "../outside", candidate)


@pytest.mark.parametrize(
    "artifact,payload",
    [
        ("manifest.json", []),
        ("summary.json", []),
        ("summary.json", {"scoring_enabled": True, "case_status_counts": []}),
        ("cases/case/queries/result.json", []),
        ("cases/case/queries/result.json", {"queries": {}}),
        ("cases/case/queries/result.json", {"queries": [None]}),
        ("cases/case/queries/result.json", {"queries": [{"query_id": "a", "answer_result": "bad"}]}),
        ("cases/case/queries/result.json", {"queries": [{"query_id": "a", "answer_result": []}]}),
        ("cases/case/memory_construction/result.json", []),
        ("cases/case/failure.json", []),
    ],
)
def test_malformed_history_is_skipped_without_losing_valid_comparisons(tmp_path, artifact, payload):
    broken = save_run(tmp_path, "a-broken", {"a": query(1)})
    base = save_run(tmp_path, "base", {"a": query(1)})
    candidate = save_run(tmp_path, "candidate", {"a": query(0.5)})
    write_json(broken / artifact, payload)
    with pytest.raises(ValueError):
        compare_validations(tmp_path, broken, candidate)
    published = publish_comparisons(tmp_path, candidate)
    assert published["comparison_count"] == published["skipped_count"] == 1
    table = read_json(Path(published["path"]))
    assert table["candidate_validation_path"] == candidate.relative_to(tmp_path).as_posix()
    assert table["rows"][0]["base_validation_path"] == base.relative_to(tmp_path).as_posix()
    assert table["rows"][0]["score"]["delta"] == -0.5
    assert set(table) == {"schema_version", "candidate_validation_path", "note", "rows", "skipped_count"}


@pytest.mark.parametrize("history_count,valid_current", [(0, False), (2, False), (2, True)])
def test_table_reuses_valid_current_evidence_without_changing_results(
    tmp_path,
    monkeypatch,
    history_count,
    valid_current,
):
    history = [save_run(tmp_path, f"base-{index}", {"a": query(index, 100)}) for index in range(history_count)]
    candidate = save_run(tmp_path, "candidate", {"a": query(0.5, 60)})
    expected = []
    if valid_current:
        for base in history:
            row = compact_comparison(compare_validations(tmp_path, base, candidate), tmp_path)
            row.pop("candidate_validation_path")
            expected.append(row)
    else:
        write_json(candidate / "summary.json", [])
    load = comparison._load
    current_loads = []

    def tracked_load(home, path):
        if path == candidate:
            current_loads.append(path)
        return load(home, path)

    monkeypatch.setattr(comparison, "_load", tracked_load)
    published = publish_comparisons(tmp_path, candidate)
    table = read_json(Path(published["path"]))
    assert table["rows"] == expected
    assert published["comparison_count"] == len(expected)
    assert table["skipped_count"] == published["skipped_count"] == history_count - len(expected)
    assert len(current_loads) == (min(1, history_count) if valid_current else history_count)


def test_table_retries_current_evidence_after_transient_read_failure(tmp_path, monkeypatch):
    save_run(tmp_path, "base-0", {"a": query(0)})
    base = save_run(tmp_path, "base-1", {"a": query(1)})
    candidate = save_run(tmp_path, "candidate", {"a": query(0.5)})
    load = comparison._load
    failed = False

    def flaky_load(home, path):
        nonlocal failed
        if path == candidate and not failed:
            failed = True
            raise OSError("Transient read failure")
        return load(home, path)

    monkeypatch.setattr(comparison, "_load", flaky_load)
    published = publish_comparisons(tmp_path, candidate)
    table = read_json(Path(published["path"]))
    assert published["comparison_count"] == published["skipped_count"] == 1
    assert table["rows"][0]["base_validation_path"] == base.relative_to(tmp_path).as_posix()
    assert table["rows"][0]["score"]["delta"] == -0.5


def test_compact_evidence_keeps_decision_metrics_without_redundant_coverage(tmp_path):
    base = save_run(tmp_path, "base", {"a": query(1, 100), "old": query(1, 50)})
    candidate = save_run(tmp_path, "candidate", {"a": query(0.5, 60), "new": query(None)})
    compact = compact_comparison(compare_validations(tmp_path, base, candidate), tmp_path)
    assert compact["coverage"] == {
        "base_query_count": 2,
        "candidate_query_count": 2,
        "common_case_count": 1,
        "common_query_count": 1,
    }
    assert compact["score"]["delta"] == -0.5
    assert compact["score"]["regressed"] == 1
    assert compact["tokens_by_model"]["model"] == {"pairs": 1, "base": 100, "candidate": 60, "delta": -40}
    assert compact["same_snapshot_queries"] == 1
    assert "rows" not in compact and "by_case" not in compact
    # Relative paths remain directly usable by inspection/comparison tools.
    rebuilt = compare_validations(tmp_path, compact["base_validation_path"], compact["candidate_validation_path"])
    assert rebuilt["totals"]["mean_score_delta"] == -0.5


async def call(app, name, **kwargs):
    result = await app.run_job(name, **kwargs)
    assert result.success, result.answer
    return result.answer


@pytest.mark.asyncio
@pytest.mark.parametrize("error_type", [OSError, AttributeError])
async def test_comparison_failure_does_not_fail_completed_evaluation(app, prepared, monkeypatch, error_type):
    def fail_write(*args, **kwargs):
        raise error_type("private comparison error")

    monkeypatch.setattr(comparison, "write_json", fail_write)
    logger = Mock()
    monkeypatch.setattr(app.context.components["home.evaluation"]["default"], "logger", logger)
    response = await app.run_job("home_evaluate")
    assert response.success, response.answer
    assert response.answer["status"] == "completed"
    assert "paired_comparisons" not in response.answer
    assert "private comparison error" not in str(response.answer) + str(response.metadata)
    assert "reme.job_error" not in response.metadata
    run = Path(response.answer["result_dir"])
    logger.exception.assert_called_once_with(f"Paired comparison publication failed for completed evaluation {run}")
    manifest = read_json(run / "manifest.json")
    assert manifest["status"] == "completed" and manifest["success"] is True
    assert (run / "summary.json").is_file()
    assert not (run / "paired_comparisons.json").exists()
    assert verify_final(
        prepared,
        OptimizationResult(
            branch_name="init",
            commit_sha=response.answer["commit_sha"],
            evaluation_path=str(run),
            report="Completed evaluation remains eligible without derived comparisons",
        ),
    ) == str(run)


@pytest.mark.asyncio
async def test_comparison_cancellation_propagates_without_demoting_evaluation(app, prepared, monkeypatch):
    def cancel(*args, **kwargs):
        raise asyncio.CancelledError()

    monkeypatch.setattr(comparison, "publish_comparisons", cancel)
    with pytest.raises(asyncio.CancelledError):
        await app.run_job("home_evaluate")
    manifests = list((prepared / "evaluations").glob("*/*/*/manifest.json"))
    assert len(manifests) == 1
    manifest = read_json(manifests[0])
    assert manifest["status"] == "completed" and manifest["success"] is True
    assert (manifests[0].parent / "summary.json").is_file()


@pytest.mark.asyncio
async def test_formal_evaluation_pins_selectable_commit_and_publishes_table(app, prepared):
    baseline = await call(app, "home_evaluate")
    await call(app, "home_create_candidate_branch", branch_name="candidate")
    repo = prepared / "code/repo"
    target = repo / "plugins/beam/src/reme_beam/__init__.py"
    target.write_text("value = 2\n")
    first = await call(app, "home_commit_candidate", message="First")
    evaluated = await call(app, "home_evaluate")
    pin = "evaluated-" + first["commit_sha"]
    assert evaluated["evaluated_branch"] == pin
    assert evaluated["paired_comparisons"]["comparison_count"] == 1
    target.write_text("value = 3\n")
    await call(app, "home_commit_candidate", message="Continue same branch")
    await call(app, "home_switch_candidate_branch", branch_name=first["commit_sha"][:7].upper())
    assert command(repo, "rev-parse", "HEAD") == first["commit_sha"]
    selection = OptimizationResult(
        branch_name=pin[:17],
        commit_sha=first["commit_sha"][:7].upper(),
        evaluation_path=evaluated["result_dir"],
        report="Select the preserved evaluated version",
    )
    assert verify_final(prepared, selection) == evaluated["result_dir"]
    assert selection.commit_sha == first["commit_sha"] and selection.branch_name == pin
    catalog = await call(app, "home_list_validation_results", branch_name=pin)
    assert catalog["row_count"] == 1
    assert catalog["rows"][0]["validation_path"] == evaluated["result_dir"]
    for name, kwargs in (
        ("home_commit_candidate", {"message": "Forbidden"}),
        ("home_merge_candidate_branch", {"branch_name": "candidate"}),
        ("home_create_candidate_branch", {"branch_name": "evaluated-fake"}),
    ):
        assert not (await app.run_job(name, **kwargs)).success
    paired = await call(
        app,
        "home_compare_validations",
        base_validation_path=baseline["result_dir"],
        candidate_validation_path=evaluated["result_dir"],
    )
    assert paired["coverage"]["common_query_count"] == 2
    assert paired["score"]["unknown_pairs"] == 0
    assert "rows" not in paired and "by_case" not in paired
    await call(app, "home_create_candidate_branch", branch_name="continue-from-pin")
    target.write_text("value = 4\n")
    await call(app, "home_commit_candidate", message="Allowed fork")
    assert command(repo, "rev-parse", pin) == first["commit_sha"]


def test_ambiguous_commit_prefix_is_not_guessed(monkeypatch, tmp_path):
    commits = ["a" * 40, "ab" + "0" * 38]
    monkeypatch.setattr(repository, "git", lambda *args: "\n".join(commits))
    assert repository.resolve_commit_prefix(tmp_path, "AB") == commits[1]
    with pytest.raises(ValueError, match="2 matches"):
        repository.resolve_commit_prefix(tmp_path, "a")
    with pytest.raises(ValueError, match="0 matches"):
        repository.resolve_commit_prefix(tmp_path, "ff")
