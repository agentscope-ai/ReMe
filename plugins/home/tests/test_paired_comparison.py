"""Paired evidence and immutable selection using isolated artifacts and Git repos."""

# pylint: disable=missing-function-docstring,redefined-outer-name,unused-argument,protected-access

from pathlib import Path

import pytest

from reme_home.agent.optimizer import OptimizationResult, verify_final
from reme_home.code import repository
from reme_home.evaluation.artifacts import read_json, write_json
from reme_home.evaluation.comparison import compare_validations, publish_comparisons

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


async def call(app, name, **kwargs):
    result = await app.run_job(name, **kwargs)
    assert result.success, result.answer
    return result.answer


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
