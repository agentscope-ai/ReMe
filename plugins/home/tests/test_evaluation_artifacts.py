"""Reviewed dev_meta result content, compact statistics and rebuildability."""

# pylint: disable=missing-function-docstring

import asyncio
from copy import deepcopy
import json

import pytest

from reme_home.evaluation.artifacts import average_tokens, read_json, rebuild_summary, summarize, write_json
from reme_home.evaluation.compaction import compact_construction, compact_query
from reme_home.preparation.data.models import CaseSpec, QuerySpec


def test_query_compaction_preserves_evidence_without_duplicate_prompts_or_costs():
    raw = {
        "question": "What happened?",
        "golden_answer": "An event.",
        "answer": "The event happened.",
        "score": 1.0,
        "error": None,
        "answer_result": {
            "success": True,
            "answer": "The event happened.",
            "error": None,
            "job_call_counts": {"agentic_answer": 2, "search": 3},
            "token_usage": {
                "answer": {"input_tokens": 10, "output_tokens": 3, "total_tokens": 13},
                "empty": {"input_tokens": None, "output_tokens": None, "total_tokens": None},
            },
            "metadata": {
                "query": "What happened?",
                "response": "The event happened.",
                "query_time": "2026-09-12T10:00:00",
                "sys_prompt": "reconstructable prompt",
                "nested": [{"sys_prompt": "nested prompt", "evidence": "keep"}],
            },
        },
        "judge_result": {
            "success": True,
            "answer": "yes",
            "job_call_counts": {"answer_judge": 1},
            "token_usage": {"judge": {"total_tokens": 100}},
            "metadata": {
                "answer_judgement": "yes",
                "raw_answer_judgement": "Yes, the answer is correct.",
                "agent_answer": "The event happened.",
                "golden_answer": "An event.",
                "llm_response": "The event happened.",
                "probing_question": "What happened?",
                "query": "What happened?",
                "question_type": "fact",
                "rubric": ["An event"],
            },
        },
    }
    original = deepcopy(raw)
    result = compact_query(raw)
    assert raw == original
    assert result["answer_result"] == {
        "success": True,
        "job_call_counts": {"agentic_answer": 1, "search": 3},
        "token_usage": {"answer": {"input_tokens": 10, "output_tokens": 3, "total_tokens": 13}},
        "metadata": {"query_time": "2026-09-12T10:00:00", "nested": [{"evidence": "keep"}]},
    }
    assert result["judge_result"] == {
        "success": True,
        "metadata": {"raw_answer_judgement": "Yes, the answer is correct."},
    }
    assert result["question"] == raw["question"]
    assert result["golden_answer"] == raw["golden_answer"]
    assert result["answer"] == raw["answer"]


def test_distinct_failure_details_and_failed_judge_response_are_retained():
    raw = {
        "answer": "partial answer",
        "error": "judge failed",
        "score": None,
        "answer_result": {"answer": "different", "error": "answer failed"},
        "judge_result": {"answer": "invalid judgement", "metadata": {"reason": "diagnostic"}},
        "exception": {"type": "TimeoutError", "causes": [{"type": "ReadTimeout"}]},
    }
    assert compact_query(raw) == raw


def test_construction_compaction_preserves_jobs_and_only_drops_reconstructable_data():
    raw = {
        "success": True,
        "jobs": [
            {
                "job": "auto_memory",
                "memory_checkpoint": {"commit_sha": "abc", "message": "session: s1"},
                "result": {
                    "success": True,
                    "job_call_counts": {"auto_memory": 2, "daily_list": 1},
                    "token_usage": {
                        "answer": {"input_tokens": 10, "output_tokens": 3, "total_tokens": 13},
                        "empty": {"total_tokens": None},
                    },
                    "metadata": {
                        "source_conversation": "[[session/dialog/s1.jsonl]]",
                        "details": [
                            {"session_id": "s1", "source_conversation": "[[session/dialog/s1.jsonl]]"},
                            {"session_id": "s2", "source_conversation": "[[session/dialog/other.jsonl]]"},
                        ],
                        "plugin_payload": {"index": {"notes": ["keep", "this"]}},
                        "index": {
                            "date": "2026-01-01",
                            "path": "custom/daily/2026-01-01.md",
                            "created": False,
                            "changed": True,
                            "notes": [{"text": "old"}, {"text": "new"}],
                        },
                    },
                },
            },
            {
                "job": "index_update",
                "result": {
                    "job_call_counts": {"index_update": 1},
                    "token_usage": {"answer": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}},
                },
            },
            {"job": "index_update", "result": {"token_usage": {"empty": {"total_tokens": None}}}},
        ],
    }
    original = deepcopy(raw)
    result = compact_construction(raw)
    assert raw == original
    job = result["jobs"][0]
    assert "memory_checkpoint" not in job
    assert job["result"]["job_call_counts"] == {"auto_memory": 1, "daily_list": 1}
    assert set(job["result"]["token_usage"]) == {"answer"}
    metadata = job["result"]["metadata"]
    assert metadata["source_conversation"] == "[[session/dialog/s1.jsonl]]"
    assert metadata["details"] == [
        {"session_id": "s1"},
        {"session_id": "s2", "source_conversation": "[[session/dialog/other.jsonl]]"},
    ]
    assert metadata["plugin_payload"] == {"index": {"notes": ["keep", "this"]}}
    assert metadata["index"] == {
        "date": "2026-01-01",
        "path": "custom/daily/2026-01-01.md",
        "created": False,
        "changed": True,
        "notes_count": 2,
    }
    assert result["jobs"][1]["result"] == {"token_usage": 0}
    assert result["jobs"][2]["result"] == {}


def test_summary_matches_reviewed_dev_meta_fields_and_denominators(tmp_path):
    cases = [
        CaseSpec(
            case_id=case_id,
            sessions=[],
            queries=[QuerySpec(query_id=query_id, question="?", golden_answer="a") for query_id in ids],
        )
        for case_id, ids in [("one", ["q1", "q2"]), ("two", ["q3"])]
    ]
    write_json(
        tmp_path / "cases/one/memory_construction/result.json",
        {
            "status": "partial_success",
            "jobs": [{"result": {"success": True}}, {"result": {"success": False}}],
        },
    )
    write_json(
        tmp_path / "cases/one/queries/result.json",
        {
            "queries": [
                {
                    "query_id": "q1",
                    "question_type": "fact",
                    "score": 0.5,
                    "answer_result": {"token_usage": {"answer": {"total_tokens": 12}}},
                },
                {
                    "query_id": "q2",
                    "question_type": "fact",
                    "score": None,
                    "error": "candidate bug",
                    "answer_result": {},
                },
            ],
        },
    )
    write_json(tmp_path / "cases/two/memory_construction/result.json", {"status": "failure", "jobs": []})
    write_json(
        tmp_path / "cases/two/failure.json",
        {"stage": "construction", "failure_kind": "candidate_failure", "error": "build failed"},
    )
    summary = summarize(tmp_path, cases)
    assert summary == {
        "case_count": 2,
        "case_status_counts": {"completed": 0, "candidate_failure": 2, "infra_error": 0},
        "construction_status_counts": {"success": 0, "partial_success": 1, "failure": 1},
        "failure_stage_case_counts": {"construction": 2, "query": 1},
        "expected_query_count": 3,
        "scored_query_count": 1,
        "query_status_counts": {"completed": 1, "candidate_failure": 1, "infra_error": 0, "unscored": 2},
        "mean_query_score": 0.5 / 3,
        "observed_mean_query_score": 0.5,
        "low_score_cases": [{"one": 0.25}],
        "question_type_statistics": {
            "fact": {"query_count": 2, "average_score": 0.5, "average_query_total_token_usage": {"answer": 6.0}},
        },
        "cases": {
            "one": {
                "construction": {"status": "partial_success", "success_job_count": 1, "fail_job_count": 1},
                "mean_score": 0.25,
                "average_query_total_token_usage": {"answer": 6.0},
                "queries": {
                    "q1": {"question_type": "fact", "score": 0.5, "status": "completed"},
                    "q2": {"question_type": "fact", "score": None, "status": "candidate_failure"},
                },
                "construction_error_count": 1,
                "query_error_count": 1,
            },
            "two": {
                "construction": {"status": "failure", "success_job_count": 0, "fail_job_count": 0},
                "mean_score": 0,
                "average_query_total_token_usage": {},
                "queries": {},
                "stage": "construction",
            },
        },
    }
    write_json(
        tmp_path / "manifest.json",
        {
            "status": "completed",
            "mode": "scored",
            "case_ids": ["one", "two"],
            "query_ids": {"one": ["q1", "q2"], "two": ["q3"]},
        },
    )
    assert rebuild_summary(tmp_path) == summary
    assert read_json(tmp_path / "summary.json") == summary
    failure_path = tmp_path / "cases/two/failure.json"
    failure = read_json(failure_path)
    failure["failure_kind"] = "infra_error"
    write_json(failure_path, failure)
    infra = rebuild_summary(tmp_path)
    assert infra["mean_query_score"] is None
    assert infra["observed_mean_query_score"] == 0.5
    assert infra["cases"]["two"]["mean_score"] is None
    assert infra["cases"]["one"]["mean_score"] == 0.25


def test_token_averages_exclude_zero_unknown_and_judge_cost_but_count_all_returned_queries():
    assert average_tokens(
        [
            {
                "answer_result": {"token_usage": {"answer": {"total_tokens": 12}, "zero": {"total_tokens": 0}}},
                "judge_result": {"token_usage": {"judge": {"total_tokens": 100}}},
            },
            {"answer_result": {"token_usage": {"unknown": {"total_tokens": None}, "bad": {"total_tokens": True}}}},
            {},
        ],
    ) == {"answer": 4.0}


def test_compact_json_round_trips_strings_and_keeps_small_leaves_inline(tmp_path):
    value = {"result": {"token_usage": 0}, "unicode": ["你好", 'quote"\n'], "large": [1, 2, 3]}
    path = tmp_path / "result.json"
    write_json(path, value, compact_leaf_containers=True)
    assert read_json(path) == value
    assert '"result": {"token_usage": 0}' in path.read_text()
    assert "\\u4f60" not in path.read_text()
    with pytest.raises(ValueError):
        write_json(path, {"score": float("nan")}, compact_leaf_containers=True)
    assert json.loads(path.read_text()) == value


@pytest.mark.asyncio
@pytest.mark.parametrize("stop", [RuntimeError("judge unavailable"), asyncio.CancelledError()])
async def test_judge_interruption_preserves_all_frozen_answers_and_compacts_each_response_once(tmp_path, stop):
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    from deploy.benchmark_docker.pool import CaseResult
    from reme_home.evaluation.component import HomeEvaluation
    from reme_home.evaluation.inputs import EvaluationSpec

    case = CaseSpec(
        case_id="case",
        sessions=[],
        queries=[QuerySpec(query_id=f"q{i}", question="?", golden_answer="a") for i in range(2)],
    )
    answers = [
        {
            "query_id": query.query_id,
            "answer": "frozen answer",
            "error": None,
            "answer_result": {"answer": "frozen answer", "job_call_counts": {"agentic_answer": 2}},
        }
        for query in case.queries
    ]
    original = deepcopy(answers)
    result = CaseResult("case", build={"success": True, "jobs": []}, queries={"queries": answers})
    spec = EvaluationSpec(fail_fast=True)
    await HomeEvaluation._persist_case(tmp_path, case, result, None, spec)  # pylint: disable=protected-access
    judge = SimpleNamespace(evaluate=AsyncMock(side_effect=stop))
    with pytest.raises(type(stop)):
        await HomeEvaluation._score_case(  # pylint: disable=protected-access
            tmp_path,
            case,
            "beam",
            judge,
            asyncio.Semaphore(1),
            spec,
            answers,
        )
    rows = read_json(tmp_path / "cases/case/queries/result.json")["queries"]
    assert [row["answer"] for row in rows] == ["frozen answer", "frozen answer"]
    assert [row["answer_result"]["job_call_counts"] for row in rows] == [{"agentic_answer": 1}] * 2
    assert answers == original
    assert not (tmp_path / "cases/case/queries/answers.json").exists()


def test_rebuild_rejects_missing_completed_answers_without_replacing_summary(tmp_path):
    write_json(
        tmp_path / "manifest.json",
        {"status": "completed", "case_ids": ["case"], "query_ids": {"case": ["q"]}},
    )
    write_json(tmp_path / "cases/case/memory_construction/result.json", {"status": "success", "jobs": []})
    write_json(tmp_path / "summary.json", {"preserve": "existing summary"})
    with pytest.raises(ValueError, match="incomplete query results"):
        rebuild_summary(tmp_path)
    assert read_json(tmp_path / "summary.json") == {"preserve": "existing summary"}
