"""Current-workflow benchmark inputs, rejected actions and native usage accounting."""

# pylint: disable=missing-function-docstring,protected-access

from copy import deepcopy
from pathlib import Path

import pytest
import yaml

from benchmark.proactive.cases import CASES, MEMORY, META, seed
from benchmark.proactive.evaluate import grade, percentile, report
from benchmark.proactive.prompts import freeze
from benchmark.proactive.run import initialize, make_tasks
from benchmark.proactive.runtime import configuration, files
from reme.config import resolve_app_config


def record_for(root, case_id):
    seed(root, CASES[case_id])
    original = files(root)
    jobs = [
        {
            "name": name,
            "args": {"content": "", "kind": "normal"} if name == "send_reminder" else {},
            "response": {"success": True},
        }
        for name in (
            "get_proactive_messages",
            "read_conversation",
            "read_task_results",
            "get_reminder",
            "send_reminder",
        )
    ]
    return {
        "fixture": CASES[case_id],
        "before": original,
        "after": deepcopy(original),
        "jobs": jobs,
        "turns": [],
        "tool_events": [],
        "delivered": [],
        "completed": True,
        "duration_s": 1,
    }


def test_current_bundle_freezes_all_three_roles_and_shared_memory():
    repo = Path(__file__).resolve().parents[2]
    bundle = freeze(repo)
    assert set(bundle) == {"current"} and set(bundle["current"]) == {"pa", "proactive", "task", "memory"}
    prompts = yaml.safe_load((repo / "reme/steps/proactive/proactive_agent.yaml").read_text())
    assert bundle["current"]["proactive"] == prompts


def test_matrix_is_deterministic_and_uses_current_roles(tmp_path):
    manifest = initialize(tmp_path, models=["fixture-model"], repeats=2)
    tasks = make_tasks(manifest)
    assert len(tasks) == 2 * len(CASES) and tasks == make_tasks(manifest)
    assert {row["role"] for row in tasks} == {"pa", "proactive"}
    with pytest.raises(ValueError, match="Model selection"):
        initialize(tmp_path, models=["changed"])


def test_old_manifest_is_preserved_and_rejected(tmp_path):
    path = tmp_path / "manifest.json"
    original = '{"profile":{"effort":"medium"}}'
    path.write_text(original)
    with pytest.raises(ValueError, match="new output directory"):
        initialize(tmp_path)
    assert path.read_text() == original


def test_provider_model_ids_use_distinct_contained_output_paths(tmp_path):
    models = ["provider/model", "provider_model", "../escape", "/absolute"]
    tasks = make_tasks(initialize(tmp_path, models=models, repeats=1))
    assert len({task["id"] for task in tasks}) == len(models) * len(CASES)
    for task in tasks:
        path = tmp_path / "cases" / task["id"]
        assert path.parent == tmp_path / "cases"
        assert task["model"] in models


def test_native_fixture_paths_and_toolsets_match_production(tmp_path):
    paths = seed(tmp_path, CASES["proactive_evidence"])
    assert paths and all(name.startswith("resource/fixtures/") for name in paths)
    assert (tmp_path / MEMORY).is_file() and (tmp_path / META / "session_ids.json").is_file()
    repo = Path(__file__).resolve().parents[2]
    cfg = configuration(
        resolve_app_config(config="proactive", log_config=False),
        "fixture-model",
        tmp_path,
        freeze(repo)["current"],
        CASES["proactive_evidence"],
    )
    assert cfg["jobs"]["task_agent"]["steps"][0]["job_tools"] == ["screen_capture", "finance_news"]
    assert len(cfg["jobs"]["pa_proactive"]["steps"][0]["job_tools"]) == 7
    assert "reminder_delivery" not in cfg["jobs"]


def test_empty_decision_is_valid_silence_but_nonempty_rejected_attempt_is_not(tmp_path):
    record = record_for(tmp_path, "proactive_known")
    assert grade("proactive_known", record)["score"] == 100
    record["tool_events"] = [
        {
            "event": "start",
            "tool_call_name": "mcp__reme_jobs__send_reminder",
            "chunk": {"arguments": {"content": "should not send"}},
        },
    ]
    assert not grade("proactive_known", record)["checks"]["send_attempts"]


def test_incomplete_or_missing_protocol_cannot_score_for_inaction(tmp_path):
    record = record_for(tmp_path, "proactive_known")
    record["completed"] = False
    assert grade("proactive_known", record)["score"] == 0
    record["completed"] = True
    record["jobs"] = []
    assert grade("proactive_known", record)["score"] == 0


def test_only_configured_task_snapshot_paths_are_allowed_new_files(tmp_path):
    from benchmark.proactive.cases import frozen_now

    record = record_for(tmp_path, "proactive_evidence")
    snapshot = f"daily/{frozen_now():%Y-%m-%d}/cron/fixture-0/{frozen_now():%H%M%S}.md"
    record["after"][snapshot] = "completed cumulative result"
    assert grade("proactive_evidence", record)["checks"]["no_unexpected_files"]
    record["after"][snapshot.replace("fixture-0", "unknown")] = "unexpected"
    assert not grade("proactive_evidence", record)["checks"]["no_unexpected_files"]


def test_forgetting_detects_copies_and_keeps_unrelated_facts(tmp_path):
    record = record_for(tmp_path, "pa_forgetting")
    record["after"][MEMORY] = "中文，简短；长期研究 ReMe，周末徒步。"
    assert grade("pa_forgetting", record)["checks"]["forgets_requested_memory"]
    record["after"]["daily/another.md"] = "海棠计划预算7300元。"
    verdict = grade("pa_forgetting", record)["checks"]
    assert not verdict["no_forgotten_fact_copy"] and verdict["keeps_valid_facts"]


@pytest.mark.parametrize("portable", [True, False])
def test_report_deltas_native_cache_and_reasoning_counters(tmp_path, portable):
    turns = []
    for index, (tokens, cached, reasoning) in enumerate(((100, 4, 2), (220, 9, 6))):
        turn = {
            "session_id": "same",
            "native_token_usage": {
                "tokenUsage": {
                    "total": {
                        "inputTokens": tokens,
                        "outputTokens": 20 * (index + 1),
                        "cachedInputTokens": cached,
                        "reasoningOutputTokens": reasoning,
                    },
                },
            },
        }
        if portable:
            turn["usage"] = {
                "input_tokens": 100 if index == 0 else 120,
                "output_tokens": 20,
                "total_tokens": 120 if index == 0 else 140,
            }
        turns.append(turn)
    result = report(
        tmp_path,
        results=[
            {
                "model": "fixture",
                "prompt": "current",
                "case": "pa_preferences",
                "role": "pa",
                "repeat": 0,
                "grade": {"score": 100},
                "record": {"completed": True, "duration_s": 1, "turns": turns},
            },
        ],
    )
    usage = result["ranking"][0]["usage"]
    assert usage["cached_input_tokens"] == 9 and usage["reasoning_output_tokens"] == 6
    assert usage["input_tokens"] == 220


def test_quantiles_include_singletons():
    assert percentile([3], 0.9) == 3 and percentile([0, 10], 0.9) == 9
