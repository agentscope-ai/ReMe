"""Current-workflow benchmark inputs, rejected actions and native usage accounting."""

# pylint: disable=missing-function-docstring,protected-access

from copy import deepcopy
from pathlib import Path
import json

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
            "args": {"content": "", "kind": "normal"} if name == "send_chat_message" else {},
            "response": {"success": True},
        }
        for name in (
            "read_conversation",
            "read_proactive_inbox",
            "read_cron_results",
            "get_chat_messages",
            "send_chat_message",
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
    assert set(bundle) == {"current"}
    assert set(bundle["current"]) == {"pa", "proactive", "cron", "temporary", "memory"}
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
    assert cfg["jobs"]["cron_agent"]["steps"][0]["job_tools"] == [
        "screen_capture",
        "finance_news",
        "get_chat_messages",
        "send_chat_message_immediate",
    ]
    assert set(cfg["jobs"]["cron_agent"]["steps"][0]["job_tools"]) <= cfg["jobs"].keys()
    assert all(
        set(task) == {"id", "cron", "query", "enabled"}
        for task in json.loads((tmp_path / META / "cron.json").read_text())
    )
    assert len(cfg["jobs"]["pa_proactive"]["steps"][0]["job_tools"]) == 7
    assert set(cfg["jobs"]["pa_proactive"]["steps"][0]["job_tools"]) <= cfg["jobs"].keys()
    assert "reminder_delivery" not in cfg["jobs"]


def test_empty_decision_is_valid_silence_but_nonempty_rejected_attempt_is_not(tmp_path):
    record = record_for(tmp_path, "proactive_known")
    assert grade("proactive_known", record)["score"] == 100
    record["tool_events"] = [
        {
            "event": "start",
            "tool_call_name": "mcp__reme_jobs__send_chat_message",
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
    snapshot = f"{META}/cron/fixture-0/{frozen_now():%Y-%m-%d}/{frozen_now():%H%M%S}.md"
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


def test_quality_fixtures_use_current_clock_and_discoverable_results(tmp_path):
    from datetime import datetime
    from benchmark.proactive.reminder_quality import scenarios
    from reme.schema.pa_proactive import PaProactiveConfig
    from reme.components._pa_tasks import TaskFiles
    from types import SimpleNamespace

    now = datetime.fromisoformat("2026-11-09T14:00:00+08:00")
    cases = scenarios(now)
    assert len(cases) == 11
    case = cases["proactive_new"]
    seed(tmp_path, case)
    config = json.loads((tmp_path / META / "proactive.json").read_text())
    assert not PaProactiveConfig.model_validate(config).hints_enabled
    runtime = SimpleNamespace(
        workspace_path=tmp_path,
        _now=lambda: now,
        _metadata_file=lambda name: tmp_path / META / name,
    )
    paths = TaskFiles(runtime).paths("fixture-0")
    assert len(paths) == 1 and "37项本地测试" in paths[0].read_text()
    assert all(row["created_at"] == now.isoformat() for case in cases.values() for row in case["messages"])
    notes = json.loads((tmp_path / META / "session_ids.json").read_text())
    assert notes[0]["last_active_at"] == now.isoformat()


def test_quality_checks_reject_incomplete_silence_and_out_of_phase_attempts():
    from benchmark.proactive.reminder_quality import checks

    case = {"role": "cron", "expected_send": False, "required": [], "max_chars": 240, "reminders": []}
    record = {"jobs": [], "completed": False, "before": {MEMORY: "x"}, "after": {MEMORY: "x"}}
    assert not checks(case, record)["completed"]
    record.update(completed=True)
    record["jobs"] = [{"name": "get_chat_messages", "args": {}, "turn_index": 0, "error": "wrong phase"}]
    verdict = checks(case, record)
    assert not verdict["tool_calls_succeeded"] and not verdict["message_phase_only"]
    assert not verdict["no_unneeded_history"]
    record["jobs"] = [
        {"name": "send_chat_message_immediate", "args": {"content": "new"}, "turn_index": 2},
        {"name": "get_chat_messages", "args": {}, "turn_index": 2},
    ]
    assert not checks(case, record)["history_before_send"]
    assert not checks(case, record)["expected_decision"]


@pytest.mark.asyncio
async def test_fixture_owner_preserves_empty_chat_and_native_append_contract():
    from types import SimpleNamespace
    from unittest.mock import AsyncMock
    from benchmark.proactive.runtime import bind_owner
    from benchmark.proactive.cases import SESSION_ID
    from reme.components.codex_app_server import SessionChangedError

    agent = SimpleNamespace(
        read_session=AsyncMock(),
        get_session_status=AsyncMock(),
        append_session=AsyncMock(),
        reply=AsyncMock(),
    )
    record = {"delivered": []}
    bind_owner(agent, {"messages": [], "session_status": "not_loaded"}, record)
    assert (await agent.read_session(SESSION_ID)).last_turn_id is None
    with pytest.raises(SessionChangedError, match="changed"):
        await agent.append_session(SESSION_ID, "结果", message_id="one", expected_turn_id="stale")
    first = await agent.append_session(SESSION_ID, "结果", message_id="one", expected_turn_id="")
    assert first.id == "reme-proactive-one"
    assert (await agent.read_session(SESSION_ID)).last_turn_id is None
    assert await agent.append_session(SESSION_ID, "结果", message_id="one") == first
    with pytest.raises(ValueError, match="different content"):
        await agent.append_session(SESSION_ID, "不同结果", message_id="one")
    assert len(record["delivered"]) == 1


@pytest.mark.asyncio
async def test_busy_fixture_owner_is_retryable_instead_of_a_permanent_failure():
    from types import SimpleNamespace
    from unittest.mock import AsyncMock
    from benchmark.proactive.runtime import bind_owner
    from benchmark.proactive.cases import SESSION_ID
    from reme.components.codex_app_server import SessionChangedError

    agent = SimpleNamespace(
        read_session=AsyncMock(),
        get_session_status=AsyncMock(),
        append_session=AsyncMock(),
        reply=AsyncMock(),
    )
    bind_owner(agent, {"messages": [], "session_status": "active"}, {"delivered": []})
    with pytest.raises(SessionChangedError, match="busy"):
        await agent.append_session(SESSION_ID, "结果", message_id="one")
