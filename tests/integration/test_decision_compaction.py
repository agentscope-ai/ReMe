"""Two live decision-model scenarios; explicitly opt in to paid API calls.

REME_RUN_DECISION_INTEGRATION=1 pytest tests/integration/test_decision_compaction.py -v -s
Credentials come from DECISION_MODEL_* in the environment or the repository .env.
Only synthetic history is sent. No tools in these histories are actually executed.
"""

# pylint: disable=redefined-outer-name

import asyncio
from collections import Counter
from copy import deepcopy
import json
import os
from pathlib import Path

import pytest

from reme import Application
from reme.config import resolve_app_config
from reme.schema.compaction import CompactionRequest
from reme.steps.compaction.messages import collect_pairs
from reme.steps.compaction.state import build_batches, estimate_decision_tokens
from reme.utils.env_utils import load_env

pytestmark = pytest.mark.skipif(
    os.environ.get("REME_RUN_DECISION_INTEGRATION") != "1",
    reason="Set REME_RUN_DECISION_INTEGRATION=1 to authorize live decision API calls.",
)


@pytest.fixture
def live_config():
    """Load credentials only after opt-in; never print or persist them."""
    load_env(Path(__file__).resolve().parents[2] / ".env", override=False)
    if not os.environ.get("DECISION_MODEL_API_KEY", "").strip():
        pytest.skip("DECISION_MODEL_API_KEY is required for live decisions.")
    pytest.importorskip("agentscope.classifier")
    config = resolve_app_config(config="decision_compaction", log_config=False)
    component = config["components"]["as_decision_model"]["compaction"]
    component["timeout"] = 30
    return {
        "service": {"backend": "cli"},
        "jobs": {"compact_context": config["jobs"]["compact_context"]},
        "components": {"as_decision_model": {"compaction": component}},
        "enable_logo": False,
        "log_to_console": False,
        "log_to_file": False,
    }


def _rows(prefix, count=100):
    """Synthetic, reproducible tool output; no production or personal data."""
    return "\n".join(f"{prefix} row={i:04d} status=ok sample=synthetic trace=sample-{i:04d}" for i in range(count))


def _pair(history, call_id, tool, arguments, output, note, *, result_state="success", split=False):
    """Include realistic metadata and optionally span two messages."""
    call = {
        "type": "tool_call",
        "id": call_id,
        "name": tool,
        "input": json.dumps(arguments),
        "state": "finished",
    }
    result = {
        "type": "tool_result",
        "id": call_id,
        "name": tool,
        "output": output,
        "state": result_state,
        "metadata": {"source": "synthetic-fixture"},
    }
    history.append(
        {
            "id": call_id + "-call",
            "role": "assistant",
            "metadata": {"fixture": True},
            "content": [{"type": "text", "text": note}, call] + ([] if split else [result]),
        },
    )
    if split:
        history.append({"id": call_id + "-result", "role": "assistant", "content": [result]})


def _blocks(messages, block_type):
    return [block for message in messages for block in message["content"] if block["type"] == block_type]


async def _run_case(
    config,
    tmp_path,
    *,
    name,
    history,
    goal,
    eligible,
    protected_calls,
    protected_messages,
    must_keep,
    irrelevant,
):
    original = deepcopy(history)
    archive = tmp_path / "original-history.json"
    archive.write_text(json.dumps(history, ensure_ascii=False), encoding="utf-8")
    archived_bytes = archive.read_bytes()
    request = CompactionRequest(
        messages=history,
        goal=goal,
        source_refs={block["id"]: archive.as_uri() + "#" + block["id"] for block in _blocks(history, "tool_call")},
        eligible_tool_names=eligible,
        protected_call_ids=protected_calls,
        protected_message_ids=protected_messages,
        options={"preserve_recent_messages": 2, "max_state_tokens": 6000},
    )
    pairs = collect_pairs(request)
    # Each batch carries only its evidence, rather than repeatedly copying the whole history.
    state, questions = build_batches(request, pairs)[0]
    options = request.options.model_dump()
    options.update(
        max_request_tokens=max(1800, estimate_decision_tokens({"state": state, "questions": questions}) // 3),
        max_concurrency=2,
        total_timeout_seconds=90,
    )
    async with Application(**config, workspace_dir=str(tmp_path / "workspace")) as app:
        response = await app.run_job("compact_context", **request.model_dump(exclude={"options"}), options=options)
    assert response.success, "Live compact_context failed; inspect the configured endpoint and SDK."
    result = response.answer
    assert history == original
    assert archive.read_bytes() == archived_bytes
    assert _blocks(result["messages"], "text") == _blocks(original, "text")
    decisions = {item["call_id"]: item for item in result["decisions"]}
    for call_id in must_keep:
        assert decisions[call_id]["action"] == "keep", f"Current verbatim evidence was lost: {call_id}"
    assert any(decisions[call_id]["action"] == "preview" for call_id in irrelevant)
    remaining = {message["id"]: message for message in result["messages"]}
    protected_ids = {original[0]["id"], *(message["id"] for message in original[-2:]), *protected_messages}
    for message in original:
        if message["id"] in protected_ids:
            assert remaining[message["id"]] == message
    for pair in pairs:
        if pair.protected:
            for mi, bi in (pair.call_location, pair.result_location):
                message = original[mi]
                assert message["content"][bi] in remaining[message["id"]]["content"]
    original_tools = [
        block for message in original for block in message["content"] if block["type"] in {"tool_call", "tool_result"}
    ]
    proposed_tools = [
        block
        for message in result["messages"]
        for block in message["content"]
        if block["type"] in {"tool_call", "tool_result"}
    ]
    untouched = {
        block["id"]
        for block in original_tools
        if block["id"] not in decisions or decisions[block["id"]]["reason"] == "protected"
    }
    for call_id in untouched:
        assert [block for block in proposed_tools if block["id"] == call_id] == [
            block for block in original_tools if block["id"] == call_id
        ]
    for item in decisions.values():
        if item["probability"] is not None:
            assert 0 <= item["probability"] <= 1
    stats = result["stats"]
    assert result["changed"] and stats["chars_after"] < stats["chars_before"]
    assert stats["requests"] >= 2
    assert stats["usage"], "The live provider should report token usage."
    print(
        json.dumps(
            {
                "case": name,
                "actions": dict(Counter(item["action"] for item in decisions.values())),
                "chars_before": stats["chars_before"],
                "chars_after": stats["chars_after"],
                "saved_percent": round(100 * (1 - stats["chars_after"] / stats["chars_before"]), 1),
                "requests": stats["requests"],
                "usage": stats["usage"],
                "decisions": {
                    key: {field: value[field] for field in ("action", "probability")}
                    for key, value in decisions.items()
                },
            },
            ensure_ascii=False,
        ),
    )


def test_payment_retry_patch_keeps_source_and_previews_completed_ui_task(live_config, tmp_path):
    """Exact next-edit evidence competes with large, completed unrelated work."""
    history = [
        {
            "id": "first",
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "Earlier we finished a storefront theme task. "
                    "Now fix duplicate charges after payment retries. "
                    "Keep approval receipts intact; do not deploy anything.",
                },
            ],
        },
    ]
    _pair(
        history,
        "old_css",
        "read_file",
        {"path": "web/theme/archive.css"},
        _rows("unused-theme"),
        "The storefront theme task was completed and accepted. This CSS has no payment logic.",
    )
    _pair(
        history,
        "old_icons",
        "search_code",
        {"pattern": "logo", "path": "web/icons"},
        _rows("icon-index"),
        "Historical icon search from the finished UI task; unrelated to the current retry defect.",
        split=True,
    )
    _pair(
        history,
        "retry_source",
        "read_file",
        {"path": "src/payments/retry.py"},
        "def retry_charge(payment, gateway):\n    for attempt in range(3):\n"
        "        try:\n            return gateway.charge(payment)\n"
        "        except TimeoutError:\n            continue\n" + _rows("source-context", 25),
        "Current retry source. The next edit must match its exact original lines; no further reads before patching.",
        split=True,
    )
    _pair(
        history,
        "retry_tests",
        "read_file",
        {"path": "tests/payments/test_retry.py"},
        'def test_timeout_reuses_idempotency_key():\n    assert keys == ["payment-42"] * 3\n' + _rows("tests", 30),
        "Current regression-test file. Keep its original lines to place the next patch correctly.",
    )
    _pair(
        history,
        "obsolete_inventory",
        "query_database",
        {"table": "catalog.old_inventory", "month": "2024-01"},
        _rows("old-inventory", 180),
        "Completed inventory task; these obsolete SKU rows cannot explain payment retries.",
    )
    _pair(
        history,
        "billing_receipt",
        "query_database",
        {"table": "billing.events", "payment": "synthetic-42"},
        _rows("billing-evidence", 35),
        "Explicitly pinned billing evidence for the incident report.",
    )
    _pair(
        history,
        "approval",
        "deploy_release",
        {"release": "synthetic-old-release"},
        "approved by fixture operator",
        "Historical approval receipt: a side-effecting tool, never eligible for removal.",
    )
    _pair(
        history,
        "failed_probe",
        "query_database",
        {"table": "unavailable_replica"},
        "replica unreachable",
        "Failed diagnostic attempt; preserve the failure record.",
        result_state="error",
    )
    _pair(
        history,
        "pending_probe",
        "query_database",
        {"table": "billing.pending"},
        "still running",
        "Unfinished work is protected.",
        result_state="running",
    )
    for _ in range(2):
        _pair(
            history,
            "ambiguous",
            "read_file",
            {"path": "unknown-snapshot"},
            "ambiguous duplicate identifier",
            "Duplicate call identifiers are intentionally ineligible.",
        )
    _pair(
        history,
        "recent_manifest",
        "read_file",
        {"path": "pyproject.toml"},
        _rows("manifest", 15),
        "Most recent tool exchange; retain it by the recency rule.",
    )
    goal = (
        "Next action: produce an exact minimal patch to src/payments/retry.py and its regression test, fixing "
        "idempotency-key reuse after timeouts. You must use the verbatim source and test lines already read, "
        "without another file read before the patch. The completed storefront theme and obsolete inventory "
        "tasks are irrelevant. Do not deploy. Preserve pinned billing evidence and all approval receipts."
    )
    history.append({"id": "last", "role": "user", "content": [{"type": "text", "text": goal}]})
    asyncio.run(
        _run_case(
            live_config,
            tmp_path,
            name="payment_retry_patch",
            history=history,
            goal=goal,
            eligible=["read_file", "search_code", "query_database"],
            protected_calls=["billing_receipt"],
            protected_messages=["approval-call"],
            must_keep=["retry_source", "retry_tests"],
            irrelevant=["old_css", "old_icons", "obsolete_inventory"],
        ),
    )


def test_replication_incident_keeps_verbatim_evidence_and_previews_unrelated_reports(live_config, tmp_path):
    """A forensic report needs exact event records amid mixed operational history."""
    history = [
        {
            "id": "first",
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "Investigate replica lag after the synthetic region-A failover. "
                    "Preserve audit evidence and do not roll back.",
                },
            ],
        },
    ]
    _pair(
        history,
        "old_campaign",
        "query_database",
        {"table": "marketing.completed_campaign", "year": 2023},
        _rows("completed-ad-campaign", 180),
        "Finished marketing report; no relation to this database incident.",
    )
    _pair(
        history,
        "old_weather",
        "read_file",
        {"path": "personal/completed-trip-weather.txt"},
        _rows("weather", 100),
        "Historical holiday planning from a completed task; entirely unrelated to database replication.",
    )
    _pair(
        history,
        "old_loadtest",
        "query_metrics",
        {"region": "retired-region-Z", "period": "2021-load-test"},
        _rows("retired-system", 140),
        "Old load test of a decommissioned system, not the production replica.",
    )
    _pair(
        history,
        "incident_logs",
        "query_logs",
        {"region": "region-A", "window": "10:00-10:05", "service": "wal"},
        "10:00:01 event=F001 leader_switch term=17\n10:00:04 event=W002 apply_paused slot=replica-a\n"
        "10:04:59 event=L003 lag_bytes=67108864\n" + _rows("wal-log-context", 50),
        "Current incident log records. The next report must quote the exact event IDs, timestamps and lines; "
        "summaries or later retrieval cannot replace these records.",
        split=True,
    )
    _pair(
        history,
        "wal_offsets",
        "query_database",
        {"table": "replication_slots", "region": "region-A"},
        "slot=replica-a restart_lsn=0/7000000 confirmed_lsn=0/3000000 active=false\n" + _rows("slot-evidence", 40),
        "Current WAL-offset evidence. Keep the exact row for a verbatim evidence table in the next answer.",
    )
    _pair(
        history,
        "old_costs",
        "query_metrics",
        {"dashboard": "finished-2022-advertising-budget"},
        _rows("ad-cost", 120),
        "Completed advertising cost dashboard; unrelated to region-A failover or WAL offsets.",
    )
    _pair(
        history,
        "audit_approval",
        "read_file",
        {"path": "audit/synthetic-change-approval.txt"},
        "ticket=SYN-008 approved_region=region-A action=failover",
        "Pinned audit record; retain the entire message.",
    )
    _pair(
        history,
        "rollback_receipt",
        "rollback_release",
        {"release": "old-synthetic-release"},
        "old rollback receipt",
        "Historical side effect, not an eligible read-only query.",
    )
    _pair(
        history,
        "chart",
        "query_metrics",
        {"chart": "replication"},
        [{"type": "data", "source": {"type": "base64", "media_type": "image/png", "data": "AAAA"}}],
        "Non-text chart output must remain unchanged.",
    )
    _pair(
        history,
        "failed_metrics",
        "query_metrics",
        {"region": "region-A", "metric": "disk"},
        "query failed",
        "Failure record must remain intact.",
        result_state="error",
    )
    _pair(
        history,
        "recent_alert",
        "query_logs",
        {"alert": "synthetic-replica-lag"},
        _rows("recent-alert", 20),
        "Newest exchange remains protected by recency.",
    )
    goal = (
        "Next action: write the final region-A failover root-cause report using the exact incident_logs events "
        "and wal_offsets rows already obtained. Quote every incident event line and the slot row verbatim; "
        "do not perform additional log, database or metric queries before the report. Correlate the leader switch "
        "with paused WAL apply and the offset gap. Ignore completed marketing, holiday, retired-system load-test "
        "and advertising-budget tasks. Preserve the audit approval, chart and failed probes. Do not roll back."
    )
    history.append({"id": "last", "role": "user", "content": [{"type": "text", "text": goal}]})
    asyncio.run(
        _run_case(
            live_config,
            tmp_path,
            name="replication_incident",
            history=history,
            goal=goal,
            eligible=["read_file", "query_database", "query_logs", "query_metrics"],
            protected_calls=[],
            protected_messages=["audit_approval-call"],
            must_keep=["incident_logs", "wal_offsets"],
            irrelevant=["old_campaign", "old_weather", "old_loadtest", "old_costs"],
        ),
    )
