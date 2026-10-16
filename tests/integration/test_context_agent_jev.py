"""Two real-Jev, file-backed assistance cases with a deterministic host harness.

Opt in explicitly: REME_RUN_JEV_INTEGRATION=1 python -m pytest
tests/integration/test_context_agent_jev.py -v -s

Only synthetic memory and task snapshots are sent to Jev. The main-agent plan
is scripted: these cases validate candidate selection, useful evidence and host
delivery, not the success rate of an independent generative main-agent model.
"""

# pylint: disable=redefined-outer-name,missing-function-docstring

import asyncio
import json
import os
from pathlib import Path

import pytest

from reme.application import Application
from reme.components.file_chunker import MarkdownFileChunker
from reme.config import resolve_app_config
from reme.schema import ContextEvent
from reme.utils import load_env

REPOSITORY = Path(__file__).resolve().parents[2]
pytestmark = pytest.mark.skipif(
    os.getenv("REME_RUN_JEV_INTEGRATION") != "1",
    reason="Set REME_RUN_JEV_INTEGRATION=1 to authorize real Jev requests using repository .env",
)

NETWORK_MEMORY = {
    "digest/atlas/prod-egress.md": """# Atlas production Frankfurt egress incident, 2026-09-24
Atlas production Frankfurt uses the environment alias blue-ring, with gateway edge-gw.internal:8443.
After the September mesh migration, HTTPS must use HTTP CONNECT through this gateway.
Keep TLS SNI atlas.service.internal; bypassing the gateway hits an unreachable retired listener.
A valid certificate and passing DNS do not test this route. Next diagnostic: compare direct curl
with curl --proxy http://edge-gw.internal:8443 https://atlas.service.internal/health.
Do not restart production pods to diagnose a missing CONNECT route; inspect egress configuration first.
""",
    "digest/atlas/prod-dns.md": """# Atlas production Frankfurt split DNS, 2026-09-25
For Atlas production blue-ring, atlas.service.internal is resolved inside the CONNECT gateway.
The laptop resolver returns the public retired listener. Do not pin its IP in /etc/hosts or use
curl --resolve for the new mesh route: that bypass was removed in September. Keep the hostname
and SNI intact and check CONNECT routing. This applies to production Frankfurt only.
""",
    "digest/atlas/retired-july-fix.md": """# Atlas production Frankfurt July incident, OBSOLETE
Before the September mesh migration, Atlas production Frankfurt used direct TLS egress.
The July workaround was to disable the proxy and pin atlas.service.internal to 10.7.0.4.
This workaround is obsolete after September and must not be used for the current blue-ring mesh.
""",
    "digest/atlas/staging-route.md": """# Atlas staging Frankfurt route
Atlas staging Frankfurt runs outside blue-ring and still uses direct access to port 9443.
For staging only, a localhost proxy sometimes breaks the route. Unset HTTPS_PROXY in staging.
This is a staging-only recipe and provides no production gateway configuration.
""",
    "digest/atlas/already-known.md": """# Atlas production Frankfurt certificate inventory
The current atlas.service.internal certificate is valid through December; DNS resolves and
certificate-chain validation passes. These inventory facts alone do not diagnose route failure.
""",
    "digest/atlas/unrelated.md": """# Atlas UI navigation preferences
The Atlas dashboard sidebar should use compact labels. This is a frontend styling preference
and has no connection to production Frankfurt network routing or the blue-ring mesh.
""",
}

METRICS_MEMORY = {
    "digest/northstar/finance-contract.md": """# Northstar CFO net revenue contract, effective 2026-09-01
For the Northstar CFO September report, revenue means recognized net revenue, not gross bookings.
Use settled ledger entries in revenue_ledger_v3. Include negative refund adjustments in the same
recognition period, and exclude test merchants and chargeback reversals already offset by a refund.
Do not sum order_events: retries duplicate order IDs, bookings include unsettled payments, and a
refund can arrive in a later month. The legacy order_events contract expired on August 31.
""",
    "digest/northstar/reporting-calendar.md": """# Northstar CFO recognition calendar
Northstar CFO monthly reporting uses Asia/Shanghai business time. September is the half-open interval
2026-09-01 00:00:00+08:00 to 2026-10-01 00:00:00+08:00, applied to recognized_at in revenue_ledger_v3.
Convert these endpoints to UTC before filtering the UTC ledger. The start is 2026-08-31 16:00:00Z,
the end is 2026-09-30 16:00:00Z. Filtering created_at by UTC calendar month misclassifies boundary entries.
""",
    "digest/northstar/legacy-bookings.md": """# Northstar CFO old dashboard definition, expired 2026-08-31
The legacy Northstar dashboard called gross order_events.amount in UTC creation month 'revenue'.
This was a bookings metric with no refund netting. It is expired for CFO reporting from September.
Keep only to reproduce old charts; do not apply to the September recognized net revenue report.
""",
    "digest/northstar/marketing.md": """# Northstar marketing acquisition dashboard
For Northstar marketing campaign ROAS, use gross bookings in order_events with UTC created_at.
Include unsettled purchases to estimate campaign conversion. This is marketing-only and is not
the CFO recognized net revenue measure or the finance month-close calendar.
""",
    "digest/northstar/already-known.md": """# Northstar warehouse inventory
The warehouse uses PostgreSQL. revenue_ledger_v3 has recognized_at, amount and merchant_id columns.
order_events has created_at, order_id and amount columns. These are schema facts, not metric definitions.
""",
    "digest/northstar/unrelated.md": """# Northstar presentation preference
Northstar slides use navy headings and one decimal place. This affects formatting, not revenue
recognition, refunds, settlement, or business timezone filters for September.
""",
}


def make_app(workspace):
    load_env(REPOSITORY / ".env", override=False)
    assert os.getenv("CLASSIFIER_API_KEY"), "CLASSIFIER_API_KEY must be set in .env or environment"
    config = resolve_app_config(config="context_agent", log_config=False)
    # Real local store + BM25 and real Jev; no remote embedding/LLM or background watchers.
    keep = {"tokenizer", "keyword_index", "file_graph", "file_store", "as_classifier", "context_agent"}
    config["components"] = {key: value for key, value in config["components"].items() if key in keep}
    config["components"]["file_store"]["default"]["tag_index"] = ""
    config["components"]["as_classifier"]["context_agent"].update(timeout=20, max_retries=0)
    config["components"]["context_agent"]["default"].update(
        options={
            "total_timeout_ms": 20000,
            "max_request_tokens": 16000,
            "max_candidates": 8,
            "max_hint_candidates": 2,
            "vector_weight": 0,
        },
        limits={"enabled": True, "mode": "hint", "cooldown_seconds": 0},
    )
    config["jobs"] = {"evaluate_context": config["jobs"]["evaluate_context"]}
    config.update(
        service={"backend": "cli"},
        workspace_dir=str(workspace),
        enable_logo=False,
        log_to_console=False,
        log_to_file=False,
    )
    return Application(**config)


async def seed_memories(app, memories):
    store = app.context.components["file_store"]["default"]
    chunker = MarkdownFileChunker(app_context=app.context)
    original = {}
    for relative, text in memories.items():
        path = Path(app.config.workspace_dir) / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        original[relative] = path.read_bytes()
        node, chunks = await chunker.chunk(path)
        await store.upsert([(node, chunks)])
    return original


async def receive_hint(app, event):
    agent = app.context.components["context_agent"]["default"]
    assert await agent.submit(event, allowed_paths=["digest"])
    # A production host does real tool work here. Offline replay waits explicitly
    # so this test remains independent of machine speed and network latency.
    await agent.wait_idle(event.session_id)
    result = await agent.last_result(event.session_id)
    assert result is not None
    assert result.stats.get("classifier_calls") == 1, result.reason_code
    hint = await agent.take_ready(event.session_id, event.task_id, event.revision, allowed_paths=["digest"])
    assert hint is not None, json.dumps(result.model_dump(), ensure_ascii=False)
    assert await agent.ack(event.session_id, hint.hint_id)
    return agent, result, hint


def assert_sources_unchanged(app, original):
    for relative, data in original.items():
        assert (Path(app.config.workspace_dir) / relative).read_bytes() == data


def report(case, result, hint, before, after):
    print(
        json.dumps(
            {
                "case": case,
                "scripted_plan_before": before,
                "scripted_plan_after": after,
                "evidence_paths": [c.path for c in hint.evidence],
                "stats": result.stats,
                "scope": "Real Jev selection and delivery; scripted host, not a main-model success benchmark",
            },
            ensure_ascii=False,
            indent=2,
        ),
    )


def test_jev_helps_production_incident_with_obsolete_and_staging_decoys(tmp_path):
    """A hidden mesh migration prevents repeating an obsolete production workaround."""

    async def run():
        async with make_app(tmp_path) as app:
            original = await seed_memories(app, NETWORK_MEMORY)
            event = ContextEvent(
                session_id="incident-session",
                task_id="restore-atlas",
                event_id="network-tool-1",
                revision=1,
                event_type="tool_start",
                task_snapshot={
                    "goal": "Diagnose Atlas production Frankfurt HTTPS timeouts after mesh migration without downtime.",
                    "recent_context": (
                        "The agent knows the certificate is valid through December, DNS resolves, and certificate "
                        "validation passes. It has not searched memory and does not know the production mesh route. "
                        "Its planned next step is direct curl and, if that fails, pinning the DNS address. "
                        "Current environment is production Frankfurt, not staging. Do not restart production pods."
                    ),
                    "current_tool": "curl -v --connect-timeout 5 https://atlas.service.internal/health",
                    "query_terms": ["Atlas production Frankfurt", "Atlas mesh migration"],
                },
            )
            agent, result, hint = await receive_hint(app, event)
            paths = {item.path for item in hint.evidence}
            assert "digest/atlas/prod-egress.md" in paths
            assert paths <= {"digest/atlas/prod-egress.md", "digest/atlas/prod-dns.md"}
            assert "edge-gw.internal:8443" in hint.text and "CONNECT" in hint.text
            before = "Try direct curl, then pin the local resolver's IP if the timeout persists."
            after = (
                "Compare HTTP CONNECT via edge-gw.internal:8443, preserving hostname/SNI; inspect route before restart."
            )
            report("production_network_incident", result, hint, before, after)

            # Once the useful facts are known, another event must not nag the agent.
            followup = event.model_copy(deep=True)
            followup.revision = 2
            followup.event_id = "network-tool-2"
            followup.event_type = "tool_result"
            followup.task_snapshot.recent_context += (
                " All current production egress and split-DNS facts in memory are now known: use HTTP CONNECT "
                "via edge-gw.internal:8443, preserve hostname and TLS SNI, do not pin IPs or restart pods. "
                "The gateway test succeeds. The incident is resolved; prepare a summary."
            )
            followup.task_snapshot.tool_result_excerpt = (
                "CONNECT test succeeded; production health endpoint returns 200."
            )
            assert await agent.submit(followup)
            await agent.wait_idle(followup.session_id)
            assert (await agent.last_result(followup.session_id)).action == "silent"
            assert await agent.take_ready(followup.session_id, followup.task_id, 2) is None
            assert_sources_unchanged(app, original)

    asyncio.run(run())


def test_jev_helps_finance_analysis_with_contract_and_timezone_traps(tmp_path):
    """Two environmental facts fix an apparently plausible but wrong finance query."""

    async def run():
        async with make_app(tmp_path) as app:
            original = await seed_memories(app, METRICS_MEMORY)
            event = ContextEvent(
                session_id="finance-session",
                task_id="september-close",
                event_id="schema-tool-1",
                revision=1,
                event_type="tool_start",
                task_snapshot={
                    "goal": (
                        "Prepare the Northstar CFO September 2026 recognized net revenue report and explain refunds."
                    ),
                    "recent_context": (
                        "The main agent knows PostgreSQL schemas: revenue_ledger_v3 has recognized_at, amount, "
                        "merchant_id; order_events has created_at, order_id, amount. It does not know the finance "
                        "metric contract or business calendar. The draft query sums order_events.amount where "
                        "created_at is in September UTC. This is the CFO close, not marketing ROAS."
                    ),
                    "current_tool": "inspect_schema(revenue_ledger_v3, order_events)",
                    "query_terms": ["Northstar CFO September revenue", "Northstar CFO recognition calendar"],
                },
            )
            _agent, result, hint = await receive_hint(app, event)
            paths = {item.path for item in hint.evidence}
            assert paths == {"digest/northstar/finance-contract.md", "digest/northstar/reporting-calendar.md"}
            assert "refund" in hint.text and "recognized_at" in hint.text
            assert "2026-08-31 16:00:00Z" in hint.text and "2026-09-30 16:00:00Z" in hint.text

            # Synthetic ledger: +100 just before the Shanghai month, +300 within,
            # -50 September refund, +200 after September. The order-event sample
            # includes bookings not recognized in this finance period.
            rows = [
                ("2026-08-31T15:59:00Z", 100),
                ("2026-08-31T16:01:00Z", 300),
                ("2026-09-15T12:00:00Z", -50),
                ("2026-09-30T16:01:00Z", 200),
            ]
            recognized_net = sum(
                amount for stamp, amount in rows if "2026-08-31T16:00:00Z" <= stamp < "2026-09-30T16:00:00Z"
            )
            orders = [
                ("2026-09-01T08:00:00Z", 100),
                ("2026-09-02T08:00:00Z", 300),
                ("2026-09-30T18:00:00Z", 200),
            ]
            gross_bookings = sum(amount for stamp, amount in orders if "2026-09-01" <= stamp < "2026-10-01")
            assert gross_bookings == 600
            assert recognized_net == 250
            before = "SUM(order_events.amount), UTC created_at month, no refund netting: synthetic bookings = 600."
            after = "Use settled ledger, recognized_at, Shanghai boundaries and refund netting: synthetic net = 250."
            report("finance_metric_and_timezone", result, hint, before, after)
            assert_sources_unchanged(app, original)

    asyncio.run(run())
