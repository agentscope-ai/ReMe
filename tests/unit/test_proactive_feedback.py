"""Unit tests for proactive delivery feedback (host -> ReMe reporting).

Covers the ``deliveries`` truth-source section, the ``proactive_feedback``
step, the reader-side withholding in ``proactive_step`` and the write-side
suppression in ``proactive_agenda_step``. Feedback is optional by contract, so
every test that asserts new behaviour is paired with one asserting that an
empty ``deliveries`` section changes nothing.
"""

import asyncio
import datetime as dt
from unittest.mock import patch

import yaml

from reme.components.application_context import ApplicationContext
from reme.components.runtime_context import RuntimeContext
from reme.schema import ProactiveDelivery, ProactiveStateFile
from reme.schema.proactive import MAX_DELIVERY_OUTCOME_CHARS
from reme.steps.evolve.proactive.feedback import ProactiveFeedbackStep
from reme.steps.evolve.proactive.proactive import ProactiveStep
from reme.steps.evolve.proactive.utils import (
    is_delivery_suppressed,
    load_state,
    suppressed_delivery_ids,
    topic_id,
    trim_deliveries,
    upsert_deliveries,
)

from .test_proactive_refresh import (
    DAY,
    EXTRACT_CLOCK,
    _FileStore,
    _interests,
    _reply,
    _reminder,
    _run_full_chain,
    _state_topic,
    _topic,
    _touch,
    _write_state,
)

NOW = dt.datetime(2026, 8, 13, 18, 0, tzinfo=dt.timezone(dt.timedelta(hours=8)))


def _write_state_with_deliveries(ws, deliveries, open_topics=None):
    """Seed the truth source with open topics plus a delivery log."""
    data = {
        "version": 1,
        "open_topics": open_topics or [],
        "resolved": [],
        "schedules": [],
        "deliveries": deliveries,
    }
    _touch(ws / "daily" / "_proactive.yaml", yaml.safe_dump(data, allow_unicode=True, sort_keys=False))


def _write_interests_with_agenda(ws, topics, agenda, day=DAY, push=True):
    """Write a v2 exposure file carrying an agenda, as the agenda step would."""
    payload = {
        "version": 2,
        "date": day,
        "generated_at": f"{day}T18:00:00+08:00",
        "push": push,
        "topics": topics,
        "agenda": agenda,
        "suppressed": [],
    }
    _touch(_interests(ws, day), yaml.safe_dump(payload, allow_unicode=True, sort_keys=False))


def _agenda_item(topic_id_value, title, kind="follow_up", **extra):
    item = {
        "topic_id": topic_id_value,
        "title": title,
        "kind": kind,
        "scenario_type": "resume_task",
        "opener": f"对了，{title}还差一步。",
        "next_action": "跑一下下一步",
        "preconditions": [],
        "delivery": "in_conversation",
        "linked_memory": [],
        "order_reason": "seeded",
    }
    item.update(extra)
    return item


# ---------------------------------------------------------------------------
# Schema and pure helpers
# ---------------------------------------------------------------------------


def test_delivery_model_fallbacks():
    """Illegal status/channel fall back; outcome is capped; identity is stable."""
    record = ProactiveDelivery(topic_id="t", status="BOGUS", channel="Inbox", outcome="x" * 900, external_ref="e1")
    assert record.status == "shown"
    assert record.channel == "inbox"
    assert len(record.outcome) == MAX_DELIVERY_OUTCOME_CHARS
    assert record.identity() == ("t", "ref:e1")

    bare = ProactiveDelivery(topic_id="t", agenda_date=DAY, channel="notification")
    assert bare.identity() == ("t", f"day:{DAY}|notification")
    assert ProactiveDelivery(topic_id="t", channel="weird").channel == "other"


def test_upsert_deliveries_is_idempotent():
    """Re-reporting the same delivery updates in place instead of appending."""
    state_file = ProactiveStateFile()
    items = [
        {"topic_id": "a", "status": "shown", "channel": "inbox", "external_ref": "ev1"},
        {"topic_id": "b", "status": "executed", "channel": "in_conversation", "outcome": "done"},
        {"topic_id": "", "status": "shown"},
        "garbage",
    ]
    assert upsert_deliveries(state_file, items, DAY, NOW) == {"added": 2, "updated": 0, "dropped": 2}
    assert upsert_deliveries(state_file, items, DAY, NOW) == {"added": 0, "updated": 2, "dropped": 2}
    assert len(state_file.deliveries) == 2


def test_upsert_deliveries_never_downgrades_terminal_status():
    """A late ``shown`` report cannot undo an ``executed`` one."""
    state_file = ProactiveStateFile()
    upsert_deliveries(state_file, [{"topic_id": "a", "status": "executed", "channel": "inbox"}], DAY, NOW)
    counts = upsert_deliveries(state_file, [{"topic_id": "a", "status": "shown", "channel": "inbox"}], DAY, NOW)
    assert counts == {"added": 0, "updated": 0, "dropped": 1}
    assert state_file.deliveries[0].status == "executed"


def test_upsert_deliveries_defaults_missing_clock_fields():
    """A minimal report still gets an agenda_date and a delivered_at."""
    state_file = ProactiveStateFile()
    upsert_deliveries(state_file, [{"topic_id": "a"}], DAY, NOW)
    record = state_file.deliveries[0]
    assert record.agenda_date == DAY
    assert record.delivered_at.startswith(DAY)
    assert record.status == "shown" and record.channel == "other"


def test_is_delivery_suppressed_window():
    """``failed`` never blocks; other statuses block inside the cooldown only."""

    def record(status, delivered_at):
        return ProactiveDelivery(topic_id="a", status=status, delivered_at=delivered_at)

    assert is_delivery_suppressed(None, DAY, 3) is False
    assert is_delivery_suppressed(record("failed", f"{DAY}T18:00:00+08:00"), DAY, 3) is False
    assert is_delivery_suppressed(record("shown", f"{DAY}T18:00:00+08:00"), DAY, 3) is True
    assert is_delivery_suppressed(record("executed", "2026-08-10T18:00:00+08:00"), DAY, 3) is True
    assert is_delivery_suppressed(record("shown", "2026-08-09T18:00:00+08:00"), DAY, 3) is False
    assert is_delivery_suppressed(record("shown", "2026-08-09T18:00:00+08:00"), DAY, 0) is False
    assert is_delivery_suppressed(record("shown", "not-a-date"), DAY, 3) is False


def test_trim_deliveries_by_age_then_size():
    """Old records go first, then the newest ``max_deliveries`` survive."""
    state_file = ProactiveStateFile()
    for index, day in enumerate(["2026-01-01", "2026-08-11", "2026-08-12", "2026-08-13"]):
        state_file.deliveries.append(
            ProactiveDelivery(topic_id=f"t{index}", delivered_at=f"{day}T18:00:00+08:00", external_ref=f"e{index}"),
        )
    assert trim_deliveries(state_file, DAY, 180, 2) == 2
    assert [r.topic_id for r in state_file.deliveries] == ["t3", "t2"]
    # retention 1 day keeps 08-12 and 08-13; retention 0 keeps only today.
    assert trim_deliveries(state_file, DAY, 1, 10) == 0
    assert trim_deliveries(state_file, DAY, 0, 10) == 1
    assert [r.topic_id for r in state_file.deliveries] == ["t3"]


def test_suppressed_delivery_ids_prefers_the_blocking_record():
    """The newest blocking record wins; a failed run does not mask it."""
    state_file = ProactiveStateFile()
    upsert_deliveries(
        state_file,
        [
            {"topic_id": "a", "status": "failed", "channel": "inbox", "external_ref": "e1"},
            {"topic_id": "a", "status": "shown", "channel": "notification", "external_ref": "e2"},
            {"topic_id": "b", "status": "failed", "channel": "inbox", "external_ref": "e3"},
        ],
        DAY,
        NOW,
    )
    blocked = suppressed_delivery_ids(state_file, DAY, 3)
    assert set(blocked) == {"a"}
    assert blocked["a"].status == "shown"


# ---------------------------------------------------------------------------
# proactive_feedback_step
# ---------------------------------------------------------------------------


def test_feedback_step_records_and_persists(tmp_path):
    """One report lands in the truth source and is summarized in the response."""

    async def run():
        ws = tmp_path
        _write_state(ws, open_topics=[_state_topic("Topic A")])
        response = await _async_feedback(
            ws,
            [
                {
                    "topic_id": "a1",
                    "status": "executed",
                    "channel": "in_conversation",
                    "outcome": "已完成",
                    "external_ref": "s1",
                },
                {"topic_id": "a2", "status": "shown", "channel": "inbox", "external_ref": "ev1"},
            ],
        )
        assert response.success is True
        assert "added 2" in response.answer
        payload = response.metadata["proactive_feedback"]
        assert payload["added"] == 2 and payload["updated"] == 0 and payload["total"] == 2
        assert payload["suppressed_topics"] == ["a1", "a2"]

        state_file, _ = load_state(ws)
        assert {r.topic_id: r.status for r in state_file.deliveries} == {"a1": "executed", "a2": "shown"}
        # The step must not disturb the other truth-source sections.
        assert [t.title for t in state_file.open_topics] == ["Topic A"]
        assert state_file.resolved == [] and state_file.schedules == []

    asyncio.run(run())


async def _async_feedback(ws, items, day=DAY, **kwargs):
    step = ProactiveFeedbackStep(app_context=ApplicationContext(workspace_dir=str(ws)), **kwargs)
    return await step(RuntimeContext(date=day, items=items, file_store=_FileStore(ws)))


def test_feedback_step_creates_missing_truth_source(tmp_path):
    """Reporting into a fresh workspace bootstraps an otherwise empty state file."""

    async def run():
        response = await _async_feedback(tmp_path, [{"topic_id": "a1", "status": "shown"}])
        assert response.success is True
        state_file, _ = load_state(tmp_path)
        assert [r.topic_id for r in state_file.deliveries] == ["a1"]
        assert state_file.open_topics == []

    asyncio.run(run())


def test_feedback_step_without_items_is_a_noop(tmp_path):
    """An empty report succeeds, writes nothing and says so."""

    async def run():
        _write_state(tmp_path, open_topics=[_state_topic("Topic A")])
        before = (tmp_path / "daily" / "_proactive.yaml").read_text(encoding="utf-8")
        response = await _async_feedback(tmp_path, [])
        assert response.success is True
        assert response.answer == "Feedback: no delivery items reported"
        assert response.metadata["proactive_feedback"]["received"] == 0
        assert (tmp_path / "daily" / "_proactive.yaml").read_text(encoding="utf-8") == before

    asyncio.run(run())


def test_feedback_step_is_idempotent_across_runs(tmp_path):
    """Retrying the same report does not duplicate records."""

    async def run():
        _write_state(tmp_path)
        items = [{"topic_id": "a1", "status": "shown", "channel": "inbox", "external_ref": "ev1"}]
        await _async_feedback(tmp_path, items)
        response = await _async_feedback(tmp_path, items)
        assert "updated 1" in response.answer
        state_file, _ = load_state(tmp_path)
        assert len(state_file.deliveries) == 1

    asyncio.run(run())


def test_feedback_step_trims_by_retention(tmp_path):
    """Records older than the retention window are pruned on write."""

    async def run():
        ws = tmp_path
        _write_state_with_deliveries(
            ws,
            [{"topic_id": "old", "delivered_at": "2026-01-01T18:00:00+08:00", "status": "shown", "external_ref": "e"}],
        )
        response = await _async_feedback(ws, [{"topic_id": "new", "status": "shown"}], retention_days=30)
        assert response.metadata["proactive_feedback"]["trimmed"] == 1
        state_file, _ = load_state(ws)
        assert [r.topic_id for r in state_file.deliveries] == ["new"]

    asyncio.run(run())


# ---------------------------------------------------------------------------
# proactive_step (reader)
# ---------------------------------------------------------------------------


def _seed_reader(ws, fid, eid):
    """Two open topics, both on today's agenda; ``eid`` already delivered."""
    follow = _state_topic("记忆检索回归", topic_id_value=fid)
    extend = _state_topic("多跳 RAG 综述", topic_id_value=eid, kind="interest_extend")
    _write_state_with_deliveries(
        ws,
        [
            {
                "topic_id": eid,
                "agenda_date": DAY,
                "delivered_at": f"{DAY}T09:00:00+08:00",
                "channel": "inbox",
                "status": "shown",
                "external_ref": "ev1",
            },
        ],
        open_topics=[follow, extend],
    )
    _write_interests_with_agenda(
        ws,
        [follow, extend],
        [_agenda_item(fid, "记忆检索回归"), _agenda_item(eid, "多跳 RAG 综述", kind="interest_extend")],
    )


def test_reader_withholds_delivered_agenda_items(tmp_path):
    """A delivered agenda item disappears from the read but stays auditable."""

    async def run():
        ws = tmp_path
        fid, eid = topic_id("记忆检索回归"), topic_id("多跳 RAG 综述")
        _seed_reader(ws, fid, eid)
        step = ProactiveStep(app_context=ApplicationContext(workspace_dir=str(ws)))
        response = await step(RuntimeContext(date=DAY, include_content=False, file_store=_FileStore(ws)))

        assert response.success is True
        assert [i["topic_id"] for i in response.answer["agenda"]] == [fid]
        assert response.answer["delivered_ids"] == [eid]
        assert "withheld 1" in response.answer["summary"]
        # Topics are material, not delivery units: they stay unfiltered.
        assert {t["id"] for t in response.answer["topics"]} == {fid, eid}
        assert response.metadata["delivered_ids"] == [eid]

    asyncio.run(run())


def test_reader_include_delivered_returns_everything(tmp_path):
    """``include_delivered`` is the escape hatch for hosts doing their own dedup."""

    async def run():
        ws = tmp_path
        fid, eid = topic_id("记忆检索回归"), topic_id("多跳 RAG 综述")
        _seed_reader(ws, fid, eid)
        step = ProactiveStep(app_context=ApplicationContext(workspace_dir=str(ws)))
        response = await step(
            RuntimeContext(date=DAY, include_content=False, include_delivered=True, file_store=_FileStore(ws)),
        )
        assert [i["topic_id"] for i in response.answer["agenda"]] == [fid, eid]
        assert "delivered_ids" not in response.answer

    asyncio.run(run())


def test_reader_cooldown_expiry_lets_an_item_back(tmp_path):
    """A merely-shown item returns once the cooldown has passed."""

    async def run():
        ws = tmp_path
        fid, eid = topic_id("记忆检索回归"), topic_id("多跳 RAG 综述")
        follow = _state_topic("记忆检索回归", topic_id_value=fid)
        extend = _state_topic("多跳 RAG 综述", topic_id_value=eid, kind="interest_extend")
        # The single-day exposure is read at DAY, so the delivery is backdated
        # instead: shown a week ago, default cooldown is three days.
        _write_state_with_deliveries(
            ws,
            [
                {
                    "topic_id": eid,
                    "agenda_date": "2026-08-06",
                    "delivered_at": "2026-08-06T09:00:00+08:00",
                    "channel": "inbox",
                    "status": "shown",
                },
            ],
            open_topics=[follow, extend],
        )
        _write_interests_with_agenda(
            ws,
            [follow, extend],
            [_agenda_item(fid, "记忆检索回归"), _agenda_item(eid, "多跳 RAG 综述", kind="interest_extend")],
        )
        step = ProactiveStep(app_context=ApplicationContext(workspace_dir=str(ws)))
        response = await step(RuntimeContext(date=DAY, include_content=False, file_store=_FileStore(ws)))
        assert [i["topic_id"] for i in response.answer["agenda"]] == [fid, eid]
        assert "delivered_ids" not in response.answer

    asyncio.run(run())


def test_reader_without_feedback_is_unchanged(tmp_path):
    """No ``deliveries`` section means no withholding (backwards compatible)."""

    async def run():
        ws = tmp_path
        fid, eid = topic_id("记忆检索回归"), topic_id("多跳 RAG 综述")
        follow = _state_topic("记忆检索回归", topic_id_value=fid)
        extend = _state_topic("多跳 RAG 综述", topic_id_value=eid, kind="interest_extend")
        _write_state(ws, open_topics=[follow, extend])
        _write_interests_with_agenda(
            ws,
            [follow, extend],
            [_agenda_item(fid, "记忆检索回归"), _agenda_item(eid, "多跳 RAG 综述", kind="interest_extend")],
        )
        step = ProactiveStep(app_context=ApplicationContext(workspace_dir=str(ws)))
        response = await step(RuntimeContext(date=DAY, include_content=False, file_store=_FileStore(ws)))
        assert [i["topic_id"] for i in response.answer["agenda"]] == [fid, eid]
        assert "delivered_ids" not in response.answer
        assert "withheld" not in response.answer["summary"]

    asyncio.run(run())


def test_reader_reminder_still_outranks_and_survives(tmp_path):
    """Withholding a discovery item never reorders or drops a pinned reminder."""

    async def run():
        ws = tmp_path
        rid, eid = topic_id("牙医预约"), topic_id("多跳 RAG 综述")
        reminder = _state_topic("牙医预约", topic_id_value=rid, kind="reminder", confidence=0.9)
        extend = _state_topic("多跳 RAG 综述", topic_id_value=eid, kind="interest_extend")
        _write_state_with_deliveries(
            ws,
            [
                {
                    "topic_id": eid,
                    "agenda_date": DAY,
                    "delivered_at": f"{DAY}T09:00:00+08:00",
                    "channel": "inbox",
                    "status": "shown",
                },
            ],
            open_topics=[reminder, extend],
        )
        _write_interests_with_agenda(
            ws,
            [reminder, extend],
            [
                _agenda_item(rid, "牙医预约", kind="reminder", scenario_type="prepare_upcoming"),
                _agenda_item(eid, "多跳 RAG 综述", kind="interest_extend"),
            ],
        )
        step = ProactiveStep(app_context=ApplicationContext(workspace_dir=str(ws)))
        response = await step(RuntimeContext(date=DAY, include_content=False, file_store=_FileStore(ws)))
        assert [i["topic_id"] for i in response.answer["agenda"]] == [rid]
        assert response.answer["agenda"][0]["kind"] == "reminder"

    asyncio.run(run())


def test_reader_reminder_is_exempt_from_the_cooldown(tmp_path):
    """A reminder delivered inside the window is served again: due means due.

    Reminder ids derive from a stable schedule title, so withholding them like
    discovery items would silence a recurring reminder for the whole cooldown
    after a single delivery.
    """

    async def run():
        ws = tmp_path
        rid = topic_id("牙医预约")
        reminder = _state_topic("牙医预约", topic_id_value=rid, kind="reminder", confidence=0.9)
        _write_state_with_deliveries(
            ws,
            [
                {
                    "topic_id": rid,
                    "agenda_date": DAY,
                    "delivered_at": f"{DAY}T09:00:00+08:00",
                    "channel": "inbox",
                    "status": "shown",
                },
            ],
            open_topics=[reminder],
        )
        _write_interests_with_agenda(
            ws,
            [reminder],
            [_agenda_item(rid, "牙医预约", kind="reminder", scenario_type="prepare_upcoming")],
        )
        step = ProactiveStep(app_context=ApplicationContext(workspace_dir=str(ws)))
        response = await step(RuntimeContext(date=DAY, include_content=False, file_store=_FileStore(ws)))
        assert [i["topic_id"] for i in response.answer["agenda"]] == [rid]
        assert response.answer["agenda"][0]["kind"] == "reminder"
        assert "delivered_ids" not in response.answer
        assert "withheld" not in response.answer["summary"]

    asyncio.run(run())


# ---------------------------------------------------------------------------
# proactive_agenda_step (writer)
# ---------------------------------------------------------------------------


def test_agenda_step_suppresses_delivered_candidate(tmp_path):
    """A reported candidate leaves the agenda with an auditable reason."""

    async def run():
        ws = tmp_path
        _touch(ws / "daily" / DAY / "session.md", "we discussed the eval plan")
        _write_state_with_deliveries(
            ws,
            [
                {
                    "topic_id": topic_id("任务一"),
                    "agenda_date": DAY,
                    "delivered_at": f"{DAY}T09:00:00+08:00",
                    "channel": "in_conversation",
                    "status": "executed",
                    "external_ref": "s1",
                },
            ],
        )
        extract_reply = _reply(
            follow_ups=[
                _topic("任务一", confidence=0.9, paths=[f"daily/{DAY}/session.md"]),
                _topic("任务二", confidence=0.8, paths=[f"daily/{DAY}/session.md"]),
            ],
        )
        _, _, _, context, resps = await _run_full_chain(ws, [extract_reply])
        assert all(r.success for r in resps)

        state = context.get("proactive")
        assert state["suppressed_delivered"] == 1
        data = yaml.safe_load(_interests(ws).read_text(encoding="utf-8"))
        assert [i["title"] for i in data["agenda"]] == ["任务二"]
        suppressed = {s["topic_id"]: s["reason"] for s in data["suppressed"]}
        reason = suppressed[topic_id("任务一")]
        assert reason.startswith("already delivered on")
        assert "in_conversation" in reason and "executed" in reason

    asyncio.run(run())


def test_agenda_step_reminder_is_exempt_from_the_cooldown(tmp_path):
    """A reported reminder is pinned again while a reported discovery item is not."""

    async def run():
        ws = tmp_path
        _touch(ws / "daily" / DAY / "session.md", "牙医复诊在即，评审材料还差一版")
        path = f"daily/{DAY}/session.md"
        rid = topic_id("牙医复诊")
        _write_state_with_deliveries(
            ws,
            [
                {
                    "topic_id": rid,
                    "agenda_date": DAY,
                    "delivered_at": f"{DAY}T09:00:00+08:00",
                    "channel": "inbox",
                    "status": "shown",
                },
                {
                    "topic_id": topic_id("任务一"),
                    "agenda_date": DAY,
                    "delivered_at": f"{DAY}T09:00:00+08:00",
                    "channel": "inbox",
                    "status": "shown",
                },
            ],
        )
        extract_reply = _reply(
            reminders=[_reminder("牙医复诊", due_at="2026-08-14T15:00", paths=[path])],
            follow_ups=[
                _topic("任务一", confidence=0.9, paths=[path]),
                _topic("任务二", confidence=0.8, paths=[path]),
            ],
        )
        with patch(EXTRACT_CLOCK, return_value=NOW):
            _, _, _, context, resps = await _run_full_chain(ws, [extract_reply])
        assert all(r.success for r in resps)

        # only the discovery candidate counts as suppressed by the cooldown
        assert context.get("proactive")["suppressed_delivered"] == 1
        data = yaml.safe_load(_interests(ws).read_text(encoding="utf-8"))
        titles = [i["title"] for i in data["agenda"]]
        assert "牙医复诊" in titles
        assert "任务一" not in titles
        assert data["agenda"][0]["kind"] == "reminder"  # still pinned first
        suppressed = {s["topic_id"] for s in data["suppressed"]}
        assert rid not in suppressed

    asyncio.run(run())


def test_agenda_step_without_feedback_is_unchanged(tmp_path):
    """No delivery log means the deterministic fallback agenda stays intact."""

    async def run():
        ws = tmp_path
        _touch(ws / "daily" / DAY / "session.md", "material")
        extract_reply = _reply(
            follow_ups=[
                _topic("任务一", confidence=0.9, paths=[f"daily/{DAY}/session.md"]),
                _topic("任务二", confidence=0.7, paths=[f"daily/{DAY}/session.md"]),
            ],
        )
        _, _, _, context, resps = await _run_full_chain(ws, [extract_reply])
        assert all(r.success for r in resps)
        assert context.get("proactive")["suppressed_delivered"] == 0
        data = yaml.safe_load(_interests(ws).read_text(encoding="utf-8"))
        assert [i["title"] for i in data["agenda"]] == ["任务一", "任务二"]
        assert data["suppressed"] == []

    asyncio.run(run())


def test_agenda_step_cooldown_zero_only_blocks_same_day(tmp_path):
    """``redelivery_cooldown_days=0`` still blocks a delivery reported today."""

    async def run():
        ws = tmp_path
        _touch(ws / "daily" / DAY / "session.md", "material")
        _write_state_with_deliveries(
            ws,
            [
                {
                    "topic_id": topic_id("任务一"),
                    "agenda_date": DAY,
                    "delivered_at": f"{DAY}T09:00:00+08:00",
                    "channel": "inbox",
                    "status": "shown",
                },
            ],
        )
        extract_reply = _reply(follow_ups=[_topic("任务一", confidence=0.9, paths=[f"daily/{DAY}/session.md"])])
        _, _, _, context, resps = await _run_full_chain(
            ws,
            [extract_reply],
            agenda_kwargs={"redelivery_cooldown_days": 0},
        )
        assert all(r.success for r in resps)
        assert context.get("proactive")["suppressed_delivered"] == 1
        data = yaml.safe_load(_interests(ws).read_text(encoding="utf-8"))
        assert data["agenda"] == []
        assert data["push"] is False

    asyncio.run(run())


# ---------------------------------------------------------------------------
# Host interop: the report shape a consumer actually sends
# ---------------------------------------------------------------------------


def test_consumer_report_shape_round_trips(tmp_path):
    """A host report in the wire shape the consumer sends is stored verbatim and withholds.

    Locks the interop contract on three points that are easy to break silently:
    every field of the report survives (an unknown ``consumer`` label must not be
    dropped, a known ``channel`` must not be coerced to ``other``), a UTC
    ``delivered_at`` still suppresses on a later local business day, and the
    reader publishes ``delivered_ids`` in metadata for the host to count.
    """

    async def run():
        ws = tmp_path
        fid, eid = topic_id("记忆检索回归"), topic_id("多跳 RAG 综述")
        follow = _state_topic("记忆检索回归", topic_id_value=fid)
        extend = _state_topic("多跳 RAG 综述", topic_id_value=eid, kind="interest_extend")
        _write_state(ws, open_topics=[follow, extend])
        _write_interests_with_agenda(
            ws,
            [follow, extend],
            [_agenda_item(fid, "记忆检索回归"), _agenda_item(eid, "多跳 RAG 综述", kind="interest_extend")],
        )

        # 23:30Z on the previous UTC day is 07:30 on DAY in +08:00: the consumer
        # stamps UTC, the business clock is local, so the date prefix lags.
        report = {
            "topic_id": eid,
            "agenda_date": DAY,
            "delivered_at": "2026-08-12T23:30:00.123456+00:00",
            "channel": "in_conversation",
            "consumer": "qwenpaw",
            "status": "shown",
            "outcome": "",
            "external_ref": "session-7f3a",
        }
        response = await _async_feedback(ws, [report])
        assert response.success is True and "added 1" in response.answer

        state_file, _ = load_state(ws)
        assert len(state_file.deliveries) == 1
        stored = state_file.deliveries[0]
        assert stored.consumer == "qwenpaw"  # a free-form host label is kept as sent
        assert stored.channel == "in_conversation"  # in the vocabulary, so not coerced
        assert stored.external_ref == "session-7f3a"
        assert stored.agenda_date == DAY
        assert stored.delivered_at.startswith("2026-08-12T23:30:00")
        assert is_delivery_suppressed(stored, DAY, 3) is True

        # Re-reporting the same delivery updates in place instead of piling up.
        again = await _async_feedback(ws, [dict(report, status="executed")])
        assert again.success is True
        state_file, _ = load_state(ws)
        assert len(state_file.deliveries) == 1
        assert state_file.deliveries[0].status == "executed"

        reader = ProactiveStep(app_context=ApplicationContext(workspace_dir=str(ws)))
        read = await reader(RuntimeContext(date=DAY, include_content=False, file_store=_FileStore(ws)))
        assert read.success is True
        assert [item["topic_id"] for item in read.answer["agenda"]] == [fid]
        assert read.metadata["delivered_ids"] == [eid]

    asyncio.run(run())
