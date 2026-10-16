# pylint: disable=too-many-lines
"""Unit tests for the proactive refresh chain (PROACTIVE_SPEC.md A8 mapping).

Coverage: F1 contracts, F2 discovery, F3 isolation, F4 budget/timeout,
F5 read extensions, plus the M2 extends branch and semantic dedup.
"""

import asyncio
import datetime as dt
import os
import tempfile
import time
from pathlib import Path
from unittest.mock import patch

import yaml
from agentscope.model import ChatModelBase

from reme.components.agent_wrapper.base_agent_wrapper import BaseAgentWrapper
from reme.components.application_context import ApplicationContext
from reme.components.file_catalog import BaseFileCatalog
from reme.components.file_store import BaseFileStore
from reme.components.runtime_context import RuntimeContext
from reme.schema import FileNode, ProactiveSchedule, ProactiveStateFile, ProactiveTopic
from reme.steps.evolve.dream.extract import DreamExtractStep
from reme.steps.evolve.dream.utils import pack_paths
from reme.steps.evolve.proactive.agenda import ProactiveAgendaStep, _pin_reason
from reme.steps.evolve.proactive.extract import ProactiveExtractStep
from reme.steps.evolve.proactive.finish import ProactiveFinishStep
from reme.steps.evolve.proactive.plan import ProactivePlanStep
from reme.steps.evolve.proactive.proactive import ProactiveStep
from reme.steps.evolve.proactive.refine import parse_refine_reply
from reme.steps.evolve.proactive.schedules import (
    apply_schedule_updates,
    canonical_due,
    merge_reminders,
    parse_when,
    reminder_candidates_from_view,
    schedule_view,
    trim_schedules,
)
from reme.steps.evolve.proactive.topics import ProactiveTopicsStep
from reme.steps.evolve.proactive.utils import (
    load_carry_forward,
    load_personal_profile_block,
    load_state,
    load_yaml_topics,
    normalize_topic,
    parse_interests_topics,
    parse_extract_reply,
    scan_material_daily,
    scan_material_resource,
    strip_frontmatter,
    topic_id,
    trim_state_file,
)

DAY = "2026-08-13"
# Fixed business clock for the reminder branch: 08:00 on DAY (+08:00).
NOW = dt.datetime(2026, 8, 13, 8, 0, tzinfo=dt.timezone(dt.timedelta(hours=8)))
EXTRACT_CLOCK = "reme.steps.evolve.proactive.extract.current_now"
# Stage-2 refine probes: the pass is recognised by its user message, and an
# unscripted refine call answers "keep everything" (fail-open by contract).
REFINE_MARKERS = ("kind under review:", "本次裁决的 kind")
KEEP_ALL_VERDICTS = "```yaml\nverdicts: []\n```"


def _is_refine_call(inputs) -> bool:
    text = str(inputs or "")
    return any(marker in text for marker in REFINE_MARKERS)


def _verdicts(*verdicts) -> str:
    body = yaml.safe_dump({"verdicts": list(verdicts)}, allow_unicode=True, sort_keys=False)
    return f"```yaml\n{body}```"


def _verdict(cid, decision="drop", to_kind="", confidence=None, reason="not grounded"):
    doc = {"id": cid, "decision": decision, "reason": reason}
    if to_kind:
        doc["to_kind"] = to_kind
    if confidence is not None:
        doc["confidence"] = confidence
    return doc


def _touch(path: Path, text: str = "x") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


class _Catalog(BaseFileCatalog):
    """In-memory catalog stub recording every mutation."""

    def __init__(self, nodes=None):
        super().__init__()
        self.nodes = list(nodes or [])
        self.upserts = []
        self.dumps = 0
        self.deleted = []

    async def upsert(self, nodes):
        self.upserts.extend(nodes)
        known = {n.path: n for n in self.nodes}
        for node in nodes:
            known[node.path] = node
        self.nodes = list(known.values())

    async def delete(self, path):
        paths = path if isinstance(path, list) else [path]
        self.deleted.extend(paths)
        drop = set(paths)
        self.nodes = [n for n in self.nodes if n.path not in drop]

    async def get_nodes(self, paths=None):
        if paths is None:
            return list(self.nodes)
        wanted = set(paths)
        return [n for n in self.nodes if n.path in wanted]

    async def dump(self):
        self.dumps += 1


class _FileStore(BaseFileStore):
    def __init__(self, workspace: Path):
        super().__init__()
        self._workspace_path = workspace

    @property
    def workspace_path(self) -> Path:
        return self._workspace_path

    async def upsert(self, files):
        return None

    async def delete(self, path):
        return None

    async def clear(self):
        return None

    async def get_nodes(self, paths=None):
        return []

    async def get_outlinks(self, path, scope=None):
        return []

    async def get_inlinks(self, path, scope=None):
        return []

    async def vector_search(self, query, limit, search_filter):
        return []

    async def keyword_search(self, query, limit, search_filter):
        return []


class _AgentWrapper(BaseAgentWrapper):
    """Deterministic LLM stub with scripted replies and call accounting.

    Stage-2 refine calls are routed to their own queue and default to "keep
    everything", so a test that only cares about stage 1 (or about plan/agenda)
    stays aligned while refine is enabled by default.
    """

    def __init__(self, replies=None, delay: float = 0.0, refine_replies=None):
        super().__init__()
        self.replies = list(replies or [])
        self.refine_replies = list(refine_replies or [])
        self.delay = delay
        self.calls = 0
        self.refine_calls = 0
        self.last_inputs = []
        self.last_system_prompts = []

    async def reply(self, inputs, **kwargs):
        self.calls += 1
        self.last_inputs.append(inputs)
        self.last_system_prompts.append(str(kwargs.get("system_prompt") or ""))
        if _is_refine_call(inputs):
            self.refine_calls += 1
            queue, fallback = self.refine_replies, KEEP_ALL_VERDICTS
        else:
            queue, fallback = self.replies, ""
        item = queue.pop(0) if queue else fallback
        if isinstance(item, tuple):
            text, delay = item
        else:
            text, delay = item, self.delay
        if delay:
            await asyncio.sleep(delay)
        return {
            "session_id": "stub",
            "result": text,
            "last_message": {"content": [{"type": "text", "text": text}]},
        }


class _DummyModel(ChatModelBase):
    """Placeholder model satisfying dream extract's llm_available check."""

    def __init__(self):
        pass


class _FakeEmbedding:
    """Scripted embedding oracle: exact text -> vector, else orthogonal default.

    Declares ``kwargs``/``dimensions`` so the topics step can fingerprint the
    "model" against ``known_threshold_calibrated_for`` (defaults match).
    """

    def __init__(self, vectors: dict[str, list[float]], model: str = "text-embedding-v4", dimensions: int = 1024):
        self.vectors = vectors
        self.calls = 0
        self.kwargs = {"model": model}
        self.dimensions = dimensions

    async def __call__(self, inputs, **kwargs):
        self.calls += 1
        return [list(self.vectors.get(text, [0.0, 1.0])) for text in inputs]


def _reply(follow_ups=None, extends=None, reminders=None, new_ideas=None, updates=None, schedule_updates=None) -> str:
    doc = {}
    if follow_ups is not None:
        doc["follow_ups"] = follow_ups
    if extends is not None:
        doc["extends"] = extends
    if reminders is not None:
        doc["reminders"] = reminders
    if new_ideas is not None:
        doc["new_ideas"] = new_ideas
    if updates is not None:
        doc["updates"] = updates
    if schedule_updates is not None:
        doc["schedule_updates"] = schedule_updates
    body = yaml.safe_dump(doc, allow_unicode=True, sort_keys=False)
    return f"```yaml\n{body}```"


def _topic(title, reason="because", confidence=0.7, paths=None, evidence=None):
    doc = {
        "title": title,
        "reason": reason,
        "confidence": confidence,
        "paths": paths or [],
    }
    if evidence:
        doc["evidence"] = evidence
    return doc


def _state_topic(title, day=DAY, kind="follow_up", confidence=0.8, topic_id_value=None, status="open"):
    return {
        "id": topic_id_value or topic_id(title),
        "title": title,
        "reason": "seeded",
        "kind": kind,
        "confidence": confidence,
        "status": status,
        "first_seen": day,
        "last_evidence_at": day,
        "evidence": "",
        "paths": [],
    }


def _reminder(title, due_at="", reason="快到期了", confidence=0.8, paths=None, **extra):
    doc = {"title": title, "reason": reason, "confidence": confidence, "paths": paths or []}
    if due_at:
        doc["due_at"] = due_at
    doc.update(extra)
    return doc


def _schedule(title, due_at="", schedule_id=None, status="open", **extra):
    entry = {
        "id": schedule_id or topic_id(title),
        "title": title,
        "due_at": due_at,
        "status": status,
        "confidence": 0.8,
        "first_seen": DAY,
        "last_seen_at": DAY,
    }
    entry.update(extra)
    return entry


def _write_state(
    ws,
    budget=None,
    exposure=None,
    open_topics=None,
    resolved=None,
    schedules=None,
    omit_open_topics=False,
):
    data = {"version": 1}
    if budget is not None:
        data["budget"] = budget
    if exposure is not None:
        data["exposure"] = exposure
    if not omit_open_topics:
        data["open_topics"] = open_topics or []
    if resolved is not None:
        data["resolved"] = resolved
    if schedules is not None:
        data["schedules"] = schedules
    _touch(ws / "daily" / "_proactive.yaml", yaml.safe_dump(data, allow_unicode=True, sort_keys=False))


def _read_state(ws) -> dict:
    return yaml.safe_load((ws / "daily" / "_proactive.yaml").read_text(encoding="utf-8"))


def _interests(ws, day=DAY) -> Path:
    return ws / "daily" / day / "interests.yaml"


def _write_interests_v1(ws, day, topics_yaml: list[dict]):
    payload = {"date": day, "topic_count": len(topics_yaml), "topics": topics_yaml}
    _touch(_interests(ws, day), yaml.safe_dump(payload, allow_unicode=True, sort_keys=False))


def _write_interests_v2(ws, day, topics: list[dict], push=True, skip_reason=""):
    payload = {
        "version": 2,
        "date": day,
        "generated_at": f"{day}T09:00:00+08:00",
        "push": push,
        "skip_reason": skip_reason,
        "topics": topics,
    }
    _touch(_interests(ws, day), yaml.safe_dump(payload, allow_unicode=True, sort_keys=False))


async def _run_chain(
    ws,
    replies=None,
    *,
    wrapper=None,
    catalog=None,
    app=None,
    extract_kwargs=None,
    topics_kwargs=None,
    extra_ctx=None,
    context=None,
):
    app = app or ApplicationContext(workspace_dir=str(ws))
    wrapper = wrapper or _AgentWrapper(replies or [])
    catalog = catalog or _Catalog()
    context = context or RuntimeContext(
        date=DAY,
        file_catalog=catalog,
        file_store=_FileStore(ws),
        agent_wrapper=wrapper,
        **(extra_ctx or {}),
    )
    extract = ProactiveExtractStep(app_context=app, **(extract_kwargs or {}))
    resp_extract = await extract(context)
    topics = ProactiveTopicsStep(app_context=app, **(topics_kwargs or {}))
    resp_topics = await topics(context)
    finish = ProactiveFinishStep(app_context=app)
    resp_finish = await finish(context)
    return app, wrapper, catalog, context, (resp_extract, resp_topics, resp_finish)


def test_known_threshold_model_binding(tmp_path):
    """known_threshold is model-bound (v5.1): fingerprint mismatch degrades.

    The oracle maps every text to the same vector (sim 1.0 >= known_threshold),
    but the "model" differs from the calibration, so the semantic gate must be
    disabled: a paraphrase survives while an exact normalize duplicate of a
    historical interests title still drops.
    """

    async def run():
        ws = tmp_path
        _touch(ws / "daily" / DAY / "session.md", "notes")
        _write_state(ws, open_topics=[_state_topic("检索引用质量评测")])
        _write_interests_v1(ws, "2026-08-12", [{"title": "向量数据库选型", "reason": "done"}])
        embedder = _FakeEmbedding({}, model="some-other-model-v9")
        reply = _reply(
            follow_ups=[
                _topic("检索引用可信度的评测方法", paths=[f"daily/{DAY}/session.md"]),
                _topic("向量数据库选型", paths=[f"daily/{DAY}/session.md"]),
            ],
        )
        _, _, _, context, _ = await _run_chain(ws, [reply], extra_ctx={"as_embedding": embedder})
        state = context.get("proactive")
        assert state["dropped_known"] == 0  # semantic gate disabled by mismatch
        assert state["dropped_duplicate"] == 1  # exact compare still active
        titles = [t["title"] for t in _read_state(ws)["open_topics"]]
        assert "检索引用可信度的评测方法" in titles
        assert "向量数据库选型" not in titles

    asyncio.run(run())


def test_overage_remention_restarts(tmp_path):
    """Over-age open topics re-mentioned today restart instead of vanishing.

    Same-id candidate refreshes first_seen so trim keeps the topic (unfinished
    business must keep being re-executed). After the restart the revived title
    is back in the comparison set, so a paraphrase of it drops as known.
    A dying topic with NO same-id candidate stays excluded from comparison:
    its paraphrase re-mention survives and opens a fresh topic.
    """

    async def run():
        ws = tmp_path
        _touch(ws / "daily" / DAY / "session.md", "notes")
        old_day = "2026-07-28"  # 16 days before DAY, over-age at carry_forward_days=14
        _write_state(
            ws,
            open_topics=[_state_topic("旧任务确认", day=old_day), _state_topic("遗留事项", day=old_day)],
        )
        # Discriminative oracle: only the legacy-task / its-supplement pair scores sim 1.0;
        # every other pair is orthogonal. If the dying title leaked into the
        # comparison set, the paraphrase below would drop as known.
        embedder = _FakeEmbedding({"遗留事项的补充。because": [1.0, 0.0], "遗留事项。seeded": [1.0, 0.0]})
        reply = _reply(
            follow_ups=[
                _topic("旧任务确认", paths=[f"daily/{DAY}/session.md"]),
                _topic("旧任务确认的后续安排", paths=[f"daily/{DAY}/session.md"]),
                _topic("遗留事项的补充", paths=[f"daily/{DAY}/session.md"]),
            ],
        )
        _, _, _, context, _ = await _run_chain(ws, [reply], extra_ctx={"as_embedding": embedder})
        state = context.get("proactive")
        assert state["dropped_known"] == 1  # paraphrase of the REVIVED topic (default sim 1.0 vs it)
        topics = {t["id"]: t for t in _read_state(ws)["open_topics"]}
        assert topics[topic_id("旧任务确认")]["first_seen"] == DAY  # restarted, not trimmed
        assert topic_id("旧任务确认的后续安排") not in topics  # known vs revived topic
        assert topic_id("遗留事项的补充") in topics  # dying title excluded from comparison
        assert topic_id("遗留事项") not in topics  # over-age, never re-mentioned -> trimmed

    asyncio.run(run())


def test_update_evidence_anchor(tmp_path):
    """v5.2: action=update must anchor this round's new material (hard check).

    An update citing a file outside the round's changed material degrades to
    keep (no evidence rewrite, no freshness refresh); an update citing a
    changed file applies, and last_evidence_at is parsed from the evidence
    path date instead of defaulting to today.
    """

    async def run():
        ws = tmp_path
        _touch(ws / "daily" / DAY / "session.md", "today notes")
        _touch(ws / "daily" / "2026-08-12" / "old-note.md", "yesterday notes")
        _write_state(
            ws,
            open_topics=[_state_topic("证据改挂主题"), _state_topic("昨日证据主题")],
        )
        reply = _reply(
            updates=[
                {"id": topic_id("证据改挂主题"), "action": "update", "evidence": "daily/2026-08-11/absent.md"},
                {
                    "id": topic_id("昨日证据主题"),
                    "action": "update",
                    "evidence": "daily/2026-08-12/old-note.md#L5",
                    "confidence": 0.9,
                },
            ],
        )
        _, _, _, context, _ = await _run_chain(ws, [reply])
        state = context.get("proactive")
        assert state["updates_applied"] == 1
        topics = {t["id"]: t for t in _read_state(ws)["open_topics"]}
        rejected = topics[topic_id("证据改挂主题")]
        assert rejected["last_evidence_at"] == DAY  # untouched, degraded to keep
        assert rejected["evidence"] != "daily/2026-08-11/absent.md"
        applied = topics[topic_id("昨日证据主题")]
        assert applied["last_evidence_at"] == "2026-08-12"  # parsed from evidence path
        assert applied["evidence"] == "daily/2026-08-12/old-note.md#L5"
        assert applied["confidence"] == 0.9

    asyncio.run(run())


def test_no_embedding_configured(tmp_path):
    """No as_embedding anywhere: the chain runs and dedups exactly (BM25-only).

    Paraphrase-level candidates survive (no semantic gate), exact normalize
    duplicates of historical exposure still drop; nothing crashes or skips.
    """

    async def run():
        ws = tmp_path
        _touch(ws / "daily" / DAY / "session.md", "notes")
        _write_state(ws, open_topics=[_state_topic("已知主题")])
        _write_interests_v1(ws, "2026-08-12", [{"title": "历史曝光主题", "reason": "x"}])
        reply = _reply(
            follow_ups=[
                _topic("已知主题的不同说法", paths=[f"daily/{DAY}/session.md"]),
                _topic("历史曝光主题", paths=[f"daily/{DAY}/session.md"]),
            ],
        )
        _, _, _, context, _ = await _run_chain(ws, [reply])  # no as_embedding in context/app
        state = context.get("proactive")
        assert state["dropped_known"] == 0
        assert state["dropped_duplicate"] == 1
        titles = [t["title"] for t in _read_state(ws)["open_topics"]]
        assert "已知主题的不同说法" in titles
        assert "历史曝光主题" not in titles

    asyncio.run(run())


# ---------------------------------------------------------------------------
# F1 contracts
# ---------------------------------------------------------------------------


def test_normalize_and_topic_id_frozen():
    """A7 frozen contract: NFKC+casefold, keep L/N only; id is a 12-hex sha1 prefix."""
    assert normalize_topic("Memory Search: 可解释性评估!") == "memorysearch可解释性评估"
    assert topic_id("A") == topic_id("a") == topic_id("  A  ")
    assert topic_id("A") != topic_id("B")
    assert len(topic_id("anything")) == 12


def test_legacy_file_parse(tmp_path):
    """v1 files parse without error; new fields take deterministic defaults (A2)."""
    content = (
        'date: "2026-08-13"\n'
        "topic_count: 1\n"
        "topics:\n"
        "  - title: Retrieval quality\n"
        "    reason: Search behavior changed.\n"
        "    evidence: daily/2026-08-13/session.md\n"
    )
    _touch(_interests(tmp_path), content)
    topics, is_v1, push = parse_interests_topics(yaml.safe_load(content), DAY)
    assert is_v1 is True and push is True
    topic = topics[0]
    assert topic.id == topic_id("Retrieval quality")
    assert topic.kind == "interest_extend"
    assert topic.confidence == 0.5
    assert topic.first_seen == DAY
    assert topic.last_evidence_at == DAY

    async def run():
        step = ProactiveStep(app_context=ApplicationContext(workspace_dir=str(tmp_path)))
        response = await step(RuntimeContext(date=DAY, include_content=False, file_store=_FileStore(tmp_path)))
        assert response.success is True
        assert response.metadata["push"] is True

    asyncio.run(run())


def test_field_fallback():
    """Illegal kind/confidence values fall back deterministically."""
    topic = ProactiveTopic(title="T", reason="R", kind="bogus", confidence="not-a-number")
    assert topic.kind == "interest_extend"
    assert topic.confidence == 0.5
    assert ProactiveTopic(confidence=1.7).confidence == 1.0
    assert ProactiveTopic(confidence=-3).confidence == 0.0


def test_resolved_registry(tmp_path):
    """Resolved ids disappear from carry-forward and from F5 reads."""

    async def run():
        ws = tmp_path
        id_a, id_b = topic_id("Topic A"), topic_id("Topic B")
        _write_state(
            ws,
            open_topics=[_state_topic("Topic A", topic_id_value=id_a), _state_topic("Topic B", topic_id_value=id_b)],
            resolved=[{"id": id_b, "title": "Topic B", "resolved_at": DAY, "evidence": ""}],
        )
        state_file, needs_bootstrap = load_state(ws)
        assert needs_bootstrap is False
        carry_all, _ = await load_carry_forward(ws, state_file, DAY, 14, 20)
        assert [t.title for t in carry_all] == ["Topic A"]

        _write_interests_v2(
            ws,
            DAY,
            [_state_topic("Topic A", topic_id_value=id_a), _state_topic("Topic B", topic_id_value=id_b)],
        )
        step = ProactiveStep(app_context=ApplicationContext(workspace_dir=str(ws)))
        response = await step(RuntimeContext(date=DAY, include_content=False, file_store=_FileStore(ws)))
        assert [t["title"] for t in response.answer["topics"]] == ["Topic A"]

    asyncio.run(run())


def test_carry_forward_bootstrap(tmp_path):
    """First run (missing state file) bootstraps from interests.yaml history; first_seen = min."""

    async def run():
        ws = tmp_path
        _write_interests_v1(ws, "2026-08-11", [_topic("Topic T", paths=["daily/2026-08-11/a.md"])])
        _write_interests_v1(
            ws,
            "2026-08-12",
            [_topic("Topic T", paths=["daily/2026-08-12/a.md"]), _topic("Topic U", paths=["daily/2026-08-12/b.md"])],
        )

        # Upgrade path: no _proactive.yaml yet -> one-time bootstrap from history.
        state_file, needs_bootstrap = load_state(ws)
        assert needs_bootstrap is True
        carry_all, _ = await load_carry_forward(ws, state_file, DAY, 14, 20, needs_bootstrap=needs_bootstrap)
        by_title = {t.title: t for t in carry_all}
        assert set(by_title) == {"Topic T", "Topic U"}
        assert by_title["Topic T"].first_seen == "2026-08-11"  # min across days, not latest
        assert by_title["Topic U"].first_seen == "2026-08-12"

        persisted = _read_state(ws)
        assert len(persisted["open_topics"]) == 2
        _, needs_bootstrap2 = load_state(ws)
        assert needs_bootstrap2 is False  # no re-bootstrap once the key exists

        # Empty open_topics list is a normal state, not a bootstrap trigger.
        _write_state(ws, open_topics=[])
        _, needs_bootstrap3 = load_state(ws)
        assert needs_bootstrap3 is False

        # An existing file lacking the key (e.g. hand-edited) still triggers it.
        _write_state(ws, budget={}, omit_open_topics=True)
        _, needs_bootstrap4 = load_state(ws)
        assert needs_bootstrap4 is True

    asyncio.run(run())


def test_state_file_corrupt_no_bootstrap(tmp_path):
    """Corrupt _proactive.yaml rebuilds empty WITHOUT bootstrap (spec F1.3/A2/A5)."""

    async def run():
        ws = tmp_path
        _write_interests_v1(ws, "2026-08-12", [_topic("Topic U", paths=["daily/2026-08-12/b.md"])])
        _touch(ws / "daily" / "_proactive.yaml", "version: [unclosed")

        state_file, needs_bootstrap = load_state(ws)
        assert needs_bootstrap is False
        assert state_file.open_topics == []
        carry_all, _ = await load_carry_forward(ws, state_file, DAY, 14, 20, needs_bootstrap=needs_bootstrap)
        assert carry_all == []

    asyncio.run(run())


def test_bootstrap_upgrade_chain(tmp_path):
    """Upgrade path: the first refresh run seeds the truth source from interests history."""

    async def run():
        ws = tmp_path
        _write_interests_v1(ws, "2026-08-12", [_topic("历史主题", paths=["daily/2026-08-12/a.md"])])
        _touch(ws / "daily" / DAY / "s1.md", "new evidence")
        reply = _reply(follow_ups=[_topic("新主题", confidence=0.9, paths=[f"daily/{DAY}/s1.md"])])
        _, _, _, context, responses = await _run_chain(ws, [reply])
        assert all(r.success for r in responses)

        assert context.get("proactive")["carry_forward_count"] == 1  # history reached the chain context

        truth_titles = {t["title"] for t in _read_state(ws)["open_topics"]}
        assert truth_titles == {"历史主题", "新主题"}
        data = yaml.safe_load(_interests(ws).read_text(encoding="utf-8"))
        assert {t["title"] for t in data["topics"]} == {"历史主题", "新主题"}

    asyncio.run(run())


def test_carry_forward_expiry(tmp_path):
    """Topics older than carry_forward_days drop from carry-forward and are pruned."""

    async def run():
        ws = tmp_path
        _write_state(ws, open_topics=[_state_topic("Old Topic", day="2026-07-01")])
        state_file, _ = load_state(ws)
        carry_all, _ = await load_carry_forward(ws, state_file, DAY, 14, 20)
        assert carry_all == []

        _touch(ws / "daily" / DAY / "note.md", "new evidence")
        reply = _reply(follow_ups=[], updates=[])
        _, _, _, context, responses = await _run_chain(ws, [reply])
        assert all(r.success for r in responses)
        assert context.get("proactive")["carry_forward_count"] == 0
        assert _read_state(ws)["open_topics"] == []  # pruned from the truth source

    asyncio.run(run())


def test_material_filter(tmp_path):
    """Day indexes, ``_*`` files and interests.yaml never enter the material set (INV-11)."""
    ws = tmp_path
    _touch(ws / "daily" / f"{DAY}.md", "day index")
    _touch(ws / "daily" / DAY / "_hidden.md", "underscore file")
    _touch(ws / "daily" / DAY / "interests.yaml", "topics: []\n")
    _touch(ws / "daily" / DAY / "good.md", "material")
    assert scan_material_daily(ws, DAY, "daily", 2) == [f"daily/{DAY}/good.md"]


# ---------------------------------------------------------------------------
# F2 discovery + chain behaviour
# ---------------------------------------------------------------------------


def test_refresh_chain_e2e(tmp_path):
    """extract -> topics -> finish runs end to end."""

    async def run():
        ws = tmp_path
        note = _touch(ws / "daily" / DAY / "session.md", "we discussed the eval plan; nothing landed")
        reply = _reply(
            follow_ups=[_topic("记忆检索的可解释性评估", confidence=0.9, paths=["daily/2026-08-13/session.md"])],
        )
        app = ApplicationContext(workspace_dir=str(ws))
        wrapper = _AgentWrapper([reply])
        catalog = _Catalog()
        context = RuntimeContext(date=DAY, file_catalog=catalog, file_store=_FileStore(ws), agent_wrapper=wrapper)

        extract = ProactiveExtractStep(app_context=app)
        resp_extract = await extract(context)
        topics = ProactiveTopicsStep(app_context=app)
        resp_topics = await topics(context)
        finish = ProactiveFinishStep(app_context=app)
        resp_finish = await finish(context)

        assert resp_extract.success and resp_topics.success and resp_finish.success
        state = context.get("proactive")
        assert state["llm_calls"] == 1 + state["refine_calls"]  # stage 1 + one pass per non-empty kind
        assert state["push"] is True
        assert state["file_skip_reason"] == ""
        assert state["interests_written"] is True

        data = yaml.safe_load(_interests(ws).read_text(encoding="utf-8"))
        assert data["version"] == 2
        assert data["push"] is True
        assert data["topics"][0]["title"] == "记忆检索的可解释性评估"
        assert data["topics"][0]["kind"] == "follow_up"

        truth = _read_state(ws)
        assert len(truth["open_topics"]) == 1
        checkpoint_paths = {n.path for n in catalog.upserts}
        assert checkpoint_paths == {note.relative_to(ws).as_posix()}  # interests.yaml not checkpointed (R6)
        assert catalog.dumps == 1

    asyncio.run(run())


def test_open_loop_follow_up(tmp_path):
    """Branch A emits follow_ups with stable ids; paths restricted to M."""

    async def run():
        ws = tmp_path
        _touch(ws / "daily" / DAY / "session.md", "open loop content")
        reply = _reply(
            follow_ups=[
                _topic("未决事项甲", confidence=0.9, paths=[f"daily/{DAY}/session.md"]),
                _topic("越界事项", confidence=0.9, paths=["daily/2026-08-01/absent.md"]),
            ],
        )
        _, _, _, context, _ = await _run_chain(ws, [reply])
        state = context.get("proactive")
        assert len(state["follow_ups"]) == 1
        follow_up = state["follow_ups"][0]
        assert follow_up["kind"] == "follow_up"
        assert follow_up["id"] == topic_id("未决事项甲")
        assert state["dropped_missing"] == 1  # out-of-M paths dropped, never repaired
        data = yaml.safe_load(_interests(ws).read_text(encoding="utf-8"))
        assert [t["title"] for t in data["topics"]] == ["未决事项甲"]

    asyncio.run(run())


def test_updates_actions(tmp_path):
    """keep/update/resolve act on the truth source; unknown ids ignored."""

    async def run():
        ws = tmp_path
        id_a, id_b, id_c = topic_id("Topic A"), topic_id("Topic B"), topic_id("Topic C")
        _write_state(
            ws,
            open_topics=[
                {**_state_topic("Topic A", day="2026-08-12", topic_id_value=id_a)},
                {**_state_topic("Topic B", day="2026-08-12", topic_id_value=id_b)},
                {**_state_topic("Topic C", day="2026-08-12", topic_id_value=id_c)},
            ],
        )
        _touch(ws / "daily" / DAY / "session.md", "new evidence today")
        evidence = f"daily/{DAY}/session.md"
        reply = _reply(
            follow_ups=[],
            updates=[
                {"id": id_a, "action": "keep"},
                {"id": id_b, "action": "update", "evidence": evidence},
                {"id": id_c, "action": "resolve", "evidence": evidence},
                {"id": "ffffffffffff", "action": "resolve"},
            ],
        )
        _, _, _, context, responses = await _run_chain(ws, [reply])
        assert all(r.success for r in responses)
        state = context.get("proactive")
        assert state["updates_applied"] == 1
        assert state["updates_resolved"] == 1

        truth = _read_state(ws)
        by_id = {t["id"]: t for t in truth["open_topics"]}
        assert set(by_id) == {id_a, id_b}
        assert by_id[id_a]["last_evidence_at"] == "2026-08-12"  # keep: untouched
        assert by_id[id_b]["last_evidence_at"] == DAY
        assert by_id[id_b]["evidence"] == evidence
        resolved_ids = {r["id"] for r in truth["resolved"]}
        assert resolved_ids == {id_c}

        data = yaml.safe_load(_interests(ws).read_text(encoding="utf-8"))
        assert {t["title"] for t in data["topics"]} == {"Topic A", "Topic B"}

    asyncio.run(run())


def test_resolved_resurrect(tmp_path):
    """A candidate hitting a tombstone resurrects it: tombstone removed, first_seen kept."""

    async def run():
        ws = tmp_path
        tid = topic_id("Revived Topic")
        _write_state(
            ws,
            resolved=[
                {
                    "id": tid,
                    "title": "Revived Topic",
                    "resolved_at": "2026-08-12",
                    "first_seen": "2026-08-08",
                    "evidence": "",
                },
            ],
        )
        _touch(ws / "daily" / DAY / "session.md", "the plan was reopened today")
        reply = _reply(follow_ups=[_topic("Revived Topic", confidence=0.7, paths=[f"daily/{DAY}/session.md"])])
        _, _, _, context, _ = await _run_chain(ws, [reply])
        state = context.get("proactive")
        assert [c["title"] for c in state["candidates"]] == ["Revived Topic"]

        truth = _read_state(ws)
        assert truth["resolved"] == []  # tombstone removed
        by_id = {t["id"]: t for t in truth["open_topics"]}
        assert tid in by_id
        assert by_id[tid]["first_seen"] == "2026-08-08"  # original age anchor kept
        assert by_id[tid]["last_evidence_at"] == DAY

    asyncio.run(run())


def test_trim_freshness_first(tmp_path):
    """max_topics trims by the freshness-first order, regardless of kind (v5 aging)."""

    async def run():
        ws = tmp_path
        stale = [
            {**_state_topic(f"Stale Follow {i}", day="2026-08-05", kind="follow_up"), "last_evidence_at": "2026-08-05"}
            for i in range(3)
        ]
        fresh = [_state_topic(f"Fresh Extend {i}", day=DAY, kind="interest_extend") for i in range(2)]
        _write_state(ws, open_topics=stale + fresh)
        _touch(ws / "daily" / DAY / "session.md", "material")
        reply = _reply(follow_ups=[_topic("New Follow", confidence=0.9, paths=[f"daily/{DAY}/session.md"])])
        _, _, _, context, _ = await _run_chain(ws, [reply], topics_kwargs={"max_topics": 4})
        titles = [t["title"] for t in context.get("proactive")["topics_out"]]
        assert len(titles) == 4
        assert titles[0] == "New Follow"  # today's follow_up first
        assert set(titles[1:3]) == {"Fresh Extend 0", "Fresh Extend 1"}  # today's extends next
        assert titles[3].startswith("Stale Follow")  # only one stale topic survives the cap

    asyncio.run(run())


def test_dedup_fallback(tmp_path):
    """M1 fallback: normalize_topic exact match against recent interests drops duplicates."""

    async def run():
        ws = tmp_path
        _write_interests_v1(ws, "2026-08-12", [_topic("Retrieval Evaluation", paths=["daily/2026-08-12/a.md"])])
        # Seed an empty truth source so the history topic stays out of open_topics
        # (otherwise first-run bootstrap would merge the same-id candidate instead
        # of dropping it as a duplicate).
        _write_state(ws, open_topics=[])
        _touch(ws / "daily" / DAY / "session.md", "material")
        reply = _reply(
            follow_ups=[_topic("retrieval evaluation!!", confidence=0.9, paths=[f"daily/{DAY}/session.md"])],
        )
        _, _, _, context, _ = await _run_chain(ws, [reply])
        state = context.get("proactive")
        assert state["dropped_duplicate"] == 1
        assert state["candidates"] == []
        assert state["file_skip_reason"] == "all_duplicates"  # metadata-only since v5 (R7)
        data = yaml.safe_load(_interests(ws).read_text(encoding="utf-8"))
        assert data["push"] is False
        assert data["topics"] == []

    asyncio.run(run())


def test_skip_low_confidence(tmp_path):
    """Candidates under min_push_confidence persist but are not pushed."""

    async def run():
        ws = tmp_path
        _touch(ws / "daily" / DAY / "session.md", "weak signal")
        reply = _reply(follow_ups=[_topic("弱信号主题", confidence=0.3, paths=[f"daily/{DAY}/session.md"])])
        _, _, _, context, _ = await _run_chain(ws, [reply])
        state = context.get("proactive")
        assert state["push"] is False
        assert state["file_skip_reason"] == "low_confidence"
        data = yaml.safe_load(_interests(ws).read_text(encoding="utf-8"))
        assert data["push"] is False
        assert "skip_reason" not in data  # metadata-only since v5 (R7)
        assert [t["title"] for t in data["topics"]] == ["弱信号主题"]  # rendered from truth source
        assert [t["title"] for t in _read_state(ws)["open_topics"]] == ["弱信号主题"]  # persisted

    asyncio.run(run())


def test_no_new_evidence_zero_llm(tmp_path):
    """No changed material -> early exit with zero LLM calls."""

    async def run():
        ws = tmp_path
        note = _touch(ws / "daily" / DAY / "session.md", "already seen")
        rel = note.relative_to(ws).as_posix()
        catalog = _Catalog([FileNode(path=rel, st_mtime=note.stat().st_mtime)])
        wrapper = _AgentWrapper(["should never be consumed"])
        _, wrapper, catalog, context, responses = await _run_chain(
            ws,
            wrapper=wrapper,
            catalog=catalog,
        )
        state = context.get("proactive")
        assert state["early_exit"] == "no_new_evidence"
        assert wrapper.calls == 0
        assert not _interests(ws).exists()
        assert catalog.upserts == [] and catalog.dumps == 0
        assert all(r.success for r in responses)

    asyncio.run(run())


def test_push_sticky(tmp_path):
    """Within one day push only goes up: derived from the cumulative truth source (R1)."""

    async def run():
        ws = tmp_path
        _touch(ws / "daily" / DAY / "s1.md", "strong evidence")
        reply1 = _reply(follow_ups=[_topic("高置信主题", confidence=0.9, paths=[f"daily/{DAY}/s1.md"])])
        _, _, _, context1, _ = await _run_chain(ws, [reply1])
        assert context1.get("proactive")["push"] is True
        assert yaml.safe_load(_interests(ws).read_text(encoding="utf-8"))["push"] is True

        _touch(ws / "daily" / DAY / "s2.md", "weak evidence")
        reply2 = _reply(follow_ups=[_topic("低置信主题", confidence=0.3, paths=[f"daily/{DAY}/s2.md"])])
        _, _, _, context2, _ = await _run_chain(ws, [reply2])
        state2 = context2.get("proactive")
        assert state2["push"] is True  # sticky: the 0.9 topic discovered today persists in truth source
        assert state2["file_skip_reason"] == ""
        assert yaml.safe_load(_interests(ws).read_text(encoding="utf-8"))["push"] is True

    asyncio.run(run())


def test_idempotent_write(tmp_path):
    """Same render skips writing; nightly v1 is adopted via bootstrap (v5 R1)."""

    async def run():
        ws = tmp_path
        _touch(ws / "daily" / DAY / "s1.md", "evidence one")
        reply1 = _reply(follow_ups=[_topic("主题甲", confidence=0.9, paths=[f"daily/{DAY}/s1.md"])])
        _, _, _, context1, _ = await _run_chain(ws, [reply1])
        assert context1.get("proactive")["interests_written"] is True
        mtime_before = _interests(ws).stat().st_mtime

        _touch(ws / "daily" / DAY / "s2.md", "evidence two, no new topics")
        reply2 = _reply(follow_ups=[], updates=[])
        time.sleep(0.01)
        _, _, _, context2, _ = await _run_chain(ws, [reply2])
        assert context2.get("proactive")["interests_written"] is False  # identical render
        assert _interests(ws).stat().st_mtime == mtime_before

    asyncio.run(run())

    async def run_v1_nightly_adoption():
        """v5 R1: no v1 guard - first-run bootstrap adopts the nightly topic."""
        with tempfile.TemporaryDirectory() as tmp:
            ws = Path(tmp)
            nightly = [
                {
                    "title": "Nightly topic",
                    "reason": "written by dream",
                    "evidence": f"daily/{DAY}/n.md",
                    "paths": [f"daily/{DAY}/n.md"],
                },
            ]
            _write_interests_v1(ws, DAY, nightly)
            _touch(ws / "daily" / DAY / "s1.md", "material")
            reply = _reply(follow_ups=[], updates=[])
            _, _, _, context, _ = await _run_chain(ws, [reply])
            state = context.get("proactive")
            # Bootstrap gives the nightly topic first_seen = file date (today);
            # push is derived true because its fallback confidence 0.5 passes.
            assert state["push"] is True
            assert state["interests_written"] is True
            data = yaml.safe_load(_interests(ws).read_text(encoding="utf-8"))
            assert data["version"] == 2
            assert [t["title"] for t in data["topics"]] == ["Nightly topic"]

    asyncio.run(run_v1_nightly_adoption())


# ---------------------------------------------------------------------------
# F3 isolation / interleave
# ---------------------------------------------------------------------------


def test_refresh_isolation(tmp_path):
    """Refresh never writes digest, never touches the dream catalog or day index."""

    async def run():
        ws = tmp_path
        digest_file = _touch(ws / "digest" / "wiki" / "existing.md", "digest content")
        day_index = _touch(ws / "daily" / f"{DAY}.md", "index content")
        _touch(ws / "daily" / DAY / "session.md", "conversation")
        reply = _reply(follow_ups=[_topic("隔离主题", confidence=0.9, paths=[f"daily/{DAY}/session.md"])])
        proactive_catalog = _Catalog()
        dream_catalog = _Catalog()
        _, _, _, _, responses = await _run_chain(ws, [reply], catalog=proactive_catalog)
        assert all(r.success for r in responses)

        assert digest_file.read_text(encoding="utf-8") == "digest content"
        assert day_index.read_text(encoding="utf-8") == "index content"
        assert not dream_catalog.upserts and dream_catalog.dumps == 0 and not dream_catalog.deleted
        assert proactive_catalog.dumps == 1
        assert {n.path for n in proactive_catalog.upserts} == {f"daily/{DAY}/session.md"}

    asyncio.run(run())


def test_nightly_overwrite_continuity(tmp_path):
    """Nightly v1 overwrite cannot break continuity: truth source restores exposure."""

    async def run():
        ws = tmp_path
        _touch(ws / "daily" / DAY / "s1.md", "round one")
        reply1 = _reply(follow_ups=[_topic("持续主题", confidence=0.9, paths=[f"daily/{DAY}/s1.md"])])
        await _run_chain(ws, [reply1])
        truth_before = _read_state(ws)["open_topics"]

        # Nightly dream overwrites the exposure product with a v1 file.
        _write_interests_v1(ws, DAY, [_topic("Nightly only", paths=[f"daily/{DAY}/s1.md"])])

        _touch(ws / "daily" / DAY / "s2.md", "round two")
        reply2 = _reply(follow_ups=[], updates=[])
        _, _, _, context, _ = await _run_chain(ws, [reply2])
        assert context.get("proactive")["interests_written"] is True

        assert _read_state(ws)["open_topics"] == truth_before  # truth source intact
        data = yaml.safe_load(_interests(ws).read_text(encoding="utf-8"))
        assert data["version"] == 2  # exposure restored from the truth source
        assert [t["title"] for t in data["topics"]] == ["持续主题"]

    asyncio.run(run())


def test_refresh_nightly_interleave(tmp_path):
    """Refresh and nightly writes interleave without corrupting state or continuity."""

    async def run():
        ws = tmp_path
        _touch(ws / "daily" / DAY / "s1.md", "round one")
        reply1 = _reply(follow_ups=[_topic("连续主题", confidence=0.9, paths=[f"daily/{DAY}/s1.md"])])
        await _run_chain(ws, [reply1])

        # Nightly dream overwrites interests.yaml with a v1 file.
        _write_interests_v1(ws, DAY, [_topic("Nightly topic", paths=[f"daily/{DAY}/s1.md"])])
        nightly = yaml.safe_load(_interests(ws).read_text(encoding="utf-8"))
        assert nightly.get("version") is None

        # Next refresh round re-renders from the intact truth source.
        _touch(ws / "daily" / DAY / "s2.md", "round two")
        reply2 = _reply(follow_ups=[], updates=[])
        _, _, _, _, responses = await _run_chain(ws, [reply2])
        assert all(r.success for r in responses)
        data = yaml.safe_load(_interests(ws).read_text(encoding="utf-8"))
        assert data["version"] == 2
        assert data["push"] is True
        assert [t["title"] for t in data["topics"]] == ["连续主题"]
        assert [t["title"] for t in _read_state(ws)["open_topics"]] == ["连续主题"]

    asyncio.run(run())


def test_nightly_unit_set_invariant():
    """Dream extract sees identical units whether or not refresh ran first (INV-9)."""

    dream_reply = (
        "```yaml\n"
        "units:\n"
        "  - name: eval-plan-gap\n"
        "    bucket: wiki\n"
        "    summary: The evaluation plan is discussed but never lands.\n"
        f"    paths: [daily/{DAY}/session.md]\n"
        "topics: []\n"
        "```"
    )

    async def dream_units(ws) -> list[dict]:
        catalog = _Catalog()
        wrapper = _AgentWrapper([dream_reply])
        step = DreamExtractStep(scan_days=1, app_context=ApplicationContext(workspace_dir=str(ws)))
        with patch("reme.steps.evolve.dream.extract.refresh_day_index", return_value={}):
            response = await step(
                RuntimeContext(
                    date=DAY,
                    file_catalog=catalog,
                    file_store=_FileStore(ws),
                    agent_wrapper=wrapper,
                    as_llm=_DummyModel(),
                ),
            )
        assert response.success is True
        return response.metadata["dream"]["units"]

    async def run():
        with tempfile.TemporaryDirectory() as tmp_a, tempfile.TemporaryDirectory() as tmp_b:
            ws_a, ws_b = Path(tmp_a), Path(tmp_b)
            for ws in (ws_a, ws_b):
                _touch(ws / "daily" / DAY / "session.md", "same discussion content")

            # Run the full proactive chain on ws_a only.
            reply = _reply(follow_ups=[_topic("主动主题", confidence=0.9, paths=[f"daily/{DAY}/session.md"])])
            await _run_chain(ws_a, [reply])

            units_a = await dream_units(ws_a)
            units_b = await dream_units(ws_b)
            assert units_a == units_b
            assert len(units_a) == 1

    asyncio.run(run())


# ---------------------------------------------------------------------------
# F4 short-circuit, budget, timeout
# ---------------------------------------------------------------------------


def test_llm_timeout(tmp_path):
    """A hanging reply times out, short-circuits the round, and retries next round."""

    async def run():
        ws = tmp_path
        _touch(ws / "daily" / DAY / "session.md", "fresh material")
        wrapper = _AgentWrapper(["never returned"], delay=0.5)
        catalog = _Catalog()
        _, wrapper, catalog, context, responses = await _run_chain(
            ws,
            wrapper=wrapper,
            catalog=catalog,
            extract_kwargs={"llm_timeout_seconds": 0.1},
        )
        assert context.get("proactive_skip")["reason"] == "llm_timeout"
        assert wrapper.calls == 1
        assert not _interests(ws).exists()
        assert catalog.upserts == [] and catalog.dumps == 0
        assert all(r.success for r in responses)

    asyncio.run(run())


def test_parse_failure_retry_then_empty(tmp_path):
    """Unparseable output retries once; still failing yields empty, not error."""

    async def run():
        ws = tmp_path
        _touch(ws / "daily" / DAY / "session.md", "material")
        wrapper = _AgentWrapper(["this is not structured at all", _reply(follow_ups=[], updates=[])])
        _, wrapper, _, context, responses = await _run_chain(ws, wrapper=wrapper)
        assert wrapper.calls == 2
        assert all(r.success for r in responses)
        assert context.get("proactive")["follow_ups"] == []

    asyncio.run(run())


def test_parse_double_failure_keeps_material_uncheckpointed(tmp_path):
    """Two unusable replies skip without checkpoint; material retries next round (audit #1)."""

    async def run():
        ws = tmp_path
        _touch(ws / "daily" / DAY / "session.md", "material")
        wrapper = _AgentWrapper(["garbage one", "garbage two"])
        catalog = _Catalog()
        app = ApplicationContext(workspace_dir=str(ws))
        context = RuntimeContext(date=DAY, file_catalog=catalog, file_store=_FileStore(ws), agent_wrapper=wrapper)
        extract = ProactiveExtractStep(app_context=app)
        resp_extract = await extract(context)
        topics = ProactiveTopicsStep(app_context=app)
        resp_topics = await topics(context)
        finish = ProactiveFinishStep(app_context=app)
        resp_finish = await finish(context)
        assert all(r.success for r in (resp_extract, resp_topics, resp_finish))
        assert wrapper.calls == 2
        assert context.get("proactive_skip") == {"reason": "extract_parse_failed"}
        assert not catalog.upserts  # nothing checkpointed; material stays changed

    asyncio.run(run())


def test_parse_extract_reply_schema_gate():
    """Non-empty replies without contract sections count as parse failures (audit #1)."""
    assert parse_extract_reply("followups:\n  - title: misspelled section\n") == {}
    assert parse_extract_reply("foo: 1") == {}
    assert parse_extract_reply("```yaml\nfollow_ups: []\n```") == {"follow_ups": []}
    assert parse_extract_reply("follow_ups: []\nupdates: []") == {"follow_ups": [], "updates": []}


def test_schema_error_reply_retries(tmp_path):
    """Non-empty YAML with wrong section names retries once like a parse failure (audit #1)."""

    async def run():
        ws = tmp_path
        _touch(ws / "daily" / DAY / "session.md", "material")
        bad = "followups:\n  - title: 拼错段名的话题\n    reason: because\n"
        good = _reply(follow_ups=[_topic("补救话题", paths=[f"daily/{DAY}/session.md"])])
        wrapper = _AgentWrapper([bad, good])
        _, wrapper, _, context, responses = await _run_chain(ws, wrapper=wrapper)
        state = context.get("proactive")
        assert wrapper.calls == 2 + state["refine_calls"]  # 2 stage-1 attempts + refine
        assert all(r.success for r in responses)
        assert [t["title"] for t in state["follow_ups"]] == ["补救话题"]

    asyncio.run(run())


def test_pack_paths_total_budget(tmp_path):
    """max_total_chars keeps the first file and reports how many were omitted (audit #5)."""
    for name in ("a.md", "b.md", "c.md"):
        _touch(tmp_path / name, "x" * 100)
    packed = pack_paths(tmp_path, ["a.md", "b.md", "c.md"], limit_per_file=60000, max_total_chars=150)
    assert "### a.md" in packed
    assert "### b.md" not in packed
    assert "(omitted 2 file(s)" in packed


def test_extract_material_budget(tmp_path):
    """Blob packing honors max_total_chars; newest daily material is packed first (audit #5)."""

    async def run():
        ws = tmp_path
        _touch(ws / "daily" / "2026-08-12" / "big-old.md", "x" * 500)
        _touch(ws / "daily" / DAY / "fresh.md", "fresh material")
        reply = _reply(follow_ups=[_topic("预算话题", paths=[f"daily/{DAY}/fresh.md"])])
        wrapper = _AgentWrapper([reply])
        await _run_chain(ws, wrapper=wrapper, extract_kwargs={"max_total_chars": 200})
        blob = wrapper.last_inputs[0]
        assert "### daily/2026-08-13/fresh.md" in blob  # newest day survives the budget
        assert "### daily/2026-08-12/big-old.md" not in blob  # omitted from the material section
        assert "omitted" in blob

    asyncio.run(run())


# ---------------------------------------------------------------------------
# F5 read interface
# ---------------------------------------------------------------------------


def test_proactive_backward_compat(tmp_path):
    """Default params on a v1 file keep the legacy answer field-for-field."""

    async def run():
        content = (
            "date: 2026-05-28\n"
            "topics:\n"
            "  - title: Retrieval quality\n"
            "    reason: Search behavior changed repeatedly.\n"
            "    evidence: daily/2026-05-28/session.md\n"
        )
        _touch(tmp_path / "daily" / "2026-05-28" / "interests.yaml", content)
        step = ProactiveStep(app_context=ApplicationContext(workspace_dir=str(tmp_path)))
        response = await step(
            RuntimeContext(date="2026-05-28", include_content=True, file_store=_FileStore(tmp_path)),
        )
        assert response.success is True
        assert response.answer == {
            "summary": "Read 1 proactive topic(s) from daily/2026-05-28/interests.yaml",
            "topics": [
                {
                    "title": "Retrieval quality",
                    "reason": "Search behavior changed repeatedly.",
                    "evidence": "daily/2026-05-28/session.md",
                    "paths": [],
                },
            ],
            "content": content,
        }
        assert response.metadata["topics"] == response.answer["topics"]
        assert response.metadata["content"] == content
        assert response.metadata["push"] is True  # additive keys only

    asyncio.run(run())


def test_proactive_answer_omits_unrequested_content(tmp_path):
    """Raw YAML is absent from the primary answer when include_content is false."""

    async def run():
        _touch(tmp_path / "daily" / DAY / "interests.yaml", "topics:\n  - title: Topic\n    reason: Reason\n")
        step = ProactiveStep(app_context=ApplicationContext(workspace_dir=str(tmp_path)))

        response = await step(RuntimeContext(date=DAY, include_content=False, file_store=_FileStore(tmp_path)))

        assert response.success is True
        assert "content" not in response.answer
        assert response.answer["topics"][0]["title"] == "Topic"
        assert response.metadata["content"] == ""

    asyncio.run(run())


def test_proactive_missing_file_is_an_explicit_skip(tmp_path):
    """A missing single-day exposure remains distinguishable without metadata."""

    async def run():
        step = ProactiveStep(app_context=ApplicationContext(workspace_dir=str(tmp_path)))
        response = await step(RuntimeContext(date=DAY, file_store=_FileStore(tmp_path)))

        assert response.success is True
        assert response.answer == f"Skipped: interests file not found at daily/{DAY}/interests.yaml"
        assert response.metadata["skipped"] is True

    asyncio.run(run())


def test_strict_topic_loading_rejects_lossy_fields(tmp_path):
    """Strict legacy loading rejects values that would otherwise be discarded."""
    target = _touch(tmp_path / "interests.yaml", "topics:\n  - title: Topic\n    reason: Reason\n    custom: keep me\n")

    try:
        load_yaml_topics(target, strict=True)
    except ValueError as exc:
        assert "unknown field(s): custom" in str(exc)
    else:
        raise AssertionError("strict topic loading accepted a lossy field")


def test_strict_topic_loading_rejects_invalid_field_types(tmp_path):
    """Strict legacy loading rejects list fields that would otherwise be normalized away."""
    target = _touch(
        tmp_path / "interests.yaml",
        "topics:\n  - title: Topic\n    reason: Reason\n    paths: daily/source.md\n",
    )

    try:
        load_yaml_topics(target, strict=True)
    except ValueError as exc:
        assert "topics[0].paths must be a list of non-empty strings" in str(exc)
    else:
        raise AssertionError("strict topic loading accepted an invalid paths type")


def test_horizon_merge(tmp_path):
    """Horizon>1 reads the truth source filtered by evidence recency (v5 R4)."""

    async def run():
        ws = tmp_path
        id_x, id_y = topic_id("Topic X"), topic_id("Topic Y")
        _write_state(
            ws,
            open_topics=[
                _state_topic("Topic X", day=DAY, topic_id_value=id_x, confidence=0.9),
                _state_topic("Topic Y", day="2026-08-09", topic_id_value=id_y, kind="interest_extend", confidence=0.6),
            ],
        )
        step = ProactiveStep(app_context=ApplicationContext(workspace_dir=str(ws)), horizon_days=3)
        response = await step(RuntimeContext(date=DAY, include_content=False, file_store=_FileStore(ws)))
        assert response.success is True
        assert [t["title"] for t in response.answer["topics"]] == ["Topic X"]  # Y evidence too old
        assert response.metadata["path"] == "daily/_proactive.yaml"

        # A wider horizon includes the older topic again.
        response2 = await step(
            RuntimeContext(date=DAY, include_content=False, file_store=_FileStore(ws), horizon_days=7),
        )
        assert {t["title"] for t in response2.answer["topics"]} == {"Topic X", "Topic Y"}

        # Historical reads are bounded on both sides and cannot expose topics
        # whose evidence is newer than the requested date.
        historical = await step(
            RuntimeContext(date="2026-08-10", include_content=False, file_store=_FileStore(ws), horizon_days=7),
        )
        assert [t["title"] for t in historical.answer["topics"]] == ["Topic Y"]

        # Empty truth source -> skipped, not an error.
        with tempfile.TemporaryDirectory() as tmp:
            empty_ws = Path(tmp)
            response3 = await step(
                RuntimeContext(date=DAY, include_content=False, file_store=_FileStore(empty_ws), horizon_days=3),
            )
            assert response3.success is True
            assert response3.metadata["skipped"] is True

    asyncio.run(run())


def test_min_confidence(tmp_path):
    """min_confidence filters v2 topics; v1 topics fall back to 0.5."""

    async def run():
        ws = tmp_path
        _write_interests_v2(
            ws,
            DAY,
            [
                _state_topic("High topic", confidence=0.7),
                _state_topic("Low topic", kind="interest_extend", confidence=0.3),
            ],
        )
        step = ProactiveStep(app_context=ApplicationContext(workspace_dir=str(ws)), min_confidence=0.5)
        response = await step(RuntimeContext(date=DAY, include_content=False, file_store=_FileStore(ws)))
        assert [t["title"] for t in response.answer["topics"]] == ["High topic"]

        # Default min_confidence=0.4 sits below the 0.5 fallback but above weak 0.3.
        step_default = ProactiveStep(app_context=ApplicationContext(workspace_dir=str(ws)))
        default_response = await step_default(
            RuntimeContext(date=DAY, include_content=False, file_store=_FileStore(ws)),
        )
        assert [t["title"] for t in default_response.answer["topics"]] == ["High topic"]

        with tempfile.TemporaryDirectory() as tmp:
            ws_v1 = Path(tmp)
            _write_interests_v1(ws_v1, DAY, [_topic("Legacy topic", paths=[f"daily/{DAY}/a.md"])])
            step_v1 = ProactiveStep(app_context=ApplicationContext(workspace_dir=str(ws_v1)))
            ok = await step_v1(
                RuntimeContext(date=DAY, include_content=False, file_store=_FileStore(ws_v1), min_confidence=0.5),
            )
            assert [t["title"] for t in ok.answer["topics"]] == ["Legacy topic"]  # fallback 0.5 passes
            filtered = await step_v1(
                RuntimeContext(date=DAY, include_content=False, file_store=_FileStore(ws_v1), min_confidence=0.6),
            )
            assert filtered.answer["topics"] == []

    asyncio.run(run())


# ---------------------------------------------------------------------------
# M2: extends branch + semantic dedup
# ---------------------------------------------------------------------------


def test_extend_from_daily_material(tmp_path):
    """Branch B infers extends from daily material; out-of-M paths are dropped."""

    async def run():
        ws = tmp_path
        _touch(ws / "daily" / DAY / "session.md", "discussion about source attribution")
        reply = _reply(
            follow_ups=[],
            extends=[
                _topic("引用质量评测", confidence=0.5, paths=[f"daily/{DAY}/session.md"]),
                _topic("越界扩展", confidence=0.5, paths=["daily/2026-08-01/absent.md"]),
            ],
        )
        _, wrapper, _, context, _ = await _run_chain(ws, [reply])
        assert "extends" in wrapper.last_inputs[0]  # extends section rendered
        state = context.get("proactive")
        assert len(state["extends"]) == 1
        extend = state["extends"][0]
        assert extend["kind"] == "interest_extend"
        assert f"daily/{DAY}/session.md" in extend["paths"]
        assert state["dropped_missing"] == 1
        data = yaml.safe_load(_interests(ws).read_text(encoding="utf-8"))
        assert [t["title"] for t in data["topics"]] == ["引用质量评测"]

    asyncio.run(run())


def test_extends_disabled_m1_shape():
    """extends_enabled=False restores the M1 prompt shape and ignores extends output."""

    async def run_once(extends_enabled: bool):
        with tempfile.TemporaryDirectory() as tmp:
            ws = Path(tmp)
            _touch(ws / "daily" / DAY / "s1.md", "material")
            reply = _reply(
                follow_ups=[_topic("Follow Up Topic", confidence=0.9, paths=[f"daily/{DAY}/s1.md"])],
                extends=[_topic("Extends Topic", confidence=0.9, paths=[f"daily/{DAY}/s1.md"])],
            )
            _, wrapper, _, context, _ = await _run_chain(
                ws,
                [reply],
                extract_kwargs={"extends_enabled": extends_enabled},
            )
            return wrapper, context.get("proactive")

    async def run():
        wrapper_on, state_on = await run_once(True)
        assert "extends:" in wrapper_on.last_system_prompts[0]  # M2 default renders branch B
        assert [t["title"] for t in state_on["extends"]] == ["Extends Topic"]

        wrapper_off, state_off = await run_once(False)
        system_off = wrapper_off.last_system_prompts[0]
        assert "extends" not in system_off.replace("interest_extend", "")  # M1 prompt shape
        assert state_off["extends"] == []  # extends entries in the reply are ignored
        assert [t["title"] for t in state_off["topics_out"]] == ["Follow Up Topic"]

    asyncio.run(run())


def test_semantic_dedup(tmp_path):
    """With embeddings, similarity below known_threshold is kept; no LLM tier (v5 R2)."""

    async def run():
        ws = tmp_path
        _write_state(ws, open_topics=[_state_topic("Gray Comparison", kind="interest_extend")])
        _touch(ws / "daily" / DAY / "session.md", "material")
        reply = _reply(
            follow_ups=[
                _topic("Candidate Topic", confidence=0.8, paths=[f"daily/{DAY}/session.md"]),
            ],
        )
        fake = _FakeEmbedding(
            {
                # v5.2 scheme: candidate and known sides both embed "title + CJK period + reason"
                "Candidate Topic。because": [1.0, 0.0],
                "Gray Comparison。seeded": [0.8, 0.6],  # cos = 0.800 < 0.85 -> kept, no second tier
            },
        )
        _, wrapper, _, context, _ = await _run_chain(ws, [reply], extra_ctx={"as_embedding": fake})
        state = context.get("proactive")
        assert state["dropped_known"] == 0
        assert state["dropped_duplicate"] == 0
        assert [c["title"] for c in state["candidates"]] == ["Candidate Topic"]
        assert fake.calls >= 1
        assert wrapper.calls == 1 + state["refine_calls"]  # extract (+refine); topics is pure computation

    asyncio.run(run())


def test_known_drop(tmp_path):
    """sim >= known_threshold drops the candidate as already known."""

    async def run():
        ws = tmp_path
        _write_state(ws, open_topics=[_state_topic("Known Topic", kind="interest_extend")])
        _touch(ws / "daily" / DAY / "session.md", "material")
        reply = _reply(
            follow_ups=[_topic("Fresh Candidate", confidence=0.8, paths=[f"daily/{DAY}/session.md"])],
        )
        fake = _FakeEmbedding(
            {
                "Fresh Candidate。because": [1.0, 0.0],
                "Known Topic。seeded": [1.0, 0.0],  # identical -> sim 1.0 >= 0.85
            },
        )
        _, _, _, context, _ = await _run_chain(ws, [reply], extra_ctx={"as_embedding": fake})
        state = context.get("proactive")
        assert state["dropped_known"] == 1
        assert state["candidates"] == []
        data = yaml.safe_load(_interests(ws).read_text(encoding="utf-8"))
        assert [t["title"] for t in data["topics"]] == ["Known Topic"]

    asyncio.run(run())


# ---------------------------------------------------------------------------
# F2.5/F2.6 plan + agenda steps
# ---------------------------------------------------------------------------


def _plan_reply(cards: list[dict]) -> str:
    body = yaml.safe_dump({"cards": cards}, allow_unicode=True, sort_keys=False)
    return f"```yaml\n{body}```"


def _agenda_reply(agenda: list[dict], suppressed: list[dict] | None = None) -> str:
    doc = {"agenda": agenda}
    if suppressed is not None:
        doc["suppressed"] = suppressed
    body = yaml.safe_dump(doc, allow_unicode=True, sort_keys=False)
    return f"```yaml\n{body}```"


def _card(topic_id_value, index=0, scenario="resume_task"):
    return {
        "topic_id": topic_id_value,
        "scenario_type": scenario,
        "opener": f"对了，上次那件事还差一步（{index}），要不要现在动起来？",
        "next_action": f"执行第 {index} 步",
        "preconditions": [],
        "delivery": "in_conversation",
    }


async def _run_full_chain(
    ws,
    replies=None,
    *,
    wrapper=None,
    catalog=None,
    app=None,
    extract_kwargs=None,
    topics_kwargs=None,
    plan_kwargs=None,
    agenda_kwargs=None,
    extra_ctx=None,
    context=None,
):
    app = app or ApplicationContext(workspace_dir=str(ws))
    wrapper = wrapper or _AgentWrapper(replies or [])
    catalog = catalog or _Catalog()
    context = context or RuntimeContext(
        date=DAY,
        file_catalog=catalog,
        file_store=_FileStore(ws),
        agent_wrapper=wrapper,
        **(extra_ctx or {}),
    )
    extract = ProactiveExtractStep(app_context=app, **(extract_kwargs or {}))
    resp_extract = await extract(context)
    topics = ProactiveTopicsStep(app_context=app, **(topics_kwargs or {}))
    resp_topics = await topics(context)
    plan = ProactivePlanStep(app_context=app, **(plan_kwargs or {}))
    resp_plan = await plan(context)
    agenda = ProactiveAgendaStep(app_context=app, **(agenda_kwargs or {}))
    resp_agenda = await agenda(context)
    finish = ProactiveFinishStep(app_context=app)
    resp_finish = await finish(context)
    return app, wrapper, catalog, context, (resp_extract, resp_topics, resp_plan, resp_agenda, resp_finish)


def test_plan_agenda_chain_e2e(tmp_path):
    """extract -> topics -> plan -> agenda -> finish writes the enriched v2 file."""

    async def run():
        ws = tmp_path
        _touch(ws / "daily" / DAY / "session.md", "we discussed the eval plan")
        extract_reply = _reply(
            follow_ups=[_topic("记忆检索的可解释性评估", confidence=0.9, paths=[f"daily/{DAY}/session.md"])],
            extends=[_topic("多跳 RAG 综述", confidence=0.5, paths=[f"daily/{DAY}/session.md"])],
        )
        fid, eid = topic_id("记忆检索的可解释性评估"), topic_id("多跳 RAG 综述")
        plan_reply = _plan_reply(
            [
                {
                    "topic_id": fid,
                    "scenario_type": "resume_task",
                    "opener": "对了，上次那个评估还差一步，要不要现在跑起来？",
                    "next_action": "运行评估脚本",
                    "preconditions": ["数据集已就绪"],
                    "delivery": "agenda_item",
                },
                _card(eid, 1, scenario="explore_interest"),
            ],
        )
        agenda_text = _agenda_reply(
            [{"topic_id": fid, "order_reason": "卡点已解除，今天最适合收尾"}],
            [{"topic_id": eid, "reason": f"merged into {fid}"}],
        )
        _, wrapper, _, context, resps = await _run_full_chain(ws, [extract_reply, plan_reply, agenda_text])
        assert all(r.success for r in resps)
        state = context.get("proactive")
        assert wrapper.calls == 3 + state["refine_calls"]  # extract + plan + agenda
        assert state["llm_calls"] == 1 + state["refine_calls"]  # extract only
        assert state["plan_llm_calls"] == 2  # plan + agenda
        assert state["push"] is True

        data = yaml.safe_load(_interests(ws).read_text(encoding="utf-8"))
        assert data["version"] == 2
        assert data["push"] is True
        assert len(data["agenda"]) == 1
        item = data["agenda"][0]
        assert item["topic_id"] == fid
        assert item["title"] == "记忆检索的可解释性评估"
        assert item["scenario_type"] == "resume_task"
        assert item["opener"].startswith("对了")
        assert item["next_action"] == "运行评估脚本"
        assert item["preconditions"] == ["数据集已就绪"]
        assert item["delivery"] == "agenda_item"
        assert item["linked_memory"] == [f"daily/{DAY}/session.md"]
        assert item["order_reason"] == "卡点已解除，今天最适合收尾"
        assert data["suppressed"] == [
            {"topic_id": eid, "title": "多跳 RAG 综述", "reason": f"merged into {fid}"},
        ]
        assert {t["id"] for t in data["topics"]} == {fid, eid}  # topics view intact

    asyncio.run(run())


def test_plan_agenda_fallback_without_llm(tmp_path):
    """Unusable plan/agenda replies degrade to fallback cards + deterministic agenda."""

    async def run():
        ws = tmp_path
        _touch(ws / "daily" / DAY / "session.md", "material")
        extract_reply = _reply(
            follow_ups=[
                _topic("任务一", confidence=0.9, paths=[f"daily/{DAY}/session.md"]),
                _topic("任务二", confidence=0.7, paths=[f"daily/{DAY}/session.md"]),
            ],
        )
        # Only the extract reply is scripted; plan/agenda receive "" and must degrade.
        _, wrapper, _, context, resps = await _run_full_chain(ws, [extract_reply])
        assert all(r.success for r in resps)
        state = context.get("proactive")
        assert wrapper.calls == 3 + state["refine_calls"]
        assert state["plan_llm_calls"] == 2  # attempted but unusable
        assert len(state["scenario_cards"]) == 2

        data = yaml.safe_load(_interests(ws).read_text(encoding="utf-8"))
        assert data["push"] is True
        assert [i["title"] for i in data["agenda"]] == ["任务一", "任务二"]
        assert all(i["order_reason"] == "deterministic fallback: freshness/confidence order" for i in data["agenda"])
        assert data["suppressed"] == []
        assert all(i["scenario_type"] == "resume_task" for i in data["agenda"])
        assert all(i["opener"] for i in data["agenda"])

    asyncio.run(run())


def test_agenda_no_candidates_keeps_topics_render(tmp_path):
    """Without push candidates plan/agenda skip and the topics-only render stays."""

    async def run():
        ws = tmp_path
        _touch(ws / "daily" / DAY / "session.md", "material")
        extract_reply = _reply(
            follow_ups=[_topic("低置信话题", confidence=0.3, paths=[f"daily/{DAY}/session.md"])],
        )
        _, wrapper, _, context, resps = await _run_full_chain(ws, [extract_reply])
        assert all(r.success for r in resps)
        state = context.get("proactive")
        assert wrapper.calls == 1 + state["refine_calls"]  # extract only; plan/agenda made zero calls
        assert state["push"] is False
        assert state["push_candidates"] == []
        assert state["scenario_cards"] == []
        assert state["agenda"] == []
        assert state["suppressed"] == []

        data = yaml.safe_load(_interests(ws).read_text(encoding="utf-8"))
        assert data["push"] is False
        assert "agenda" not in data and "suppressed" not in data

    asyncio.run(run())


def test_agenda_single_candidate_skips_llm(tmp_path):
    """Exactly one candidate is auto-agenda without an agenda LLM call."""

    async def run():
        ws = tmp_path
        _touch(ws / "daily" / DAY / "session.md", "material")
        extract_reply = _reply(
            follow_ups=[_topic("唯一候选", confidence=0.9, paths=[f"daily/{DAY}/session.md"])],
        )
        fid = topic_id("唯一候选")
        plan_reply = _plan_reply([_card(fid, scenario="answer_pending")])
        _, wrapper, _, context, _ = await _run_full_chain(ws, [extract_reply, plan_reply])
        state = context.get("proactive")
        assert wrapper.calls == 2 + state["refine_calls"]  # extract + plan only
        assert state["plan_llm_calls"] == 1

        data = yaml.safe_load(_interests(ws).read_text(encoding="utf-8"))
        assert len(data["agenda"]) == 1
        assert data["agenda"][0]["topic_id"] == fid
        assert data["agenda"][0]["order_reason"] == "single push candidate; auto-agenda without LLM"
        assert data["suppressed"] == []

    asyncio.run(run())


def test_agenda_merge_and_full_accounting(tmp_path):
    """LLM merges two entrances of one matter; every candidate is accounted for."""

    async def run():
        ws = tmp_path
        _touch(ws / "daily" / DAY / "session.md", "material")
        titles = ["GPU 排期确认", "跑 benchmark 的 GPU 档期", "多跳引用可信度"]
        extract_reply = _reply(
            follow_ups=[
                _topic(titles[0], confidence=0.9, paths=[f"daily/{DAY}/session.md"]),
                _topic(titles[1], confidence=0.7, paths=[f"daily/{DAY}/session.md"]),
                _topic(titles[2], confidence=0.9, paths=[f"daily/{DAY}/session.md"]),
            ],
        )
        ids = [topic_id(t) for t in titles]
        plan_reply = _plan_reply([_card(tid, i) for i, tid in enumerate(ids)])
        agenda_text = _agenda_reply(
            [{"topic_id": ids[0], "order_reason": "主入口"}, {"topic_id": ids[2], "order_reason": "独立事项"}],
            [{"topic_id": ids[1], "reason": f"merged into {ids[0]}"}],
        )
        _, _, _, context, _ = await _run_full_chain(ws, [extract_reply, plan_reply, agenda_text])
        state = context.get("proactive")
        assert state["push"] is True

        data = yaml.safe_load(_interests(ws).read_text(encoding="utf-8"))
        assert [i["topic_id"] for i in data["agenda"]] == [ids[0], ids[2]]
        assert data["suppressed"] == [
            {"topic_id": ids[1], "title": titles[1], "reason": f"merged into {ids[0]}"},
        ]
        accounted = {i["topic_id"] for i in data["agenda"]} | {x["topic_id"] for x in data["suppressed"]}
        assert accounted == set(ids)

    asyncio.run(run())


def test_agenda_invalid_ids_and_missing_accounting(tmp_path):
    """Bogus agenda ids are dropped; candidates the LLM forgot are still suppressed."""

    async def run():
        ws = tmp_path
        _touch(ws / "daily" / DAY / "session.md", "material")
        titles = ["候选甲", "候选乙"]
        extract_reply = _reply(
            follow_ups=[
                _topic(titles[0], confidence=0.9, paths=[f"daily/{DAY}/session.md"]),
                _topic(titles[1], confidence=0.7, paths=[f"daily/{DAY}/session.md"]),
            ],
        )
        ids = [topic_id(t) for t in titles]
        plan_reply = _plan_reply([_card(tid, i) for i, tid in enumerate(ids)])
        agenda_text = _agenda_reply(
            [{"topic_id": "nonexistent01", "order_reason": "bogus"}, {"topic_id": ids[0], "order_reason": "real"}],
            [],
        )
        _, _, _, _, _ = await _run_full_chain(ws, [extract_reply, plan_reply, agenda_text])
        data = yaml.safe_load(_interests(ws).read_text(encoding="utf-8"))
        assert [i["topic_id"] for i in data["agenda"]] == [ids[0]]
        assert data["suppressed"] == [
            {"topic_id": ids[1], "title": titles[1], "reason": "not selected for today's agenda"},
        ]

    asyncio.run(run())


def test_plan_over_budget_suppressed(tmp_path):
    """Candidates beyond max_plan_topics stay card-less and are suppressed with reason."""

    async def run():
        ws = tmp_path
        _touch(ws / "daily" / DAY / "session.md", "material")
        titles = ["高置信一", "高置信二", "高置信三"]
        extract_reply = _reply(
            follow_ups=[
                _topic(titles[0], confidence=0.9, paths=[f"daily/{DAY}/session.md"]),
                _topic(titles[1], confidence=0.8, paths=[f"daily/{DAY}/session.md"]),
                _topic(titles[2], confidence=0.7, paths=[f"daily/{DAY}/session.md"]),
            ],
        )
        ids = [topic_id(t) for t in titles]
        # deterministic sort order: confidence desc -> ids[0], ids[1] selected; ids[2] over budget
        plan_reply = _plan_reply([_card(ids[0], 0), _card(ids[1], 1)])
        agenda_text = _agenda_reply(
            [{"topic_id": ids[0], "order_reason": "first"}, {"topic_id": ids[1], "order_reason": "second"}],
            [],
        )
        _, _, _, context, _ = await _run_full_chain(
            ws,
            [extract_reply, plan_reply, agenda_text],
            plan_kwargs={"max_plan_topics": 2},
        )
        state = context.get("proactive")
        assert len(state["scenario_cards"]) == 2
        data = yaml.safe_load(_interests(ws).read_text(encoding="utf-8"))
        assert [i["topic_id"] for i in data["agenda"]] == [ids[0], ids[1]]
        assert data["suppressed"] == [
            {
                "topic_id": ids[2],
                "title": titles[2],
                "reason": "over plan budget: not expanded into a scenario card",
            },
        ]

    asyncio.run(run())


def test_read_side_agenda_passthrough(tmp_path):
    """F5 read passes the agenda through, filtering resolved/unknown/low-confidence ids."""

    async def run():
        ws = tmp_path
        id_a, id_b = topic_id("议程主题"), topic_id("已解决主题")
        payload = {
            "version": 2,
            "date": DAY,
            "generated_at": f"{DAY}T09:00:00",
            "push": True,
            "topics": [
                _state_topic("议程主题", topic_id_value=id_a, confidence=0.9),
                _state_topic("已解决主题", topic_id_value=id_b, confidence=0.9),
            ],
            "agenda": [
                {"topic_id": id_a, "opener": "对了，那件事可以动了"},
                {"topic_id": id_b, "opener": "resolved, must be filtered"},
                {"topic_id": "unknown000000", "opener": "not in kept"},
            ],
            "suppressed": [],
        }
        _touch(_interests(ws, DAY), yaml.safe_dump(payload, allow_unicode=True, sort_keys=False))
        _write_state(ws, resolved=[{"id": id_b, "title": "已解决主题", "resolved_at": DAY, "evidence": ""}])

        step = ProactiveStep(app_context=ApplicationContext(workspace_dir=str(ws)))
        response = await step(RuntimeContext(date=DAY, include_content=False, file_store=_FileStore(ws)))
        assert [t["title"] for t in response.answer["topics"]] == ["议程主题"]
        assert response.answer["agenda"] == [{"topic_id": id_a, "opener": "对了，那件事可以动了"}]

    asyncio.run(run())


# ---------------------------------------------------------------------------
# F2.7 personal profile (digest/personal) + configured daily_dir
# ---------------------------------------------------------------------------


def test_strip_frontmatter_and_profile_loader(tmp_path):
    """digest/personal/*.md -> profile block; per-file budget; fallbacks."""
    ws = tmp_path
    personal = ws / "digest" / "personal"
    personal.mkdir(parents=True)
    (personal / "user-profile.md").write_text(
        "---\ndescription: 用户张三的身份画像\nname: user-profile\n---\n\n用户张三，偏好西瓜。\n",
        encoding="utf-8",
    )
    (personal / "user-position.md").write_text(
        "---\ndescription: 持仓约束\nname: user-position\n---\n\n持有小米，浮亏较深。\n",
        encoding="utf-8",
    )

    meta, body = strip_frontmatter((personal / "user-profile.md").read_text(encoding="utf-8"))
    assert meta["name"] == "user-profile"
    assert body == "用户张三，偏好西瓜。"

    block = load_personal_profile_block(ws, "digest", 4000)
    assert "### user-profile" in block and "### user-position" in block
    assert "用户张三的身份画像" in block and "偏好西瓜" in block
    assert "持仓约束" in block

    tiny = load_personal_profile_block(ws, "digest", 300)
    assert len(tiny) <= 300

    # absent personal dir -> legacy single-file fallback -> sentinel
    empty_ws = tmp_path / "empty"
    empty_ws.mkdir()
    assert load_personal_profile_block(empty_ws, "digest", 1000) == "(no user profile)"
    (empty_ws / "profile.md").write_text("legacy profile", encoding="utf-8")
    assert load_personal_profile_block(empty_ws, "digest", 1000, "profile.md") == "legacy profile"
    assert load_personal_profile_block(empty_ws, "digest", 0, "profile.md") == "(no user profile)"


def test_chain_with_configured_daily_dir(tmp_path):
    """The full chain honours daily_dir=memory (paths, truth source, prompts)."""

    async def run():
        ws = tmp_path
        _touch(ws / "memory" / DAY / "session.md", "小米财报讨论")
        _touch(
            ws / "digest" / "personal" / "user-profile.md",
            "---\ndescription: 张三画像\nname: user-profile\n---\n\n张三关注小米与中概股。\n",
        )
        extract_reply = _reply(
            follow_ups=[_topic("小米Q2财报跟进", confidence=0.9, paths=[f"memory/{DAY}/session.md"])],
        )
        fid = topic_id("小米Q2财报跟进")
        plan_reply = _plan_reply([_card(fid)])
        agenda_text = _agenda_reply([{"topic_id": fid, "order_reason": "only candidate"}], [])

        app = ApplicationContext(workspace_dir=str(ws), daily_dir="memory", digest_dir="digest")
        wrapper = _AgentWrapper([extract_reply, plan_reply, agenda_text])
        catalog = _Catalog()
        context = RuntimeContext(date=DAY, file_catalog=catalog, file_store=_FileStore(ws), agent_wrapper=wrapper)
        extract = ProactiveExtractStep(app_context=app)
        resp_extract = await extract(context)
        topics = ProactiveTopicsStep(app_context=app)
        resp_topics = await topics(context)
        plan = ProactivePlanStep(app_context=app)
        resp_plan = await plan(context)
        agenda = ProactiveAgendaStep(app_context=app)
        resp_agenda = await agenda(context)
        finish = ProactiveFinishStep(app_context=app)
        resp_finish = await finish(context)
        assert all(r.success for r in (resp_extract, resp_topics, resp_plan, resp_agenda, resp_finish))

        state = context.get("proactive")
        assert state["daily_dir"] == "memory"
        assert state["changed_paths"] == [f"memory/{DAY}/session.md"]
        # truth source + exposure land under memory/, not daily/
        assert (ws / "memory" / "_proactive.yaml").is_file()
        data = yaml.safe_load((ws / "memory" / DAY / "interests.yaml").read_text(encoding="utf-8"))
        assert data["push"] is True
        assert [t["id"] for t in data["topics"]] == [fid]
        assert not (ws / "daily").exists()
        # profile + configured daily dir reached the prompts
        assert any("张三画像" in inp or "张三关注小米" in inp for inp in wrapper.last_inputs)
        assert any(f"memory/{DAY}" in inp for inp in wrapper.last_inputs)
        assert any("memory/<date>" in sp for sp in wrapper.last_system_prompts)

    asyncio.run(run())


def test_finish_defers_paths_modified_during_round(tmp_path):
    """Files edited while the LLM calls run are left for the next round (audit 3)."""

    async def run():
        ws = tmp_path
        _touch(ws / "daily" / DAY / "a.md", "first note")
        note_b = _touch(ws / "daily" / DAY / "b.md", "second note")
        reply = _reply(follow_ups=[], extends=[])
        app = ApplicationContext(workspace_dir=str(ws))
        wrapper = _AgentWrapper([reply])
        catalog = _Catalog()
        context = RuntimeContext(date=DAY, file_catalog=catalog, file_store=_FileStore(ws), agent_wrapper=wrapper)

        resp_extract = await ProactiveExtractStep(app_context=app)(context)
        assert resp_extract.success is True
        state = context.get("proactive")
        assert set(state["changed_paths"]) == {f"daily/{DAY}/a.md", f"daily/{DAY}/b.md"}
        assert state["changed_mtimes"][f"daily/{DAY}/b.md"] == note_b.stat().st_mtime

        # The user edits b.md while plan/agenda LLM calls are still running.
        note_b.write_text("second note, edited mid-round", encoding="utf-8")
        future = time.time() + 5
        os.utime(note_b, (future, future))

        resp_finish = await ProactiveFinishStep(app_context=app)(context)
        assert resp_finish.success is True
        final_state = context.get("proactive")
        assert final_state["checkpoint_paths"] == [f"daily/{DAY}/a.md"]
        assert "deferred 1" in resp_finish.answer
        assert {n.path for n in catalog.upserts} == {f"daily/{DAY}/a.md"}

    asyncio.run(run())


def test_read_single_corrupt_yaml_is_read_only_failure(tmp_path):
    """Corrupt interests.yaml: the read fails loudly and never moves the file (audit 2)."""

    async def run():
        ws = tmp_path
        corrupt = "topics: [unclosed"
        path = _touch(ws / "daily" / DAY / "interests.yaml", corrupt)
        step = ProactiveStep(app_context=ApplicationContext(workspace_dir=str(ws)))
        response = await step(RuntimeContext(date=DAY, file_store=_FileStore(ws)))
        assert response.success is False
        assert response.metadata.get("error")
        assert path.is_file()
        assert path.read_text(encoding="utf-8") == corrupt
        assert not list(path.parent.glob("interests.corrupt-*"))

    asyncio.run(run())


def test_carry_forward_expiry_boundary_consistent(tmp_path):
    """Age == carry_forward_days is over-age in both prompt loading and trim (audit 9)."""

    async def run():
        ws = tmp_path
        edge_day = (dt.date.fromisoformat(DAY) - dt.timedelta(days=14)).isoformat()
        _write_state(ws, open_topics=[_state_topic("Edge Topic", day=edge_day)])
        state_file, _ = load_state(ws)
        carry_all, _ = await load_carry_forward(ws, state_file, DAY, 14, 20)
        assert carry_all == []

        trim_state_file(state_file, DAY, 14)
        assert state_file.open_topics == []

    asyncio.run(run())


# ---------------------------------------------------------------------------
# Four-kind extraction: follow_up / interest_extend / reminder / new_idea
# ---------------------------------------------------------------------------


def test_extract_prompt_covers_the_four_branches():
    """All four branches render by default; disabling one removes its whole contract."""

    async def run_once(**extract_kwargs):
        with tempfile.TemporaryDirectory() as tmp:
            ws = Path(tmp)
            _touch(ws / "daily" / DAY / "s1.md", "material")
            _, wrapper, _, _context, _ = await _run_chain(ws, [_reply(follow_ups=[])], extract_kwargs=extract_kwargs)
            return wrapper.last_system_prompts[0], wrapper.last_inputs[0]

    async def run():
        system_on, user_on = await run_once()
        for heading in (
            "Branch A - follow_ups",
            "Branch B - extends",
            "Branch C - reminders",
            "Branch D - new_ideas",
        ):
            assert heading in system_on
        for section in ("follow_ups:", "extends:", "reminders:", "new_ideas:", "updates:", "schedule_updates:"):
            assert section in system_on  # output shape lists every active section
        assert "now (current clock" in user_on
        assert "schedules (long-lived registry" in user_on
        assert "capability_inventory" in user_on

        system_off, user_off = await run_once(
            extends_enabled=False,
            reminders_enabled=False,
            new_ideas_enabled=False,
        )
        for heading in ("Branch B", "Branch C", "Branch D"):
            assert heading not in system_off
        for section in ("extends:", "reminders:", "new_ideas:", "schedule_updates:"):
            assert section not in system_off
        assert "[" not in system_off.split("## Output format")[-1]  # no unrendered flag leaked
        assert "schedules" not in user_off
        assert "capability_inventory" not in user_off

    asyncio.run(run())


def test_kind_caps_and_new_idea_kind(tmp_path):
    """Discovery kinds are capped and tagged; reminders are not capped."""

    async def run():
        ws = tmp_path
        _touch(ws / "daily" / DAY / "session.md", "material")
        path = f"daily/{DAY}/session.md"
        reply = _reply(
            follow_ups=[_topic(f"任务{i}", paths=[path]) for i in range(4)],
            new_ideas=[_topic(f"可以委托的事{i}", paths=[path]) for i in range(3)],
            reminders=[_reminder(f"日程{i}", due_at="2026-08-14", paths=[path]) for i in range(4)],
        )
        with patch(EXTRACT_CLOCK, return_value=NOW):
            _, _, _, context, _ = await _run_chain(
                ws,
                [reply],
                extract_kwargs={"max_follow_ups": 2, "max_new_ideas": 1},
            )
        state = context.get("proactive")
        assert [t["title"] for t in state["follow_ups"]] == ["任务0", "任务1"]
        assert all(t["kind"] == "follow_up" for t in state["follow_ups"])
        assert [t["title"] for t in state["new_ideas"]] == ["可以委托的事0"]
        assert state["new_ideas"][0]["kind"] == "new_idea"
        assert len(state["reminders"]) == 4  # uncapped
        assert all(t["kind"] == "reminder" for t in state["reminders"])
        assert state["dropped_over_cap"] == 2 + 2  # follow_ups + new_ideas

    asyncio.run(run())


def test_reminder_clock_only_round(tmp_path):
    """A due schedule reaches today's agenda with zero extract LLM calls and no material."""

    async def run():
        ws = tmp_path
        _write_state(ws, schedules=[_schedule("提交季度报告", "2026-08-14T09:00")])
        with patch(EXTRACT_CLOCK, return_value=NOW):
            _, wrapper, catalog, context, resps = await _run_full_chain(ws, [])
        assert all(r.success for r in resps)
        state = context.get("proactive")
        assert state["reminder_only"] is True
        assert state["llm_calls"] == 0  # extract made no call; the single call is plan's
        assert wrapper.calls == 1
        assert [r["title"] for r in state["reminders"]] == ["提交季度报告"]
        assert state["reminders"][0]["trigger"] == "due_soon"
        assert state["push"] is True
        assert catalog.upserts == [] and catalog.dumps == 0  # no material to checkpoint

        data = yaml.safe_load(_interests(ws).read_text(encoding="utf-8"))
        assert data["push"] is True
        assert [item["title"] for item in data["agenda"]] == ["提交季度报告"]
        item = data["agenda"][0]
        assert item["kind"] == "reminder"
        assert item["scenario_type"] == "prepare_upcoming"
        assert item["due_at"] == "2026-08-14T09:00"  # timed schedules keep their clock time
        assert item["schedule_id"] == topic_id("提交季度报告")
        assert item["order_reason"].startswith("reminder pinned")
        assert data["topics"][0]["kind"] == "reminder"

    asyncio.run(run())


def test_pin_reason_names_the_actual_cause():
    """A pinned reminder's audit trail says whether the clock or freshness pinned it."""
    day = "2026-10-12"
    overdue = _pin_reason({"due_at": "2026-10-06", "first_seen": "2026-09-19"}, day)
    assert overdue == "reminder pinned: a due schedule always enters today's agenda (due 2026-10-06)"
    due_today = _pin_reason({"due_at": "2026-10-12T18:00", "first_seen": day}, day)
    assert due_today.startswith("reminder pinned: a due schedule")
    fresh = _pin_reason({"due_at": "2026-11-06", "first_seen": day}, day)
    assert fresh == "reminder pinned: a new reminder always enters today's agenda (due 2026-11-06)"
    carried = _pin_reason({"due_at": "2026-11-06", "first_seen": "2026-09-19"}, day)
    assert carried.startswith("reminder pinned: an open reminder")
    undated = _pin_reason({"first_seen": day}, day)
    assert undated.endswith("today's agenda") and "(due" not in undated
    # Without an agenda date nothing can be claimed about the clock.
    assert _pin_reason({"due_at": "2026-11-06"}, "").startswith("reminder pinned: an open reminder")


def test_reminder_registry_maintenance_and_anchoring(tmp_path):
    """schedule_updates maintain the registry; unanchored changes are rejected."""

    async def run():
        ws = tmp_path
        _touch(ws / "daily" / DAY / "session.md", "用户说评审会改期，下周三要牙医复诊")
        _write_state(
            ws,
            schedules=[
                _schedule("季度评审会", "2026-08-20T14:00", schedule_id="sch-1"),
                _schedule("旧体检", "2026-08-18", schedule_id="sch-2"),
            ],
        )
        path = f"daily/{DAY}/session.md"
        reply = _reply(
            reminders=[_reminder("牙医复诊", due_at="2026年8月19日 15:00", paths=[path])],
            schedule_updates=[
                {
                    "id": "sch-1",
                    "action": "reschedule",
                    "due_at": "2026-08-21T10:00",
                    "reason": "改期",
                    "evidence": path,
                },
                {"id": "sch-2", "action": "done", "reason": "已完成", "evidence": "daily/2026-01-01/old.md"},
                {"id": "", "action": "add", "title": "牙医复诊", "due_at": "2026-08-19T15:00", "evidence": path},
            ],
        )
        with patch(EXTRACT_CLOCK, return_value=NOW):
            _, _, _, context, _ = await _run_full_chain(ws, [reply])
        state = context.get("proactive")
        assert state["schedules_added"] == 1
        assert state["schedules_updated"] == 1
        assert state["schedules_closed"] == 0  # the unanchored `done` was rejected
        assert state["reminders"][0]["due_at"] == "2026-08-19T15:00"  # CJK due normalized

        registry = {entry["id"]: entry for entry in _read_state(ws)["schedules"]}
        assert registry["sch-1"]["due_at"] == "2026-08-21T10:00"
        assert registry["sch-1"]["history"][-1]["action"] == "reschedule"
        assert registry["sch-1"]["history"][-1]["from"] == "2026-08-20T14:00"
        assert registry["sch-2"]["status"] == "open"  # untouched
        added = registry[topic_id("牙医复诊")]
        assert added["due_at"] == "2026-08-19T15:00"
        assert added["history"][-1]["action"] == "add"
        assert added["evidence"] == path
        # ``schedules_out`` is the slim open-registry view published in metadata.
        assert {entry["id"] for entry in state["schedules_out"]} == set(registry)
        assert all("history" not in entry and "paths" not in entry for entry in state["schedules_out"])

    asyncio.run(run())


def test_reminder_registration_adopts_the_explicit_entry(tmp_path):
    """One notice under two titles registers once: the derived channel adopts the explicit id."""

    async def run():
        ws = tmp_path
        _touch(ws / "daily" / DAY / "session.md", "航司通知 MU5137 起飞时间变更")
        path = f"daily/{DAY}/session.md"
        reply = _reply(
            reminders=[_reminder("MU5137 航班起飞时间调整", due_at="2026-08-19T15:30", paths=[path])],
            schedule_updates=[
                {
                    "id": "",
                    "action": "add",
                    "title": "MU5137 上海虹桥-北京首都航班",
                    "due_at": "2026-08-19T15:30",
                    "evidence": path,
                },
            ],
        )
        with patch(EXTRACT_CLOCK, return_value=NOW):
            _, _, _, context, _ = await _run_full_chain(ws, [reply])
        state = context.get("proactive")
        assert state["schedules_added"] == 1  # the explicit one; no twin behind a second title
        registry = {entry["id"]: entry for entry in _read_state(ws)["schedules"]}
        assert len(registry) == 1
        kept = registry[topic_id("MU5137 上海虹桥-北京首都航班")]
        assert kept["title"] == "MU5137 上海虹桥-北京首都航班"
        assert [h["action"] for h in kept["history"]] == ["add"]
        # The reminder republishes the adopted id, so the topic and the agenda
        # card cannot point at a schedule that was never registered.
        assert state["reminders"][0]["schedule_id"] == kept["id"]
        agenda = yaml.safe_load(_interests(ws).read_text(encoding="utf-8"))["agenda"]
        assert [item["schedule_id"] for item in agenda] == [kept["id"]]

    asyncio.run(run())


def test_one_notice_two_matters_keeps_both_schedules(tmp_path):
    """Distinct dues from one file are distinct matters, so adoption must not merge them."""

    async def run():
        ws = tmp_path
        _touch(ws / "daily" / DAY / "session.md", "航司通知 MU5137 起飞时间变更，改签需今天前办理")
        path = f"daily/{DAY}/session.md"
        reply = _reply(
            reminders=[
                _reminder("MU5137 航班起飞时间调整", due_at="2026-08-19T15:30", paths=[path]),
                _reminder("MU5137 改签决策截止", due_at="2026-08-14T18:00", paths=[path]),
            ],
            schedule_updates=[
                {
                    "id": "",
                    "action": "add",
                    "title": "MU5137 上海虹桥-北京首都航班",
                    "due_at": "2026-08-19T15:30",
                    "evidence": path,
                },
            ],
        )
        with patch(EXTRACT_CLOCK, return_value=NOW):
            _, _, _, context, _ = await _run_full_chain(ws, [reply])
        state = context.get("proactive")
        assert state["schedules_added"] == 2  # the flight plus the decision deadline
        registry = {entry["id"]: entry for entry in _read_state(ws)["schedules"]}
        assert len(registry) == 2
        assert registry[topic_id("MU5137 改签决策截止")]["due_at"] == "2026-08-14T18:00"
        # Reminders are sorted by due date, so look them up by title.
        published = {r["title"]: r["schedule_id"] for r in state["reminders"]}
        assert published["MU5137 航班起飞时间调整"] == topic_id("MU5137 上海虹桥-北京首都航班")
        assert published["MU5137 改签决策截止"] == topic_id("MU5137 改签决策截止")

    asyncio.run(run())


def test_reminder_pins_agenda_ahead_of_higher_confidence(tmp_path):
    """A low-confidence reminder outranks a 0.95 follow-up on today's agenda."""

    async def run():
        ws = tmp_path
        _touch(ws / "daily" / DAY / "session.md", "material")
        path = f"daily/{DAY}/session.md"
        reply = _reply(
            follow_ups=[_topic("高置信任务", confidence=0.95, paths=[path])],
            reminders=[_reminder("明天到期的报告", due_at="2026-08-14", confidence=0.4, paths=[path])],
        )
        with patch(EXTRACT_CLOCK, return_value=NOW):
            _, _, _, context, _ = await _run_full_chain(ws, [reply])
        state = context.get("proactive")
        assert [c["kind"] for c in state["push_candidates"]] == ["reminder", "follow_up"]

        data = yaml.safe_load(_interests(ws).read_text(encoding="utf-8"))
        assert [item["kind"] for item in data["agenda"]] == ["reminder", "follow_up"]
        assert data["agenda"][0]["due_at"] == "2026-08-14"
        assert data["agenda"][1]["order_reason"] == "single push candidate; auto-agenda without LLM"
        assert data["suppressed"] == []

    asyncio.run(run())


def test_llm_timeout_still_delivers_due_reminder(tmp_path):
    """An extract timeout degrades to a clock-only round and never consumes material."""

    async def run():
        ws = tmp_path
        _touch(ws / "daily" / DAY / "session.md", "unread material")
        _write_state(ws, schedules=[_schedule("提交季度报告", "2026-08-14T09:00")])
        wrapper = _AgentWrapper(["never returned"], delay=0.5)
        with patch(EXTRACT_CLOCK, return_value=NOW):
            _, _, catalog, context, resps = await _run_full_chain(
                ws,
                wrapper=wrapper,
                extract_kwargs={"llm_timeout_seconds": 0.05},
            )
        assert all(r.success for r in resps)
        assert context.get("proactive_skip") is None  # the chain was not short-circuited
        state = context.get("proactive")
        assert state["reminder_only"] is True
        assert state["changed_paths"] == []
        assert catalog.upserts == [] and catalog.dumps == 0  # material stays changed for next round

        data = yaml.safe_load(_interests(ws).read_text(encoding="utf-8"))
        assert [item["title"] for item in data["agenda"]] == ["提交季度报告"]

    asyncio.run(run())


def test_recurring_reminder_keeps_registry_anchor(tmp_path):
    """A weekly schedule reminds on its next occurrence without rewriting the anchor."""

    async def run():
        ws = tmp_path
        _write_state(ws, schedules=[_schedule("周报", "2026-08-06", recurrence="weekly", schedule_id="sch-w")])
        with patch(EXTRACT_CLOCK, return_value=NOW):
            _, _, _, context, _ = await _run_full_chain(ws, [])
        state = context.get("proactive")
        assert [r["title"] for r in state["reminders"]] == ["周报"]
        assert state["reminders"][0]["trigger"] == "due_today"
        assert state["schedules_added"] == 0 and state["schedules_updated"] == 0

        registry = _read_state(ws)["schedules"][0]
        assert registry["due_at"] == "2026-08-06"  # anchor untouched
        assert registry["history"] == []
        topic = yaml.safe_load(_interests(ws).read_text(encoding="utf-8"))["topics"][0]
        assert topic["due_at"] == "2026-08-13"  # published as the next occurrence
        assert topic["schedule_id"] == "sch-w"

    asyncio.run(run())


def test_reminder_bypasses_known_gate(tmp_path):
    """A due reminder fires even when the same title was published recently."""

    async def run():
        ws = tmp_path
        title = "提交季度报告"
        _write_interests_v2(
            ws,
            "2026-08-12",
            [{"id": topic_id(title), "title": title, "reason": "旧提醒", "kind": "reminder", "confidence": 0.8}],
        )
        _write_state(
            ws,
            schedules=[_schedule(title, "2026-08-14T09:00", schedule_id="sch-1")],
        )
        embedder = _FakeEmbedding({})  # every text maps to the same vector -> sim 1.0 >= threshold
        with patch(EXTRACT_CLOCK, return_value=NOW):
            _, _, _, context, _ = await _run_chain(ws, [], extra_ctx={"as_embedding": embedder})
        state = context.get("proactive")
        assert state["dropped_known"] == 0
        assert [t["title"] for t in state["topics_out"]] == [title]
        assert state["push"] is True

    asyncio.run(run())


def test_read_exempts_reminders_from_min_confidence(tmp_path):
    """proactive_read keeps a low-confidence reminder but filters a weak discovery topic."""

    async def run():
        ws = tmp_path
        topics = [
            {
                "id": topic_id("低置信提醒"),
                "title": "低置信提醒",
                "reason": "明天到期",
                "kind": "reminder",
                "confidence": 0.35,
                "first_seen": DAY,
                "last_evidence_at": DAY,
                "evidence": "",
                "paths": [],
                "due_at": "2026-08-14",
            },
            {
                "id": topic_id("低置信延伸"),
                "title": "低置信延伸",
                "reason": "弱联想",
                "kind": "interest_extend",
                "confidence": 0.35,
                "first_seen": DAY,
                "last_evidence_at": DAY,
                "evidence": "",
                "paths": [],
            },
            {
                # A dateless reminder is an external notice / key event, i.e. a
                # judgement call rather than a clock fact: the floor applies.
                "id": topic_id("无日期关键事件"),
                "title": "无日期关键事件",
                "reason": "外部信源关键事件",
                "kind": "reminder",
                "confidence": 0.35,
                "first_seen": DAY,
                "last_evidence_at": DAY,
                "evidence": f"resource/{DAY}/feed.md",
                "paths": [f"resource/{DAY}/feed.md"],
            },
        ]
        _write_interests_v2(ws, DAY, topics)
        step = ProactiveStep(app_context=ApplicationContext(workspace_dir=str(ws)))
        response = await step(RuntimeContext(date=DAY, include_content=False, file_store=_FileStore(ws)))
        assert response.success is True
        assert [t["title"] for t in response.answer["topics"]] == ["低置信提醒"]
        assert response.answer["topics"][0]["due_at"] == "2026-08-14"

    asyncio.run(run())


# ---------------------------------------------------------------------------
# Schedule registry helpers (clock maths, conflicts, lifecycle)
# ---------------------------------------------------------------------------


def test_parse_when_and_canonical_due():
    """Due parsing tolerates ISO and CJK wording; year-less values roll forward."""
    assert parse_when("2026-08-14") == (dt.date(2026, 8, 14), None)
    assert parse_when("2026-08-14T09:30") == (dt.date(2026, 8, 14), dt.datetime(2026, 8, 14, 9, 30))
    assert parse_when("2026-08-14 09:30:00+08:00")[1] == dt.datetime(2026, 8, 14, 9, 30)
    assert parse_when("2026年8月14日 9点")[1] == dt.datetime(2026, 8, 14, 9, 0)
    assert parse_when("8月14日", dt.date(2026, 8, 13))[0] == dt.date(2026, 8, 14)
    assert parse_when("8月14日", dt.date(2026, 9, 1))[0] == dt.date(2027, 8, 14)
    assert parse_when(" soon ") == (None, None)
    assert canonical_due("2026年8月14日 9点") == "2026-08-14T09:00"
    assert canonical_due("无法解析") == "无法解析"


def test_schedule_view_conflicts_and_triggers():
    """Overlaps need a timed side; the earliest clash announces the conflict."""
    schedules = [
        ProactiveSchedule(id="a", title="评审(全天)", due_at="2026-08-14"),
        ProactiveSchedule(id="b", title="客户会议", due_at="2026-08-14T10:00", due_end_at="2026-08-14T11:00"),
        ProactiveSchedule(id="c", title="午餐", due_at="2026-08-14T12:00"),
        ProactiveSchedule(id="d", title="远期限期", due_at="2026-12-01"),
        ProactiveSchedule(id="e", title="逾期项", due_at="2026-08-01"),
        ProactiveSchedule(id="f", title="无日期项", due_at=""),
        ProactiveSchedule(id="g", title="已完成", due_at="2026-08-14", status="done"),
    ]
    view = {entry["id"]: entry for entry in schedule_view(schedules, NOW, 3)}
    assert view["a"]["trigger"] == "conflict"  # all-day vs timed meeting on the same date
    assert view["b"]["trigger"] == "due_soon"  # the clash is already announced by "a"
    assert view["b"]["conflict_with"] == ["评审(全天)"]
    # A timed appointment inside an all-day block is still a possible clash; it
    # keeps ``due_soon`` only because the clash was announced once already.
    assert view["c"]["trigger"] == "due_soon" and view["c"]["conflict_with"] == ["评审(全天)"]
    assert view["d"]["trigger"] == "later"
    assert view["e"]["trigger"] == "overdue" and view["e"]["due_in_days"] < 0
    assert view["f"]["trigger"] == "unscheduled"
    assert view["g"]["trigger"] == "closed"

    titles = [c["title"] for c in reminder_candidates_from_view(list(view.values()), DAY)]
    assert titles == ["逾期项", "评审(全天)", "客户会议", "午餐"]  # nearest due first, then by hour
    assert "已完成" not in titles and "远期限期" not in titles and "无日期项" not in titles


def test_merge_reminders_keeps_clock_facts():
    """The model may reword a reminder but cannot move the registry anchor."""
    seeded = [
        {
            "id": topic_id("评审会"),
            "title": "评审会",
            "reason": "模板理由",
            "kind": "reminder",
            "confidence": 0.6,
            "schedule_id": "sch-1",
            "due_at": "2026-08-20T14:00",
            "next_due": "2026-08-20",
            "paths": ["daily/2026-08-01/a.md"],
            "trigger": "due_soon",
        },
    ]
    emitted = [
        {
            "id": topic_id("评审会"),
            "title": "评审会",
            "reason": "素材显示材料还没齐",
            "kind": "reminder",
            "confidence": 0.9,
            "schedule_id": "sch-1",
            "due_at": "2026-09-01",
            "next_due": "",
            "paths": [f"daily/{DAY}/session.md"],
        },
    ]
    merged = merge_reminders(seeded, emitted)
    assert len(merged) == 1
    assert merged[0]["reason"] == "素材显示材料还没齐"
    assert merged[0]["confidence"] == 0.9
    assert merged[0]["due_at"] == "2026-08-20T14:00"  # anchor kept
    assert merged[0]["next_due"] == "2026-08-20"
    assert merged[0]["paths"] == ["daily/2026-08-01/a.md", f"daily/{DAY}/session.md"]
    assert merged[0]["trigger"] == "due_soon"


def test_apply_schedule_updates_requires_anchor():
    """Registry changes need evidence inside this round's material."""
    state_file = ProactiveStateFile()
    updates = [
        {"id": "", "action": "add", "title": "新日程", "due_at": "2026-08-20", "evidence": "daily/2026-08-13/s.md"},
        {"id": "", "action": "add", "title": "旧证据日程", "due_at": "2026-08-21", "evidence": "daily/2026-01-01/o.md"},
        {"id": "unknown", "action": "cancel", "evidence": "daily/2026-08-13/s.md"},
        {"id": "", "action": "nonsense", "title": "x"},
    ]
    counts, touched = apply_schedule_updates(state_file, updates, DAY, {"daily/2026-08-13/s.md"})
    assert counts == {"added": 1, "updated": 0, "closed": 0, "skipped": 3}
    # Only the accepted update is protected from the reminder safety net; the
    # three skipped ones never touched the registry.
    assert touched == {topic_id("新日程")}
    assert [s.title for s in state_file.schedules] == ["新日程"]


def test_trim_schedules_lifecycle():
    """The registry outlives topics: only aged closures and hopeless overdues go."""
    state_file = ProactiveStateFile(
        schedules=[
            ProactiveSchedule(id="open-future", title="未来", due_at="2027-01-01", status="open", last_seen_at=DAY),
            ProactiveSchedule(
                id="open-overdue",
                title="逾期半年",
                due_at="2026-01-01",
                status="open",
                last_seen_at=DAY,
            ),
            ProactiveSchedule(
                id="closed-old",
                title="旧完成",
                due_at="2025-01-01",
                status="done",
                last_seen_at="2025-01-02",
            ),
            ProactiveSchedule(
                id="chatty",
                title="多次改期",
                due_at="2026-09-01",
                status="open",
                last_seen_at=DAY,
                history=[{"at": DAY, "action": "reschedule"} for _ in range(20)],
            ),
        ],
    )
    dropped = trim_schedules(state_file, DAY, retention_days=180, max_schedules=80, max_history=8)
    kept = {s.id: s for s in state_file.schedules}
    assert dropped == 1
    assert "closed-old" not in kept
    assert kept["open-overdue"].status == "cancelled"  # hopeless overdue auto-closed
    assert kept["open-overdue"].history[-1]["action"] == "expire"
    assert kept["open-future"].status == "open"
    assert len(kept["chatty"].history) == 8  # change log capped, newest kept


def test_scan_material_resource_window_and_filters(tmp_path):
    """Only in-window, text-readable external records join the notice set."""
    ws = tmp_path
    _touch(ws / "resource" / DAY / "flight-change.eml", "MU5137 改期")
    _touch(ws / "resource" / DAY / "scan.pdf", "binary")
    _touch(ws / "resource" / DAY / "_internal.md", "not material")
    _touch(ws / "resource" / "2026-01-02" / "old.md", "out of window")
    inbox = _touch(ws / "resource" / "inbox.md", "undated root input")
    fresh = dt.datetime(2026, 8, 13, 9, 0).timestamp()
    os.utime(inbox, (fresh, fresh))
    stale = _touch(ws / "resource" / "stale.md", "landed long ago")
    ancient = dt.datetime(2026, 1, 2, 9, 0).timestamp()
    os.utime(stale, (ancient, ancient))

    paths, skipped = scan_material_resource(ws, DAY, 2)
    assert paths == [f"resource/{DAY}/flight-change.eml", "resource/inbox.md"]
    assert skipped == 1  # the pdf: auto_resource already interprets binaries into daily cards

    with tempfile.TemporaryDirectory() as empty:
        assert scan_material_resource(Path(empty), DAY, 2) == ([], 0)


def test_resource_notice_feeds_reminder_branch_only(tmp_path):
    """A notice alone keeps the round alive; only reminders may cite the resource layer."""

    async def run():
        ws = tmp_path
        notice = f"resource/{DAY}/airline-notice.md"
        _touch(ws / notice, "MU5137 改期到 8 月 15 日")
        reply = _reply(
            follow_ups=[_topic("跟进改签", paths=[notice])],
            reminders=[_reminder("航班改期通知", due_at="2026-08-15", paths=[notice])],
        )
        with patch(EXTRACT_CLOCK, return_value=NOW):
            _, wrapper, catalog, context, resps = await _run_chain(ws, [reply])
        assert all(r.success for r in resps)
        state = context.get("proactive")
        assert (state["resource_scanned"], state["resource_skipped"]) == (1, 0)
        assert state["resource_changed"] == [notice]
        assert state["changed_paths"] == []  # no daily note changed this round
        # the external notice alone triggered the round: stage 1 + its refine pass
        assert state["llm_calls"] == 1 + state["refine_calls"]
        assert [t["title"] for t in state["reminders"]] == ["航班改期通知"]
        assert state["reminders"][0]["paths"] == [notice]
        assert state["follow_ups"] == [] and state["dropped_missing"] == 1  # untraceable kind
        assert [n.path for n in catalog.upserts] == [notice]  # shared watermark checkpoint
        assert "resource_paths" in wrapper.last_inputs[0]
        assert "# External notices" in wrapper.last_inputs[0]
        assert "3. Notices and key events - scan `resource_notices`" in wrapper.last_system_prompts[0]

    asyncio.run(run())


def test_resources_disabled_keeps_notice_layer_out(tmp_path):
    """``resources_enabled=False`` removes the scan, the prompt block and the trigger."""

    async def run():
        ws = tmp_path
        _touch(ws / "resource" / DAY / "airline-notice.md", "MU5137 改期")
        with patch(EXTRACT_CLOCK, return_value=NOW):
            _, wrapper, catalog, context, resps = await _run_chain(
                ws,
                [_reply(follow_ups=[])],
                extract_kwargs={"resources_enabled": False},
            )
        assert all(r.success for r in resps)
        state = context.get("proactive")
        assert (state["resource_scanned"], state["resource_changed"]) == (0, [])
        assert state["early_exit"] == "no_new_evidence"  # the notice is not material
        assert wrapper.calls == 0
        assert catalog.upserts == [] and catalog.dumps == 0

        _touch(ws / "daily" / DAY / "session.md", "material")
        _, wrapper2, _, _, _ = await _run_chain(
            ws,
            [_reply(follow_ups=[])],
            extract_kwargs={"resources_enabled": False},
        )
        assert "resource_paths" not in wrapper2.last_inputs[0]
        assert "# External notices" not in wrapper2.last_inputs[0]
        assert "resource_notices" not in wrapper2.last_system_prompts[0]  # branch C job 3 gone too
        assert "[" not in wrapper2.last_inputs[0].split("# Material")[-1]  # no unrendered flag leaked

    asyncio.run(run())


def test_degraded_round_does_not_consume_notices(tmp_path):
    """A timeout still delivers due reminders and leaves the unread notice for next round."""

    async def run():
        ws = tmp_path
        _touch(ws / "resource" / DAY / "airline-notice.md", "MU5137 改期")
        _write_state(ws, schedules=[_schedule("提交季度报告", "2026-08-14T09:00")])
        wrapper = _AgentWrapper(["never returned"], delay=0.5)
        with patch(EXTRACT_CLOCK, return_value=NOW):
            _, _, catalog, context, resps = await _run_full_chain(
                ws,
                wrapper=wrapper,
                extract_kwargs={"llm_timeout_seconds": 0.05},
            )
        assert all(r.success for r in resps)
        state = context.get("proactive")
        assert state["reminder_only"] is True
        assert state["changed_paths"] == [] and state["resource_changed"] == []
        assert catalog.upserts == [] and catalog.dumps == 0
        data = yaml.safe_load(_interests(ws).read_text(encoding="utf-8"))
        assert [item["title"] for item in data["agenda"]] == ["提交季度报告"]

    asyncio.run(run())


# ---------------------------------------------------------------------------
# Stage 2: per-kind adjudication (precision pass)
# ---------------------------------------------------------------------------


def test_refine_stage_drops_moves_and_rescores(tmp_path):
    """Stage 2 adjudicates each kind's own candidates: drop, move and rescore."""

    async def run():
        ws = tmp_path
        path = f"daily/{DAY}/session.md"
        _touch(ws / path, "material")
        titles = ("值得跟进的事", "其实是可委托的活", "站不住脚的猜测")
        reply = _reply(follow_ups=[_topic(title, paths=[path]) for title in titles])
        refine = _verdicts(
            _verdict(topic_id("站不住脚的猜测"), "drop", reason="no visible open loop"),
            _verdict(topic_id("其实是可委托的活"), "move", to_kind="new_idea", reason="delegatable work"),
            _verdict(topic_id("值得跟进的事"), "keep", confidence=0.9, reason=""),
        )
        wrapper = _AgentWrapper([reply], refine_replies=[refine])
        _, _, _, context, _ = await _run_chain(ws, wrapper=wrapper)
        state = context.get("proactive")
        assert (state["refine_calls"], state["refine_degraded"]) == (1, 0)
        assert wrapper.refine_calls == 1  # only follow_up produced candidates
        assert [t["title"] for t in state["follow_ups"]] == ["值得跟进的事"]
        assert state["follow_ups"][0]["confidence"] == 0.9  # rescored per rubric
        assert [t["title"] for t in state["new_ideas"]] == ["其实是可委托的活"]
        assert state["new_ideas"][0]["kind"] == "new_idea"  # re-bucketed by the move
        assert [d["title"] for d in state["refine_dropped"]] == ["站不住脚的猜测"]
        assert state["refine_moved"][0]["to_kind"] == "new_idea"
        assert "kind under review: follow_up" in wrapper.last_inputs[-1]

    asyncio.run(run())


def test_refine_stage_is_fail_open(tmp_path):
    """An unusable or timed-out refine pass keeps the stage-1 candidates."""

    async def run():
        ws = tmp_path
        path = f"daily/{DAY}/session.md"
        _touch(ws / path, "material")
        reply = _reply(follow_ups=[_topic("候选一", paths=[path])])

        _, _, _, context, _ = await _run_chain(
            ws,
            wrapper=_AgentWrapper([reply], refine_replies=["not a verdict block"]),
        )
        state = context.get("proactive")
        assert state["refine_degraded"] == 1 and state["refine_calls"] == 1
        assert [t["title"] for t in state["follow_ups"]] == ["候选一"]

        ws2 = tmp_path / "second"
        _touch(ws2 / "daily" / DAY / "session.md", "material")
        _, _, _, context2, _ = await _run_chain(
            ws2,
            wrapper=_AgentWrapper([reply], refine_replies=[(KEEP_ALL_VERDICTS, 0.5)]),
            extract_kwargs={"refine_timeout_seconds": 0.05},
        )
        state2 = context2.get("proactive")
        assert state2["refine_degraded"] == 1
        assert [t["title"] for t in state2["follow_ups"]] == ["候选一"]

    asyncio.run(run())


def test_parse_refine_reply_shapes():
    """Only a genuinely unparseable reply is unusable; an empty verdict list is not."""
    assert parse_refine_reply("```yaml\nverdicts:\n```") == []  # YAML's spelling of []
    assert parse_refine_reply(KEEP_ALL_VERDICTS) == []
    assert parse_refine_reply("```yaml\nfoo: 1\n```") is None  # format not followed
    assert parse_refine_reply("no fenced block here") is None
    assert parse_refine_reply("```yaml\nverdicts: {id: x}\n```") is None  # not a list
    assert parse_refine_reply(_verdicts({"id": "x1", "decision": "drop", "reason": "weak"})) == [
        {"id": "x1", "decision": "drop", "to_kind": "", "reason": "weak"},
    ]


def test_refine_empty_verdicts_is_not_a_degrade(tmp_path):
    """ "Nothing to change" is the expected clean answer, so it must not count as degraded."""

    async def run():
        ws = tmp_path
        path = f"daily/{DAY}/session.md"
        _touch(ws / path, "material")
        reply = _reply(follow_ups=[_topic("候选一", paths=[path])])
        _, _, _, context, _ = await _run_chain(
            ws,
            wrapper=_AgentWrapper([reply], refine_replies=["```yaml\nverdicts:\n```"]),
        )
        state = context.get("proactive")
        assert (state["refine_calls"], state["refine_degraded"]) == (1, 0)
        assert [t["title"] for t in state["follow_ups"]] == ["候选一"]

    asyncio.run(run())


def test_refine_disabled_keeps_stage1_output(tmp_path):
    """``refine_enabled=False`` restores the single-call round."""

    async def run():
        ws = tmp_path
        path = f"daily/{DAY}/session.md"
        _touch(ws / path, "material")
        reply = _reply(follow_ups=[_topic("候选一", paths=[path])])
        wrapper = _AgentWrapper([reply])
        _, _, _, context, _ = await _run_chain(ws, wrapper=wrapper, extract_kwargs={"refine_enabled": False})
        state = context.get("proactive")
        assert (state["refine_calls"], wrapper.refine_calls) == (0, 0)
        assert state["llm_calls"] == 1
        assert [t["title"] for t in state["follow_ups"]] == ["候选一"]

    asyncio.run(run())


def test_refine_never_touches_clock_reminders(tmp_path):
    """Seeded clock reminders are merged after stage 2, so no pass can drop them."""

    async def run():
        ws = tmp_path
        path = f"daily/{DAY}/session.md"
        _touch(ws / path, "material")
        _write_state(ws, schedules=[_schedule("提交季度报告", "2026-08-14T09:00")])
        reply = _reply(follow_ups=[_topic("候选一", paths=[path])])
        hostile = _verdicts(_verdict(topic_id("候选一"), "drop", reason="rejected"))
        wrapper = _AgentWrapper([reply], refine_replies=[hostile])
        with patch(EXTRACT_CLOCK, return_value=NOW):
            _, _, _, context, resps = await _run_full_chain(ws, wrapper=wrapper)
        assert all(r.success for r in resps)
        state = context.get("proactive")
        assert state["follow_ups"] == []  # the drop was applied
        assert [r["title"] for r in state["reminders"]] == ["提交季度报告"]
        assert wrapper.refine_calls == 1  # the reminder bucket was empty at stage 2
        data = yaml.safe_load(_interests(ws).read_text(encoding="utf-8"))
        assert [item["title"] for item in data["agenda"]] == ["提交季度报告"]

    asyncio.run(run())


def test_key_event_reminder_without_due_date(tmp_path):
    """A resource key event may be dateless; it is capped and never registered."""

    async def run():
        ws = tmp_path
        notice = f"resource/{DAY}/policy-feed.md"
        _touch(ws / notice, "监管新规发布，影响用户持续跟踪的品种")
        reply = _reply(
            reminders=[
                _reminder(f"关键事件{i}", reason="外部信源关键事件", confidence=0.8, paths=[notice]) for i in range(4)
            ],
        )
        with patch(EXTRACT_CLOCK, return_value=NOW):
            _, wrapper, _, context, _ = await _run_chain(ws, [reply])
        state = context.get("proactive")
        assert len(state["reminders"]) == 3  # max_notice_reminders default
        assert state["dropped_over_cap"] == 1
        assert all(not r["due_at"] for r in state["reminders"])
        assert state["schedules_added"] == 0  # dateless events do not enter the registry
        assert "key event" in wrapper.last_system_prompts[0]

        # A reminder anchored on daily material still needs a due date: only the
        # external-notice sub-type may be dateless.
        daily_note = f"daily/{DAY}/session.md"
        daily_only = _reply(reminders=[_reminder("无日期日程", reason="x", paths=[daily_note])])
        ws2 = tmp_path / "second"
        _touch(ws2 / daily_note, "material")
        with patch(EXTRACT_CLOCK, return_value=NOW):
            _, _, _, context2, _ = await _run_chain(ws2, [daily_only])
        assert context2.get("proactive")["reminders"] == []
        assert context2.get("proactive")["dropped_missing"] == 1

    asyncio.run(run())


def test_state_file_bare_yaml_dates_load(tmp_path):
    """Unquoted YAML dates must not invalidate the long-lived registry.

    ``yaml.safe_load`` reads a bare ``2026-08-05`` as ``datetime.date``. The
    truth source is hand-editable and the registry has to outlive topics by
    months, so those scalars are coerced back to ISO text instead of failing
    validation and taking every schedule down with them.
    """
    raw = (
        "version: 1\n"
        "open_topics:\n"
        "  - id: t1\n"
        "    title: 历史主题\n"
        "    kind: follow_up\n"
        "    first_seen: 2026-07-01\n"
        "    last_evidence_at: 2026-07-02\n"
        "resolved: []\n"
        "schedules:\n"
        "  - id: sch-tax\n"
        "    title: 个税补充申报\n"
        "    due_at: 2026-08-05\n"
        "    status: open\n"
        "    first_seen: 2026-07-01\n"
        "    last_seen_at: 2026-07-01\n"
        "    history:\n"
        "      - at: 2026-07-01\n"
        "        action: add\n"
        "        to: 2026-08-05T09:00:00\n"
        "deliveries:\n"
        "  - topic_id: t1\n"
        "    agenda_date: 2026-08-12\n"
        "    delivered_at: 2026-08-12T09:00:00\n"
        "    status: shown\n"
    )
    _touch(tmp_path / "daily" / "_proactive.yaml", raw)

    state_file, needs_bootstrap = load_state(tmp_path)

    assert needs_bootstrap is False
    topic = state_file.open_topics[0]
    assert (topic.first_seen, topic.last_evidence_at) == ("2026-07-01", "2026-07-02")
    schedule = state_file.schedules[0]
    assert schedule.id == "sch-tax" and schedule.due_at == "2026-08-05"
    assert (schedule.first_seen, schedule.last_seen_at) == ("2026-07-01", "2026-07-01")
    assert schedule.history[0]["at"] == "2026-07-01"
    assert schedule.history[0]["to"] == "2026-08-05T09:00:00"
    delivery = state_file.deliveries[0]
    assert delivery.agenda_date == "2026-08-12"
    assert delivery.delivered_at == "2026-08-12T09:00:00"


def test_state_file_salvages_unreadable_entries(tmp_path):
    """One malformed entry costs that entry, never the whole truth source."""
    raw = (
        "version: 1\n"
        "open_topics:\n"
        "  - id: good\n"
        "    title: 正常主题\n"
        "    kind: follow_up\n"
        "  - id: bad\n"
        "    title:\n"
        "      nested: mapping\n"
        "schedules:\n"
        "  - id: sch-bad\n"
        "    title:\n"
        "      nested: mapping\n"
        "    due_at: 2026-08-14\n"
        "  - id: sch-earnings-call\n"
        "    title: 财报电话会\n"
        "    due_at: 2026-08-14T20:00\n"
        "    first_seen: 2026-07-01\n"
        "resolved:\n"
        "  - id: tomb\n"
        "  - not-a-mapping\n"
    )
    _touch(tmp_path / "daily" / "_proactive.yaml", raw)

    state_file, needs_bootstrap = load_state(tmp_path)

    # Salvage is not a first run, so the interests.yaml bootstrap stays off.
    assert needs_bootstrap is False
    assert [t.id for t in state_file.open_topics] == ["good"]
    assert [s.id for s in state_file.schedules] == ["sch-earnings-call"]
    assert state_file.schedules[0].due_at == "2026-08-14T20:00"
    assert [r["id"] for r in state_file.resolved] == ["tomb"]


def test_refine_cannot_demote_a_notice_reminder(tmp_path):
    """Stage 2 may not move a resource-anchored reminder into a kind that cannot quote it."""

    async def run():
        ws = tmp_path
        notice = f"resource/{DAY}/policy-feed.md"
        daily = f"daily/{DAY}/session.md"
        _touch(ws / notice, "监管新规发布，实质影响用户跟踪的品种")
        _touch(ws / daily, "material")
        reply = _reply(
            reminders=[
                _reminder("监管新规", reason="外部信源关键事件", paths=[notice]),
                _reminder("内部评审", reason="临近到期", due_at="2026-08-14T09:00", paths=[daily]),
            ],
            follow_ups=[_topic("待跟进", paths=[daily])],
        )
        hostile = _verdicts(
            _verdict(topic_id("监管新规"), "move", to_kind="interest_extend", reason="只是背景阅读"),
            _verdict(topic_id("内部评审"), "move", to_kind="follow_up", reason="其实是未闭环事项"),
        )
        wrapper = _AgentWrapper([reply], refine_replies=[hostile])
        with patch(EXTRACT_CLOCK, return_value=NOW):
            _, _, _, context, _ = await _run_chain(ws, wrapper=wrapper)
        state = context.get("proactive")

        # The notice stays a reminder: only that branch may cite resource/.
        assert [r["title"] for r in state["reminders"]] == ["监管新规"]
        assert [t["title"] for t in state["extends"]] == []
        # The daily-anchored move is legal, and leaves no dangling registry link.
        moved = next(t for t in state["follow_ups"] if t["title"] == "内部评审")
        assert moved["kind"] == "follow_up"
        leaked = [key for key in ("schedule_id", "due_at", "next_due", "trigger", "conflict_with") if key in moved]
        assert leaked == []
        assert [m["to_kind"] for m in state["refine_moved"]] == ["follow_up"]

    asyncio.run(run())


def test_refine_refuses_an_impossible_move_into_reminder(tmp_path):
    """Stage 2 cannot attach a due date, so a dateless daily candidate stays put."""

    async def run():
        ws = tmp_path
        path = f"daily/{DAY}/session.md"
        _touch(ws / path, "material")
        reply = _reply(follow_ups=[_topic("带日期的承诺", paths=[path])])
        refine = _verdicts(_verdict(topic_id("带日期的承诺"), "move", to_kind="reminder", reason="其实是日程"))
        wrapper = _AgentWrapper([reply], refine_replies=[refine])
        _, _, _, context, _ = await _run_chain(ws, wrapper=wrapper)
        state = context.get("proactive")

        assert [t["title"] for t in state["follow_ups"]] == ["带日期的承诺"]
        assert state["reminders"] == []
        assert state["refine_moved"] == []  # refused outright, not recorded as a move

    asyncio.run(run())


def test_reminder_cannot_revert_an_explicit_reschedule(tmp_path):
    """``schedule_updates`` is the only channel that may move a due date.

    The clock scan reads the registry *before* the LLM reports a reschedule, so
    the reminder it seeds still carries the old due value. Writing that back
    would silently undo the change the material actually reported.
    """

    async def run():
        ws = tmp_path
        path = f"daily/{DAY}/session.md"
        _touch(ws / path, "投顾通知复盘延后半小时")
        _write_state(ws, schedules=[_schedule("季度持仓复盘", "2026-08-13T21:00", schedule_id="sch-review")])
        reply = _reply(
            reminders=[
                _reminder(
                    "季度持仓复盘",
                    due_at="2026-08-13T21:00",
                    reason="今晚复盘，投顾通知可能延后半小时",
                    paths=[path],
                    schedule_id="sch-review",
                ),
            ],
            schedule_updates=[
                {
                    "action": "reschedule",
                    "id": "sch-review",
                    "title": "季度持仓复盘",
                    "due_at": "2026-08-13T21:30",
                    "evidence": path,
                    "reason": "投顾通知延后半小时开始",
                },
            ],
        )
        with patch(EXTRACT_CLOCK, return_value=NOW):
            _, _, _, context, resps = await _run_full_chain(ws, [reply])
        assert all(r.success for r in resps)
        state = context.get("proactive")
        assert (state["schedules_added"], state["schedules_updated"]) == (0, 1)
        # The clock scan seeded this reminder with the pre-update value; the
        # published due date a reader sorts by has to follow the registry.
        assert state["reminders_reanchored"] == 1
        assert [r["due_at"] for r in state["reminders"]] == ["2026-08-13T21:30"]

        entry = next(s for s in _read_state(ws)["schedules"] if s["id"] == "sch-review")
        assert entry["due_at"] == "2026-08-13T21:30"  # the stale reminder did not revert it
        assert [(h["action"], h["to"]) for h in entry["history"]] == [("reschedule", "2026-08-13T21:30")]
        published = yaml.safe_load(_interests(ws).read_text(encoding="utf-8"))
        assert [t["due_at"] for t in published["topics"] if t["kind"] == "reminder"] == ["2026-08-13T21:30"]

    asyncio.run(run())


def test_reminder_heavy_day_keeps_discovery_slots(tmp_path):
    """Pinned reminders never starve the discovery kinds of cards and slots."""

    async def run():
        ws = tmp_path
        path = f"daily/{DAY}/session.md"
        _touch(ws / path, "material")
        # Seven schedules due today: more than max_agenda_items (6), which used
        # to leave zero discovery slots and card-less, suppressed candidates.
        _write_state(
            ws,
            schedules=[_schedule(f"今日日程{i}", "2026-08-13T09:00", schedule_id=f"sch-{i}") for i in range(7)],
        )
        reply = _reply(follow_ups=[_topic("待跟进事项", paths=[path])])
        with patch(EXTRACT_CLOCK, return_value=NOW):
            _, _, _, context, resps = await _run_full_chain(ws, [reply])
        assert all(r.success for r in resps)
        state = context.get("proactive")
        assert len(state["reminders"]) == 7

        carded = {c["topic_id"]: c["kind"] for c in state["scenario_cards"]}
        assert list(carded.values()).count("follow_up") == 1  # expanded, not left card-less
        data = yaml.safe_load(_interests(ws).read_text(encoding="utf-8"))
        kinds = [item["kind"] for item in data["agenda"]]
        assert kinds.count("reminder") == 7 and kinds.count("follow_up") == 1
        assert all(item["kind"] != "follow_up" for item in data["suppressed"] if "kind" in item)
        assert not any(s["title"] == "待跟进事项" for s in data["suppressed"])

    asyncio.run(run())


def test_schedule_update_may_anchor_on_a_notice(tmp_path):
    """The registry accepts notice-anchored updates; discovery topics still may not.

    Only the reminder branch may cite the ``resource/`` layer, and
    ``clean_schedule_update`` already accepts those paths - so the registry
    anchor has to agree, or a dated external notice could never be registered
    through the explicit channel and would lose its detail/end-time/recurrence.
    """

    async def run():
        ws = tmp_path
        notice = f"resource/{DAY}/airline.md"
        daily = f"daily/{DAY}/session.md"
        _touch(ws / notice, "MU5137 改期，改签截止 2026-08-14 18:00")
        _touch(ws / daily, "material")
        _write_state(ws, open_topics=[_state_topic("既有主题")])
        reply = _reply(
            schedule_updates=[
                {
                    "action": "add",
                    "title": "改签截止",
                    "due_at": "2026-08-14T18:00",
                    "due_end_at": "",
                    "detail": "免手续费",
                    "evidence": notice,
                    "reason": "航司通知改期",
                },
            ],
            updates=[{"id": topic_id("既有主题"), "action": "update", "evidence": notice, "confidence": 0.9}],
        )
        with patch(EXTRACT_CLOCK, return_value=NOW):
            _, _, _, context, resps = await _run_full_chain(ws, [reply])
        assert all(r.success for r in resps)
        state = context.get("proactive")

        schedules = _read_state(ws)["schedules"]
        assert [s["title"] for s in schedules] == ["改签截止"]
        assert schedules[0]["due_at"] == "2026-08-14T18:00"
        assert schedules[0]["detail"] == "免手续费"  # only the explicit channel carries this
        assert state["schedules_added"] == 1
        # A discovery topic still may not take its evidence from the notice layer.
        assert state["updates_applied"] == 0

    asyncio.run(run())
