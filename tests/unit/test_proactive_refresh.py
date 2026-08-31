"""Unit tests for the proactive refresh chain (PROACTIVE_SPEC.md A8 mapping).

Coverage: F1 contracts, F2 discovery, F3 isolation, F4 idle gate/budget/timeout,
F5 read extensions, plus the M2 extends branch and semantic dedup.
"""

import asyncio
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
from reme.schema import FileNode, ProactiveTopic
from reme.steps.evolve.dream.extract import DreamExtractStep
from reme.steps.evolve.dream.utils import pack_paths
from reme.steps.evolve.proactive.agenda import ProactiveAgendaStep
from reme.steps.evolve.proactive.extract import ProactiveExtractStep
from reme.steps.evolve.proactive.finish import ProactiveFinishStep
from reme.steps.evolve.proactive.plan import ProactivePlanStep
from reme.steps.evolve.proactive.proactive import ProactiveStep
from reme.steps.evolve.proactive.topics import ProactiveTopicsStep
from reme.steps.evolve.proactive.utils import (
    load_carry_forward,
    load_state,
    normalize_topic,
    parse_interests_topics,
    parse_extract_reply,
    scan_material_daily,
    topic_id,
)
from reme.steps.evolve.wait_for_idle import WaitForIdleStep

DAY = "2026-08-13"


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
    """Deterministic LLM stub with scripted replies and call accounting."""

    def __init__(self, replies=None, delay: float = 0.0):
        super().__init__()
        self.replies = list(replies or [])
        self.delay = delay
        self.calls = 0
        self.last_inputs = []
        self.last_system_prompts = []

    async def reply(self, inputs, **kwargs):
        self.calls += 1
        self.last_inputs.append(inputs)
        self.last_system_prompts.append(str(kwargs.get("system_prompt") or ""))
        item = self.replies.pop(0) if self.replies else ""
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


def _reply(follow_ups=None, extends=None, updates=None) -> str:
    doc = {}
    if follow_ups is not None:
        doc["follow_ups"] = follow_ups
    if extends is not None:
        doc["extends"] = extends
    if updates is not None:
        doc["updates"] = updates
    body = yaml.safe_dump(doc, allow_unicode=True, sort_keys=False)
    return f"```yaml\n{body}```"


def _topic(title, reason="because", confidence=0.7, paths=None, evidence=None, keywords=None):
    return {
        "title": title,
        "reason": reason,
        "confidence": confidence,
        "evidence": evidence or (paths[0] if paths else ""),
        "keywords": keywords or [],
        "paths": paths or [],
    }


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
        "keywords": [],
        "paths": [],
    }


def _write_state(ws, budget=None, exposure=None, open_topics=None, resolved=None, omit_open_topics=False):
    data = {"version": 1}
    if budget is not None:
        data["budget"] = budget
    if exposure is not None:
        data["exposure"] = exposure
    if not omit_open_topics:
        data["open_topics"] = open_topics or []
    if resolved is not None:
        data["resolved"] = resolved
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

        carry = context.get("proactive")["carry_forward_all"]
        assert [t["title"] for t in carry] == ["历史主题"]  # history reached the chain context

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
        assert context.get("proactive")["carry_forward_all"] == []
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
    """wait_for_idle -> extract -> topics -> finish runs end to end."""

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

        wait = WaitForIdleStep(app_context=app, max_wait=5, poll_interval=0.05)
        resp_wait = await wait(context)
        assert resp_wait.success is True
        assert "proactive_skip" not in context

        extract = ProactiveExtractStep(app_context=app)
        resp_extract = await extract(context)
        topics = ProactiveTopicsStep(app_context=app)
        resp_topics = await topics(context)
        finish = ProactiveFinishStep(app_context=app)
        resp_finish = await finish(context)

        assert resp_extract.success and resp_topics.success and resp_finish.success
        state = context.get("proactive")
        assert state["llm_calls"] == 1
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
                    "keywords": [],
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
# F4 idle gate, short-circuit, budget, timeout
# ---------------------------------------------------------------------------


def test_idle_gate(tmp_path):
    """Running/quiet-window jobs block; idle trunk proceeds; fnmatch patterns apply."""

    async def run():
        now_mono = time.monotonic()
        app = ApplicationContext(workspace_dir=str(tmp_path))
        app.metadata["__job_last_run"] = {
            "auto_memory_batch": {"running": True, "last_start": now_mono, "last_end": now_mono - 999},
            "unrelated_job": {"running": True, "last_start": now_mono, "last_end": now_mono},
        }
        step = WaitForIdleStep(app_context=app, max_wait=0.3, poll_interval=0.05)
        context = RuntimeContext()
        response = await step(context)
        assert response.success is True  # giving up is not a failure
        flag = context.get("proactive_skip")
        assert flag["reason"] == "busy"
        assert flag["busy_jobs"] == ["auto_memory_batch"]  # unrelated_job not matched

        app.metadata["__job_last_run"] = {
            "auto_memory_batch": {"running": False, "last_start": now_mono - 900, "last_end": now_mono - 900},
        }
        context2 = RuntimeContext()
        response2 = await WaitForIdleStep(app_context=app, quiet_window=120)(context2)
        assert response2.success is True
        assert "proactive_skip" not in context2

    asyncio.run(run())


def test_idle_timeout_skip(tmp_path):
    """Waiting past max_wait gives up the round: success=True, skip flag set, no writes."""

    async def run():
        app = ApplicationContext(workspace_dir=str(tmp_path))
        app.metadata["__job_last_run"] = {
            "dream_cron": {"running": True, "last_start": time.monotonic(), "last_end": 0.0},
        }
        step = WaitForIdleStep(app_context=app, max_wait=0.2, poll_interval=0.05)
        context = RuntimeContext()
        started = time.monotonic()
        response = await step(context)
        assert time.monotonic() - started >= 0.2
        assert response.success is True
        assert "Skipped" in str(response.answer)
        assert context.get("proactive_skip")["reason"] == "busy"
        assert not _interests(tmp_path).exists()

    asyncio.run(run())


def test_wait_for_idle_skip_key(tmp_path):
    """The short-circuit flag key is parameterized, not hardcoded."""

    async def run():
        app = ApplicationContext(workspace_dir=str(tmp_path))
        app.metadata["__job_last_run"] = {
            "dream_cron": {"running": True, "last_start": time.monotonic(), "last_end": 0.0},
        }
        step = WaitForIdleStep(app_context=app, max_wait=0.15, poll_interval=0.05, skip_key="custom_skip")
        context = RuntimeContext()
        await step(context)
        assert context.get("custom_skip")["reason"] == "busy"
        assert context.get("proactive_skip") is None

    asyncio.run(run())


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
        assert wrapper.calls == 2
        assert all(r.success for r in responses)
        assert [t["title"] for t in context.get("proactive")["follow_ups"]] == ["补救话题"]

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
                    "keywords": [],
                    "paths": [],
                },
            ],
            "content": content,
        }
        assert response.metadata["topics"] == response.answer["topics"]
        assert response.metadata["content"] == content
        assert response.metadata["push"] is True  # additive keys only

    asyncio.run(run())


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
                _topic("Candidate Topic", confidence=0.8, paths=[f"daily/{DAY}/session.md"], keywords=["alpha"]),
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
        assert wrapper.calls == 1  # topics step is pure computation now

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
        assert wrapper.calls == 3
        state = context.get("proactive")
        assert state["llm_calls"] == 1  # extract only
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
        assert wrapper.calls == 3
        state = context.get("proactive")
        assert state["plan_llm_calls"] == 2  # attempted but unusable
        assert len(state["scenario_cards"]) == 2

        data = yaml.safe_load(_interests(ws).read_text(encoding="utf-8"))
        assert data["push"] is True
        assert [i["title"] for i in data["agenda"]] == ["任务一", "任务二"]
        assert all(
            i["order_reason"] == "deterministic fallback: freshness/confidence order" for i in data["agenda"]
        )
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
        assert wrapper.calls == 1  # extract only; plan/agenda made zero LLM calls
        state = context.get("proactive")
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
        assert wrapper.calls == 2  # extract + plan only
        state = context.get("proactive")
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
        _, _, _, context, _ = await _run_full_chain(ws, [extract_reply, plan_reply, agenda_text])
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
            "topics": [_state_topic("议程主题", topic_id_value=id_a, confidence=0.9),
                       _state_topic("已解决主题", topic_id_value=id_b, confidence=0.9)],
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
