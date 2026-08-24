"""Unit tests for the refactored dream package."""

# pylint: disable=protected-access

import asyncio
import tempfile
from pathlib import Path
from unittest.mock import patch

import yaml

from reme.components.application_context import ApplicationContext
from reme.components.agent_wrapper import BaseAgentWrapper
from reme.components.file_catalog import BaseFileCatalog
from reme.components.file_store import BaseFileStore
from reme.components.runtime_context import RuntimeContext
from reme.schema import DreamState, FileNode
from reme.steps.evolve.dream.extract import DreamExtractStep
from reme.steps.evolve.dream.finish import DreamFinishStep
from reme.steps.evolve.proactive.proactive import ProactiveStep
from reme.steps.evolve.dream.topics import DreamTopicsStep
from reme.steps.evolve.dream.utils import load_yaml_topics, parse_structured_reply, recent_dates, scan_day_files


def _touch(path: Path, text: str = "x") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


class _Catalog(BaseFileCatalog):
    def __init__(self):
        super().__init__()
        self.upserts = []
        self.dumps = 0

    async def upsert(self, nodes):
        self.upserts.extend(nodes)

    async def delete(self, path):
        return None

    async def get_nodes(self, paths=None):
        return []

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


class _ReplyAgent(BaseAgentWrapper):
    """Small agent double that can return, mutate files, or raise."""

    def __init__(self, result=None, *, on_reply=None, error=None):
        super().__init__()
        self.result = result or {"result": "{}"}
        self.on_reply = on_reply
        self.error = error
        self.calls = 0

    async def reply(self, _message, **_kwargs):
        self.calls += 1
        if self.on_reply:
            self.on_reply()
        if self.error:
            raise self.error
        return self.result


class _SequenceAgent(BaseAgentWrapper):
    """Return a fixed sequence of replies or exceptions."""

    def __init__(self, *outcomes):
        super().__init__()
        self.outcomes = list(outcomes)
        self.calls = 0

    async def reply(self, _message, **_kwargs):
        self.calls += 1
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def test_scan_day_files_includes_nested_md_and_excludes_interests():
    """Scan day files."""
    with tempfile.TemporaryDirectory() as tmp:
        workspace = Path(tmp)
        _touch(workspace / "daily" / "2026-05-28.md")
        _touch(workspace / "daily" / "2026-05-28" / "session.md")
        _touch(workspace / "daily" / "2026-05-28" / "auth-refactor" / "notes.md")
        _touch(workspace / "daily" / "2026-05-28" / "interests.yaml")

        assert scan_day_files(workspace, "2026-05-28", "daily") == [
            "daily/2026-05-28.md",
            "daily/2026-05-28/auth-refactor/notes.md",
            "daily/2026-05-28/session.md",
        ]


def test_dream_extract_matches_posix_catalog_paths(tmp_path):
    """Unchanged nested files retain their POSIX catalog entries on every platform."""

    class Catalog(_Catalog):
        """Catalog seeded with POSIX paths and recording deletions."""

        def __init__(self, nodes):
            super().__init__()
            self.nodes = nodes
            self.deleted = []

        async def delete(self, path):
            self.deleted.extend(path if isinstance(path, list) else [path])

        async def get_nodes(self, paths=None):
            return self.nodes

    async def run():
        note = _touch(tmp_path / "daily" / "2026-05-28" / "nested" / "session.md")
        rel_path = note.relative_to(tmp_path).as_posix()
        catalog = Catalog([FileNode(path=rel_path, st_mtime=note.stat().st_mtime)])
        step = DreamExtractStep(scan_days=1, app_context=ApplicationContext(workspace_dir=str(tmp_path)))

        with patch("reme.steps.evolve.dream.extract.refresh_day_index", return_value={}):
            response = await step(
                RuntimeContext(date="2026-05-28", file_catalog=catalog, file_store=_FileStore(tmp_path)),
            )

        dream = response.metadata["dream"]
        assert response.success is True
        assert dream["unchanged_paths"] == [rel_path]
        assert dream["changed_paths"] == []
        assert dream["deleted_paths"] == []
        assert not catalog.deleted

    asyncio.run(run())


def test_recent_dates_includes_anchor_and_previous_days():
    """Recent date window is inclusive and chronological."""
    assert recent_dates("2026-05-28", 3) == ["2026-05-26", "2026-05-27", "2026-05-28"]
    assert recent_dates("2026-05-28", 1) == ["2026-05-28"]


def test_parse_structured_reply_handles_fenced_yaml_and_scalar_fallback():
    """Parse a JSON/YAML object from an agent reply, including fenced blocks."""
    data = parse_structured_reply(
        "```yaml\n"
        "action: REFINE\n"
        "target_path: digest/personal/no-trailing-summary.md\n"
        "note: Extended node. Core rule unchanged: answer directly and stop.\n"
        "```",
    )
    assert data["action"] == "REFINE"
    assert data["target_path"] == "digest/personal/no-trailing-summary.md"
    assert data["note"].startswith("Extended node")


def test_integrate_prompts_require_wikilinks_in_contextual_prose():
    """Every digest bucket rejects standalone relation fields and bare links."""
    prompt_path = Path(__file__).parents[2] / "reme" / "steps" / "evolve" / "dream" / "integrate.yaml"
    prompts = yaml.safe_load(prompt_path.read_text(encoding="utf-8"))

    for bucket in ("procedure", "personal", "wiki"):
        english = prompts[f"integrate_system_prompt_{bucket}"]
        chinese = prompts[f"integrate_system_prompt_{bucket}_zh"]
        assert "Never emit standalone relation fields or bare-link lines" in english
        assert "Digest links belong inside the sentence" in english
        assert "Before returning, read the target" in english
        assert "禁止生成 `relates_to:: [[...]]`" in chinese
        assert "Digest 链接必须融入解释关系的句子" in chinese
        assert "返回前读取目标文件" in chinese


def test_extract_clean_output_respects_max_units():
    """Extract cleaning caps valid units at max_units."""
    state = DreamState(changed_paths=["daily/a.md"])
    meta = {
        "units": [
            {"name": f"unit-{i}", "bucket": "wiki", "summary": f"summary {i}", "paths": ["daily/a.md"]}
            for i in range(7)
        ],
    }

    DreamExtractStep().clean_output(state, meta, max_units=5)

    assert len(state.units) == 5
    assert [unit["name"] for unit in state.units] == [f"unit-{i}" for i in range(5)]


def test_extract_unusable_receipt_is_a_warning_not_a_failure(tmp_path):
    """Malformed best-effort extraction output does not fail the nightly job."""

    async def run():
        _touch(tmp_path / "daily" / "2026-05-28" / "session.md")
        step = DreamExtractStep(scan_days=1, app_context=ApplicationContext(workspace_dir=str(tmp_path)))
        step.agent_wrapper = _ReplyAgent()

        with (
            patch("reme.steps.evolve.dream.extract.refresh_day_index", return_value={}),
            patch("reme.steps.evolve.dream.extract.llm_available", return_value=True),
        ):
            response = await step(
                RuntimeContext(
                    date="2026-05-28",
                    agent_wrapper=step.agent_wrapper,
                    file_catalog=_Catalog(),
                    file_store=_FileStore(tmp_path),
                ),
            )

        dream = response.metadata["dream"]
        assert response.success is True
        assert dream["units"] == []
        assert dream["failed_paths"] == []
        assert dream["warnings"] == [
            "dream extract skipped unusable agent receipt after retry; expected units and topics lists",
        ]
        assert step.agent_wrapper.calls == 2

    asyncio.run(run())


def test_extract_retries_one_unusable_receipt(tmp_path):
    """A transient malformed extraction receipt gets one bounded retry."""

    async def run():
        _touch(tmp_path / "daily" / "2026-05-28" / "session.md")
        agent = _SequenceAgent({"result": "{}"}, {"result": '{"units": [], "topics": []}'})
        step = DreamExtractStep(
            scan_days=1,
            app_context=ApplicationContext(workspace_dir=str(tmp_path)),
            agent_wrapper=agent,
        )

        with (
            patch("reme.steps.evolve.dream.extract.refresh_day_index", return_value={}),
            patch("reme.steps.evolve.dream.extract.llm_available", return_value=True),
        ):
            response = await step(
                RuntimeContext(date="2026-05-28", file_catalog=_Catalog(), file_store=_FileStore(tmp_path)),
            )

        assert response.success is True
        assert response.metadata["dream"]["warnings"] == []
        assert agent.calls == 2

    asyncio.run(run())


def test_integrate_invalid_receipt_without_file_change_is_skipped(tmp_path):
    """An unusable integration receipt is a skipped unit, not a failed unit."""

    async def run():
        unit = {"name": "unit", "bucket": "wiki", "summary": "summary", "paths": ["daily/a.md"]}
        state = DreamState(units=[unit])
        step = DreamIntegrateStep()
        step.agent_wrapper = _ReplyAgent()

        await step._integrate_one(state, unit, 1, tmp_path, "digest")  # pylint: disable=protected-access

        assert state.integrate_results == []
        assert len(state.skipped_units) == 1
        assert state.failed_units == []
        assert not state.failed_paths
        assert len(state.warnings) == 1
        assert step.agent_wrapper.calls == 2

    asyncio.run(run())


def test_integrate_retries_one_invalid_receipt(tmp_path):
    """A transient invalid integration receipt gets one bounded retry."""

    async def run():
        unit = {"name": "unit", "bucket": "wiki", "summary": "summary", "paths": ["daily/a.md"]}
        state = DreamState(units=[unit])
        target = _touch(tmp_path / "digest" / "wiki" / "unit.md", "integrated")
        agent = _SequenceAgent(
            {"result": "{}"},
            {
                "result": (
                    '{"action": "REFINE", "target_path": "digest/wiki/unit.md", ' '"note": "updated after retry"}'
                ),
            },
        )
        step = DreamIntegrateStep(agent_wrapper=agent)

        with patch(
            "reme.steps.evolve.dream.integrate._snapshot_digest",
            wraps=_snapshot_digest,
        ) as snapshot:
            await step._integrate_one(state, unit, 1, tmp_path, "digest")  # pylint: disable=protected-access

        assert target.is_file()
        assert state.integrate_results[0]["target_path"] == "digest/wiki/unit.md"
        assert state.skipped_units == []
        assert agent.calls == 2
        assert snapshot.call_count == 3

    asyncio.run(run())


def test_integrate_invalid_receipt_recovers_one_changed_digest_file(tmp_path):
    """The file-native side effect wins when the agent's final receipt is malformed."""

    async def run():
        unit = {"name": "unit", "bucket": "wiki", "summary": "summary", "paths": ["daily/a.md"]}
        state = DreamState(units=[unit])
        target = tmp_path / "digest" / "wiki" / "unit.md"

        def write_digest():
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text("integrated", encoding="utf-8")

        step = DreamIntegrateStep()
        step.agent_wrapper = _ReplyAgent(on_reply=write_digest)

        await step._integrate_one(state, unit, 1, tmp_path, "digest")  # pylint: disable=protected-access

        assert state.skipped_units == []
        assert state.failed_units == []
        assert state.integrate_results[0]["action"] == "CREATE"
        assert state.integrate_results[0]["target_path"] == "digest/wiki/unit.md"
        assert state.nodes_created == ["digest/wiki/unit.md"]
        assert state.modified_paths == ["digest/wiki/unit.md"]
        assert len(state.warnings) == 1

    asyncio.run(run())


def test_integrate_recovers_an_update_to_another_bucket(tmp_path):
    """A malformed receipt can recover an update to an existing cross-bucket node."""

    async def run():
        unit = {"name": "unit", "bucket": "wiki", "summary": "summary", "paths": ["daily/a.md"]}
        state = DreamState(units=[unit])
        target = tmp_path / "digest" / "procedure" / "unit.md"
        _touch(target, "existing procedure")

        def write_digest():
            target.write_text("updated procedure", encoding="utf-8")

        step = DreamIntegrateStep()
        step.agent_wrapper = _ReplyAgent(on_reply=write_digest)

        await step._integrate_one(state, unit, 1, tmp_path, "digest")  # pylint: disable=protected-access

        assert state.skipped_units == []
        assert state.integrate_results[0]["action"] == "UPDATED"
        assert state.integrate_results[0]["target_path"] == "digest/procedure/unit.md"
        assert state.nodes_updated == ["digest/procedure/unit.md"]
        assert state.modified_paths == ["digest/procedure/unit.md"]
        assert step.agent_wrapper.calls == 1

    asyncio.run(run())


def test_integrate_does_not_recover_a_create_in_another_bucket(tmp_path):
    """A malformed receipt cannot recover a cross-bucket create."""

    async def run():
        unit = {"name": "unit", "bucket": "wiki", "summary": "summary", "paths": ["daily/a.md"]}
        state = DreamState(units=[unit])
        target = tmp_path / "digest" / "procedure" / "unit.md"

        def write_digest():
            _touch(target, "created procedure")

        step = DreamIntegrateStep(agent_wrapper=_ReplyAgent(on_reply=write_digest))

        await step._integrate_one(state, unit, 1, tmp_path, "digest")  # pylint: disable=protected-access

        assert state.integrate_results == []
        assert len(state.skipped_units) == 1
        assert state.nodes_created == []
        assert state.modified_paths == ["digest/procedure/unit.md"]

    asyncio.run(run())


def test_integrate_accepts_cross_bucket_update_receipt(tmp_path):
    """An UPDATE receipt may target an existing node in any supported bucket."""

    async def run():
        unit = {"name": "unit", "bucket": "wiki", "summary": "summary", "paths": ["daily/a.md"]}
        state = DreamState(units=[unit])
        _touch(tmp_path / "digest" / "procedure" / "unit.md", "updated procedure")
        agent = _ReplyAgent(
            result={
                "result": (
                    '{"action": "REFINE", "target_path": "digest/procedure/unit.md", '
                    '"note": "updated existing procedure"}'
                ),
            },
        )
        step = DreamIntegrateStep(agent_wrapper=agent)

        with patch(
            "reme.steps.evolve.dream.integrate._snapshot_digest",
            wraps=_snapshot_digest,
        ) as snapshot:
            await step._integrate_one(state, unit, 1, tmp_path, "digest")  # pylint: disable=protected-access

        assert state.skipped_units == []
        assert state.integrate_results[0]["action"] == "REFINE"
        assert state.nodes_updated == ["digest/procedure/unit.md"]
        assert state.modified_paths == []
        assert snapshot.call_count == 2

    asyncio.run(run())


def test_integrate_uses_one_application_wide_lock(tmp_path):
    """Concurrent AutoDream jobs in one Application serialize integration."""
    app_context = ApplicationContext(workspace_dir=str(tmp_path))
    first = DreamIntegrateStep(app_context=app_context)
    second = DreamIntegrateStep(app_context=app_context)

    assert first._integration_lock() is second._integration_lock()  # pylint: disable=protected-access


def test_extract_without_llm_marks_changed_paths_failed(tmp_path):
    """A missing LLM must not let finish checkpoint unprocessed source files."""

    async def run():
        note = _touch(tmp_path / "daily" / "2026-05-28" / "session.md")
        step = DreamExtractStep(app_context=ApplicationContext(workspace_dir=str(tmp_path)))

        response = await step(
            RuntimeContext(date="2026-05-28", file_catalog=_Catalog(), file_store=_FileStore(tmp_path)),
        )

        dream = response.metadata["dream"]
        assert response.success is False
        assert note.relative_to(tmp_path).as_posix() in dream["changed_paths"]
        assert dream["failed_paths"] == dream["changed_paths"]

    asyncio.run(run())


def test_topics_step_writes_only_target_date_interests():
    """Topics are written only to ``state.date`` even when scan dates span multiple days."""

    async def run():
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            _touch(workspace / "daily" / "2026-05-26" / "old.md")
            _touch(workspace / "daily" / "2026-05-28" / "today.md")
            old_interests = workspace / "daily" / "2026-05-26" / "interests.yaml"
            _touch(old_interests, "date: 2026-05-26\ntopics: []\n")
            state = DreamState(
                date="2026-05-28",
                dates=["2026-05-26", "2026-05-27", "2026-05-28"],
                workspace=str(workspace),
                daily_dir="daily",
                topics=[
                    {
                        "title": "Old changed topic",
                        "reason": "Old daily material changed.",
                        "paths": ["daily/2026-05-26/old.md"],
                    },
                    {
                        "title": "Today changed topic",
                        "reason": "Today's daily material changed.",
                        "paths": ["daily/2026-05-28/today.md"],
                    },
                ],
            )
            step = DreamTopicsStep()
            resp = await step(RuntimeContext(dream=state.model_dump(), file_store=_FileStore(workspace)))

            target = workspace / "daily" / "2026-05-28" / "interests.yaml"
            dream = resp.metadata["dream"]
            assert resp.success is True
            assert target.is_file()
            assert old_interests.read_text(encoding="utf-8") == "date: 2026-05-26\ntopics: []\n"
            assert dream["interests_paths"] == ["daily/2026-05-28/interests.yaml"]
            assert dream["modified_paths"] == ["daily/2026-05-28/interests.yaml"]
            assert yaml.safe_load(target.read_text(encoding="utf-8"))["date"] == "2026-05-28"

    asyncio.run(run())


def test_topics_same_content_is_not_modified(tmp_path):
    """Rewriting deterministic interests content does not count as a user-visible change."""

    async def run():
        topic = {"title": "Topic", "reason": "Reason", "paths": ["daily/source.md"]}
        step = DreamTopicsStep()

        with patch("reme.steps.evolve.dream.topics.refresh_day_index", return_value={}):
            first = await step(
                RuntimeContext(
                    dream=DreamState(
                        date="2026-05-28",
                        workspace=str(tmp_path),
                        daily_dir="daily",
                        topics=[topic],
                    ).model_dump(),
                    file_store=_FileStore(tmp_path),
                ),
            )
            second = await step(
                RuntimeContext(
                    dream=DreamState(
                        date="2026-05-28",
                        workspace=str(tmp_path),
                        daily_dir="daily",
                        topics=[topic],
                    ).model_dump(),
                    file_store=_FileStore(tmp_path),
                ),
            )
            third = await step(
                RuntimeContext(
                    dream=DreamState(
                        date="2026-05-28",
                        workspace=str(tmp_path),
                        daily_dir="daily",
                        topics=[topic],
                    ).model_dump(),
                    file_store=_FileStore(tmp_path),
                ),
            )

        assert first.metadata["dream"]["modified_paths"] == ["daily/2026-05-28/interests.yaml"]
        assert second.metadata["dream"]["modified_paths"] == ["daily/2026-05-28/interests.yaml"]
        assert third.metadata["dream"]["modified_paths"] == []

    asyncio.run(run())


def test_topics_agent_failure_falls_back_to_candidates(tmp_path):
    """Topic ranking remains best-effort when the optional agent is unavailable."""

    async def run():
        state = DreamState(
            date="2026-05-28",
            workspace=str(tmp_path),
            daily_dir="daily",
            topics=[{"title": "Topic", "reason": "Reason", "paths": ["daily/source.md"]}],
        )
        step = DreamTopicsStep()
        step.agent_wrapper = _ReplyAgent(error=RuntimeError("temporary model failure"))

        with (
            patch("reme.steps.evolve.dream.topics.refresh_day_index", return_value={}),
            patch("reme.steps.evolve.dream.topics.llm_available", return_value=True),
        ):
            response = await step(
                RuntimeContext(
                    dream=state.model_dump(),
                    file_store=_FileStore(tmp_path),
                    agent_wrapper=step.agent_wrapper,
                ),
            )

        dream = response.metadata["dream"]
        assert response.success is True
        assert dream["topics_written"] == 1
        assert "deterministic fallback" in dream["warnings"][0]

    asyncio.run(run())


def test_topics_does_not_overwrite_invalid_existing_yaml(tmp_path):
    """A malformed user-owned interests file is preserved instead of treated as empty."""

    async def run():
        target = _touch(tmp_path / "daily" / "2026-05-28" / "interests.yaml", "topics: [\n")
        state = DreamState(
            date="2026-05-28",
            workspace=str(tmp_path),
            daily_dir="daily",
            topics=[{"title": "Topic", "reason": "Reason", "paths": ["daily/source.md"]}],
        )
        step = DreamTopicsStep()

        response = await step(RuntimeContext(dream=state.model_dump(), file_store=_FileStore(tmp_path)))

        assert response.success is False
        assert target.read_text(encoding="utf-8") == "topics: [\n"
        assert "Invalid interests YAML" in response.answer

    asyncio.run(run())


def test_topics_does_not_overwrite_invalid_existing_topic_entry(tmp_path):
    """Strict loading rejects entries that lenient loading would discard."""

    async def run():
        content = "topics:\n  - title: User topic\n    paths:\n      - daily/source.md\n"
        target = _touch(tmp_path / "daily" / "2026-05-28" / "interests.yaml", content)
        state = DreamState(
            date="2026-05-28",
            workspace=str(tmp_path),
            daily_dir="daily",
            topics=[{"title": "New topic", "reason": "New reason", "paths": ["daily/source.md"]}],
        )
        step = DreamTopicsStep()

        response = await step(RuntimeContext(dream=state.model_dump(), file_store=_FileStore(tmp_path)))

        assert response.success is False
        assert target.read_text(encoding="utf-8") == content
        assert "topics[0].reason must be a non-empty string" in response.answer

    asyncio.run(run())


def test_strict_topic_loading_rejects_lossy_fields(tmp_path):
    """Strict mode rejects values and fields that clean_topic would silently lose."""
    target = _touch(tmp_path / "interests.yaml", "topics:\n  - title: Topic\n    reason: Reason\n    custom: keep me\n")

    try:
        load_yaml_topics(target, strict=True)
    except ValueError as exc:
        assert "unknown field(s): custom" in str(exc)
    else:
        raise AssertionError("strict topic loading accepted a lossy field")


def test_strict_topic_loading_rejects_invalid_field_types(tmp_path):
    """Strict mode rejects list fields that would otherwise be normalized away."""
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


def test_proactive_answer_includes_topics_and_requested_content(tmp_path):
    """Successful proactive reads expose useful data through the primary answer."""

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

        response = await step(RuntimeContext(date="2026-05-28", include_content=True, file_store=_FileStore(tmp_path)))

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

    asyncio.run(run())


def test_proactive_answer_omits_unrequested_content(tmp_path):
    """Raw YAML is absent from the primary answer when include_content is false."""

    async def run():
        _touch(tmp_path / "daily" / "2026-05-28" / "interests.yaml", "topics:\n  - title: Topic\n    reason: Reason\n")
        step = ProactiveStep(app_context=ApplicationContext(workspace_dir=str(tmp_path)))

        response = await step(RuntimeContext(date="2026-05-28", include_content=False, file_store=_FileStore(tmp_path)))

        assert response.success is True
        assert "content" not in response.answer
        assert response.answer["topics"][0]["title"] == "Topic"
        assert response.metadata["content"] == ""

    asyncio.run(run())


def test_proactive_keeps_skipped_and_error_answers_explicit(tmp_path):
    """Empty and failure outcomes remain distinguishable without reading metadata."""

    async def run():
        step = ProactiveStep(app_context=ApplicationContext(workspace_dir=str(tmp_path)))
        skipped = await step(RuntimeContext(date="2026-05-28", file_store=_FileStore(tmp_path)))

        assert skipped.success is True
        assert skipped.answer == "Skipped: interests file not found at daily/2026-05-28/interests.yaml"
        assert skipped.metadata["skipped"] is True

        _touch(tmp_path / "daily" / "2026-05-28" / "interests.yaml", "topics: []\n")
        with patch("reme.steps.evolve.proactive.proactive.load_yaml_topics", side_effect=ValueError("bad topics")):
            failed = await step(RuntimeContext(date="2026-05-28", file_store=_FileStore(tmp_path)))

        assert failed.success is False
        assert failed.answer == "Error: ValueError: bad topics"
        assert failed.metadata["error"] == "ValueError: bad topics"

    asyncio.run(run())


def test_finish_does_not_checkpoint_failed_changed_paths():
    """Finish does not checkpoint failed changed paths."""

    async def run():
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            ok = _touch(workspace / "daily" / "2026-05-28" / "ok.md")
            failed = _touch(workspace / "daily" / "2026-05-28" / "failed.md")
            day_index = _touch(workspace / "daily" / "2026-05-28.md")
            interests = _touch(workspace / "daily" / "2026-05-28" / "interests.yaml")
            state = DreamState(
                date="2026-05-28",
                dates=["2026-05-26", "2026-05-27", "2026-05-28"],
                workspace=str(workspace),
                daily_dir="daily",
                changed_paths=[ok.relative_to(workspace).as_posix(), failed.relative_to(workspace).as_posix()],
                failed_paths=[failed.relative_to(workspace).as_posix()],
                interests_paths=[interests.relative_to(workspace).as_posix()],
                modified_paths=["digest/procedure/example.md"],
                integrate_results=[
                    {
                        "action": "CREATE",
                        "target_path": "digest/procedure/example.md",
                        "note": "Created a concise procedure node.",
                    },
                ],
            )
            step, catalog = DreamFinishStep(), _Catalog()
            resp = await step(RuntimeContext(dream=state.model_dump(), file_catalog=catalog))

            upserted = [n.path for n in catalog.upserts]
            assert resp.success is True
            assert resp.answer.startswith("AutoDream completed\n\n")
            assert "action:" not in resp.answer
            assert "topics:" not in resp.answer
            assert "Changes:" in resp.answer
            assert "- [digest/procedure/example.md][CREATE]: Created a concise procedure node." in resp.answer
            assert ok.relative_to(workspace).as_posix() in upserted
            assert failed.relative_to(workspace).as_posix() not in upserted
            assert interests.relative_to(workspace).as_posix() in upserted
            assert day_index.relative_to(workspace).as_posix() in upserted
            assert catalog.dumps == 1
            assert resp.metadata["modified"] is True

    asyncio.run(run())


def test_finish_does_not_readd_a_failed_day_index(tmp_path):
    """The generated day-index list cannot bypass the failed-path filter."""

    async def run():
        day_index = _touch(tmp_path / "daily" / "2026-05-28.md")
        rel_path = day_index.relative_to(tmp_path).as_posix()
        state = DreamState(
            date="2026-05-28",
            dates=["2026-05-28"],
            workspace=str(tmp_path),
            daily_dir="daily",
            changed_paths=[rel_path],
            failed_paths=[rel_path],
            errors=["extract unavailable"],
        )
        step, catalog = DreamFinishStep(), _Catalog()

        response = await step(RuntimeContext(dream=state.model_dump(), file_catalog=catalog))

        assert response.success is False
        assert not catalog.upserts
        assert response.metadata["dream"]["checkpoint_paths"] == []
        assert response.metadata["modified"] is False

    asyncio.run(run())


def test_finish_keeps_skipped_agent_output_successful(tmp_path):
    """Best-effort agent skips remain successful and checkpointed after their bounded retry."""

    async def run():
        rel_path = "daily/2026-05-28/source.md"
        _touch(tmp_path / rel_path)
        state = DreamState(
            date="2026-05-28",
            dates=["2026-05-28"],
            workspace=str(tmp_path),
            daily_dir="daily",
            changed_paths=[rel_path],
            skipped_units=[{"name": "unit", "reason": "unusable receipt"}],
            warnings=["unit 'unit' skipped: unusable agent receipt"],
        )
        step, catalog = DreamFinishStep(), _Catalog()

        response = await step(RuntimeContext(dream=state.model_dump(), file_catalog=catalog))

        assert response.success is True
        assert response.answer.startswith("AutoDream completed with warnings\n\n")
        assert "- Integrated: 0 ok, 1 skipped, 0 failed" in response.answer
        assert response.metadata["dream"]["checkpoint_paths"] == [rel_path]

    asyncio.run(run())
