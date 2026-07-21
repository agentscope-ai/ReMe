"""Unit tests for the refactored dream package."""

import asyncio
import tempfile
from pathlib import Path
from unittest.mock import patch

import yaml

from reme.components.application_context import ApplicationContext
from reme.components.file_catalog import BaseFileCatalog
from reme.components.file_store import BaseFileStore
from reme.components.runtime_context import RuntimeContext
from reme.schema import DreamState
from reme.steps.evolve.dream.extract import DreamExtractStep
from reme.steps.evolve.dream.finish import DreamFinishStep
from reme.steps.evolve.dream.proactive import ProactiveStep
from reme.steps.evolve.dream.topics import DreamTopicsStep
from reme.steps.evolve.dream.utils import parse_structured_reply, recent_dates, scan_day_files


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


def test_extract_clean_output_respects_max_units():
    """Extract cleaning caps valid units at max_units."""
    state = DreamState(changed_paths=["daily/a.md"])
    meta = {
        "units": [
            {
                "name": f"unit-{i}",
                "bucket": "wiki",
                "summary": f"summary {i}",
                "paths": ["daily/a.md"],
            }
            for i in range(7)
        ],
    }

    DreamExtractStep().clean_output(state, meta, max_units=5)

    assert len(state.units) == 5
    assert [unit["name"] for unit in state.units] == [f"unit-{i}" for i in range(5)]


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
            resp = await step(
                RuntimeContext(dream=state.model_dump(), file_store=_FileStore(workspace)),
            )

            target = workspace / "daily" / "2026-05-28" / "interests.yaml"
            dream = resp.metadata["dream"]
            assert resp.success is True
            assert target.is_file()
            assert old_interests.read_text(encoding="utf-8") == "date: 2026-05-26\ntopics: []\n"
            assert dream["interests_paths"] == ["daily/2026-05-28/interests.yaml"]
            assert yaml.safe_load(target.read_text(encoding="utf-8"))["date"] == "2026-05-28"

    asyncio.run(run())


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

    asyncio.run(run())


def test_proactive_answer_omits_unrequested_content(tmp_path):
    """Raw YAML is absent from the primary answer when include_content is false."""

    async def run():
        _touch(
            tmp_path / "daily" / "2026-05-28" / "interests.yaml",
            "topics:\n  - title: Topic\n    reason: Reason\n",
        )
        step = ProactiveStep(app_context=ApplicationContext(workspace_dir=str(tmp_path)))

        response = await step(
            RuntimeContext(date="2026-05-28", include_content=False, file_store=_FileStore(tmp_path)),
        )

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
        with patch("reme.steps.evolve.dream.proactive.load_yaml_topics", side_effect=ValueError("bad topics")):
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
                changed_paths=[str(ok.relative_to(workspace)), str(failed.relative_to(workspace))],
                failed_paths=[str(failed.relative_to(workspace))],
                interests_paths=[str(interests.relative_to(workspace))],
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
            assert str(ok.relative_to(workspace)) in upserted
            assert str(failed.relative_to(workspace)) not in upserted
            assert str(interests.relative_to(workspace)) in upserted
            assert str(day_index.relative_to(workspace)) in upserted
            assert catalog.dumps == 1

    asyncio.run(run())
