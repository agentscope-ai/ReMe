"""Unit tests for the refactored dream package."""

import asyncio
import tempfile
from pathlib import Path

from reme.components.file_catalog import BaseFileCatalog
from reme.components.runtime_context import RuntimeContext
from reme.steps.evolve.dream.finish import DreamFinishStep
from reme.steps.evolve.dream.schema import DreamState
from reme.steps.evolve.dream.utils import parse_structured_reply, scan_day_files


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


def test_finish_does_not_checkpoint_failed_changed_paths():
    """Finish does not checkpoint failed changed paths."""

    async def run():
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            ok = _touch(workspace / "daily" / "2026-05-28" / "ok.md")
            failed = _touch(workspace / "daily" / "2026-05-28" / "failed.md")
            interests = _touch(workspace / "daily" / "2026-05-28" / "interests.yaml")
            state = DreamState(
                date="2026-05-28",
                workspace=str(workspace),
                changed_paths=[str(ok.relative_to(workspace)), str(failed.relative_to(workspace))],
                failed_paths=[str(failed.relative_to(workspace))],
                interests_path=str(interests.relative_to(workspace)),
            )
            step, catalog = DreamFinishStep(), _Catalog()
            resp = await step(RuntimeContext(dream=state.model_dump(), file_catalog=catalog))

            upserted = [n.path for n in catalog.upserts]
            assert resp.success is True
            assert str(ok.relative_to(workspace)) in upserted
            assert str(failed.relative_to(workspace)) not in upserted
            assert str(interests.relative_to(workspace)) in upserted
            assert catalog.dumps == 1

    asyncio.run(run())
