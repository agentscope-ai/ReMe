"""Regression tests for front matter handling in auto-dream extraction input (issue #730)."""

# pylint: disable=protected-access

from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from reme.components.application_context import ApplicationContext
from reme.components.job import BaseJob
from reme.config import resolve_app_config
from reme.steps.evolve.dream.utils import pack_paths

from .test_auto_dream import _Catalog, _FileStore, _SequenceAgent


def _write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def test_pack_paths_strips_frontmatter(tmp_path):
    """Daily-note front matter must not reach the extraction prompt."""
    note = _write(
        tmp_path / "daily" / "2026-05-28" / "session.md",
        "---\n"
        "name: router-troubleshooting\n"
        "description: a durable lesson about WG routing\n"
        "tags: [troubleshooting, network]\n"
        "---\n\n"
        "# Body\n\n"
        "The actual reusable abstraction lives here.\n",
    )

    blob = pack_paths(tmp_path, [note.relative_to(tmp_path).as_posix()])

    assert "The actual reusable abstraction lives here." in blob
    for key in ("name:", "description:", "tags:"):
        assert key not in blob, f"front-matter key leaked into material: {key}"


def test_pack_paths_keeps_body_without_frontmatter(tmp_path):
    """Plain notes are packed verbatim; stripping must not eat real content."""
    note = _write(tmp_path / "daily" / "2026-05-28" / "plain.md", "no front matter here\nsecond line\n")

    blob = pack_paths(tmp_path, [note.relative_to(tmp_path).as_posix()])

    assert "no front matter here" in blob
    assert "second line" in blob


def test_pack_paths_truncation_budget_counts_body_only(tmp_path):
    """The per-file budget applies to the packed body, not the discarded front matter."""
    big = "---\nname: n\ndescription: " + "d" * 5000 + "\n---\n\nbody text\n"
    note = _write(tmp_path / "daily" / "2026-05-28" / "big.md", big)

    blob = pack_paths(tmp_path, [note.relative_to(tmp_path).as_posix()], limit_per_file=200)

    assert "body text" in blob
    assert "dddd" not in blob


def test_pack_paths_handles_multiple_frontmatter_blocks(tmp_path):
    """Consecutive front matter blocks are all removed, matching the shared helper."""
    note = _write(
        tmp_path / "daily" / "2026-05-28" / "double.md",
        "---\nname: first\n---\n---\ntags: [a, b]\n---\n\nreal body\n",
    )

    blob = pack_paths(tmp_path, [note.relative_to(tmp_path).as_posix()])

    assert "real body" in blob
    assert "name:" not in blob
    assert "tags:" not in blob


def test_pack_paths_preserves_body_that_opens_with_a_thematic_break(tmp_path):
    """Review case (#738): a body-level ``---`` is Markdown, not another block.

    A block is only consumed once it is known to be a mapping; otherwise the
    whole recovery instruction used to be dropped and the note still got
    checkpointed as processed.
    """
    note = _write(
        tmp_path / "daily" / "2026-05-28" / "recovery.md",
        "---\n"
        "name: recovery\n"
        "description: recovery procedure\n"
        "---\n\n"
        "---\n"
        "# Recovery procedure\n"
        "Rotate the recovery key before restarting the gateway.\n"
        "---\n",
    )

    blob = pack_paths(tmp_path, [note.relative_to(tmp_path).as_posix()])

    assert "Rotate the recovery key before restarting the gateway." in blob
    assert "# Recovery procedure" in blob
    assert "name:" not in blob


def test_pack_paths_survives_unconstructible_metadata_value(tmp_path):
    """Review case (#738): ``safe_load`` raises ``ValueError`` on bad timestamps."""
    note = _write(
        tmp_path / "daily" / "2026-05-28" / "bad-date.md",
        "---\n"
        "name: recovery\n"
        "review_date: 2026-02-30\n"
        "---\n\n"
        "Rotate the recovery key before restarting the gateway.\n",
    )

    blob = pack_paths(tmp_path, [note.relative_to(tmp_path).as_posix()])

    assert "Rotate the recovery key before restarting the gateway." in blob
    # Ambiguous metadata is retained, not silently discarded on constructor errors.
    assert note.read_text(encoding="utf-8").strip() in blob


@pytest.mark.parametrize("prefix", ["", "---\nname: recovery\n---\n\n"])
def test_pack_paths_preserves_markdown_that_is_invalid_yaml(tmp_path, prefix):
    """A YAML parser error must not turn a Markdown body into discarded metadata."""
    body = "---\n# Recovery procedure\nInstruction: [keep this unfinished example\n---"
    note = _write(tmp_path / "daily/2026-05-28/recovery.md", prefix + body)

    blob = pack_paths(tmp_path, [note.relative_to(tmp_path).as_posix()])

    assert body in blob
    assert "name: recovery" not in blob


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "raw,expected_body",
    [
        (
            "---\nname: recovery\n---\n\n---\n# Recovery procedure\nRotate the recovery key.\n---\n",
            "---\n# Recovery procedure\nRotate the recovery key.\n---",
        ),
        (
            "---\nname: recovery\nreview_date: 2026-02-30\n---\n\nRotate the recovery key.\n",
            "---\nname: recovery\nreview_date: 2026-02-30\n---\n\nRotate the recovery key.",
        ),
        (
            "---\nname: recovery\n---\n\n---\nInstruction: [keep this unfinished example\n---\n",
            "---\nInstruction: [keep this unfinished example\n---",
        ),
    ],
    ids=["thematic-break", "invalid-timestamp", "invalid-yaml-body"],
)
async def test_configured_dream_sees_body_before_checkpointing(tmp_path, monkeypatch, raw, expected_body):
    """Empty extraction is valid only after the model has received the note's body."""

    class Catalog(_Catalog):
        """Expose checkpointed versions to the next configured job invocation."""

        async def get_nodes(self, paths=None):
            return self.upserts

    rel = "daily/2026-05-28/session.md"
    note = _write(tmp_path / rel, raw)
    _write(tmp_path / "daily/2026-05-28.md", "day index")
    catalog = Catalog()
    agent = _SequenceAgent({"result": '{"units": []}'})
    config = resolve_app_config(config="default", log_config=False)["jobs"]["auto_dream"]
    job = BaseJob(
        name="auto_dream",
        steps=config["steps"],
        app_context=ApplicationContext(workspace_dir=str(tmp_path)),
    )
    monkeypatch.setattr("reme.steps.evolve.dream.extract.refresh_day_index", AsyncMock(return_value={}))
    monkeypatch.setattr("reme.steps.evolve.dream.extract.llm_available", lambda _step: True)
    await job.start()
    try:
        inputs = {
            "date": "2026-05-28",
            "scan_days": 1,
            "agent_wrapper": agent,
            "file_catalog": catalog,
            "file_store": _FileStore(tmp_path),
        }
        response = await job(**inputs)

        assert agent.calls == 1
        assert expected_body in agent.requests[0][0]
        assert response.success
        assert rel in response.metadata["dream"]["checkpoint_paths"]
        assert note.read_text(encoding="utf-8") == raw

        response = await job(**inputs)
        assert response.success
        assert response.metadata["dream"]["files_changed"] == 0
        assert agent.calls == 1
    finally:
        await job.close()
