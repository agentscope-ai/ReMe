"""Regression tests for front matter handling in auto-dream extraction input (issue #730)."""

# pylint: disable=protected-access

from pathlib import Path

from reme.steps.evolve.dream.utils import pack_paths


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
