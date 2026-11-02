"""Filename selection must preserve notes when generated titles collide."""

import asyncio

import frontmatter
import pytest

from reme_daily_paper.base import reserve_unique_note_path, resolve_unique_note_path, write_markdown


@pytest.mark.asyncio
@pytest.mark.parametrize("suffix", ["（2610.12345）", "（2026-10-31）"])
async def test_repeated_disk_collision_preserves_existing_notes(tmp_path, suffix):
    """A second occupied candidate must not be replaced."""
    title = "论文总结"
    occupied = [tmp_path / f"{title}.md", tmp_path / f"{title}{suffix}.md"]
    for path in occupied:
        path.write_text("unrelated user-owned note", encoding="utf-8")

    selected_title, destination = resolve_unique_note_path(
        tmp_path,
        title,
        taken=set(),
        taken_suffix=suffix,
        disk_suffix=suffix,
        existing=None,
    )
    await write_markdown(destination, "new generated content", {"kind": "daily-paper-analysis"})

    assert selected_title == f"{title}{suffix}{suffix}"
    assert destination == tmp_path / f"{selected_title}.md"
    assert all(path.read_text(encoding="utf-8") == "unrelated user-owned note" for path in occupied)


def test_reserved_title_and_disk_collision_both_advance_candidate(tmp_path):
    """Skip names reserved by this run as well as files on disk."""
    title = "论文总结"
    suffix = "（paper-id）"
    first = f"{title}{suffix}"
    second = f"{first}{suffix}"
    (tmp_path / f"{first}.md").write_text("existing note", encoding="utf-8")

    selected_title, destination = resolve_unique_note_path(
        tmp_path,
        title,
        taken={title, first, second},
        taken_suffix=suffix,
        disk_suffix=suffix,
        existing=None,
    )

    assert selected_title == f"{second}{suffix}"
    assert destination == tmp_path / f"{selected_title}.md"
    assert not destination.exists()


def test_explicitly_owned_note_remains_updateable(tmp_path):
    """The matched note for the same paper remains an update target."""
    path = tmp_path / "论文总结.md"
    path.write_text("old owned note", encoding="utf-8")

    selected_title, destination = resolve_unique_note_path(
        tmp_path,
        "论文总结",
        taken=set(),
        taken_suffix="（id）",
        disk_suffix="（id）",
        existing=path,
    )

    assert selected_title == "论文总结"
    assert destination == path


def test_single_collision_keeps_existing_suffix_convention(tmp_path):
    """The first collision still uses the established suffix."""
    title = "论文总结"
    suffix = "（paper-id）"
    (tmp_path / f"{title}.md").write_text("unrelated note", encoding="utf-8")

    selected_title, destination = resolve_unique_note_path(
        tmp_path,
        title,
        taken=set(),
        taken_suffix=suffix,
        disk_suffix=suffix,
        existing=None,
    )

    assert selected_title == f"{title}{suffix}"
    assert destination == tmp_path / f"{selected_title}.md"
    assert not destination.exists()


@pytest.mark.asyncio
async def test_concurrent_notes_reserve_distinct_paths(tmp_path):
    """Concurrent runs select and publish under one directory-scoped lock."""

    async def create_note(arxiv_id):
        async with reserve_unique_note_path(
            tmp_path,
            "论文总结",
            taken=set(),
            taken_suffix=f"（{arxiv_id}）",
            disk_suffix=f"（{arxiv_id}）",
            existing=None,
        ) as (title, path):
            await write_markdown(path, f"body from {arxiv_id}", {"arxiv_id": arxiv_id, "title": title})
        return path

    paths = await asyncio.gather(create_note("paper-a"), create_note("paper-b"))

    assert len(set(paths)) == 2
    assert {frontmatter.load(path).metadata["arxiv_id"] for path in paths} == {"paper-a", "paper-b"}
