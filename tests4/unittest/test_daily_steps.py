"""Tests for daily-aware steps: daily_resolve / daily_create / daily_list / daily_reindex.

Sets up a small ``daily/`` tree with mixed dates and exercises note
genesis + list + index-rebuild operations. Body reads / writes are
generic CRUD (covered in test_crud_steps); arbitrary frontmatter
mutation is covered in test_property_steps.

A daily note is the single file ``daily/<YYYY-MM-DD>/<slug>.md``
(no folder, no sibling materials). ``daily_resolve`` ensures the day
folder ``daily/<today>/`` exists and returns the vault-relative path
to the note file, reporting whether it already ``exists`` — it
does **not** create the file itself. ``daily_create`` writes the
note stub with minimal ``name`` frontmatter and refreshes the day
index.

``daily_list`` and ``daily_reindex`` both call ``refresh_day_index``
(daily_list as a side effect; daily_reindex as its primary act). They
differ in payload shape: daily_list returns the per-note inventory
(read view), daily_reindex returns the write-result fields (write view).

Note: status / lifecycle / scope / role / source are no longer
core-reserved fields — the reme schema reserves only name /
description (both optional). Opinionated state machines belong
to the plugin layer.
"""

# pylint: disable=protected-access

import asyncio
import os
import tempfile
from datetime import date as _date
from pathlib import Path

import warnings

from reme4.components.file_store import LocalFileStore
from reme4.steps.daily import (
    resolve as daily_resolve_step,
    create as daily_create_step,
    list as daily_list_step,
    reindex as daily_reindex_step,
)

warnings.filterwarnings("ignore", category=DeprecationWarning, module="jieba")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="pkg_resources")


class temp_chdir:
    """Context manager: chdir into a path on enter, restore on exit."""

    def __init__(self, path):
        self.path = path
        self.old = None

    def __enter__(self):
        self.old = os.getcwd()
        os.chdir(self.path)
        return self

    def __exit__(self, *_):
        os.chdir(self.old)


def _today() -> str:
    return _date.today().isoformat()


async def _make_store_with_dailies(entries: list[tuple[str, str, str]]) -> LocalFileStore:
    """Seed the vault with daily notes.

    entries: list of (date, slug, body). Each tuple creates
    ``daily/<date>/<slug>.md`` with a minimal ``name``-only
    frontmatter — no opinionated status / lifecycle axes.
    """
    store = LocalFileStore(store_name="t", embedding_model="")
    await store.start()
    for day, slug, body in entries:
        day_dir = Path.cwd() / "daily" / day
        day_dir.mkdir(parents=True, exist_ok=True)
        text = f"---\nname: {slug}\n---\n{body}\n"
        (day_dir / f"{slug}.md").write_text(text, encoding="utf-8")
    return store


def _metadata(step) -> dict:
    return step.context.response.metadata


async def _seed_note(date: str, slug: str, name: str = "", description: str = "") -> None:
    """Write ``daily/<date>/<slug>.md`` with optional frontmatter."""
    day_dir = Path.cwd() / "daily" / date
    day_dir.mkdir(parents=True, exist_ok=True)
    fm_lines = [f"name: {name or slug}"]
    if description:
        fm_lines.append(f"description: {description}")
    text = "---\n" + "\n".join(fm_lines) + "\n---\nbody\n"
    (day_dir / f"{slug}.md").write_text(text, encoding="utf-8")


# -- daily_list_step ----------------------------------------------------------


def test_daily_list_default_date_is_today():
    """No ``date`` arg ⇒ falls back to today; only today's notes returned."""

    async def run():
        with tempfile.TemporaryDirectory() as tmp, temp_chdir(tmp):
            store = await _make_store_with_dailies(
                [
                    (_today(), "today-a", "today a"),
                    (_today(), "today-b", "today b"),
                    ("2026-05-17", "yesterday", "y"),
                ],
            )
            step = daily_list_step.DailyListStep(file_store=store)
            await step()
            payload = _metadata(step)
            assert payload["date"] == _today()
            paths = sorted(n["path"] for n in payload["notes"])
            assert paths == [
                f"daily/{_today()}/today-a.md",
                f"daily/{_today()}/today-b.md",
            ]
            await store.close()
        print("✓ test_daily_list_default_date_is_today passed")

    asyncio.run(run())


def test_daily_list_filters_by_date():
    """Explicit ``date`` scopes to that day's folder."""

    async def run():
        with tempfile.TemporaryDirectory() as tmp, temp_chdir(tmp):
            store = await _make_store_with_dailies(
                [
                    ("2026-05-18", "a", "a"),
                    ("2026-05-17", "b", "b"),
                ],
            )
            step = daily_list_step.DailyListStep(file_store=store)
            await step(date="2026-05-18")
            payload = _metadata(step)
            assert payload["date"] == "2026-05-18"
            paths = [n["path"] for n in payload["notes"]]
            assert paths == ["daily/2026-05-18/a.md"]
            await store.close()
        print("✓ test_daily_list_filters_by_date passed")

    asyncio.run(run())


def test_daily_list_returns_path_name_description():
    """Each note row exposes path / name / description (and nothing else)."""

    async def run():
        with tempfile.TemporaryDirectory() as tmp, temp_chdir(tmp):
            store = LocalFileStore(store_name="t", embedding_model="")
            await store.start()
            await _seed_note(
                "2026-05-18",
                "alpha",
                name="Alpha Project",
                description="JWT auth migration",
            )
            step = daily_list_step.DailyListStep(file_store=store)
            await step(date="2026-05-18")
            payload = _metadata(step)
            assert payload["notes"] == [
                {
                    "path": "daily/2026-05-18/alpha.md",
                    "name": "Alpha Project",
                    "description": "JWT auth migration",
                },
            ]
            await store.close()
        print("✓ test_daily_list_returns_path_name_description passed")

    asyncio.run(run())


def test_daily_list_ignores_subdirectories():
    """Subdirectories under the day folder are skipped — only direct .md files count."""

    async def run():
        with tempfile.TemporaryDirectory() as tmp, temp_chdir(tmp):
            store = await _make_store_with_dailies(
                [
                    ("2026-05-18", "main", "main body"),
                ],
            )
            # Stray subdir (e.g. left over from an old folder-shaped layout)
            # should not be picked up as a note.
            stray = Path(tmp) / "daily" / "2026-05-18" / "old-folder"
            stray.mkdir(parents=True, exist_ok=True)
            (stray / "old-folder.md").write_text(
                "---\nname: old\n---\nstale\n",
                encoding="utf-8",
            )

            step = daily_list_step.DailyListStep(file_store=store)
            await step(date="2026-05-18")
            payload = _metadata(step)
            paths = [n["path"] for n in payload["notes"]]
            assert paths == ["daily/2026-05-18/main.md"]
            await store.close()
        print("✓ test_daily_list_ignores_subdirectories passed")

    asyncio.run(run())


def test_daily_list_empty_when_no_daily_dir():
    """No daily/ folder ⇒ empty notes list, no crash."""

    async def run():
        with tempfile.TemporaryDirectory() as tmp, temp_chdir(tmp):
            store = LocalFileStore(store_name="t", embedding_model="")
            await store.start()
            step = daily_list_step.DailyListStep(file_store=store)
            await step(date="2026-05-18")
            payload = _metadata(step)
            assert payload == {"date": "2026-05-18", "notes": []}
            await store.close()
        print("✓ test_daily_list_empty_when_no_daily_dir passed")

    asyncio.run(run())


def test_daily_list_triggers_index_refresh_as_side_effect():
    """Calling daily_list also rebuilds daily/<date>.md (the index page)."""

    async def run():
        with tempfile.TemporaryDirectory() as tmp, temp_chdir(tmp):
            store = await _make_store_with_dailies(
                [
                    ("2026-05-18", "alpha", "a"),
                    ("2026-05-18", "beta", "b"),
                ],
            )
            index_path = Path(tmp) / "daily" / "2026-05-18.md"
            assert not index_path.exists()

            step = daily_list_step.DailyListStep(file_store=store)
            await step(date="2026-05-18")

            assert index_path.is_file()
            text = index_path.read_text(encoding="utf-8")
            assert "[[daily/2026-05-18/alpha.md]]" in text
            assert "[[daily/2026-05-18/beta.md]]" in text
            await store.close()
        print("✓ test_daily_list_triggers_index_refresh_as_side_effect passed")

    asyncio.run(run())


def test_daily_list_response_excludes_index_page_fields():
    """daily_list is the read view — no `path` / `created` fields leak through."""

    async def run():
        with tempfile.TemporaryDirectory() as tmp, temp_chdir(tmp):
            store = await _make_store_with_dailies(
                [
                    ("2026-05-18", "alpha", "a"),
                ],
            )
            step = daily_list_step.DailyListStep(file_store=store)
            await step(date="2026-05-18")
            payload = _metadata(step)
            assert set(payload.keys()) == {"date", "notes"}
            await store.close()
        print("✓ test_daily_list_response_excludes_index_page_fields passed")

    asyncio.run(run())


# -- daily_resolve_step -------------------------------------------------------


def test_daily_resolve_ensures_day_folder_and_reports_missing_file():
    """daily_resolve on a fresh name creates the day folder, leaves the note
    file unwritten, and reports ``exists=False``."""

    async def run():
        with tempfile.TemporaryDirectory() as tmp, temp_chdir(tmp):
            store = await _make_store_with_dailies([])
            step = daily_resolve_step.DailyResolveStep(file_store=store)
            await step(name="kickoff")
            payload = _metadata(step)

            assert payload["exists"] is False
            assert payload["name"] == "kickoff"
            assert payload["date"] == _today()
            assert payload["path"] == f"daily/{_today()}/kickoff.md"
            assert "message" not in payload

            day_dir = Path(tmp) / "daily" / _today()
            assert day_dir.is_dir()
            # The note file itself is NOT created by resolve.
            assert not (day_dir / "kickoff.md").exists()
            # No index page either.
            assert not (Path(tmp) / "daily" / f"{_today()}.md").exists()
            await store.close()
        print("✓ test_daily_resolve_ensures_day_folder_and_reports_missing_file passed")

    asyncio.run(run())


def test_daily_resolve_idempotent_when_file_exists():
    """Existing note file ⇒ ``exists=True`` + message; file contents untouched."""

    async def run():
        with tempfile.TemporaryDirectory() as tmp, temp_chdir(tmp):
            store = await _make_store_with_dailies(
                [(_today(), "ongoing", "morning thoughts")],
            )
            file_path = Path(tmp) / "daily" / _today() / "ongoing.md"
            before = file_path.read_text(encoding="utf-8")

            step = daily_resolve_step.DailyResolveStep(file_store=store)
            await step(name="ongoing")
            payload = _metadata(step)

            assert payload["exists"] is True
            assert payload["name"] == "ongoing"
            assert payload["path"] == f"daily/{_today()}/ongoing.md"
            assert "already exists" in payload["message"]

            # Contents unchanged.
            assert file_path.read_text(encoding="utf-8") == before
            await store.close()
        print("✓ test_daily_resolve_idempotent_when_file_exists passed")

    asyncio.run(run())


def test_daily_resolve_rejects_empty_name():
    """Empty name ⇒ error payload, success=False, no day folder created."""

    async def run():
        with tempfile.TemporaryDirectory() as tmp, temp_chdir(tmp):
            store = await _make_store_with_dailies([])
            step = daily_resolve_step.DailyResolveStep(file_store=store)
            await step(name="")
            payload = _metadata(step)

            assert "error" in payload
            assert "required" in payload["error"]
            assert step.context.response.success is False
            assert not (Path(tmp) / "daily" / _today()).exists()
            await store.close()
        print("✓ test_daily_resolve_rejects_empty_name passed")

    asyncio.run(run())


def test_daily_resolve_rejects_windows_invalid_chars():
    """Windows-reserved characters in name ⇒ error, no day folder created."""

    async def run():
        with tempfile.TemporaryDirectory() as tmp, temp_chdir(tmp):
            store = await _make_store_with_dailies([])
            step = daily_resolve_step.DailyResolveStep(file_store=store)
            for bad in (
                "foo/bar",
                "foo:bar",
                "foo*bar",
                "foo?bar",
                "foo|bar",
                "foo<bar",
                "foo>bar",
                'foo"bar',
                "foo\\bar",
            ):
                await step(name=bad)
                payload = _metadata(step)
                assert "error" in payload, f"expected error for {bad!r}, got {payload!r}"
                assert "invalid characters" in payload["error"]
                assert step.context.response.success is False
            await store.close()
        print("✓ test_daily_resolve_rejects_windows_invalid_chars passed")

    asyncio.run(run())


def test_daily_resolve_rejects_windows_reserved_names():
    """Windows device-name stems (CON / PRN / AUX / NUL / COM1-9 / LPT1-9) are rejected."""

    async def run():
        with tempfile.TemporaryDirectory() as tmp, temp_chdir(tmp):
            store = await _make_store_with_dailies([])
            step = daily_resolve_step.DailyResolveStep(file_store=store)
            for bad in ("CON", "prn", "Aux", "NUL", "COM1", "lpt9", "CON.notes", "com5.txt"):
                await step(name=bad)
                payload = _metadata(step)
                assert "error" in payload, f"expected error for {bad!r}, got {payload!r}"
                assert "reserved" in payload["error"]
            await store.close()
        print("✓ test_daily_resolve_rejects_windows_reserved_names passed")

    asyncio.run(run())


def test_daily_resolve_rejects_trailing_dot_or_whitespace():
    """Trailing '.' / leading-or-trailing whitespace are rejected."""

    async def run():
        with tempfile.TemporaryDirectory() as tmp, temp_chdir(tmp):
            store = await _make_store_with_dailies([])
            step = daily_resolve_step.DailyResolveStep(file_store=store)
            for bad in ("foo.", "foo ", " foo", "  bar  "):
                await step(name=bad)
                payload = _metadata(step)
                assert "error" in payload, f"expected error for {bad!r}, got {payload!r}"
            await store.close()
        print("✓ test_daily_resolve_rejects_trailing_dot_or_whitespace passed")

    asyncio.run(run())


# -- daily_create_step --------------------------------------------------------


def test_daily_create_writes_file_and_refreshes_index():
    """Fresh slug ⇒ note file created with `name` frontmatter + index refreshed."""

    async def run():
        with tempfile.TemporaryDirectory() as tmp, temp_chdir(tmp):
            store = await _make_store_with_dailies([])
            step = daily_create_step.DailyCreateStep(file_store=store)
            await step(slug="kickoff", date="2026-05-18", name="Kickoff Task", body="hello body")
            payload = _metadata(step)

            assert payload["created"] is True
            assert payload["date"] == "2026-05-18"
            assert payload["slug"] == "kickoff"
            assert payload["path"] == "daily/2026-05-18/kickoff.md"

            note = Path(tmp) / "daily" / "2026-05-18" / "kickoff.md"
            assert note.is_file()
            text = note.read_text(encoding="utf-8")
            assert "name: Kickoff Task" in text
            assert "hello body" in text

            # Index page refreshed.
            index = Path(tmp) / "daily" / "2026-05-18.md"
            assert index.is_file()
            assert "[[daily/2026-05-18/kickoff.md]]" in index.read_text(encoding="utf-8")
            await store.close()
        print("✓ test_daily_create_writes_file_and_refreshes_index passed")

    asyncio.run(run())


def test_daily_create_is_idempotent():
    """Existing note ⇒ `created=False`, file untouched, index still refreshes."""

    async def run():
        with tempfile.TemporaryDirectory() as tmp, temp_chdir(tmp):
            store = await _make_store_with_dailies(
                [("2026-05-18", "ongoing", "old body")],
            )
            file_path = Path(tmp) / "daily" / "2026-05-18" / "ongoing.md"
            before = file_path.read_text(encoding="utf-8")

            step = daily_create_step.DailyCreateStep(file_store=store)
            await step(slug="ongoing", date="2026-05-18", body="ignored new body")
            payload = _metadata(step)

            assert payload["created"] is False
            assert payload["path"] == "daily/2026-05-18/ongoing.md"
            # File contents unchanged.
            assert file_path.read_text(encoding="utf-8") == before
            # But the index was still rebuilt.
            assert payload["index"]["path"] == "daily/2026-05-18.md"
            await store.close()
        print("✓ test_daily_create_is_idempotent passed")

    asyncio.run(run())


def test_daily_create_name_falls_back_to_slug():
    """Omitted ``name`` arg ⇒ frontmatter ``name`` defaults to slug."""

    async def run():
        with tempfile.TemporaryDirectory() as tmp, temp_chdir(tmp):
            store = await _make_store_with_dailies([])
            step = daily_create_step.DailyCreateStep(file_store=store)
            await step(slug="auth-refactor", date="2026-05-18")

            note = Path(tmp) / "daily" / "2026-05-18" / "auth-refactor.md"
            assert "name: auth-refactor" in note.read_text(encoding="utf-8")
            await store.close()
        print("✓ test_daily_create_name_falls_back_to_slug passed")

    asyncio.run(run())


# -- day index: daily/<date>.md ------------------------------------------


def _day_index_text(tmp: str, day: str) -> str:
    return (Path(tmp) / "daily" / f"{day}.md").read_text(encoding="utf-8")


def test_day_index_lists_each_note():
    """Multiple notes all show up in the index notes block with name."""

    async def run():
        with tempfile.TemporaryDirectory() as tmp, temp_chdir(tmp):
            store = LocalFileStore(store_name="t", embedding_model="")
            await store.start()
            await _seed_note("2026-05-18", "alpha", name="Alpha Project")
            await _seed_note("2026-05-18", "beta", name="Beta Project")

            await daily_reindex_step.DailyReindexStep(file_store=store)(date="2026-05-18")
            text = _day_index_text(tmp, "2026-05-18")
            assert "[[daily/2026-05-18/alpha.md]]" in text
            assert "[[daily/2026-05-18/beta.md]]" in text
            # Note names show on the indented sub-line.
            assert "Alpha Project" in text
            assert "Beta Project" in text
            await store.close()
        print("✓ test_day_index_lists_each_note passed")

    asyncio.run(run())


def test_day_index_includes_note_descriptions():
    """Note ``description`` fields land in the rendered block so the
    index reads as a one-glance "what's happening today" summary.
    """

    async def run():
        with tempfile.TemporaryDirectory() as tmp, temp_chdir(tmp):
            store = LocalFileStore(store_name="t", embedding_model="")
            await store.start()
            cases = [
                ("alpha", "Alpha Project", "实现 JWT auth 中间件，迁移 session middleware"),
                ("beta", "beta", "调研增值税新政对 SaaS 的影响"),  # name == slug
                ("gamma", "Gamma", ""),  # no description
            ]
            for slug, name, description in cases:
                await _seed_note("2026-05-18", slug, name=name, description=description)

            await daily_reindex_step.DailyReindexStep(file_store=store)(date="2026-05-18")
            text = _day_index_text(tmp, "2026-05-18")
            # name + description rendered together
            assert "Alpha Project — 实现 JWT auth 中间件" in text
            # name == slug → only description shown (no redundant "beta")
            assert "调研增值税新政对 SaaS 的影响" in text
            # no description → only name shown, no trailing em-dash
            assert "  Gamma\n" in text or text.rstrip().endswith("Gamma")
            await store.close()
        print("✓ test_day_index_includes_note_descriptions passed")

    asyncio.run(run())


def test_day_index_description_is_note_count():
    """The typed ``description`` field carries a one-line note-count digest."""

    async def run():
        with tempfile.TemporaryDirectory() as tmp, temp_chdir(tmp):
            store = await _make_store_with_dailies(
                [
                    ("2026-05-18", "a", "a body"),
                    ("2026-05-18", "b", "b body"),
                ],
            )
            step = daily_reindex_step.DailyReindexStep(file_store=store)
            await step(date="2026-05-18")

            text = _day_index_text(tmp, "2026-05-18")
            assert "description:" in text
            assert "2 篇笔记" in text
            await store.close()
        print("✓ test_day_index_description_is_note_count passed")

    asyncio.run(run())


def test_day_index_preserves_manual_segment():
    """The ``## 备忘`` (manual) segment is preserved across refreshes."""

    async def run():
        with tempfile.TemporaryDirectory() as tmp, temp_chdir(tmp):
            store = LocalFileStore(store_name="t", embedding_model="")
            await store.start()
            await _seed_note("2026-05-18", "alpha")
            reindex = daily_reindex_step.DailyReindexStep(file_store=store)
            await reindex(date="2026-05-18")

            # Inject a manual annotation into the index file's body.
            index_path = Path(tmp) / "daily" / "2026-05-18.md"
            text = index_path.read_text(encoding="utf-8")
            patched = text.replace(
                "（人工记录区，刷新索引时不会动）",
                "MY HAND-WRITTEN NOTE\n这是我手写的备忘，不该被覆盖",
            )
            index_path.write_text(patched, encoding="utf-8")

            # Adding a sibling note + refresh — manual segment must survive.
            await _seed_note("2026-05-18", "beta")
            await reindex(date="2026-05-18")
            after = index_path.read_text(encoding="utf-8")
            assert "MY HAND-WRITTEN NOTE" in after
            assert "这是我手写的备忘" in after
            # Auto block was updated with the new note.
            assert "[[daily/2026-05-18/beta.md]]" in after
            await store.close()
        print("✓ test_day_index_preserves_manual_segment passed")

    asyncio.run(run())


# -- daily_reindex_step -----------------------------------------------------


def test_daily_reindex_returns_write_view():
    """daily_reindex returns {date, path, created, notes_count}."""

    async def run():
        with tempfile.TemporaryDirectory() as tmp, temp_chdir(tmp):
            store = await _make_store_with_dailies(
                [
                    ("2026-05-18", "alpha", "a body"),
                    ("2026-05-18", "beta", "b body"),
                ],
            )
            # Index doesn't exist yet.
            assert not (Path(tmp) / "daily" / "2026-05-18.md").exists()

            step = daily_reindex_step.DailyReindexStep(file_store=store)
            await step(date="2026-05-18")
            payload = _metadata(step)

            assert set(payload.keys()) == {"date", "path", "created", "notes_count"}
            assert payload["date"] == "2026-05-18"
            assert payload["path"] == "daily/2026-05-18.md"
            assert payload["created"] is True
            assert payload["notes_count"] == 2

            text = _day_index_text(tmp, "2026-05-18")
            assert "[[daily/2026-05-18/alpha.md]]" in text
            assert "[[daily/2026-05-18/beta.md]]" in text
            await store.close()
        print("✓ test_daily_reindex_returns_write_view passed")

    asyncio.run(run())


def test_daily_reindex_created_flag_flips_on_rerun():
    """First call creates the index (created=True); re-run reports created=False."""

    async def run():
        with tempfile.TemporaryDirectory() as tmp, temp_chdir(tmp):
            store = await _make_store_with_dailies(
                [("2026-05-18", "alpha", "a")],
            )
            step = daily_reindex_step.DailyReindexStep(file_store=store)

            await step(date="2026-05-18")
            payload_first = _metadata(step)
            assert payload_first["created"] is True

            await step(date="2026-05-18")
            payload_second = _metadata(step)
            assert payload_second["created"] is False
            assert payload_second["notes_count"] == 1
            await store.close()
        print("✓ test_daily_reindex_created_flag_flips_on_rerun passed")

    asyncio.run(run())


if __name__ == "__main__":
    print("\n=== Daily step tests ===")
    test_daily_list_default_date_is_today()
    test_daily_list_filters_by_date()
    test_daily_list_returns_path_name_description()
    test_daily_list_ignores_subdirectories()
    test_daily_list_empty_when_no_daily_dir()
    test_daily_list_triggers_index_refresh_as_side_effect()
    test_daily_list_response_excludes_index_page_fields()
    test_daily_resolve_ensures_day_folder_and_reports_missing_file()
    test_daily_resolve_idempotent_when_file_exists()
    test_daily_resolve_rejects_empty_name()
    test_daily_resolve_rejects_windows_invalid_chars()
    test_daily_resolve_rejects_windows_reserved_names()
    test_daily_resolve_rejects_trailing_dot_or_whitespace()
    test_daily_create_writes_file_and_refreshes_index()
    test_daily_create_is_idempotent()
    test_daily_create_name_falls_back_to_slug()
    test_day_index_lists_each_note()
    test_day_index_includes_note_descriptions()
    test_day_index_description_is_note_count()
    test_day_index_preserves_manual_segment()
    test_daily_reindex_returns_write_view()
    test_daily_reindex_created_flag_flips_on_rerun()
    print("\nAll tests passed!")
