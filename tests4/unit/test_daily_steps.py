"""Tests for daily-aware steps: daily_create / daily_list / daily_reindex.

Sets up a small ``daily/`` tree with mixed dates and exercises note
provision / listing / index-rebuild operations. Body authoring and
frontmatter mutations are generic CRUD (covered in test_crud_steps
and test_property_steps).

A daily note is the single file ``daily/<YYYY-MM-DD>/<session_id>.md``
(no folder, no sibling materials). ``daily_create`` validates the
session_id, writes an empty-body note with default
``{name: session_id}`` frontmatter when the file is absent, and
refreshes the day index. When the file already exists it is a no-op
write (``created=False``) — the body is filled in afterwards via
``file_write`` / ``file_edit`` / ``frontmatter_update`` or a native
editor.

``daily_list`` is a **pure read** — it never refreshes the index.
Use ``daily_reindex`` explicitly when the index page needs to be
rebuilt (e.g. after batch flows or a ``frontmatter_update`` that
touched ``name`` / ``description``).

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
from reme4.steps.file_io import (
    daily_create as daily_create_step,
    daily_list as daily_list_step,
    daily_reindex as daily_reindex_step,
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

    entries: list of (date, session_id, body). Each tuple creates
    ``daily/<date>/<session_id>.md`` with a minimal ``name``-only
    frontmatter — no opinionated status / lifecycle axes.
    """
    store = LocalFileStore(name="t", embedding_store="")
    await store.start()
    for day, session_id, body in entries:
        day_dir = Path.cwd() / "daily" / day
        day_dir.mkdir(parents=True, exist_ok=True)
        text = f"---\nname: {session_id}\n---\n{body}\n"
        (day_dir / f"{session_id}.md").write_text(text, encoding="utf-8")
    return store


def _metadata(step) -> dict:
    return step.context.response.metadata


async def _seed_note(date: str, session_id: str, name: str = "", description: str = "") -> None:
    """Write ``daily/<date>/<session_id>.md`` with optional frontmatter."""
    day_dir = Path.cwd() / "daily" / date
    day_dir.mkdir(parents=True, exist_ok=True)
    fm_lines = [f"name: {name or session_id}"]
    if description:
        fm_lines.append(f"description: {description}")
    text = "---\n" + "\n".join(fm_lines) + "\n---\nbody\n"
    (day_dir / f"{session_id}.md").write_text(text, encoding="utf-8")


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
            assert payload["count"] == 2
            answer = step.context.response.answer
            assert f"daily/{_today()}/today-a.md" in answer
            assert f"daily/{_today()}/today-b.md" in answer
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
            assert payload["count"] == 1
            answer = step.context.response.answer
            assert "daily/2026-05-18/a.md" in answer
            await store.close()
        print("✓ test_daily_list_filters_by_date passed")

    asyncio.run(run())


def test_daily_list_returns_path_session_id_metadata():
    """Each note row exposes path / session_id / metadata (full frontmatter dict)."""

    async def run():
        with tempfile.TemporaryDirectory() as tmp, temp_chdir(tmp):
            store = LocalFileStore(name="t", embedding_store="")
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
            assert payload["count"] == 1
            answer = step.context.response.answer
            assert "daily/2026-05-18/alpha.md" in answer
            assert "Alpha Project" in answer
            assert "JWT auth migration" in answer
            await store.close()
        print("✓ test_daily_list_returns_path_session_id_metadata passed")

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
            stray = Path(tmp) / "daily" / "2026-05-18" / "old-folder"
            stray.mkdir(parents=True, exist_ok=True)
            (stray / "old-folder.md").write_text(
                "---\nname: old\n---\nstale\n",
                encoding="utf-8",
            )

            step = daily_list_step.DailyListStep(file_store=store)
            await step(date="2026-05-18")
            payload = _metadata(step)
            assert payload["count"] == 1
            answer = step.context.response.answer
            assert "daily/2026-05-18/main.md" in answer
            await store.close()
        print("✓ test_daily_list_ignores_subdirectories passed")

    asyncio.run(run())


def test_daily_list_empty_when_no_daily_dir():
    """No daily/ folder ⇒ empty notes list, no crash."""

    async def run():
        with tempfile.TemporaryDirectory() as tmp, temp_chdir(tmp):
            store = LocalFileStore(name="t", embedding_store="")
            await store.start()
            step = daily_list_step.DailyListStep(file_store=store)
            await step(date="2026-05-18")
            payload = _metadata(step)
            assert payload == {"date": "2026-05-18", "count": 0}
            await store.close()
        print("✓ test_daily_list_empty_when_no_daily_dir passed")

    asyncio.run(run())


def test_daily_list_does_not_refresh_index():
    """daily_list is a pure read — it must NOT touch daily/<date>.md."""

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

            assert not index_path.exists(), "daily_list must not refresh the day index — use daily_reindex"
            await store.close()
        print("✓ test_daily_list_does_not_refresh_index passed")

    asyncio.run(run())


def test_daily_list_response_shape():
    """daily_list returns only {date, count}."""

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
            assert set(payload.keys()) == {"date", "count"}
            await store.close()
        print("✓ test_daily_list_response_shape passed")

    asyncio.run(run())


# -- daily_create_step --------------------------------------------------------


def test_daily_create_provisions_note_and_refreshes_index():
    """Fresh session_id ⇒ empty-body note with ``{name: session_id}`` + day index refreshed."""

    async def run():
        with tempfile.TemporaryDirectory() as tmp, temp_chdir(tmp):
            store = await _make_store_with_dailies([])
            step = daily_create_step.DailyCreateStep(file_store=store)
            await step(session_id="kickoff", date="2026-05-18")
            payload = _metadata(step)

            assert step.context.response.success is True
            assert payload["created"] is True
            assert payload["date"] == "2026-05-18"
            assert payload["session_id"] == "kickoff"
            assert payload["path"] == "daily/2026-05-18/kickoff.md"

            note = Path(tmp) / "daily" / "2026-05-18" / "kickoff.md"
            text = note.read_text(encoding="utf-8")
            assert "name: kickoff" in text
            # Body is empty — file is frontmatter + trailing newline.
            assert text.rstrip().endswith("---")

            index = Path(tmp) / "daily" / "2026-05-18.md"
            assert index.is_file()
            assert "[[daily/2026-05-18/kickoff.md]]" in index.read_text(encoding="utf-8")
            await store.close()
        print("✓ test_daily_create_provisions_note_and_refreshes_index passed")

    asyncio.run(run())


def test_daily_create_is_idempotent_on_existing():
    """File exists ⇒ ``created=False``; the file body is NOT touched; index still refreshes."""

    async def run():
        with tempfile.TemporaryDirectory() as tmp, temp_chdir(tmp):
            store = await _make_store_with_dailies(
                [("2026-05-18", "ongoing", "old body")],
            )
            file_path = Path(tmp) / "daily" / "2026-05-18" / "ongoing.md"
            before = file_path.read_text(encoding="utf-8")

            step = daily_create_step.DailyCreateStep(file_store=store)
            await step(session_id="ongoing", date="2026-05-18")
            payload = _metadata(step)

            assert step.context.response.success is True
            assert payload["created"] is False
            assert payload["path"] == "daily/2026-05-18/ongoing.md"
            assert file_path.read_text(encoding="utf-8") == before
            assert payload["index"]["path"] == "daily/2026-05-18.md"
            await store.close()
        print("✓ test_daily_create_is_idempotent_on_existing passed")

    asyncio.run(run())


def test_daily_create_default_date_is_today():
    """Omitted ``date`` ⇒ today's folder."""

    async def run():
        with tempfile.TemporaryDirectory() as tmp, temp_chdir(tmp):
            store = await _make_store_with_dailies([])
            step = daily_create_step.DailyCreateStep(file_store=store)
            await step(session_id="today-task")
            payload = _metadata(step)
            assert payload["date"] == _today()
            assert payload["path"] == f"daily/{_today()}/today-task.md"
            assert payload["created"] is True
            await store.close()
        print("✓ test_daily_create_default_date_is_today passed")

    asyncio.run(run())


def test_daily_create_default_frontmatter_uses_session_id_as_name():
    """The provisioned note's frontmatter is ``{name: session_id, description: ''}`` (no body)."""

    async def run():
        with tempfile.TemporaryDirectory() as tmp, temp_chdir(tmp):
            store = await _make_store_with_dailies([])
            step = daily_create_step.DailyCreateStep(file_store=store)
            await step(session_id="auth-refactor", date="2026-05-18")

            note = Path(tmp) / "daily" / "2026-05-18" / "auth-refactor.md"
            text = note.read_text(encoding="utf-8")
            assert "name: auth-refactor" in text
            assert "description:" in text
            await store.close()
        print("✓ test_daily_create_default_frontmatter_uses_session_id_as_name passed")

    asyncio.run(run())


def test_daily_create_rejects_invalid_session_id():
    """session_id validation runs before any IO; no day folder is created on reject."""

    async def run():
        with tempfile.TemporaryDirectory() as tmp, temp_chdir(tmp):
            store = await _make_store_with_dailies([])
            step = daily_create_step.DailyCreateStep(file_store=store)
            for bad in ("foo/bar", "foo:bar", "CON", "lpt9", "foo.", " bar"):
                await step(session_id=bad, date="2026-05-18")
                assert step.context.response.success is False, f"expected reject for {bad!r}"
            assert not (Path(tmp) / "daily" / "2026-05-18").exists()
            await store.close()
        print("✓ test_daily_create_rejects_invalid_session_id passed")

    asyncio.run(run())


def test_daily_create_empty_session_id_creates_day_level_file():
    """Empty session_id creates day-level file ``daily/<date>.md``."""

    async def run():
        with tempfile.TemporaryDirectory() as tmp, temp_chdir(tmp):
            store = await _make_store_with_dailies([])
            step = daily_create_step.DailyCreateStep(file_store=store)
            await step(session_id="", date="2026-05-18")
            assert step.context.response.success is True
            meta = step.context.response.metadata
            assert meta["path"] == "daily/2026-05-18.md"
            assert meta["session_id"] == ""
            assert meta["created"] is True
            assert Path(tmp, "daily", "2026-05-18.md").is_file()
            await store.close()
        print("✓ test_daily_create_empty_session_id_creates_day_level_file passed")

    asyncio.run(run())


def test_daily_create_then_skip_round_trip():
    """First call provisions, second call is an idempotent no-op write."""

    async def run():
        with tempfile.TemporaryDirectory() as tmp, temp_chdir(tmp):
            store = await _make_store_with_dailies([])
            step = daily_create_step.DailyCreateStep(file_store=store)

            await step(session_id="probe", date="2026-05-18")
            first = _metadata(step)
            assert first["created"] is True

            note = Path(tmp) / "daily" / "2026-05-18" / "probe.md"
            before = note.read_text(encoding="utf-8")

            await step(session_id="probe", date="2026-05-18")
            second = _metadata(step)
            assert second["created"] is False
            assert note.read_text(encoding="utf-8") == before
            await store.close()
        print("✓ test_daily_create_then_skip_round_trip passed")

    asyncio.run(run())


# -- day index: daily/<date>.md ------------------------------------------


def _day_index_text(tmp: str, day: str) -> str:
    return (Path(tmp) / "daily" / f"{day}.md").read_text(encoding="utf-8")


def test_day_index_lists_each_note():
    """Multiple notes all show up in the index notes block with name."""

    async def run():
        with tempfile.TemporaryDirectory() as tmp, temp_chdir(tmp):
            store = LocalFileStore(name="t", embedding_store="")
            await store.start()
            await _seed_note("2026-05-18", "alpha", name="Alpha Project")
            await _seed_note("2026-05-18", "beta", name="Beta Project")

            await daily_reindex_step.DailyReindexStep(file_store=store)(date="2026-05-18")
            text = _day_index_text(tmp, "2026-05-18")
            assert "[[daily/2026-05-18/alpha.md]]" in text
            assert "[[daily/2026-05-18/beta.md]]" in text
            assert "Alpha Project" in text
            assert "Beta Project" in text
            await store.close()
        print("✓ test_day_index_lists_each_note passed")

    asyncio.run(run())


def test_day_index_includes_note_descriptions():
    """Each note line inlines the full frontmatter (single-line, key: value pairs)."""

    async def run():
        with tempfile.TemporaryDirectory() as tmp, temp_chdir(tmp):
            store = LocalFileStore(name="t", embedding_store="")
            await store.start()
            cases = [
                ("alpha", "Alpha Project", "实现 JWT auth 中间件，迁移 session middleware"),
                ("beta", "beta", "调研增值税新政对 SaaS 的影响"),
                ("gamma", "Gamma", ""),
            ]
            for sid, name, description in cases:
                await _seed_note("2026-05-18", sid, name=name, description=description)

            await daily_reindex_step.DailyReindexStep(file_store=store)(date="2026-05-18")
            text = _day_index_text(tmp, "2026-05-18")
            # name + description inline on the same line as the wikilink
            assert "[[daily/2026-05-18/alpha.md]] name: Alpha Project description: 实现 JWT auth 中间件" in text
            assert "[[daily/2026-05-18/beta.md]] name: beta description: 调研增值税新政对 SaaS 的影响" in text
            # gamma has no description → only name is emitted, no trailing `description:` cruft
            assert "[[daily/2026-05-18/gamma.md]] name: Gamma\n" in text or text.rstrip().endswith(
                "[[daily/2026-05-18/gamma.md]] name: Gamma",
            )
            assert "description:" not in text.split("[[daily/2026-05-18/gamma.md]]")[1].split("\n")[0]
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
            assert "2 note(s) today." in text
            await store.close()
        print("✓ test_day_index_description_is_note_count passed")

    asyncio.run(run())


def test_day_index_preserves_user_content_outside_marker():
    """Any user-authored content sitting outside the auto markers is
    preserved verbatim across refreshes."""

    async def run():
        with tempfile.TemporaryDirectory() as tmp, temp_chdir(tmp):
            store = LocalFileStore(name="t", embedding_store="")
            await store.start()
            await _seed_note("2026-05-18", "alpha")
            reindex = daily_reindex_step.DailyReindexStep(file_store=store)
            await reindex(date="2026-05-18")

            index_path = Path(tmp) / "daily" / "2026-05-18.md"
            text = index_path.read_text(encoding="utf-8")
            # Append user content AFTER the auto block; it should survive refresh.
            user_block = "\n\n## 我的笔记\nMY HAND-WRITTEN NOTE\n这是我手写的备忘，不该被覆盖\n"
            index_path.write_text(text.rstrip() + user_block, encoding="utf-8")

            await _seed_note("2026-05-18", "beta")
            await reindex(date="2026-05-18")
            after = index_path.read_text(encoding="utf-8")
            assert "MY HAND-WRITTEN NOTE" in after
            assert "这是我手写的备忘" in after
            assert "## 我的笔记" in after
            assert "[[daily/2026-05-18/beta.md]]" in after
            await store.close()
        print("✓ test_day_index_preserves_user_content_outside_marker passed")

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
    test_daily_list_returns_path_session_id_metadata()
    test_daily_list_ignores_subdirectories()
    test_daily_list_empty_when_no_daily_dir()
    test_daily_list_does_not_refresh_index()
    test_daily_list_response_shape()
    test_daily_create_provisions_note_and_refreshes_index()
    test_daily_create_is_idempotent_on_existing()
    test_daily_create_default_date_is_today()
    test_daily_create_default_frontmatter_uses_session_id_as_name()
    test_daily_create_rejects_invalid_session_id()
    test_daily_create_empty_session_id_creates_day_level_file()
    test_daily_create_then_skip_round_trip()
    test_day_index_lists_each_note()
    test_day_index_includes_note_descriptions()
    test_day_index_description_is_note_count()
    test_day_index_preserves_user_content_outside_marker()
    test_daily_reindex_returns_write_view()
    test_daily_reindex_created_flag_flips_on_rerun()
    print("\nAll tests passed!")
