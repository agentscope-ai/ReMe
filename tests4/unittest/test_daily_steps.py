"""Tests for daily-aware steps: daily_read / daily_write / daily_list / daily_reindex.

Sets up a small ``daily/`` tree with mixed dates and exercises note
read / write / list / index-rebuild operations. Arbitrary body
mid-edits, plain appends, and frontmatter mutations are generic CRUD
(covered in test_crud_steps and test_property_steps).

A daily note is the single file ``daily/<YYYY-MM-DD>/<slug>.md`` (no
folder, no sibling materials). ``daily_read`` returns body in answer
and parsed frontmatter as a dict in metadata; ``daily_write`` writes
body + frontmatter in one shot — ``overwrite=False`` (default) is an
idempotent skip-if-exists, ``overwrite=True`` is unconditional. Both
validate the slug up-front (Windows-safe filename rules) so the
path-shape contract is enforced at the daily boundary, not inside
generic CRUD.

``daily_list`` is now a **pure read** — it no longer triggers index
refresh. Use ``daily_reindex`` explicitly when the index page needs
to be rebuilt. ``daily_write`` auto-refreshes the index by default;
``frontmatter_update`` / ``file_append`` flows leave it stale and
require an explicit ``daily_reindex``.

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
    read as daily_read_step,
    write as daily_write_step,
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
    store = LocalFileStore(name="t", embedding_model="")
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


def test_daily_list_returns_path_slug_name_description():
    """Each note row exposes path / slug / name / description (and nothing else)."""

    async def run():
        with tempfile.TemporaryDirectory() as tmp, temp_chdir(tmp):
            store = LocalFileStore(name="t", embedding_model="")
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
                    "slug": "alpha",
                    "name": "Alpha Project",
                    "description": "JWT auth migration",
                },
            ]
            await store.close()
        print("✓ test_daily_list_returns_path_slug_name_description passed")

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
            paths = [n["path"] for n in payload["notes"]]
            assert paths == ["daily/2026-05-18/main.md"]
            await store.close()
        print("✓ test_daily_list_ignores_subdirectories passed")

    asyncio.run(run())


def test_daily_list_empty_when_no_daily_dir():
    """No daily/ folder ⇒ empty notes list, no crash."""

    async def run():
        with tempfile.TemporaryDirectory() as tmp, temp_chdir(tmp):
            store = LocalFileStore(name="t", embedding_model="")
            await store.start()
            step = daily_list_step.DailyListStep(file_store=store)
            await step(date="2026-05-18")
            payload = _metadata(step)
            assert payload == {"date": "2026-05-18", "notes": []}
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
    """daily_list returns only {date, notes}."""

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
        print("✓ test_daily_list_response_shape passed")

    asyncio.run(run())


# -- daily_read_step ----------------------------------------------------------


def test_daily_read_returns_body_and_frontmatter_dict():
    """daily_read on an existing note returns body in answer and parsed frontmatter dict."""

    async def run():
        with tempfile.TemporaryDirectory() as tmp, temp_chdir(tmp):
            store = LocalFileStore(name="t", embedding_model="")
            await store.start()
            day_dir = Path(tmp) / "daily" / "2026-05-18"
            day_dir.mkdir(parents=True, exist_ok=True)
            (day_dir / "alpha.md").write_text(
                "---\nname: Alpha Project\ndescription: JWT migration\n---\n## Objective\nfoo\n",
                encoding="utf-8",
            )

            step = daily_read_step.DailyReadStep(file_store=store)
            await step(slug="alpha", date="2026-05-18")
            payload = _metadata(step)

            assert step.context.response.success is True
            assert "## Objective\nfoo" in step.context.response.answer
            assert "---" not in step.context.response.answer  # frontmatter stripped
            assert payload["date"] == "2026-05-18"
            assert payload["slug"] == "alpha"
            assert payload["path"] == "daily/2026-05-18/alpha.md"
            assert payload["exists"] is True
            assert payload["frontmatter"] == {
                "name": "Alpha Project",
                "description": "JWT migration",
            }
            await store.close()
        print("✓ test_daily_read_returns_body_and_frontmatter_dict passed")

    asyncio.run(run())


def test_daily_read_default_date_is_today():
    """Omitted ``date`` ⇒ today's folder."""

    async def run():
        with tempfile.TemporaryDirectory() as tmp, temp_chdir(tmp):
            store = await _make_store_with_dailies(
                [(_today(), "live", "current body")],
            )
            step = daily_read_step.DailyReadStep(file_store=store)
            await step(slug="live")
            payload = _metadata(step)
            assert payload["date"] == _today()
            assert payload["path"] == f"daily/{_today()}/live.md"
            assert payload["exists"] is True
            assert "current body" in step.context.response.answer
            await store.close()
        print("✓ test_daily_read_default_date_is_today passed")

    asyncio.run(run())


def test_daily_read_missing_file_reports_exists_false():
    """Note absent ⇒ success=False, payload carries exists=False + path."""

    async def run():
        with tempfile.TemporaryDirectory() as tmp, temp_chdir(tmp):
            store = await _make_store_with_dailies([])
            step = daily_read_step.DailyReadStep(file_store=store)
            await step(slug="nothing-here", date="2026-05-18")
            payload = _metadata(step)

            assert step.context.response.success is False
            assert payload["exists"] is False
            assert payload["date"] == "2026-05-18"
            assert payload["slug"] == "nothing-here"
            assert payload["path"] == "daily/2026-05-18/nothing-here.md"
            await store.close()
        print("✓ test_daily_read_missing_file_reports_exists_false passed")

    asyncio.run(run())


def test_daily_read_rejects_invalid_slug():
    """Slug validation (Windows-safe filename rules) runs up-front."""

    async def run():
        with tempfile.TemporaryDirectory() as tmp, temp_chdir(tmp):
            store = await _make_store_with_dailies([])
            step = daily_read_step.DailyReadStep(file_store=store)
            for bad in ("foo/bar", "foo:bar", "foo*bar", "CON", "lpt9", "foo.", " bar"):
                await step(slug=bad)
                assert step.context.response.success is False, f"expected reject for {bad!r}"
            await store.close()
        print("✓ test_daily_read_rejects_invalid_slug passed")

    asyncio.run(run())


def test_daily_read_empty_frontmatter_dict():
    """No frontmatter ⇒ ``frontmatter`` key is the empty dict."""

    async def run():
        with tempfile.TemporaryDirectory() as tmp, temp_chdir(tmp):
            store = LocalFileStore(name="t", embedding_model="")
            await store.start()
            day_dir = Path(tmp) / "daily" / "2026-05-18"
            day_dir.mkdir(parents=True, exist_ok=True)
            (day_dir / "plain.md").write_text("just body\n", encoding="utf-8")

            step = daily_read_step.DailyReadStep(file_store=store)
            await step(slug="plain", date="2026-05-18")
            payload = _metadata(step)
            assert payload["frontmatter"] == {}
            assert step.context.response.answer.strip() == "just body"
            await store.close()
        print("✓ test_daily_read_empty_frontmatter_dict passed")

    asyncio.run(run())


# -- daily_write_step ---------------------------------------------------------


def test_daily_write_creates_note_and_refreshes_index():
    """Fresh slug ⇒ note file written + day index refreshed by default."""

    async def run():
        with tempfile.TemporaryDirectory() as tmp, temp_chdir(tmp):
            store = await _make_store_with_dailies([])
            step = daily_write_step.DailyWriteStep(file_store=store)
            await step(
                slug="kickoff",
                date="2026-05-18",
                body="## Plan\nfirst pass\n",
                frontmatter={"name": "Kickoff Task", "description": "Day-one plan"},
            )
            payload = _metadata(step)

            assert payload["created"] is True
            assert payload["overwritten"] is False
            assert payload["date"] == "2026-05-18"
            assert payload["slug"] == "kickoff"
            assert payload["path"] == "daily/2026-05-18/kickoff.md"

            note = Path(tmp) / "daily" / "2026-05-18" / "kickoff.md"
            text = note.read_text(encoding="utf-8")
            assert "name: Kickoff Task" in text
            assert "description: Day-one plan" in text
            assert "## Plan\nfirst pass" in text

            index = Path(tmp) / "daily" / "2026-05-18.md"
            assert index.is_file()
            assert "[[daily/2026-05-18/kickoff.md]]" in index.read_text(encoding="utf-8")
            await store.close()
        print("✓ test_daily_write_creates_note_and_refreshes_index passed")

    asyncio.run(run())


def test_daily_write_create_mode_is_idempotent():
    """``overwrite=False`` + file exists ⇒ created=False, file untouched, index still refreshed."""

    async def run():
        with tempfile.TemporaryDirectory() as tmp, temp_chdir(tmp):
            store = await _make_store_with_dailies(
                [("2026-05-18", "ongoing", "old body")],
            )
            file_path = Path(tmp) / "daily" / "2026-05-18" / "ongoing.md"
            before = file_path.read_text(encoding="utf-8")

            step = daily_write_step.DailyWriteStep(file_store=store)
            await step(slug="ongoing", date="2026-05-18", body="ignored new body")
            payload = _metadata(step)

            assert payload["created"] is False
            assert payload["overwritten"] is False
            assert payload["path"] == "daily/2026-05-18/ongoing.md"
            assert file_path.read_text(encoding="utf-8") == before
            assert payload["index"]["path"] == "daily/2026-05-18.md"
            await store.close()
        print("✓ test_daily_write_create_mode_is_idempotent passed")

    asyncio.run(run())


def test_daily_write_overwrite_mode_replaces_existing():
    """``overwrite=True`` ⇒ unconditional rewrite; overwritten=True."""

    async def run():
        with tempfile.TemporaryDirectory() as tmp, temp_chdir(tmp):
            store = await _make_store_with_dailies(
                [("2026-05-18", "live", "stale text")],
            )
            step = daily_write_step.DailyWriteStep(file_store=store)
            await step(
                slug="live",
                date="2026-05-18",
                body="## Updated\nfresh body\n",
                frontmatter={"name": "Live", "description": "post-merge"},
                overwrite=True,
            )
            payload = _metadata(step)

            assert payload["created"] is False
            assert payload["overwritten"] is True

            note = Path(tmp) / "daily" / "2026-05-18" / "live.md"
            text = note.read_text(encoding="utf-8")
            assert "stale text" not in text
            assert "## Updated\nfresh body" in text
            assert "description: post-merge" in text
            await store.close()
        print("✓ test_daily_write_overwrite_mode_replaces_existing passed")

    asyncio.run(run())


def test_daily_write_default_frontmatter_uses_slug_as_name():
    """Omitted ``frontmatter`` ⇒ defaults to ``{name: slug}``."""

    async def run():
        with tempfile.TemporaryDirectory() as tmp, temp_chdir(tmp):
            store = await _make_store_with_dailies([])
            step = daily_write_step.DailyWriteStep(file_store=store)
            await step(slug="auth-refactor", date="2026-05-18")

            note = Path(tmp) / "daily" / "2026-05-18" / "auth-refactor.md"
            assert "name: auth-refactor" in note.read_text(encoding="utf-8")
            await store.close()
        print("✓ test_daily_write_default_frontmatter_uses_slug_as_name passed")

    asyncio.run(run())


def test_daily_write_default_body_is_empty():
    """Omitted ``body`` ⇒ empty body, frontmatter-only note."""

    async def run():
        with tempfile.TemporaryDirectory() as tmp, temp_chdir(tmp):
            store = await _make_store_with_dailies([])
            step = daily_write_step.DailyWriteStep(file_store=store)
            await step(slug="stub", date="2026-05-18")

            note = Path(tmp) / "daily" / "2026-05-18" / "stub.md"
            text = note.read_text(encoding="utf-8")
            # Just frontmatter + a single newline after the closing ---.
            assert text.startswith("---\n")
            assert "name: stub" in text
            # Body section is empty: the post body resolves to "".
            assert text.rstrip().endswith("---")
            await store.close()
        print("✓ test_daily_write_default_body_is_empty passed")

    asyncio.run(run())


def test_daily_write_drops_empty_frontmatter_values():
    """Empty / None frontmatter values are dropped (write_step idiom)."""

    async def run():
        with tempfile.TemporaryDirectory() as tmp, temp_chdir(tmp):
            store = await _make_store_with_dailies([])
            step = daily_write_step.DailyWriteStep(file_store=store)
            await step(
                slug="trim",
                date="2026-05-18",
                frontmatter={
                    "name": "Trim",
                    "description": "   ",  # whitespace-only → drop
                    "extra": None,  # None → drop
                    "kept": "value",
                },
            )
            note = Path(tmp) / "daily" / "2026-05-18" / "trim.md"
            text = note.read_text(encoding="utf-8")
            assert "name: Trim" in text
            assert "kept: value" in text
            assert "description:" not in text
            assert "extra:" not in text
            await store.close()
        print("✓ test_daily_write_drops_empty_frontmatter_values passed")

    asyncio.run(run())


def test_daily_write_rejects_invalid_slug():
    """Slug validation runs before any IO."""

    async def run():
        with tempfile.TemporaryDirectory() as tmp, temp_chdir(tmp):
            store = await _make_store_with_dailies([])
            step = daily_write_step.DailyWriteStep(file_store=store)
            for bad in ("foo/bar", "foo:bar", "CON", "lpt9", "foo.", " bar"):
                await step(slug=bad, date="2026-05-18", body="x")
                assert step.context.response.success is False, f"expected reject for {bad!r}"
            # No day folder should be created on rejection.
            assert not (Path(tmp) / "daily" / "2026-05-18").exists()
            await store.close()
        print("✓ test_daily_write_rejects_invalid_slug passed")

    asyncio.run(run())


def test_daily_write_overwrite_default_is_false_create_then_skip():
    """First call (file absent) creates; second call without overwrite skips — proves the
    overwrite=False default mirrors the old daily_resolve idempotent probe."""

    async def run():
        with tempfile.TemporaryDirectory() as tmp, temp_chdir(tmp):
            store = await _make_store_with_dailies([])
            step = daily_write_step.DailyWriteStep(file_store=store)

            await step(slug="probe", date="2026-05-18", body="original")
            first = _metadata(step)
            assert first["created"] is True
            assert first["overwritten"] is False

            await step(slug="probe", date="2026-05-18", body="ignored")
            second = _metadata(step)
            assert second["created"] is False
            assert second["overwritten"] is False

            note = Path(tmp) / "daily" / "2026-05-18" / "probe.md"
            assert "original" in note.read_text(encoding="utf-8")
            assert "ignored" not in note.read_text(encoding="utf-8")
            await store.close()
        print("✓ test_daily_write_overwrite_default_is_false_create_then_skip passed")

    asyncio.run(run())


def test_daily_write_rejects_non_dict_frontmatter():
    """``frontmatter`` must be a dict when supplied."""

    async def run():
        with tempfile.TemporaryDirectory() as tmp, temp_chdir(tmp):
            store = await _make_store_with_dailies([])
            step = daily_write_step.DailyWriteStep(file_store=store)
            await step(slug="ok", date="2026-05-18", frontmatter="not a dict")
            assert step.context.response.success is False
            assert "dict" in (step.context.response.answer or "")
            await store.close()
        print("✓ test_daily_write_rejects_non_dict_frontmatter passed")

    asyncio.run(run())


def test_daily_write_refresh_index_can_be_disabled():
    """``refresh_index=False`` ⇒ note written but day index untouched."""

    async def run():
        with tempfile.TemporaryDirectory() as tmp, temp_chdir(tmp):
            store = await _make_store_with_dailies([])
            step = daily_write_step.DailyWriteStep(file_store=store)
            await step(
                slug="solo",
                date="2026-05-18",
                body="x",
                refresh_index=False,
            )
            note = Path(tmp) / "daily" / "2026-05-18" / "solo.md"
            assert note.is_file()
            assert "index" not in _metadata(step)
            assert not (Path(tmp) / "daily" / "2026-05-18.md").exists()
            await store.close()
        print("✓ test_daily_write_refresh_index_can_be_disabled passed")

    asyncio.run(run())


def test_daily_write_round_trips_with_daily_read():
    """A note written via daily_write must be retrievable via daily_read with the same data."""

    async def run():
        with tempfile.TemporaryDirectory() as tmp, temp_chdir(tmp):
            store = await _make_store_with_dailies([])
            body = "## Plan\nstep one\nstep two\n"
            fm = {"name": "Round-trip", "description": "CRUD smoke"}

            await daily_write_step.DailyWriteStep(file_store=store)(
                slug="round-trip",
                date="2026-05-18",
                body=body,
                frontmatter=fm,
            )
            read_step = daily_read_step.DailyReadStep(file_store=store)
            await read_step(slug="round-trip", date="2026-05-18")

            assert read_step.context.response.answer.strip() == body.strip()
            assert _metadata(read_step)["frontmatter"] == fm
            await store.close()
        print("✓ test_daily_write_round_trips_with_daily_read passed")

    asyncio.run(run())


# -- day index: daily/<date>.md ------------------------------------------


def _day_index_text(tmp: str, day: str) -> str:
    return (Path(tmp) / "daily" / f"{day}.md").read_text(encoding="utf-8")


def test_day_index_lists_each_note():
    """Multiple notes all show up in the index notes block with name."""

    async def run():
        with tempfile.TemporaryDirectory() as tmp, temp_chdir(tmp):
            store = LocalFileStore(name="t", embedding_model="")
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
    """Note ``description`` fields land in the rendered block so the
    index reads as a one-glance "what's happening today" summary.
    """

    async def run():
        with tempfile.TemporaryDirectory() as tmp, temp_chdir(tmp):
            store = LocalFileStore(name="t", embedding_model="")
            await store.start()
            cases = [
                ("alpha", "Alpha Project", "实现 JWT auth 中间件，迁移 session middleware"),
                ("beta", "beta", "调研增值税新政对 SaaS 的影响"),
                ("gamma", "Gamma", ""),
            ]
            for slug, name, description in cases:
                await _seed_note("2026-05-18", slug, name=name, description=description)

            await daily_reindex_step.DailyReindexStep(file_store=store)(date="2026-05-18")
            text = _day_index_text(tmp, "2026-05-18")
            assert "Alpha Project — 实现 JWT auth 中间件" in text
            assert "调研增值税新政对 SaaS 的影响" in text
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
            store = LocalFileStore(name="t", embedding_model="")
            await store.start()
            await _seed_note("2026-05-18", "alpha")
            reindex = daily_reindex_step.DailyReindexStep(file_store=store)
            await reindex(date="2026-05-18")

            index_path = Path(tmp) / "daily" / "2026-05-18.md"
            text = index_path.read_text(encoding="utf-8")
            patched = text.replace(
                "（人工记录区，刷新索引时不会动）",
                "MY HAND-WRITTEN NOTE\n这是我手写的备忘，不该被覆盖",
            )
            index_path.write_text(patched, encoding="utf-8")

            await _seed_note("2026-05-18", "beta")
            await reindex(date="2026-05-18")
            after = index_path.read_text(encoding="utf-8")
            assert "MY HAND-WRITTEN NOTE" in after
            assert "这是我手写的备忘" in after
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
    test_daily_list_returns_path_slug_name_description()
    test_daily_list_ignores_subdirectories()
    test_daily_list_empty_when_no_daily_dir()
    test_daily_list_does_not_refresh_index()
    test_daily_list_response_shape()
    test_daily_read_returns_body_and_frontmatter_dict()
    test_daily_read_default_date_is_today()
    test_daily_read_missing_file_reports_exists_false()
    test_daily_read_rejects_invalid_slug()
    test_daily_read_empty_frontmatter_dict()
    test_daily_write_creates_note_and_refreshes_index()
    test_daily_write_create_mode_is_idempotent()
    test_daily_write_overwrite_mode_replaces_existing()
    test_daily_write_default_frontmatter_uses_slug_as_name()
    test_daily_write_default_body_is_empty()
    test_daily_write_drops_empty_frontmatter_values()
    test_daily_write_rejects_invalid_slug()
    test_daily_write_overwrite_default_is_false_create_then_skip()
    test_daily_write_rejects_non_dict_frontmatter()
    test_daily_write_refresh_index_can_be_disabled()
    test_daily_write_round_trips_with_daily_read()
    test_day_index_lists_each_note()
    test_day_index_includes_note_descriptions()
    test_day_index_description_is_note_count()
    test_day_index_preserves_manual_segment()
    test_daily_reindex_returns_write_view()
    test_daily_reindex_created_flag_flips_on_rerun()
    print("\nAll tests passed!")
