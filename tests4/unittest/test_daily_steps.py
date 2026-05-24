"""Tests for daily-aware steps: daily_resolve_step / daily_list_step / daily_reindex_step.

Sets up a small daily/ tree with mixed dates and exercises folder
genesis + list + index-rebuild operations. Body reads / writes are
generic CRUD (covered in test_crud_steps); arbitrary frontmatter
mutation is covered in test_property_steps.

``daily_resolve`` is intentionally minimal: it creates the folder
``daily/<today>/<name>/`` if missing and returns its vault-relative
path. Body / frontmatter / day-index writes go through the generic
CRUD + daily_reindex steps.

``daily_list`` and ``daily_reindex`` both call ``refresh_day_index``
(daily_list as a side effect; daily_reindex as its primary act). They
differ in payload shape: daily_list returns the per-workspace inventory
(read view), daily_reindex returns the write-result fields (write view).

Note: status / lifecycle / scope / role / source are no longer
core-reserved fields — the reme schema reserves only name /
description (both optional). Opinionated state machines belong
to the plugin layer.
"""

# pylint: disable=protected-access

import asyncio
import json
import os
import tempfile
from datetime import date as _date
from pathlib import Path

import warnings

from reme4.components.file_store import LocalFileStore
from reme4.steps.daily import (
    resolve as daily_resolve_step,
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
    """Seed the vault with daily workspaces.

    entries: list of (date, slug, summary_body). Each tuple creates
    ``daily/<date>/<slug>/<slug>.md`` with a minimal ``name``-only
    frontmatter — no opinionated status / lifecycle axes.
    """
    store = LocalFileStore(store_name="t", embedding_model="")
    await store.start()
    for day, slug, body in entries:
        folder = Path.cwd() / "daily" / day / slug
        folder.mkdir(parents=True, exist_ok=True)
        text = f"---\nname: {slug}\n---\n{body}\n"
        (folder / f"{slug}.md").write_text(text, encoding="utf-8")
    return store


def _answer(step) -> dict:
    return json.loads(step.context.response.answer)


async def _seed_workspace(date: str, slug: str, name: str = "", description: str = "") -> None:
    """Write ``daily/<date>/<slug>/<slug>.md`` with optional frontmatter."""
    folder = Path.cwd() / "daily" / date / slug
    folder.mkdir(parents=True, exist_ok=True)
    fm_lines = [f"name: {name or slug}"]
    if description:
        fm_lines.append(f"description: {description}")
    text = "---\n" + "\n".join(fm_lines) + "\n---\nbody\n"
    (folder / f"{slug}.md").write_text(text, encoding="utf-8")


# -- daily_list_step ----------------------------------------------------------


def test_daily_list_default_date_is_today():
    """No ``date`` arg ⇒ falls back to today; only today's workspaces returned."""

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
            payload = _answer(step)
            assert payload["date"] == _today()
            paths = sorted(w["path"] for w in payload["workspaces"])
            assert paths == [
                f"daily/{_today()}/today-a/today-a.md",
                f"daily/{_today()}/today-b/today-b.md",
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
            payload = _answer(step)
            assert payload["date"] == "2026-05-18"
            paths = [w["path"] for w in payload["workspaces"]]
            assert paths == ["daily/2026-05-18/a/a.md"]
            await store.close()
        print("✓ test_daily_list_filters_by_date passed")

    asyncio.run(run())


def test_daily_list_returns_path_name_description():
    """Each workspace row exposes path / name / description (and nothing else)."""

    async def run():
        with tempfile.TemporaryDirectory() as tmp, temp_chdir(tmp):
            store = LocalFileStore(store_name="t", embedding_model="")
            await store.start()
            await _seed_workspace(
                "2026-05-18",
                "alpha",
                name="Alpha Project",
                description="JWT auth migration",
            )
            step = daily_list_step.DailyListStep(file_store=store)
            await step(date="2026-05-18")
            payload = _answer(step)
            assert payload["workspaces"] == [
                {
                    "path": "daily/2026-05-18/alpha/alpha.md",
                    "name": "Alpha Project",
                    "description": "JWT auth migration",
                },
            ]
            await store.close()
        print("✓ test_daily_list_returns_path_name_description passed")

    asyncio.run(run())


def test_daily_list_ignores_material_files():
    """Sibling material files in a workspace folder don't get listed as workspaces."""

    async def run():
        with tempfile.TemporaryDirectory() as tmp, temp_chdir(tmp):
            store = await _make_store_with_dailies(
                [
                    ("2026-05-18", "main", "main body"),
                ],
            )
            material = Path(tmp) / "daily" / "2026-05-18" / "main" / "notes.md"
            material.write_text("---\nname: notes\n---\nmaterial body\n", encoding="utf-8")

            step = daily_list_step.DailyListStep(file_store=store)
            await step(date="2026-05-18")
            payload = _answer(step)
            paths = [w["path"] for w in payload["workspaces"]]
            assert paths == ["daily/2026-05-18/main/main.md"]
            await store.close()
        print("✓ test_daily_list_ignores_material_files passed")

    asyncio.run(run())


def test_daily_list_empty_when_no_daily_dir():
    """No daily/ folder ⇒ empty workspaces list, no crash."""

    async def run():
        with tempfile.TemporaryDirectory() as tmp, temp_chdir(tmp):
            store = LocalFileStore(store_name="t", embedding_model="")
            await store.start()
            step = daily_list_step.DailyListStep(file_store=store)
            await step(date="2026-05-18")
            payload = _answer(step)
            assert payload == {"date": "2026-05-18", "workspaces": []}
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
            assert "[[daily/2026-05-18/alpha/alpha.md]]" in text
            assert "[[daily/2026-05-18/beta/beta.md]]" in text
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
            payload = _answer(step)
            assert set(payload.keys()) == {"date", "workspaces"}
            await store.close()
        print("✓ test_daily_list_response_excludes_index_page_fields passed")

    asyncio.run(run())


# -- daily_resolve_step -------------------------------------------------------


def test_daily_resolve_creates_folder():
    """daily_resolve on a fresh name creates the folder and returns the vault path."""

    async def run():
        with tempfile.TemporaryDirectory() as tmp, temp_chdir(tmp):
            store = await _make_store_with_dailies([])
            step = daily_resolve_step.DailyResolveStep(file_store=store)
            await step(name="kickoff")
            payload = _answer(step)

            assert payload["created"] is True
            assert payload["name"] == "kickoff"
            assert payload["date"] == _today()
            assert payload["path"] == f"daily/{_today()}/kickoff"
            assert "message" not in payload

            folder = Path(tmp) / "daily" / _today() / "kickoff"
            assert folder.is_dir()
            # Folder is empty — no .md, no frontmatter, no index.
            assert not list(folder.iterdir())
            assert not (Path(tmp) / "daily" / f"{_today()}.md").exists()
            await store.close()
        print("✓ test_daily_resolve_creates_folder passed")

    asyncio.run(run())


def test_daily_resolve_idempotent_when_folder_exists():
    """Existing folder returns {created:false, message:...} and leaves contents untouched."""

    async def run():
        with tempfile.TemporaryDirectory() as tmp, temp_chdir(tmp):
            store = await _make_store_with_dailies(
                [(_today(), "ongoing", "morning thoughts")],
            )
            before = (Path(tmp) / "daily" / _today() / "ongoing" / "ongoing.md").read_text(
                encoding="utf-8",
            )

            step = daily_resolve_step.DailyResolveStep(file_store=store)
            await step(name="ongoing")
            payload = _answer(step)

            assert payload["created"] is False
            assert payload["name"] == "ongoing"
            assert payload["path"] == f"daily/{_today()}/ongoing"
            assert "already exists" in payload["message"]

            # Contents unchanged.
            after = (Path(tmp) / "daily" / _today() / "ongoing" / "ongoing.md").read_text(
                encoding="utf-8",
            )
            assert before == after
            await store.close()
        print("✓ test_daily_resolve_idempotent_when_folder_exists passed")

    asyncio.run(run())


def test_daily_resolve_rejects_empty_name():
    """Empty name ⇒ error payload, success=False, no folder created."""

    async def run():
        with tempfile.TemporaryDirectory() as tmp, temp_chdir(tmp):
            store = await _make_store_with_dailies([])
            step = daily_resolve_step.DailyResolveStep(file_store=store)
            await step(name="")
            payload = _answer(step)

            assert "error" in payload
            assert "required" in payload["error"]
            assert step.context.response.success is False
            assert not (Path(tmp) / "daily" / _today()).exists()
            await store.close()
        print("✓ test_daily_resolve_rejects_empty_name passed")

    asyncio.run(run())


def test_daily_resolve_rejects_windows_invalid_chars():
    """Windows-reserved characters in name ⇒ error, no folder created."""

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
                payload = _answer(step)
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
                payload = _answer(step)
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
                payload = _answer(step)
                assert "error" in payload, f"expected error for {bad!r}, got {payload!r}"
            await store.close()
        print("✓ test_daily_resolve_rejects_trailing_dot_or_whitespace passed")

    asyncio.run(run())


# -- day index: daily/<date>.md ------------------------------------------


def _day_index_text(tmp: str, day: str) -> str:
    return (Path(tmp) / "daily" / f"{day}.md").read_text(encoding="utf-8")


def test_day_index_lists_each_workspace():
    """Multiple workspaces all show up in the index workspaces block with name."""

    async def run():
        with tempfile.TemporaryDirectory() as tmp, temp_chdir(tmp):
            store = LocalFileStore(store_name="t", embedding_model="")
            await store.start()
            await _seed_workspace("2026-05-18", "alpha", name="Alpha Project")
            await _seed_workspace("2026-05-18", "beta", name="Beta Project")

            await daily_reindex_step.DailyReindexStep(file_store=store)(date="2026-05-18")
            text = _day_index_text(tmp, "2026-05-18")
            assert "[[daily/2026-05-18/alpha/alpha.md]]" in text
            assert "[[daily/2026-05-18/beta/beta.md]]" in text
            # Workspace names show on the indented sub-line.
            assert "Alpha Project" in text
            assert "Beta Project" in text
            await store.close()
        print("✓ test_day_index_lists_each_workspace passed")

    asyncio.run(run())


def test_day_index_includes_workspace_descriptions():
    """Workspace ``description`` fields land in the rendered block so the
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
                await _seed_workspace("2026-05-18", slug, name=name, description=description)

            await daily_reindex_step.DailyReindexStep(file_store=store)(date="2026-05-18")
            text = _day_index_text(tmp, "2026-05-18")
            # name + description rendered together
            assert "Alpha Project — 实现 JWT auth 中间件" in text
            # name == slug → only description shown (no redundant "beta")
            assert "调研增值税新政对 SaaS 的影响" in text
            # no description → only name shown, no trailing em-dash
            assert "  Gamma\n" in text or text.rstrip().endswith("Gamma")
            await store.close()
        print("✓ test_day_index_includes_workspace_descriptions passed")

    asyncio.run(run())


def test_day_index_description_is_workspace_count():
    """The typed ``description`` field carries a one-line workspace-count digest."""

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
            assert "2 个工作区" in text
            await store.close()
        print("✓ test_day_index_description_is_workspace_count passed")

    asyncio.run(run())


def test_day_index_preserves_manual_segment():
    """The ``## 备忘`` (manual) segment is preserved across refreshes."""

    async def run():
        with tempfile.TemporaryDirectory() as tmp, temp_chdir(tmp):
            store = LocalFileStore(store_name="t", embedding_model="")
            await store.start()
            await _seed_workspace("2026-05-18", "alpha")
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

            # Adding a sibling workspace + refresh — manual segment must survive.
            await _seed_workspace("2026-05-18", "beta")
            await reindex(date="2026-05-18")
            after = index_path.read_text(encoding="utf-8")
            assert "MY HAND-WRITTEN NOTE" in after
            assert "这是我手写的备忘" in after
            # Auto block was updated with the new workspace.
            assert "[[daily/2026-05-18/beta/beta.md]]" in after
            await store.close()
        print("✓ test_day_index_preserves_manual_segment passed")

    asyncio.run(run())


# -- daily_reindex_step -----------------------------------------------------


def test_daily_reindex_returns_write_view():
    """daily_reindex returns {date, path, created, workspaces_count}."""

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
            payload = _answer(step)

            assert set(payload.keys()) == {"date", "path", "created", "workspaces_count"}
            assert payload["date"] == "2026-05-18"
            assert payload["path"] == "daily/2026-05-18.md"
            assert payload["created"] is True
            assert payload["workspaces_count"] == 2

            text = _day_index_text(tmp, "2026-05-18")
            assert "[[daily/2026-05-18/alpha/alpha.md]]" in text
            assert "[[daily/2026-05-18/beta/beta.md]]" in text
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
            payload_first = _answer(step)
            assert payload_first["created"] is True

            await step(date="2026-05-18")
            payload_second = _answer(step)
            assert payload_second["created"] is False
            assert payload_second["workspaces_count"] == 1
            await store.close()
        print("✓ test_daily_reindex_created_flag_flips_on_rerun passed")

    asyncio.run(run())


if __name__ == "__main__":
    print("\n=== Daily step tests ===")
    test_daily_list_default_date_is_today()
    test_daily_list_filters_by_date()
    test_daily_list_returns_path_name_description()
    test_daily_list_ignores_material_files()
    test_daily_list_empty_when_no_daily_dir()
    test_daily_list_triggers_index_refresh_as_side_effect()
    test_daily_list_response_excludes_index_page_fields()
    test_daily_resolve_creates_folder()
    test_daily_resolve_idempotent_when_folder_exists()
    test_daily_resolve_rejects_empty_name()
    test_daily_resolve_rejects_windows_invalid_chars()
    test_daily_resolve_rejects_windows_reserved_names()
    test_daily_resolve_rejects_trailing_dot_or_whitespace()
    test_day_index_lists_each_workspace()
    test_day_index_includes_workspace_descriptions()
    test_day_index_description_is_workspace_count()
    test_day_index_preserves_manual_segment()
    test_daily_reindex_returns_write_view()
    test_daily_reindex_created_flag_flips_on_rerun()
    print("\nAll tests passed!")
