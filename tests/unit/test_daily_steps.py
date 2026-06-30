"""Tests for daily-aware steps: daily_list / daily_reindex / daily_write.

Sets up a small ``daily/`` tree with mixed dates and exercises note
listing / index-rebuild operations. Body authoring and frontmatter
mutations are generic CRUD (covered in test_crud_steps and
test_property_steps).

A daily note is the single file ``daily/<YYYY-MM-DD>/<name>.md``
(no folder, no sibling materials).

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
import warnings
from pathlib import Path

import frontmatter

from reme.components import ApplicationContext
from reme.components.file_store import LocalFileStore
from reme.enumeration import ComponentEnum
from reme.steps.file_io import (
    daily_list as daily_list_step,
    daily_reindex as daily_reindex_step,
    daily_write as daily_write_step,
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
    from reme.steps.evolve import now

    return now("Asia/Shanghai").strftime("%Y-%m-%d")


async def _make_store_with_dailies(entries: list[tuple[str, str, str]]) -> LocalFileStore:
    """Seed the workspace with daily notes.

    entries: list of (date, name, body). Each tuple creates
    ``daily/<date>/<name>.md`` with a minimal ``name``-only frontmatter —
    no opinionated status / lifecycle axes.
    """
    store = LocalFileStore(name="t", embedding_store="")
    await store.start()
    for day, name, body in entries:
        day_dir = Path.cwd() / "daily" / day
        day_dir.mkdir(parents=True, exist_ok=True)
        text = f"---\nname: {name}\n---\n{body}\n"
        (day_dir / f"{name}.md").write_text(text, encoding="utf-8")
    return store


def _metadata(step) -> dict:
    return step.context.response.metadata


async def _seed_note(date: str, filename: str, name: str = "", description: str = "", **metadata) -> None:
    """Write ``daily/<date>/<filename>.md`` with optional frontmatter."""
    day_dir = Path.cwd() / "daily" / date
    day_dir.mkdir(parents=True, exist_ok=True)
    fm_lines = [f"name: {name or filename}"]
    if description:
        fm_lines.append(f"description: {description}")
    for key, value in metadata.items():
        fm_lines.append(f"{key}: {value}")
    text = "---\n" + "\n".join(fm_lines) + "\n---\nbody\n"
    (day_dir / f"{filename}.md").write_text(text, encoding="utf-8")


async def _make_daily_write_step(store: LocalFileStore, workspace_dir: str):
    """Build ``daily_write`` in a minimal app context."""
    app_context = ApplicationContext(workspace_dir=workspace_dir)
    app_context.components[ComponentEnum.FILE_STORE] = {"default": store}
    return daily_write_step.DailyWriteStep(app_context=app_context)


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
            app_context = ApplicationContext(workspace_dir=tmp)
            app_context.components[ComponentEnum.FILE_STORE] = {"default": store}
            step = daily_list_step.DailyListStep(file_store=store, app_context=app_context)
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


def test_daily_list_returns_path_and_frontmatter_notes():
    """Each note row exposes path plus the full frontmatter dict."""

    async def run():
        with tempfile.TemporaryDirectory() as tmp, temp_chdir(tmp):
            store = LocalFileStore(name="t", embedding_store="")
            await store.start()
            await _seed_note(
                "2026-05-18",
                "Alpha Project",
                name="Alpha Project",
                description="JWT auth migration",
                session_id="session-123",
                source_conversation='"[[sessions/dialog/session-123.jsonl]]"',
            )
            step = daily_list_step.DailyListStep(file_store=store)
            await step(date="2026-05-18")
            payload = _metadata(step)
            assert payload["count"] == 1
            assert payload["notes"] == [
                {
                    "path": "daily/2026-05-18/Alpha Project.md",
                    "name": "Alpha Project",
                    "description": "JWT auth migration",
                    "session_id": "session-123",
                    "source_conversation": "[[sessions/dialog/session-123.jsonl]]",
                },
            ]
            answer = step.context.response.answer
            assert "daily/2026-05-18/Alpha Project.md" in answer
            assert "Alpha Project" in answer
            assert "JWT auth migration" in answer
            assert "session-123" in answer
            assert "[[sessions/dialog/session-123.jsonl]]" in answer
            await store.close()
        print("✓ test_daily_list_returns_path_and_frontmatter_notes passed")

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
            assert payload == {"date": "2026-05-18", "count": 0, "notes": []}
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
    """daily_list returns {date, count, notes}."""

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
            assert set(payload.keys()) == {"date", "count", "notes"}
            assert payload["notes"] == [{"path": "daily/2026-05-18/alpha.md", "name": "alpha"}]
            await store.close()
        print("✓ test_daily_list_response_shape passed")

    asyncio.run(run())


# -- daily_write_step ---------------------------------------------------------


def test_daily_write_delegates_to_write_and_refreshes_index():
    """``daily_write`` writes body/frontmatter through ``write_step`` and refreshes the day index."""

    async def run():
        with tempfile.TemporaryDirectory() as tmp, temp_chdir(tmp):
            store = await _make_store_with_dailies([])
            step = await _make_daily_write_step(store, tmp)

            await step(
                name="project-plan",
                description="Plan note",
                session_id="2f85a30f4faa41b39380cabb647c1b5b",
                content="body text",
                metadata={"tag": "kept"},
            )
            payload = _metadata(step)

            assert step.context.response.success is True
            assert payload["path"] == f"daily/{_today()}/project-plan.md"
            assert payload["source_conversation"] == "[[session/dialog/2f85a30f4faa41b39380cabb647c1b5b.jsonl]]"

            note = Path(tmp) / "daily" / _today() / "project-plan.md"
            post = frontmatter.loads(note.read_text(encoding="utf-8"))
            assert post.metadata["name"] == "project-plan"
            assert post.metadata["description"] == "Plan note"
            assert post.metadata["session_id"] == "2f85a30f4faa41b39380cabb647c1b5b"
            assert post.metadata["source_conversation"] == "[[session/dialog/2f85a30f4faa41b39380cabb647c1b5b.jsonl]]"
            assert post.metadata["tag"] == "kept"
            assert post.content.strip() == "body text"

            index = Path(tmp) / "daily" / f"{_today()}.md"
            assert index.is_file()
            assert f"[[daily/{_today()}/project-plan.md]]" in index.read_text(encoding="utf-8")
            await store.close()
        print("✓ test_daily_write_delegates_to_write_and_refreshes_index passed")

    asyncio.run(run())


def test_daily_write_accepts_explicit_date():
    """``daily_write`` can target a historical daily folder when ``date`` is provided."""

    async def run():
        with tempfile.TemporaryDirectory() as tmp, temp_chdir(tmp):
            store = await _make_store_with_dailies([])
            step = await _make_daily_write_step(store, tmp)

            await step(
                date="2023-01-19",
                name="historical-memory",
                description="Historical note",
                session_id="locomo-session",
                content="body text",
            )
            payload = _metadata(step)

            assert step.context.response.success is True
            assert payload["date"] == "2023-01-19"
            assert payload["path"] == "daily/2023-01-19/historical-memory.md"
            assert (Path(tmp) / "daily" / "2023-01-19" / "historical-memory.md").is_file()
            assert (Path(tmp) / "daily" / "2023-01-19.md").is_file()
            assert not (Path(tmp) / "daily" / _today()).exists()
            await store.close()
        print("✓ test_daily_write_accepts_explicit_date passed")

    asyncio.run(run())


def test_daily_write_reserved_metadata_keys_are_overridden():
    """User metadata cannot override daily_write's fixed frontmatter fields."""

    async def run():
        with tempfile.TemporaryDirectory() as tmp, temp_chdir(tmp):
            store = await _make_store_with_dailies([])
            step = await _make_daily_write_step(store, tmp)

            await step(
                name="real-name",
                description="Real description",
                session_id="real-session",
                content="body",
                metadata={
                    "name": "bad-name",
                    "description": "bad-description",
                    "session_id": "bad-session",
                    "source_conversation": "[[bad]]",
                    "priority": 2,
                },
            )

            post = frontmatter.loads((Path(tmp) / "daily" / _today() / "real-name.md").read_text(encoding="utf-8"))
            assert post.metadata["name"] == "real-name"
            assert post.metadata["description"] == "Real description"
            assert post.metadata["session_id"] == "real-session"
            assert post.metadata["source_conversation"] == "[[session/dialog/real-session.jsonl]]"
            assert post.metadata["priority"] == 2
            await store.close()
        print("✓ test_daily_write_reserved_metadata_keys_are_overridden passed")

    asyncio.run(run())


def test_daily_write_rejects_invalid_name_and_session_id():
    """Path-bearing fields are validated before write job execution."""

    async def run():
        with tempfile.TemporaryDirectory() as tmp, temp_chdir(tmp):
            store = await _make_store_with_dailies([])
            step = await _make_daily_write_step(store, tmp)

            await step(name="bad/name", description="d", session_id="ok", content="body")
            assert step.context.response.success is False
            await step(name="ok", description="d", session_id="bad/session", content="body")
            assert step.context.response.success is False
            await step(name="ok", description="d", session_id="ok", content="body", date="2023/01/19")
            assert step.context.response.success is False
            assert not (Path(tmp) / "daily" / _today()).exists()
            await store.close()
        print("✓ test_daily_write_rejects_invalid_name_and_session_id passed")

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


def test_day_index_hides_conversation_metadata():
    """The day index omits conversation metadata while keeping it in the returned notes payload."""

    async def run():
        with tempfile.TemporaryDirectory() as tmp, temp_chdir(tmp):
            store = LocalFileStore(name="t", embedding_store="")
            await store.start()
            await _seed_note(
                "2026-05-18",
                "alpha",
                name="Alpha Project",
                description="JWT auth migration",
                session_id="session-123",
                source_conversation='"[[session/dialog/session-123.jsonl]]"',
            )

            refreshed = await daily_reindex_step.refresh_day_index(store, "2026-05-18", "daily")
            assert refreshed["notes"][0]["session_id"] == "session-123"
            assert refreshed["notes"][0]["source_conversation"] == "[[session/dialog/session-123.jsonl]]"

            text = _day_index_text(tmp, "2026-05-18")
            line = text.split("[[daily/2026-05-18/alpha.md]]", 1)[1].split("\n", 1)[0]
            assert "name: Alpha Project" in line
            assert "description: JWT auth migration" in line
            assert "session_id" not in line
            assert "session-123" not in line
            assert "source_conversation" not in line
            assert "[[session/dialog/session-123.jsonl]]" not in line
            await store.close()
        print("✓ test_day_index_hides_conversation_metadata passed")

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


def test_day_index_description_updates_when_note_count_changes():
    """Reindexing an existing day index refreshes the note-count description."""

    async def run():
        with tempfile.TemporaryDirectory() as tmp, temp_chdir(tmp):
            store = LocalFileStore(name="t", embedding_store="")
            await store.start()
            await _seed_note("2026-05-18", "alpha")
            reindex = daily_reindex_step.DailyReindexStep(file_store=store)
            await reindex(date="2026-05-18")

            await _seed_note("2026-05-18", "beta")
            await reindex(date="2026-05-18")

            text = _day_index_text(tmp, "2026-05-18")
            assert "2 note(s) today." in text
            assert "1 note(s) today." not in text
            await store.close()
        print("✓ test_day_index_description_updates_when_note_count_changes passed")

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
    test_daily_list_returns_path_and_frontmatter_notes()
    test_daily_list_ignores_subdirectories()
    test_daily_list_empty_when_no_daily_dir()
    test_daily_list_does_not_refresh_index()
    test_daily_list_response_shape()
    test_daily_write_delegates_to_write_and_refreshes_index()
    test_daily_write_accepts_explicit_date()
    test_daily_write_reserved_metadata_keys_are_overridden()
    test_daily_write_rejects_invalid_name_and_session_id()
    test_day_index_lists_each_note()
    test_day_index_includes_note_descriptions()
    test_day_index_hides_conversation_metadata()
    test_day_index_description_is_note_count()
    test_day_index_description_updates_when_note_count_changes()
    test_day_index_preserves_user_content_outside_marker()
    test_daily_reindex_returns_write_view()
    test_daily_reindex_created_flag_flips_on_rerun()
    print("\nAll tests passed!")
