# pylint: disable=too-many-lines
"""Tests for crud steps — the opaque-byte vault_dir surface plus the
text-content ops (``read`` / ``write`` / ``edit``).

Every test drives a step directly against a freshly built
``LocalFileStore`` (embedding disabled, BM25 kept) with files seeded
on disk (and, where relevant, registered in the graph so retarget's
reverse-index lookup finds inbound edges). No app config, no HTTP
server — the step's ``vault_path`` defaults to ``cwd()`` and tests
chdir into a tmpdir to scope the vault.

Covers ``stat`` / ``list`` / ``download`` / ``move`` / ``delete``
plus the text ops ``read`` / ``write`` / ``edit`` (including non-md
degraded mode + encoding edge cases).

Frontmatter-only ops live in ``test_frontmatter_steps.py``. The
``upload`` step is a passive resource-ingest entry point with its own
bucket semantics — tests for it live in ``test_resource_steps.py``.

Path-shape contract (enforced by ``read`` / ``write`` / ``edit``):
``path=`` is vault-relative by default. A bare path with no suffix
auto-appends ``.md``; non-``.md`` suffix is accepted in degraded mode.
Absolute paths are accepted with a warning.
"""

# pylint: disable=protected-access,redefined-builtin

import asyncio
import os
import tempfile
import warnings
from pathlib import Path

from reme4.components.file_store import LocalFileStore
from reme4.schema import FileNode
from reme4.steps.file_io import (
    delete as crud_delete,
    edit as crud_edit,
    list as crud_list,
    move as crud_move,
    read as crud_read,
    stat as crud_stat,
    write as crud_write,
)
from reme4.steps.transfer import download as crud_download
from reme4.utils.wikilink_handler import WikilinkHandler

warnings.filterwarnings("ignore", category=DeprecationWarning, module="jieba")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="pkg_resources")


class temp_chdir:
    """Context manager to temporarily chdir into a path and restore on exit."""

    def __init__(self, path):
        self.path = path
        self.old = None

    def __enter__(self):
        self.old = os.getcwd()
        os.chdir(self.path)
        return self

    def __exit__(self, *exc):
        os.chdir(self.old)


async def _make_store(files: dict[str, str] | None = None) -> LocalFileStore:
    """LocalFileStore seeded with files on disk + registered in the graph."""
    store = LocalFileStore(name="t", embedding_store="")
    await store.start()
    nodes: list[FileNode] = []
    for rel, content in (files or {}).items():
        abs_path = Path.cwd() / rel
        abs_path.parent.mkdir(parents=True, exist_ok=True)
        abs_path.write_text(content, encoding="utf-8")
        nodes.append(
            FileNode(
                path=rel,
                st_mtime=abs_path.stat().st_mtime,
                links=WikilinkHandler.extract_links(content, rel),
            ),
        )
    if nodes:
        await store.file_graph.upsert_nodes(nodes)
    return store


def _metadata(step) -> dict:
    return step.context.response.metadata


def _run(coro):
    """Run an async coroutine on a fresh isolated event loop."""
    asyncio.run(coro)


def _seed_md(vault_dir: Path, rel: str, body: str) -> Path:
    target = vault_dir / rel
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(body, encoding="utf-8")
    return target


# ===========================================================================
# Direct unit tests: stat / list / download / move / delete
# ===========================================================================


# -- stat ----------------------------------------------------------------


def test_stat_indexed_file():
    """stat returns size, mime, and frontmatter for an indexed .md file."""

    async def run():
        with tempfile.TemporaryDirectory() as tmp, temp_chdir(tmp):
            store = await _make_store({"topics/n.md": "---\nname: T\n---\nbody"})
            step = crud_stat.StatStep(file_store=store)
            await step(path="topics/n.md")
            payload = _metadata(step)
            assert payload["exists"] is True
            assert payload["type"] == "file"
            assert "size" in payload and payload["size"] > 0
            assert payload["mime"].startswith("text/")
            assert payload["frontmatter"] == {"name": "T"}
            await store.close()
        print("✓ test_stat_indexed_file passed")

    asyncio.run(run())


def test_stat_directory_fallback():
    """stat on a non-indexed directory falls back to a plain join + type=dir."""

    async def run():
        with tempfile.TemporaryDirectory() as tmp, temp_chdir(tmp):
            store = await _make_store()
            (Path(tmp) / "topics").mkdir(parents=True, exist_ok=True)
            step = crud_stat.StatStep(file_store=store)
            await step(path="topics")
            payload = _metadata(step)
            assert payload["exists"] is True
            assert payload["type"] == "dir"
            await store.close()
        print("✓ test_stat_directory_fallback passed")

    asyncio.run(run())


# -- list ----------------------------------------------------------------


def test_list_lists_files():
    """list returns paths relative to the vault for files under the given directory."""

    async def run():
        with tempfile.TemporaryDirectory() as tmp, temp_chdir(tmp):
            store = await _make_store(
                {
                    "topics/a.md": "x",
                    "topics/b.md": "y",
                    "topics/sub/c.md": "z",
                },
            )
            step = crud_list.ListStep(file_store=store)
            await step(path="topics", recursive=True)
            payload = _metadata(step)
            assert set(payload["items"]) == {"topics/a.md", "topics/b.md", "topics/sub/c.md"}
            assert payload["count"] == 3
            await store.close()
        print("✓ test_list_lists_files passed")

    asyncio.run(run())


def test_list_respects_limit_and_non_recursive():
    """Non-recursive list ignores subdirs; limit caps the count."""

    async def run():
        with tempfile.TemporaryDirectory() as tmp, temp_chdir(tmp):
            store = await _make_store(
                {
                    "topics/a.md": "x",
                    "topics/b.md": "y",
                    "topics/sub/c.md": "z",
                },
            )
            step = crud_list.ListStep(file_store=store)
            await step(path="topics", recursive=False, limit=1)
            payload = _metadata(step)
            assert payload["count"] == 1
            assert len(payload["items"]) == 1
            await store.close()
        print("✓ test_list_respects_limit_and_non_recursive passed")

    asyncio.run(run())


# -- download ------------------------------------------------------------


def test_download_to_explicit_path():
    """download copies the vault file to dst_path."""

    async def run():
        with tempfile.TemporaryDirectory() as tmp, temp_chdir(tmp):
            store = await _make_store({"topics/a.md": "alpha"})
            target = Path(tmp) / "out" / "a.md"
            step = crud_download.DownloadStep(file_store=store)
            await step(src_path="topics/a.md", dst_path=str(target))
            payload = _metadata(step)
            assert "error" not in payload
            assert payload["dst_path"] == str(target)
            assert target.read_text(encoding="utf-8") == "alpha"
            await store.close()
        print("✓ test_download_to_explicit_path passed")

    asyncio.run(run())


def test_download_to_temp_when_dst_path_empty():
    """Without dst_path, download lands the file in a temp file and returns the path."""

    async def run():
        with tempfile.TemporaryDirectory() as tmp, temp_chdir(tmp):
            store = await _make_store({"topics/a.md": "alpha"})
            step = crud_download.DownloadStep(file_store=store)
            await step(src_path="topics/a.md")
            payload = _metadata(step)
            assert "error" not in payload
            assert Path(payload["dst_path"]).read_text(encoding="utf-8") == "alpha"
            await store.close()
        print("✓ test_download_to_temp_when_dst_path_empty passed")

    asyncio.run(run())


# -- move ----------------------------------------------------------------


def test_move_relocates_within_vault():
    """move renames / relocates a file in place."""

    async def run():
        with tempfile.TemporaryDirectory() as tmp, temp_chdir(tmp):
            store = await _make_store({"daily/2026-05-18/foo/foo.md": "draft"})
            step = crud_move.MoveStep(file_store=store)
            await step(src_path="daily/2026-05-18/foo/foo.md", dst_path="knowledge/foo/foo.md")
            payload = _metadata(step)
            assert "error" not in payload
            assert not (Path(tmp) / "daily/2026-05-18/foo/foo.md").exists()
            assert (Path(tmp) / "knowledge/foo/foo.md").read_text(encoding="utf-8") == "draft"
            await store.close()
        print("✓ test_move_relocates_within_vault passed")

    asyncio.run(run())


def test_move_refuses_overwrite_without_flag():
    """move refuses to clobber an existing dst_path unless overwrite=True."""

    async def run():
        with tempfile.TemporaryDirectory() as tmp, temp_chdir(tmp):
            store = await _make_store(
                {
                    "a/x.md": "src",
                    "b/x.md": "dst",
                },
            )
            step = crud_move.MoveStep(file_store=store)
            await step(src_path="a/x.md", dst_path="b/x.md")
            payload = _metadata(step)
            assert "destination exists" in payload.get("error", "")
            assert (Path(tmp) / "a/x.md").exists()
            await store.close()
        print("✓ test_move_refuses_overwrite_without_flag passed")

    asyncio.run(run())


def test_move_default_retargets_inbound_links():
    """move with retarget=True (default) rewrites inbound full-path [[src_path]] → [[dst_path]]."""

    async def run():
        with tempfile.TemporaryDirectory() as tmp, temp_chdir(tmp):
            store = await _make_store(
                {
                    "daily/2026-05-18/draft/draft.md": "spec",
                    # only the literal full-path form is retargeted by design;
                    # short / no-ext forms are intentionally left alone.
                    "knowledge/notes/notes.md": (
                        "see [[daily/2026-05-18/draft/draft.md]] and again " "[[daily/2026-05-18/draft/draft.md]] twice"
                    ),
                },
            )
            step = crud_move.MoveStep(file_store=store)
            await step(
                src_path="daily/2026-05-18/draft/draft.md",
                dst_path="knowledge/draft/draft.md",
            )
            payload = _metadata(step)

            assert "error" not in payload
            assert payload["retarget"]["files_touched"] == 1
            assert payload["retarget"]["links_changed"] == 2

            notes = (Path(tmp) / "knowledge/notes/notes.md").read_text(encoding="utf-8")
            assert "[[daily/2026-05-18/draft/draft.md]]" not in notes
            assert notes.count("[[knowledge/draft/draft.md]]") == 2
            await store.close()
        print("✓ test_move_default_retargets_inbound_links passed")

    asyncio.run(run())


def test_move_opt_out_leaves_links_dangling():
    """retarget=False moves the file but leaves inbound references stale."""

    async def run():
        with tempfile.TemporaryDirectory() as tmp, temp_chdir(tmp):
            store = await _make_store(
                {
                    "daily/2026-05-18/draft/draft.md": "spec",
                    "knowledge/notes/notes.md": "see [[daily/2026-05-18/draft/draft.md]]",
                },
            )
            step = crud_move.MoveStep(file_store=store)
            await step(
                src_path="daily/2026-05-18/draft/draft.md",
                dst_path="knowledge/draft/draft.md",
                retarget=False,
            )
            payload = _metadata(step)

            assert "error" not in payload
            assert payload["retarget"] is None

            notes = (Path(tmp) / "knowledge/notes/notes.md").read_text(encoding="utf-8")
            # link UNCHANGED — caller opted out of retarget
            assert "[[daily/2026-05-18/draft/draft.md]]" in notes
            await store.close()
        print("✓ test_move_opt_out_leaves_links_dangling passed")

    asyncio.run(run())


# -- delete --------------------------------------------------------------


def test_delete_removes_file():
    """delete hard-removes the file."""

    async def run():
        with tempfile.TemporaryDirectory() as tmp, temp_chdir(tmp):
            store = await _make_store({"knowledge/draft/draft.md": "x"})
            step = crud_delete.DeleteStep(file_store=store)
            await step(path="knowledge/draft/draft.md")
            payload = _metadata(step)
            assert payload.get("deleted") is True
            assert not (Path(tmp) / "knowledge/draft/draft.md").exists()
            await store.close()
        print("✓ test_delete_removes_file passed")

    asyncio.run(run())


def test_delete_missing_returns_error():
    """delete on a nonexistent path returns error rather than raising."""

    async def run():
        with tempfile.TemporaryDirectory() as tmp, temp_chdir(tmp):
            store = await _make_store()
            step = crud_delete.DeleteStep(file_store=store)
            await step(path="knowledge/nope/nope.md")
            payload = _metadata(step)
            assert payload["error"] == "not found"
            await store.close()
        print("✓ test_delete_missing_returns_error passed")

    asyncio.run(run())


def test_delete_reports_inbound_refs():
    """delete returns the inbound wikilink list (literal full-path matches only)."""

    async def run():
        with tempfile.TemporaryDirectory() as tmp, temp_chdir(tmp):
            store = await _make_store(
                {
                    "knowledge/target/target.md": "doomed",
                    "knowledge/a/a.md": "see [[knowledge/target/target.md]]",
                    "knowledge/b/b.md": (
                        "ref [[knowledge/target/target.md]] and again " "[[knowledge/target/target.md]]"
                    ),
                    # short / no-ext forms are NOT counted by design
                    "knowledge/c/c.md": "[[target]] and [[knowledge/target/target]]",
                },
            )
            step = crud_delete.DeleteStep(file_store=store)
            await step(path="knowledge/target/target.md")
            payload = _metadata(step)

            assert payload["deleted"] is True
            assert not (Path(tmp) / "knowledge/target/target.md").exists()
            # referencing files are untouched — agent decides what to do
            assert (Path(tmp) / "knowledge/a/a.md").read_text(encoding="utf-8") == (
                "see [[knowledge/target/target.md]]"
            )

            inbound = payload["inbound"]
            paths = {item["path"] for item in inbound["by_file"]}
            assert paths == {"knowledge/a/a.md", "knowledge/b/b.md"}
            # a: 1 full-path ref; b: 2 full-path refs; c: not counted
            assert inbound["files_touched"] == 2
            assert inbound["links_total"] == 3
            await store.close()
        print("✓ test_delete_reports_inbound_refs passed")

    asyncio.run(run())


def test_delete_folder_removes_tree():
    """delete on a directory hard-removes the whole subtree."""

    async def run():
        with tempfile.TemporaryDirectory() as tmp, temp_chdir(tmp):
            store = await _make_store(
                {
                    "scratch/a.md": "alpha",
                    "scratch/sub/b.md": "beta",
                    "scratch/asset.bin": "blob",
                    "keeper/k.md": "kept",
                },
            )
            step = crud_delete.DeleteStep(file_store=store)
            await step(path="scratch")
            payload = _metadata(step)
            assert payload["deleted"] is True
            assert payload["is_dir"] is True
            assert set(payload["deleted_files"]) == {
                "scratch/a.md",
                "scratch/sub/b.md",
                "scratch/asset.bin",
            }
            assert not (Path(tmp) / "scratch").exists()
            assert (Path(tmp) / "keeper/k.md").exists()
            await store.close()
        print("✓ test_delete_folder_removes_tree passed")

    asyncio.run(run())


def test_delete_folder_reports_only_external_inbound():
    """Inbound from inside the doomed folder is suppressed; outside refs surface."""

    async def run():
        with tempfile.TemporaryDirectory() as tmp, temp_chdir(tmp):
            store = await _make_store(
                {
                    "doomed/a.md": "see [[doomed/b.md]]",  # internal — filtered out
                    "doomed/b.md": "see [[doomed/a.md]]",  # internal — filtered out
                    "outside/x.md": ("ext [[doomed/a.md]] and [[doomed/a.md]] plus [[doomed/b.md]]"),
                    "outside/y.md": "another [[doomed/a.md]]",
                },
            )
            step = crud_delete.DeleteStep(file_store=store)
            await step(path="doomed")
            payload = _metadata(step)
            assert payload["deleted"] is True
            assert payload["is_dir"] is True
            assert not (Path(tmp) / "doomed").exists()
            # outside files survive untouched
            assert (Path(tmp) / "outside/x.md").exists()

            inbound = payload["inbound"]
            # external sources: x.md, y.md (deduped) → 2 files
            assert inbound["files_touched"] == 2
            # 2 refs to doomed/a.md from x + 1 ref from y + 1 ref to b from x = 4
            assert inbound["links_total"] == 4

            by_target = {row["target"]: row for row in inbound["by_target"]}
            assert set(by_target) == {"doomed/a.md", "doomed/b.md"}
            a_sources = {row["path"]: row["count"] for row in by_target["doomed/a.md"]["by_file"]}
            assert a_sources == {"outside/x.md": 2, "outside/y.md": 1}
            b_sources = {row["path"]: row["count"] for row in by_target["doomed/b.md"]["by_file"]}
            assert b_sources == {"outside/x.md": 1}
            await store.close()
        print("✓ test_delete_folder_reports_only_external_inbound passed")

    asyncio.run(run())


def test_delete_folder_empty_has_no_inbound():
    """Empty folder delete reports zero deleted files and zero inbound."""

    async def run():
        with tempfile.TemporaryDirectory() as tmp, temp_chdir(tmp):
            (Path(tmp) / "empty").mkdir()
            store = await _make_store()
            step = crud_delete.DeleteStep(file_store=store)
            await step(path="empty")
            payload = _metadata(step)
            assert payload["deleted"] is True
            assert payload["is_dir"] is True
            assert payload["deleted_files"] == []
            assert payload["inbound"]["files_touched"] == 0
            assert payload["inbound"]["links_total"] == 0
            assert not (Path(tmp) / "empty").exists()
            await store.close()
        print("✓ test_delete_folder_empty_has_no_inbound passed")

    asyncio.run(run())


# ===========================================================================
# Direct unit tests: read / write / edit
# ===========================================================================


# -- read ----------------------------------------------------------------


async def _read(store: LocalFileStore, **kwargs):
    """Run a ReadStep against ``store`` and return its response."""
    step = crud_read.ReadStep(file_store=store)
    await step(**kwargs)
    return step.context.response


async def _write(store: LocalFileStore, **kwargs):
    """Run a WriteStep against ``store`` and return its response."""
    step = crud_write.WriteStep(file_store=store)
    await step(**kwargs)
    return step.context.response


async def _edit(store: LocalFileStore, **kwargs):
    """Run an EditStep against ``store`` and return its response."""
    step = crud_edit.EditStep(file_store=store)
    await step(**kwargs)
    return step.context.response


def test_read_relative_path():
    """`read path=Templates/Recipe.md` returns the file body from the vault."""

    async def run():
        with tempfile.TemporaryDirectory() as tmp, temp_chdir(tmp):
            body = "# Recipe\n\nMix flour and water.\n"
            _seed_md(Path(tmp), "Templates/Recipe.md", body)
            store = await _make_store()
            resp = await _read(store, path="Templates/Recipe.md")
            assert resp.success is True
            assert "# Recipe" in str(resp.answer)
            assert "flour and water" in str(resp.answer)
            await store.close()
        print("✓ test_read_relative_path passed")

    _run(run())


def test_read_no_suffix_autoappends_md():
    """A bare path with no suffix auto-appends `.md`."""

    async def run():
        with tempfile.TemporaryDirectory() as tmp, temp_chdir(tmp):
            _seed_md(Path(tmp), "Templates/Recipe.md", "auto-md\n")
            store = await _make_store()
            resp = await _read(store, path="Templates/Recipe")
            assert resp.success is True
            assert "auto-md" in str(resp.answer)
            await store.close()
        print("✓ test_read_no_suffix_autoappends_md passed")

    _run(run())


def test_read_line_range():
    """start_line / end_line slice the file 1-based, inclusive."""

    async def run():
        with tempfile.TemporaryDirectory() as tmp, temp_chdir(tmp):
            _seed_md(Path(tmp), "Notes.md", "L1\nL2\nL3\nL4\nL5\n")
            store = await _make_store()
            resp = await _read(store, path="Notes.md", start_line=2, end_line=4)
            assert resp.success is True
            assert "L2" in str(resp.answer)
            assert "L3" in str(resp.answer)
            assert "L4" in str(resp.answer)
            assert "L1" not in str(resp.answer)
            assert "L5" not in str(resp.answer)
            await store.close()
        print("✓ test_read_line_range passed")

    _run(run())


def test_read_absolute_path_accepted():
    """Absolute paths are accepted (a log warning is emitted but the read proceeds)."""

    async def run():
        with tempfile.TemporaryDirectory() as tmp, temp_chdir(tmp):
            target = _seed_md(Path(tmp), "Abs.md", "x\n")
            store = await _make_store()
            resp = await _read(store, path=str(target.resolve()))
            assert resp.success is True
            assert "x" in str(resp.answer)
            await store.close()
        print("✓ test_read_absolute_path_accepted passed")

    _run(run())


def test_read_non_md_degraded():
    """Paths whose suffix is not `.md` are read in compatibility mode."""

    async def run():
        with tempfile.TemporaryDirectory() as tmp, temp_chdir(tmp):
            _seed_md(Path(tmp), "data/foo.txt", "plain-text body\n")
            store = await _make_store()
            resp = await _read(store, path="data/foo.txt")
            assert resp.success is True
            assert "plain-text body" in str(resp.answer)
            await store.close()
        print("✓ test_read_non_md_degraded passed")

    _run(run())


def test_read_missing_file():
    """Reading a non-existent file should fail with a clear error."""

    async def run():
        with tempfile.TemporaryDirectory() as tmp, temp_chdir(tmp):
            store = await _make_store()
            resp = await _read(store, path="NotThere.md")
            assert resp.success is False
            assert "does not exist" in str(resp.answer).lower()
            await store.close()
        print("✓ test_read_missing_file passed")

    _run(run())


def test_read_start_after_end():
    """start_line > end_line is invalid."""

    async def run():
        with tempfile.TemporaryDirectory() as tmp, temp_chdir(tmp):
            _seed_md(Path(tmp), "Range.md", "a\nb\nc\n")
            store = await _make_store()
            resp = await _read(store, path="Range.md", start_line=3, end_line=1)
            assert resp.success is False
            assert "start_line" in str(resp.answer)
            await store.close()
        print("✓ test_read_start_after_end passed")

    _run(run())


def test_read_start_line_exceeds_total():
    """start_line beyond total line count is invalid."""

    async def run():
        with tempfile.TemporaryDirectory() as tmp, temp_chdir(tmp):
            _seed_md(Path(tmp), "Short.md", "only-one-line\n")
            store = await _make_store()
            resp = await _read(store, path="Short.md", start_line=99)
            assert resp.success is False
            assert "exceeds" in str(resp.answer).lower()
            await store.close()
        print("✓ test_read_start_line_exceeds_total passed")

    _run(run())


def test_read_truncation():
    """A file larger than DEFAULT_MAX_BYTES triggers truncation with a continuation notice."""

    async def run():
        with tempfile.TemporaryDirectory() as tmp, temp_chdir(tmp):
            # Seed > DEFAULT_MAX_BYTES (50 KiB) so the default truncation kicks in.
            body = "\n".join(f"line {i}" for i in range(8000)) + "\n"
            _seed_md(Path(tmp), "Big.md", body)
            store = await _make_store()
            resp = await _read(store, path="Big.md")
            assert resp.success is True
            assert "truncated" in str(resp.answer)
            assert "start_line=" in str(resp.answer)
            await store.close()
        print("✓ test_read_truncation passed")

    _run(run())


def test_read_empty_path_rejected():
    """An empty `path` should be rejected."""

    async def run():
        with tempfile.TemporaryDirectory() as tmp, temp_chdir(tmp):
            store = await _make_store()
            resp = await _read(store, path="")
            assert resp.success is False
            assert "required" in str(resp.answer).lower()
            await store.close()
        print("✓ test_read_empty_path_rejected passed")

    _run(run())


# -- write / edit --------------------------------------------------------


def test_write_basic_with_frontmatter():
    """`write path=... name=... description=... content=...` writes a YAML front matter block."""

    async def run():
        with tempfile.TemporaryDirectory() as tmp, temp_chdir(tmp):
            store = await _make_store()
            resp = await _write(
                store,
                path="Notes/A.md",
                name="Greetings",
                description="a friendly hello note",
                content="# Hello",
            )
            assert resp.success is True
            assert "Wrote" in str(resp.answer)
            on_disk = (Path(tmp) / "Notes/A.md").read_text(encoding="utf-8")
            assert on_disk.startswith("---\n"), on_disk
            assert "name: Greetings" in on_disk
            assert "description: a friendly hello note" in on_disk
            assert "# Hello" in on_disk
            await store.close()
        print("✓ test_write_basic_with_frontmatter passed")

    _run(run())


def test_write_rejects_invalid_path_components():
    """`resolve_path` validates each segment with the same rules as daily-note slugs:
    Windows reserved chars / device names / trailing-dot (also blocks `..` traversal)."""

    async def run():
        with tempfile.TemporaryDirectory() as tmp, temp_chdir(tmp):
            store = await _make_store()
            for bad in (
                "CON.md",  # Windows-reserved device name (with extension)
                "Notes/AUX",  # device name in a sub-segment
                "Notes/foo<bar.md",  # invalid char `<`
                "../escape.md",  # path-traversal attempt
                "Notes/ trim.md",  # leading whitespace
                "lpt9.md",  # case-insensitive device name match
            ):
                resp = await _write(store, path=bad, content="x")
                assert resp.success is False, f"expected reject for {bad!r}, got success"
            # Sanity: no `Notes/` directory got created from any of the bad attempts.
            assert not (Path(tmp) / "Notes").exists()
            await store.close()
        print("✓ test_write_rejects_invalid_path_components passed")

    _run(run())


def test_write_no_suffix_autoappends_md():
    """`path` with no suffix gets `.md` appended."""

    async def run():
        with tempfile.TemporaryDirectory() as tmp, temp_chdir(tmp):
            store = await _make_store()
            resp = await _write(store, path="Notes/My", content="x")
            assert resp.success is True
            assert (Path(tmp) / "Notes/My.md").exists()
            await store.close()
        print("✓ test_write_no_suffix_autoappends_md passed")

    _run(run())


def test_write_overwrites_with_notice():
    """Writing into an existing path overwrites the file and surfaces a system notice."""

    async def run():
        with tempfile.TemporaryDirectory() as tmp, temp_chdir(tmp):
            _seed_md(Path(tmp), "Existing.md", "old\n")
            store = await _make_store()
            resp = await _write(store, path="Existing.md", content="new")
            assert resp.success is True
            assert "Wrote" in str(resp.answer)
            assert "already existed" in str(resp.answer)
            assert "overwritten" in str(resp.answer)
            on_disk = (Path(tmp) / "Existing.md").read_text(encoding="utf-8")
            assert "new" in on_disk and "old" not in on_disk, on_disk
            await store.close()
        print("✓ test_write_overwrites_with_notice passed")

    _run(run())


def test_write_creates_parent_dirs():
    """Nested-non-existent parents are auto-created."""

    async def run():
        with tempfile.TemporaryDirectory() as tmp, temp_chdir(tmp):
            store = await _make_store()
            resp = await _write(store, path="a/b/c/D.md", content="hi")
            assert resp.success is True
            assert (Path(tmp) / "a/b/c/D.md").exists()
            await store.close()
        print("✓ test_write_creates_parent_dirs passed")

    _run(run())


def test_write_no_frontmatter_when_all_empty():
    """When both `name` and `description` are empty strings, the file is body-only.

    The CLI schema declares them required, but the step is intentionally lenient
    so manual calls without these fields don't fail catastrophically."""

    async def run():
        with tempfile.TemporaryDirectory() as tmp, temp_chdir(tmp):
            store = await _make_store()
            resp = await _write(store, path="Plain.md", name="", description="", content="# Hello")
            assert resp.success is True
            on_disk = (Path(tmp) / "Plain.md").read_text(encoding="utf-8")
            assert not on_disk.startswith("---"), on_disk
            assert "# Hello" in on_disk
            await store.close()
        print("✓ test_write_no_frontmatter_when_all_empty passed")

    _run(run())


def test_write_ignores_arbitrary_extra_fields():
    """Extra kwargs beyond name/description are silently ignored (schema is strict)."""

    async def run():
        with tempfile.TemporaryDirectory() as tmp, temp_chdir(tmp):
            store = await _make_store()
            # Extras below should NOT appear in front matter under the
            # hardcoded-fields schema.
            resp = await _write(
                store,
                path="Custom.md",
                name="My Note",
                description="short summary",
                content="body",
                title="ignored",
                author="ignored",
                tags='["x","y"]',
            )
            assert resp.success is True
            on_disk = (Path(tmp) / "Custom.md").read_text(encoding="utf-8")
            assert on_disk.startswith("---\n"), on_disk
            assert "name: My Note" in on_disk
            assert "description: short summary" in on_disk
            assert "title:" not in on_disk
            assert "author:" not in on_disk
            assert "tags:" not in on_disk
            assert "body" in on_disk
            await store.close()
        print("✓ test_write_ignores_arbitrary_extra_fields passed")

    _run(run())


def test_write_only_description_present():
    """Step is lenient: providing only `description` works; missing `name` is skipped."""

    async def run():
        with tempfile.TemporaryDirectory() as tmp, temp_chdir(tmp):
            store = await _make_store()
            resp = await _write(store, path="OnlyDesc.md", description="just a description", content="body")
            assert resp.success is True
            on_disk = (Path(tmp) / "OnlyDesc.md").read_text(encoding="utf-8")
            assert on_disk.startswith("---\n"), on_disk
            assert "description: just a description" in on_disk
            assert "name:" not in on_disk
            await store.close()
        print("✓ test_write_only_description_present passed")

    _run(run())


def test_edit_global_replace():
    """`edit` replaces every occurrence of `old` with `new`."""

    async def run():
        with tempfile.TemporaryDirectory() as tmp, temp_chdir(tmp):
            _seed_md(Path(tmp), "E.md", "foo bar foo\nfoo\n")
            store = await _make_store()
            resp = await _edit(store, path="E.md", old="foo", new="qux")
            assert resp.success is True
            assert "3" in str(resp.answer)  # 3 replacements
            assert (Path(tmp) / "E.md").read_text(encoding="utf-8") == "qux bar qux\nqux\n"
            await store.close()
        print("✓ test_edit_global_replace passed")

    _run(run())


def test_edit_old_not_found():
    """`old` absent in the file → success=False."""

    async def run():
        with tempfile.TemporaryDirectory() as tmp, temp_chdir(tmp):
            _seed_md(Path(tmp), "E.md", "hello world\n")
            store = await _make_store()
            resp = await _edit(store, path="E.md", old="absent", new="x")
            assert resp.success is False
            assert "not found" in str(resp.answer).lower()
            assert (Path(tmp) / "E.md").read_text(encoding="utf-8") == "hello world\n"
            await store.close()
        print("✓ test_edit_old_not_found passed")

    _run(run())


def test_edit_missing_file():
    """Editing a non-existent file should fail."""

    async def run():
        with tempfile.TemporaryDirectory() as tmp, temp_chdir(tmp):
            store = await _make_store()
            resp = await _edit(store, path="NotThere.md", old="x", new="y")
            assert resp.success is False
            assert "does not exist" in str(resp.answer).lower()
            await store.close()
        print("✓ test_edit_missing_file passed")

    _run(run())


def test_edit_skips_frontmatter():
    """A match present in both front matter and body is replaced only in the body."""

    async def run():
        with tempfile.TemporaryDirectory() as tmp, temp_chdir(tmp):
            body = (
                "---\n"
                "name: alpha\n"
                "description: alpha-doc\n"
                "---\n"
                "intro paragraph mentioning alpha and alpha again.\n"
            )
            _seed_md(Path(tmp), "WithFM.md", body)
            store = await _make_store()
            resp = await _edit(store, path="WithFM.md", old="alpha", new="beta")
            assert resp.success is True
            assert "2" in str(resp.answer)  # 2 body occurrences only
            on_disk = (Path(tmp) / "WithFM.md").read_text(encoding="utf-8")
            # Front matter untouched.
            assert "name: alpha" in on_disk, on_disk
            assert "description: alpha-doc" in on_disk, on_disk
            # Body fully rewritten.
            assert "beta and beta" in on_disk, on_disk
            assert "alpha and alpha" not in on_disk, on_disk
            await store.close()
        print("✓ test_edit_skips_frontmatter passed")

    _run(run())


def test_edit_match_only_in_frontmatter_fails():
    """If `old` appears ONLY inside front matter, edit reports not-found and writes nothing."""

    async def run():
        with tempfile.TemporaryDirectory() as tmp, temp_chdir(tmp):
            body = "---\nname: secret\ndescription: nope\n---\nplain body without the keyword.\n"
            _seed_md(Path(tmp), "FMOnly.md", body)
            store = await _make_store()
            resp = await _edit(store, path="FMOnly.md", old="secret", new="leaked")
            assert resp.success is False
            assert "not found" in str(resp.answer).lower()
            # File untouched.
            assert (Path(tmp) / "FMOnly.md").read_text(encoding="utf-8") == body
            await store.close()
        print("✓ test_edit_match_only_in_frontmatter_fails passed")

    _run(run())


# -- non-markdown degraded mode + encoding edge cases --------------------


def test_write_non_md_skips_frontmatter():
    """Writing to a non-md path skips name/description and emits a recommendation notice."""

    async def run():
        with tempfile.TemporaryDirectory() as tmp, temp_chdir(tmp):
            store = await _make_store()
            resp = await _write(
                store,
                path="data/notes.txt",
                name="Greetings",
                description="should be ignored",
                content="# Hello",
            )
            assert resp.success is True
            assert "Wrote" in str(resp.answer)
            assert "non-markdown" in str(resp.answer).lower()
            on_disk = (Path(tmp) / "data/notes.txt").read_text(encoding="utf-8")
            assert not on_disk.startswith("---"), on_disk
            assert "name: Greetings" not in on_disk
            assert on_disk == "# Hello"
            await store.close()
        print("✓ test_write_non_md_skips_frontmatter passed")

    _run(run())


def test_edit_non_md_full_text():
    """Editing a non-md path operates on the full file body (no frontmatter parsing)."""

    async def run():
        with tempfile.TemporaryDirectory() as tmp, temp_chdir(tmp):
            # A YAML-looking header that would otherwise be stripped as frontmatter.
            body = "---\nname: keep-me\n---\nfoo bar foo\n"
            _seed_md(Path(tmp), "data/code.txt", body)
            store = await _make_store()
            resp = await _edit(store, path="data/code.txt", old="keep-me", new="replaced")
            assert resp.success is True
            assert "1" in str(resp.answer)
            assert "non-markdown" in str(resp.answer).lower()
            on_disk = (Path(tmp) / "data/code.txt").read_text(encoding="utf-8")
            assert "name: replaced" in on_disk, on_disk
            assert "foo bar foo" in on_disk, on_disk
            await store.close()
        print("✓ test_edit_non_md_full_text passed")

    _run(run())


def test_read_non_utf8_encoding():
    """A GBK-encoded legacy file (e.g. CN-Windows CSV) is decoded via the GBK fallback."""

    async def run():
        with tempfile.TemporaryDirectory() as tmp, temp_chdir(tmp):
            target = Path(tmp) / "data.csv"
            target.parent.mkdir(parents=True, exist_ok=True)
            text = "姓名,职业\n你好世界,工程师\n"
            target.write_bytes(text.encode("gbk"))
            store = await _make_store()
            resp = await _read(store, path="data.csv")
            assert resp.success is True
            assert "你好世界" in str(resp.answer)
            await store.close()
        print("✓ test_read_non_utf8_encoding passed")

    _run(run())


def test_edit_preserves_gbk_encoding():
    """Editing a GBK file keeps the file encoded in GBK after the rewrite."""

    async def run():
        with tempfile.TemporaryDirectory() as tmp, temp_chdir(tmp):
            target = Path(tmp) / "notes.csv"
            target.parent.mkdir(parents=True, exist_ok=True)
            text = "原始内容,占位\n" * 20
            target.write_bytes(text.encode("gbk"))
            store = await _make_store()
            resp = await _edit(store, path="notes.csv", old="原始内容", new="替换后")
            assert resp.success is True
            raw = target.read_bytes()
            # File still decodes as GBK (would raise if we'd silently converted to UTF-8).
            decoded = raw.decode("gbk")
            assert "替换后" in decoded and "原始内容" not in decoded
            await store.close()
        print("✓ test_edit_preserves_gbk_encoding passed")

    _run(run())


def test_read_utf8_bom():
    """Reading a UTF-8 file with BOM strips the BOM transparently."""

    async def run():
        with tempfile.TemporaryDirectory() as tmp, temp_chdir(tmp):
            target = Path(tmp) / "bom.txt"
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(b"\xef\xbb\xbfhello world\n")
            store = await _make_store()
            resp = await _read(store, path="bom.txt")
            assert resp.success is True
            assert "hello world" in str(resp.answer)
            assert "﻿" not in str(resp.answer)
            await store.close()
        print("✓ test_read_utf8_bom passed")

    _run(run())


# -- aggregate: reuse one store for all read cases ----------------------


def test_all_read_cases_one_store():
    """Run multiple read scenarios against a single shared store for efficiency."""

    async def run():
        with tempfile.TemporaryDirectory() as tmp, temp_chdir(tmp):
            _seed_md(Path(tmp), "Templates/Recipe.md", "# Recipe\nbody\n")
            _seed_md(Path(tmp), "Notes.md", "L1\nL2\nL3\n")
            store = await _make_store()

            resp = await _read(store, path="Templates/Recipe.md")
            assert resp.success is True and "# Recipe" in str(resp.answer)

            resp = await _read(store, path="Notes")  # auto-append .md
            assert resp.success is True and "L1" in str(resp.answer)

            resp = await _read(store, path="Notes.md", start_line=2, end_line=2)
            assert resp.success is True and str(resp.answer).strip() == "L2"

            await store.close()
        print("✓ test_all_read_cases_one_store passed")

    _run(run())


if __name__ == "__main__":
    print("\n=== crud step tests (opaque-byte surface) ===")
    # stat / list / download / move / delete
    test_stat_indexed_file()
    test_stat_directory_fallback()
    test_list_lists_files()
    test_list_respects_limit_and_non_recursive()
    test_download_to_explicit_path()
    test_download_to_temp_when_dst_path_empty()
    test_move_relocates_within_vault()
    test_move_refuses_overwrite_without_flag()
    test_move_default_retargets_inbound_links()
    test_move_opt_out_leaves_links_dangling()
    test_delete_removes_file()
    test_delete_missing_returns_error()
    test_delete_reports_inbound_refs()
    test_delete_folder_removes_tree()
    test_delete_folder_reports_only_external_inbound()
    test_delete_folder_empty_has_no_inbound()
    print("\n=== crud_md (read) E2E tests ===")
    test_read_relative_path()
    test_read_no_suffix_autoappends_md()
    test_read_line_range()
    test_read_absolute_path_accepted()
    test_read_non_md_degraded()
    test_read_missing_file()
    test_read_start_after_end()
    test_read_start_line_exceeds_total()
    test_read_truncation()
    test_read_empty_path_rejected()
    print("\n=== crud_md (write/edit) E2E tests ===")
    test_write_basic_with_frontmatter()
    test_write_no_suffix_autoappends_md()
    test_write_overwrites_with_notice()
    test_write_creates_parent_dirs()
    test_write_no_frontmatter_when_all_empty()
    test_write_ignores_arbitrary_extra_fields()
    test_write_only_description_present()
    test_edit_global_replace()
    test_edit_old_not_found()
    test_edit_missing_file()
    test_edit_skips_frontmatter()
    test_edit_match_only_in_frontmatter_fails()
    print("\n=== crud_md (non-md degraded mode) E2E tests ===")
    test_write_non_md_skips_frontmatter()
    test_edit_non_md_full_text()
    test_read_non_utf8_encoding()
    test_edit_preserves_gbk_encoding()
    test_read_utf8_bom()
    print("\n所有测试通过!")
