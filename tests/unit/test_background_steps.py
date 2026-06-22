"""Tests for background steps: scan/watch/dispatch steps.

Both scan steps are subclasses of BaseStep. To exercise them without spinning up
the full ApplicationContext, we pass real (started) file_store/file_chunker via
the step's kwargs (so the BaseStep _resolve() machinery returns them).

InitChangesStep writes its result into ``context["changes"]`` for a
downstream ``update_index_step`` to consume; tests assert against that key.
"""

# pylint: disable=protected-access

import asyncio
import os
import tempfile
import warnings
from pathlib import Path
from unittest.mock import MagicMock

from watchfiles import Change

from reme.components.file_chunker import DefaultFileChunker
from reme.components.file_catalog import LocalFileCatalog
from reme.components.file_store import LocalFileStore
from reme.components.runtime_context import RuntimeContext
from reme.enumeration import ComponentEnum
from reme.steps.evolve.auto_resource import AutoResourceStep, _compute_note_stem
from reme.steps.index import (
    DEFAULT_LOW_POWER_POLL_MS,
    DEFAULT_WATCH_DEBOUNCE_MS,
    DEFAULT_WATCH_STEP_MS,
    ClearStoreStep,
    InitChangesStep,
    LogChangesStep,
    UpdateCatalogStep,
    WatchChangesStep,
)
from reme.steps.index._change_batch import bucket_changes
from reme.steps.index._watch_rules import WatchRule, build_watch_rules, collect_existing, match_file

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


def write_file(path: Path, content: str = "x") -> Path:
    """Create parent dirs and write `content` to `path`; return the path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


def _make_app_context(workspace_path: Path, daily_dir="daily", digest_dir="digest", resource_dir="resource"):
    """Create a mock app_context with app_config pointing to the given workspace."""
    ctx = MagicMock()
    ctx.app_config.workspace_dir = str(workspace_path)
    ctx.app_config.daily_dir = daily_dir
    ctx.app_config.digest_dir = digest_dir
    ctx.app_config.resource_dir = resource_dir
    return ctx


# ---------------------------------------------------------------------------
# _watch_rules module tests
# ---------------------------------------------------------------------------


def test_build_watch_rules_basic():
    """Build rules from watch_dirs and watch_suffixes."""
    app_config = MagicMock()
    app_config.daily_dir = "daily"
    app_config.digest_dir = "digest"
    app_config.resource_dir = "resource"
    workspace = Path("/fake/workspace")

    rules = build_watch_rules(app_config, workspace, watch_dirs=["daily_dir", "digest_dir"], watch_suffixes=["md"])
    assert len(rules) == 2
    assert rules[0].path == workspace / "daily"
    assert rules[0].suffixes == ["md"]
    assert rules[1].path == workspace / "digest"
    print("✓ test_build_watch_rules_basic passed")


def test_build_watch_rules_multiple_suffixes():
    """Multiple suffixes are forwarded to each rule."""
    app_config = MagicMock()
    app_config.daily_dir = "daily"
    app_config.resource_dir = "resource"
    workspace = Path("/fake/workspace")

    rules = build_watch_rules(
        app_config,
        workspace,
        watch_dirs=["daily_dir", "resource_dir"],
        watch_suffixes=["md", "jsonl"],
    )
    assert len(rules) == 2
    assert rules[0].suffixes == ["md", "jsonl"]
    assert rules[1].suffixes == ["md", "jsonl"]
    print("✓ test_build_watch_rules_multiple_suffixes passed")


def test_build_watch_rules_fallback_literal():
    """Unknown field names are used as literal directory names."""
    app_config = MagicMock(spec=[])  # no attributes
    workspace = Path("/fake/workspace")
    rules = build_watch_rules(app_config, workspace, watch_dirs=["custom_dir"], watch_suffixes=["txt"])
    assert rules[0].path == workspace / "custom_dir"
    print("✓ test_build_watch_rules_fallback_literal passed")


def test_match_file_suffix():
    """match_file accepts files matching suffix under rule path."""
    rules = [WatchRule(path=Path("/workspace/daily"), suffixes=["md"])]
    assert match_file("/workspace/daily/2026-01-01.md", rules)
    assert match_file("/workspace/daily/sub/note.md", rules)
    assert not match_file("/workspace/daily/file.txt", rules)
    assert not match_file("/workspace/other/file.md", rules)
    print("✓ test_match_file_suffix passed")


def test_match_file_no_suffix_filter():
    """Empty suffixes list means all files match."""
    rules = [WatchRule(path=Path("/workspace/resource"), suffixes=[])]
    assert match_file("/workspace/resource/anything.xyz", rules)
    assert match_file("/workspace/resource/sub/deep.pdf", rules)
    assert not match_file("/workspace/other/file.md", rules)
    print("✓ test_match_file_no_suffix_filter passed")


def test_collect_existing_filters():
    """collect_existing applies suffix rules correctly."""
    with tempfile.TemporaryDirectory() as tmpdir:
        workspace = Path(tmpdir)
        daily = workspace / "daily"
        resource = workspace / "resource"
        write_file(daily / "note.md")
        write_file(daily / "ignore.txt")
        write_file(resource / "data.json")
        write_file(resource / "binary.png")

        rules = [
            WatchRule(path=daily, suffixes=["md"]),
            WatchRule(path=resource, suffixes=["json"]),
        ]
        result = collect_existing(rules, recursive=True)
        paths = set(result.keys())
        assert str((daily / "note.md").absolute()) in paths
        assert str((daily / "ignore.txt").absolute()) not in paths
        assert str((resource / "data.json").absolute()) in paths
        assert str((resource / "binary.png").absolute()) not in paths
    print("✓ test_collect_existing_filters passed")


# ---------------------------------------------------------------------------
# InitChangesStep
# ---------------------------------------------------------------------------


def test_clear_and_scan_defaults_include_jsonl():
    """Full reindex should include jsonl files when no explicit suffix filter is passed."""

    async def run():
        with tempfile.TemporaryDirectory() as tmpdir, temp_chdir(tmpdir):
            cwd = Path.cwd()
            write_file(cwd / "daily" / "note.md", "alpha")
            write_file(cwd / "resource" / "events.jsonl", '{"a": 1}\n')
            write_file(cwd / "resource" / "ignore.txt", "skip")

            fs = LocalFileStore(name="test_store", embedding_store="")
            await fs.start()
            try:
                clear_step = ClearStoreStep(file_store=fs, app_context=_make_app_context(cwd))
                scan_step = InitChangesStep(store="file_store", file_store=fs, app_context=_make_app_context(cwd))
                ctx = RuntimeContext(watch_dirs=["daily_dir", "resource_dir"], watch_suffixes=["md", "jsonl"])
                await clear_step(ctx)
                resp = await scan_step(ctx)
                paths = {Path(item["path"]).name for item in ctx["changes"]}
                assert resp.metadata["counts"] == {"added": 2, "modified": 0, "deleted": 0}
                assert paths == {"note.md", "events.jsonl"}
            finally:
                await fs.close()
        print("✓ test_clear_and_scan_defaults_include_jsonl passed")

    asyncio.run(run())


async def _make_scan_step(workspace_path: Path, watch_dirs=None, watch_suffixes=None, recursive=True):
    fs = LocalFileStore(name="test_store", embedding_store="")
    chunker = DefaultFileChunker()
    await fs.start()
    await chunker.start()
    app_ctx = _make_app_context(workspace_path)
    step = InitChangesStep(
        store="file_store",
        recursive=recursive,
        file_store=fs,
        file_chunker=chunker,
        app_context=app_ctx,
    )
    context = RuntimeContext(
        watch_dirs=watch_dirs or ["daily_dir", "digest_dir"],
        watch_suffixes=watch_suffixes or ["md"],
    )
    return step, context, fs, chunker


async def _teardown(fs: LocalFileStore, chunker: DefaultFileChunker) -> None:
    await chunker.close()
    await fs.close()


def test_scan_changes_initial_all_added():
    """First run on a fresh store emits 'added' for every existing file."""

    async def run():
        with tempfile.TemporaryDirectory() as tmpdir, temp_chdir(tmpdir):
            cwd = Path.cwd()
            write_file(cwd / "daily" / "a.md", "alpha")
            write_file(cwd / "daily" / "b.md", "beta")
            (cwd / "digest").mkdir(parents=True, exist_ok=True)

            step, ctx, fs, chunker = await _make_scan_step(cwd)
            try:
                resp = await step(ctx)
                counts = resp.metadata["counts"]
                assert counts == {"added": 2, "modified": 0, "deleted": 0}
                assert len(ctx["changes"]) == 2
            finally:
                await _teardown(fs, chunker)
        print("✓ test_scan_changes_initial_all_added passed")

    asyncio.run(run())


def test_scan_changes_no_changes():
    """Second run over an unchanged store reports zero counts."""

    async def run():
        with tempfile.TemporaryDirectory() as tmpdir, temp_chdir(tmpdir):
            cwd = Path.cwd()
            a = write_file(cwd / "daily" / "a.md", "alpha")
            (cwd / "digest").mkdir(parents=True, exist_ok=True)

            step, ctx, fs, chunker = await _make_scan_step(cwd)
            try:
                node, chunks = await chunker.chunk(a)
                await fs.upsert([(node, chunks)])
                resp = await step(ctx)
                assert resp.metadata["counts"] == {"added": 0, "modified": 0, "deleted": 0}
                assert ctx["changes"] == []
            finally:
                await _teardown(fs, chunker)
        print("✓ test_scan_changes_no_changes passed")

    asyncio.run(run())


def test_scan_changes_detect_modify_delete():
    """Second pass distinguishes added/modified/deleted."""

    async def run():
        with tempfile.TemporaryDirectory() as tmpdir, temp_chdir(tmpdir):
            cwd = Path.cwd()
            a = write_file(cwd / "daily" / "a.md", "alpha")
            b = write_file(cwd / "daily" / "b.md", "beta")
            (cwd / "digest").mkdir(parents=True, exist_ok=True)

            step, ctx, fs, chunker = await _make_scan_step(cwd)
            try:
                for p in (a, b):
                    node, chunks = await chunker.chunk(p)
                    await fs.upsert([(node, chunks)])
                a.write_text("alpha-v2", encoding="utf-8")
                os.utime(a, (9_999_999_999, 9_999_999_999))
                b.unlink()
                write_file(cwd / "daily" / "c.md", "gamma")

                resp = await step(ctx)
                counts = resp.metadata["counts"]
                assert counts == {"added": 1, "modified": 1, "deleted": 1}
            finally:
                await _teardown(fs, chunker)
        print("✓ test_scan_changes_detect_modify_delete passed")

    asyncio.run(run())


def test_scan_changes_missing_dir_skipped():
    """Non-existent watch_dirs entries are dropped silently."""

    async def run():
        with tempfile.TemporaryDirectory() as tmpdir, temp_chdir(tmpdir):
            cwd = Path.cwd()
            (cwd / "daily").mkdir()
            # digest dir missing
            step, ctx, fs, chunker = await _make_scan_step(cwd)
            try:
                resp = await step(ctx)
                assert resp.metadata["counts"] == {"added": 0, "modified": 0, "deleted": 0}
            finally:
                await _teardown(fs, chunker)
        print("✓ test_scan_changes_missing_dir_skipped passed")

    asyncio.run(run())


def test_scan_changes_resource_dir():
    """Scanning resource_dir with multiple suffixes works."""

    async def run():
        with tempfile.TemporaryDirectory() as tmpdir, temp_chdir(tmpdir):
            cwd = Path.cwd()
            resource = cwd / "resource"
            write_file(resource / "data.json", "{}")
            write_file(resource / "note.md", "# Note")
            write_file(resource / "image.png", "binary")

            step, ctx, fs, chunker = await _make_scan_step(
                cwd,
                watch_dirs=["resource_dir"],
                watch_suffixes=["md", "json"],
            )
            try:
                resp = await step(ctx)
                assert resp.metadata["counts"]["added"] == 2
            finally:
                await _teardown(fs, chunker)
        print("✓ test_scan_changes_resource_dir passed")

    asyncio.run(run())


def test_init_changes_named_file_catalog_monitor():
    """monitor_type/monitor_name selects the requested file_catalog component."""

    async def run():
        with tempfile.TemporaryDirectory() as tmpdir, temp_chdir(tmpdir):
            cwd = Path.cwd()
            write_file(cwd / "resource" / "2026-01-01" / "a.md", "alpha")

            catalog = LocalFileCatalog(name="resource")
            await catalog.start()
            try:
                app_ctx = _make_app_context(cwd)
                app_ctx.components = {ComponentEnum.FILE_CATALOG: {"resource": catalog}}
                step = InitChangesStep(monitor_type="file_catalog", monitor_name="resource", app_context=app_ctx)
                ctx = RuntimeContext(watch_dirs=["resource_dir"], watch_suffixes=["md"])
                resp = await step(ctx)

                assert resp.metadata["counts"] == {"added": 1, "modified": 0, "deleted": 0}
                assert ctx["changes"][0]["change"] == "added"
            finally:
                await catalog.close()
        print("✓ test_init_changes_named_file_catalog_monitor passed")

    asyncio.run(run())


def test_bucket_changes_coalesces_by_final_file_state():
    """A delete+add replacement batch for an existing file becomes one modified event."""
    with tempfile.TemporaryDirectory() as tmpdir:
        p = write_file(Path(tmpdir) / "daily" / "a.md", "alpha")
        buckets = bucket_changes(
            [
                {"change": "deleted", "path": str(p)},
                {"change": "added", "path": str(p)},
            ],
        )

        assert buckets[Change.modified] == [str(p)]
        assert buckets[Change.added] == []
        assert buckets[Change.deleted] == []
    print("✓ test_bucket_changes_coalesces_by_final_file_state passed")


def test_update_catalog_relative_path_uses_workspace():
    """update_catalog_step resolves workspace-relative change paths against workspace_path."""

    async def run():
        with tempfile.TemporaryDirectory() as tmpdir, temp_chdir(tmpdir):
            cwd = Path.cwd()
            write_file(cwd / "daily" / "a.md", "alpha")

            catalog = LocalFileCatalog(name="test_catalog")
            await catalog.start()
            try:
                step = UpdateCatalogStep(file_catalog=catalog, app_context=_make_app_context(cwd))
                ctx = RuntimeContext(changes=[{"change": "added", "path": "daily/a.md"}])
                resp = await step(ctx)

                assert resp.success is True
                nodes = await catalog.get_nodes()
                assert [n.path for n in nodes] == ["daily/a.md"]
            finally:
                await catalog.close()
        print("✓ test_update_catalog_relative_path_uses_workspace passed")

    asyncio.run(run())


def test_index_update_loop_init_dispatch_updates_store_across_batches():
    """index_update_loop init scan dispatches to update_index_step and preserves final store state."""

    async def run():
        with tempfile.TemporaryDirectory() as tmpdir, temp_chdir(tmpdir):
            cwd = Path.cwd()
            daily_a = write_file(cwd / "daily" / "a.md", "alpha\n[[digest/report.md]]\n")
            write_file(cwd / "digest" / "report.md", "# Report\nbeta\n")
            write_file(cwd / "daily" / "ignore.txt", "skip")

            fs = LocalFileStore(name="default", embedding_store="")
            chunker = DefaultFileChunker()
            await fs.start()
            await chunker.start()
            try:
                app_ctx = _make_app_context(cwd)
                app_ctx.components = {
                    ComponentEnum.FILE_STORE: {"default": fs},
                    ComponentEnum.FILE_CHUNKER: {"default": chunker},
                }
                ctx = RuntimeContext(watch_dirs=["daily_dir", "digest_dir"], watch_suffixes=["md"])
                init_step = InitChangesStep(
                    monitor_type="file_store",
                    monitor_name="default",
                    dispatch_steps=["update_index_step"],
                    app_context=app_ctx,
                )

                first = await init_step(ctx)
                assert first.metadata["counts"] == {"added": 2, "modified": 0, "deleted": 0}
                nodes = {n.path: n for n in await fs.get_nodes()}
                assert set(nodes) == {"daily/a.md", "digest/report.md"}
                assert all(nodes[p].chunk_ids for p in nodes)

                daily_a.write_text("alpha v2\n[[digest/report.md]]\n", encoding="utf-8")
                os.utime(daily_a, (9_999_999_999, 9_999_999_999))
                (cwd / "digest" / "report.md").unlink()
                write_file(cwd / "daily" / "c.md", "gamma\n")

                second = await init_step(ctx)
                assert second.metadata["counts"] == {"added": 1, "modified": 1, "deleted": 1}
                assert {(c["change"], Path(c["path"]).name) for c in ctx["changes"]} == {
                    ("modified", "a.md"),
                    ("deleted", "report.md"),
                    ("added", "c.md"),
                }

                nodes = {n.path: n for n in await fs.get_nodes()}
                assert set(nodes) == {"daily/a.md", "daily/c.md"}
                assert all(nodes[p].chunk_ids for p in nodes)
                assert all(chunk.path in nodes for chunk in fs.file_chunks.values())
            finally:
                await chunker.close()
                await fs.close()
        print("✓ test_index_update_loop_init_dispatch_updates_store_across_batches passed")

    asyncio.run(run())


def test_digest_watch_loop_init_dispatch_updates_named_catalog_and_logs():
    """digest_watch_loop style config updates the digest catalog without touching resource/default catalogs."""

    async def run():
        with tempfile.TemporaryDirectory() as tmpdir, temp_chdir(tmpdir):
            cwd = Path.cwd()
            daily = write_file(cwd / "daily" / "2026-01-01.md", "day one")
            digest = write_file(cwd / "digest" / "week.md", "weekly")
            write_file(cwd / "resource" / "asset.md", "not watched by digest loop")

            digest_catalog = LocalFileCatalog(name="digest")
            resource_catalog = LocalFileCatalog(name="resource")
            await digest_catalog.start()
            await resource_catalog.start()
            try:
                app_ctx = _make_app_context(cwd)
                app_ctx.components = {
                    ComponentEnum.FILE_CATALOG: {
                        "digest": digest_catalog,
                        "resource": resource_catalog,
                    },
                }
                ctx = RuntimeContext(watch_dirs=["daily_dir", "digest_dir"], watch_suffixes=["md"])
                init_step = InitChangesStep(
                    monitor_type="file_catalog",
                    monitor_name="digest",
                    dispatch_steps=[
                        {"backend": "update_catalog_step", "file_catalog": "digest"},
                        {"backend": "log_changes_step"},
                    ],
                    app_context=app_ctx,
                )

                first = await init_step(ctx)
                assert first.metadata["counts"] == {"added": 2, "modified": 0, "deleted": 0}
                assert {n.path for n in await digest_catalog.get_nodes()} == {
                    "daily/2026-01-01.md",
                    "digest/week.md",
                }
                assert await resource_catalog.get_nodes() == []

                daily.write_text("day two", encoding="utf-8")
                os.utime(daily, (9_999_999_999, 9_999_999_999))
                digest.unlink()
                write_file(cwd / "daily" / "2026-01-02.md", "next day")

                second = await init_step(ctx)
                assert second.metadata["counts"] == {"added": 1, "modified": 1, "deleted": 1}
                assert {(c["change"], Path(c["path"]).name) for c in ctx["changes"]} == {
                    ("modified", "2026-01-01.md"),
                    ("deleted", "week.md"),
                    ("added", "2026-01-02.md"),
                }
                assert {n.path for n in await digest_catalog.get_nodes()} == {
                    "daily/2026-01-01.md",
                    "daily/2026-01-02.md",
                }
                assert await resource_catalog.get_nodes() == []
            finally:
                await resource_catalog.close()
                await digest_catalog.close()
        print("✓ test_digest_watch_loop_init_dispatch_updates_named_catalog_and_logs passed")

    asyncio.run(run())


# ---------------------------------------------------------------------------
# WatchChangesStep
# ---------------------------------------------------------------------------


def test_watch_changes_default_low_power_timing():
    """Default watcher timing favors lower resource use."""
    step = WatchChangesStep()

    assert step.debounce == DEFAULT_WATCH_DEBOUNCE_MS
    assert step.step == DEFAULT_WATCH_STEP_MS
    assert step.poll_delay_ms == DEFAULT_LOW_POWER_POLL_MS

    custom = WatchChangesStep(debounce=1000, step=250, poll_delay_ms=3000)
    assert custom.debounce == 1000
    assert custom.step == 250
    assert custom.poll_delay_ms == 3000

    print("✓ test_watch_changes_default_low_power_timing passed")


def test_watch_changes_requires_stop_event():
    """Missing stop_event in context raises a clear error."""

    async def run():
        with tempfile.TemporaryDirectory() as tmpdir, temp_chdir(tmpdir):
            cwd = Path.cwd()
            (cwd / "daily").mkdir()
            app_ctx = _make_app_context(cwd)
            step = WatchChangesStep(app_context=app_ctx)
            step.context = RuntimeContext(watch_dirs=["daily_dir"], watch_suffixes=["md"])
            try:
                await step.execute()
            except RuntimeError as e:
                assert "stop_event" in str(e)
            else:
                raise AssertionError("expected RuntimeError")
        print("✓ test_watch_changes_requires_stop_event passed")

    asyncio.run(run())


def test_watch_changes_raises_no_valid_paths():
    """With no valid watch_paths, the step raises."""

    async def run():
        with tempfile.TemporaryDirectory() as tmpdir, temp_chdir(tmpdir):
            cwd = Path.cwd()
            app_ctx = _make_app_context(cwd)
            step = WatchChangesStep(app_context=app_ctx)
            stop = asyncio.Event()
            step.context = RuntimeContext(stop_event=stop, watch_dirs=["daily_dir"], watch_suffixes=["md"])
            try:
                await step.execute()
            except RuntimeError as e:
                assert "No valid watch paths" in str(e)
            else:
                raise AssertionError("expected RuntimeError")
        print("✓ test_watch_changes_raises_no_valid_paths passed")

    asyncio.run(run())


def test_watch_changes_filter_matches_rules():
    """The internal filter uses watch rules from context."""
    with tempfile.TemporaryDirectory() as tmpdir:
        workspace = Path(tmpdir)
        (workspace / "daily").mkdir()
        (workspace / "digest").mkdir()
        (workspace / "resource").mkdir()
        app_ctx = _make_app_context(workspace)

        step = WatchChangesStep(app_context=app_ctx)
        step.context = RuntimeContext(watch_dirs=["daily_dir", "digest_dir"], watch_suffixes=["md"])
        step._rules = step._get_watch_rules()

        assert step._filter(Change.added, str(workspace / "daily/foo.md"))
        assert step._filter(Change.added, str(workspace / "digest/bar.md"))
        assert not step._filter(Change.added, str(workspace / "daily/foo.txt"))
        assert not step._filter(Change.added, str(workspace / "resource/file.md"))

    print("✓ test_watch_changes_filter_matches_rules passed")


def test_watch_changes_dispatch_steps_list():
    """dispatch_steps config is stored by BaseStep."""
    step = WatchChangesStep(dispatch_steps=["update_catalog_step", "auto_resource_step"])
    assert step.dispatch_step_specs == ["update_catalog_step", "auto_resource_step"]

    print("✓ test_watch_changes_dispatch_steps_list passed")


def test_auto_resource_batch_deleted_changes():
    """AutoResourceStep accepts a batch of change dicts from dispatch_steps."""

    async def run():
        with tempfile.TemporaryDirectory() as tmpdir, temp_chdir(tmpdir):
            cwd = Path.cwd()
            app_ctx = _make_app_context(cwd)
            fs = LocalFileStore(name="test_store", embedding_store="")
            await fs.start()
            try:
                filename = "file.md"
                note_stem = _compute_note_stem(filename)
                note_path = cwd / "daily" / "2026-01-01" / f"{note_stem}.md"
                write_file(note_path, "---\nname: test\n---\nbody\n")

                step = AutoResourceStep(app_context=app_ctx, file_store=fs)
                ctx = RuntimeContext(
                    changes=[
                        {"change": "deleted", "path": str(cwd / "resource" / "2026-01-01" / filename)},
                    ],
                )
                resp = await step(ctx)

                assert resp.success is True
                assert resp.answer == "Processed 1/1 resource change(s)"
                assert resp.metadata["processed"] == 1
                assert resp.metadata["results"][0]["path"] == "resource/2026-01-01/file.md"
                assert not note_path.exists()
            finally:
                await fs.close()
        print("✓ test_auto_resource_batch_deleted_changes passed")

    asyncio.run(run())


# LogChangesStep
# ---------------------------------------------------------------------------


def test_log_changes_step():
    """LogChangesStep logs and reports count."""

    async def run():
        step = LogChangesStep()
        changes = [
            {"change": "added", "path": "/workspace/daily/note.md"},
            {"change": "deleted", "path": "/workspace/daily/old.md"},
        ]
        ctx = RuntimeContext(changes=changes)
        resp = await step(ctx)
        assert resp.success is True
        assert resp.metadata["count"] == 2
        print("✓ test_log_changes_step passed")

    asyncio.run(run())


if __name__ == "__main__":
    print("\n=== Background Steps Tests ===")
    # _watch_rules
    test_build_watch_rules_basic()
    test_build_watch_rules_multiple_suffixes()
    test_build_watch_rules_fallback_literal()
    test_match_file_suffix()
    test_match_file_no_suffix_filter()
    test_collect_existing_filters()
    # InitChangesStep
    test_clear_and_scan_defaults_include_jsonl()
    test_scan_changes_initial_all_added()
    test_scan_changes_no_changes()
    test_scan_changes_detect_modify_delete()
    test_scan_changes_missing_dir_skipped()
    test_scan_changes_resource_dir()
    test_index_update_loop_init_dispatch_updates_store_across_batches()
    test_digest_watch_loop_init_dispatch_updates_named_catalog_and_logs()
    # WatchChangesStep
    test_watch_changes_default_low_power_timing()
    test_watch_changes_requires_stop_event()
    test_watch_changes_raises_no_valid_paths()
    test_watch_changes_filter_matches_rules()
    test_watch_changes_dispatch_steps_list()
    test_auto_resource_batch_deleted_changes()
    # LogChangesStep
    test_log_changes_step()
    print("\n所有测试通过!")
