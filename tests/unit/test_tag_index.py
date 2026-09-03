"""Focused contract tests for the file-level tag index."""

# pylint: disable=protected-access

import asyncio
import os
from pathlib import Path

import pytest

from reme.components.application_context import ApplicationContext
from reme.components.runtime_context import RuntimeContext
from reme.components.tag_index import LocalTagIndex
from reme.enumeration import ComponentEnum
from reme.schema import TagSourceRecord
from reme.steps.index import UpdateTagIndexStep
from reme.steps.index._tag_index_io import read_frontmatter, scan_watch_scope, validated_watch_rules
from reme.steps.index.tag_index_watch import TagIndexWatchStep
from reme.utils.jsonl_zst import write_jsonl_zst


def _context(tmp_path: Path) -> ApplicationContext:
    context = ApplicationContext(workspace_dir=str(tmp_path), metadata_dir="metadata", daily_dir="daily")
    context.components = {ComponentEnum.TAG_INDEX: {}}
    return context


def test_tag_normalization_and_bidirectional_mutations(tmp_path: Path) -> None:
    """Normalize tags and keep both relationship directions consistent."""

    async def run() -> None:
        index = LocalTagIndex(name="default", app_context=_context(tmp_path), max_tags_per_file=3)
        await index.start()
        await index.upsert(
            [
                TagSourceRecord(
                    path="daily/a.md",
                    mtime_ns=1,
                    tags=["Python", "PYTHON", "C++", ".NET", "ignored"],
                ),
            ],
        )
        assert [record.tags for record in await index.get_records()] == [["python", "c++", ".net"]]
        file_id = index.path_to_file_id["daily/a.md"]
        assert all(file_id in posting for _name, posting in index.tags.values())

        await index.upsert([TagSourceRecord(path="daily/a.md", mtime_ns=2, tags=["ReMe"])])
        assert [record.tags for record in await index.get_records()] == [["reme"]]
        assert set(index.tag_to_tag_id) == {"reme"}

        await index.delete(["daily/a.md", "daily/missing.md"])
        assert await index.get_records() == []
        assert index.tags == {}
        await index.close()

    asyncio.run(run())


def test_snapshot_round_trip_and_parameter_change_rebuilds(tmp_path: Path) -> None:
    """Restore valid snapshots and discard parameter-incompatible snapshots."""

    async def run() -> None:
        context = _context(tmp_path)
        first = LocalTagIndex(name="default", app_context=context)
        await first.start()
        await first.upsert([TagSourceRecord(path="daily/a.md", mtime_ns=3, tags=["A"])])
        await first.dump()
        old_file_id = first.path_to_file_id["daily/a.md"]
        await first.close()

        restored = LocalTagIndex(name="default", app_context=context)
        await restored.start()
        assert restored.loaded is True
        assert restored.path_to_file_id["daily/a.md"] == old_file_id
        await restored.close()

        incompatible = LocalTagIndex(name="default", app_context=context, max_tag_length=8)
        await incompatible.start()
        assert incompatible.loaded is False
        assert await incompatible.get_records() == []

    asyncio.run(run())


def test_bounded_frontmatter_states_and_alias_rejection(tmp_path: Path) -> None:
    """Bound prefix reads and reject aliases without treating them as I/O errors."""
    valid = tmp_path / "valid.md"
    valid.write_bytes(b"\xef\xbb\xbf---\r\ntags: [Python, C++]\r\n---\r\n" + b"x" * 100_000)
    result = read_frontmatter(valid, 1024)
    assert result.status == "parsed"
    assert result.metadata == {"tags": ["Python", "C++"]}
    assert result.bytes_read <= 1024

    missing = tmp_path / "missing.md"
    missing.write_bytes(b"body\n" + b"x" * 100_000)
    result = read_frontmatter(missing, 1024)
    assert result.status == "no_frontmatter"
    assert result.bytes_read <= 1024

    alias = tmp_path / "alias.md"
    alias.write_text("---\na: &a [x]\ntags: *a\n---\n", encoding="utf-8")
    assert read_frontmatter(alias, 1024).status == "invalid_frontmatter"

    unterminated = tmp_path / "unterminated.md"
    unterminated.write_bytes(b"---\ntags: [a]\n" + b"x" * 10_000)
    result = read_frontmatter(unterminated, 128)
    assert result.status == "invalid_frontmatter"
    assert result.bytes_read == 128


def test_update_step_uses_relative_paths_and_coalesces_stale_events(tmp_path: Path) -> None:
    """Store relative paths and coalesce stale modified events to deletion."""

    async def run() -> None:
        daily = tmp_path / "daily"
        daily.mkdir()
        note = daily / "a.md"
        note.write_text("---\ntags: [Python]\n---\nbody\n", encoding="utf-8")
        context = _context(tmp_path)
        index = LocalTagIndex(name="default", app_context=context)
        context.components[ComponentEnum.TAG_INDEX]["default"] = index
        await index.start()
        runtime = RuntimeContext(
            watch_dirs=["daily_dir"],
            watch_suffixes=["md"],
            changes=[{"change": "added", "path": str(note)}],
        )
        response = await UpdateTagIndexStep(app_context=context, tag_index="default")(runtime)
        assert response.answer[0]["path"] == "daily/a.md"
        assert (await index.get_records())[0].tags == ["python"]

        note.unlink()
        # A stale modified event is coalesced to delete and therefore removes the record.
        response = await UpdateTagIndexStep(app_context=context, tag_index="default")(
            RuntimeContext(
                watch_dirs=["daily_dir"],
                watch_suffixes=["md"],
                changes=[{"change": "modified", "path": str(note)}],
            ),
        )
        assert response.answer[0]["change"] == "deleted"
        assert await index.get_records() == []
        await index.close()

    asyncio.run(run())


def test_startup_audit_repairs_tags_even_when_mtime_is_unchanged(tmp_path: Path) -> None:
    """Audit frontmatter rather than trusting an unchanged filesystem mtime."""

    async def run() -> None:
        daily = tmp_path / "daily"
        daily.mkdir()
        note = daily / "a.md"
        note.write_text("---\ntags: [new]\n---\nbody\n", encoding="utf-8")
        mtime_ns = note.stat().st_mtime_ns
        context = _context(tmp_path)
        index = LocalTagIndex(name="default", app_context=context)
        context.components[ComponentEnum.TAG_INDEX]["default"] = index
        await index.start()
        await index.upsert([TagSourceRecord(path="daily/a.md", mtime_ns=mtime_ns, tags=["old"])])
        index.loaded = True

        runtime = RuntimeContext(
            stop_event=asyncio.Event(),
            watch_dirs=["daily_dir"],
            watch_suffixes=["md"],
        )
        step = TagIndexWatchStep(app_context=context, tag_index=index)
        step.context = runtime
        rules = validated_watch_rules(context.app_config, tmp_path, runtime)
        current = scan_watch_scope(tmp_path, rules, recursive=True)
        results = await step._initial_sync(current)

        assert (await index.get_records())[0].tags == ["new"]
        assert results[0]["change"] == "modified"
        assert runtime.response.metadata["counts"] == {
            "added": 0,
            "modified": 1,
            "deleted": 0,
            "audited": 1,
            "skipped": 0,
        }
        await index.close()

    asyncio.run(run())


def test_jsonl_zst_atomic_failure_cleans_temp_and_preserves_target(tmp_path: Path, monkeypatch) -> None:
    """A failed atomic replace leaves neither a torn target nor a temporary file."""
    target = tmp_path / "index.jsonl.zst"
    target.write_bytes(b"previous snapshot")

    def fail_replace(_source, _target) -> None:
        raise OSError("replace failed")

    monkeypatch.setattr(os, "replace", fail_replace)
    with pytest.raises(OSError, match="replace failed"):
        write_jsonl_zst(target, ['{"new":true}'])

    assert target.read_bytes() == b"previous snapshot"
    assert not list(tmp_path.glob(".index.jsonl.zst.*.tmp"))
