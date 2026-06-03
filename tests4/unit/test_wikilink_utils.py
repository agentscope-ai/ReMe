"""Tests for the wikilink helpers in ``reme4.utils.wikilink_handler``.

Two pure async helpers used by file_move / file_delete:

  * ``retarget_links(src, dst, scope?, dry_run?)`` — rewrite wikilink
    targets across the vault, using the file_graph's reverse index to
    find inbound sources (no fs scan).
  * ``find_inbound(target, scope?)`` — report inbound count without
    rewriting.

Retarget only matches the literal full-path form ``[[topics/x.md]]``;
short ``[[x]]`` and no-ext ``[[topics/x]]`` are left alone by design.
"""

# pylint: disable=protected-access

import asyncio
import os
import tempfile
import warnings
from pathlib import Path

from reme4.components.file_store import LocalFileStore
from reme4.schema import FileNode
from reme4.utils.wikilink_handler import WikilinkHandler

warnings.filterwarnings("ignore", category=DeprecationWarning, module="jieba")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="pkg_resources")


class temp_chdir:
    """Test helper: chdir to ``path`` on enter, restore previous cwd on exit."""

    def __init__(self, path):
        self.path = path
        self.old = None

    def __enter__(self):
        self.old = os.getcwd()
        os.chdir(self.path)
        return self

    def __exit__(self, *exc):
        os.chdir(self.old)


async def _store_with(files: dict[str, str]) -> LocalFileStore:
    """LocalFileStore seeded with (rel → content) files: written to disk
    AND registered in the file_graph with wikilinks parsed from the body.

    Without the parsed links the reverse-index lookup yields nothing and
    retarget becomes a no-op.
    """
    store = LocalFileStore(name="t", embedding_store="")
    await store.start()
    nodes: list[FileNode] = []
    root = Path.cwd()
    for rel, content in files.items():
        abs_path = root / rel
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


async def _empty_store() -> LocalFileStore:
    store = LocalFileStore(name="t", embedding_store="")
    await store.start()
    return store


def test_retarget_exact_full_path_match():
    """[[topics/Alice.md]] → [[people/Alice.md]]; non-matching links untouched."""

    async def run():
        with tempfile.TemporaryDirectory() as tmp, temp_chdir(tmp):
            root = Path(tmp)
            store = await _store_with(
                {
                    "note.md": "See [[topics/Alice.md]] and [[topics/Bob.md]].",
                    "people/Alice.md": "# Alice",
                },
            )
            payload = await WikilinkHandler.retarget_links(store, src="topics/Alice.md", dst="people/Alice.md")
            assert "error" not in payload
            assert payload["links_changed"] == 1
            assert payload["files_touched"] == 1
            body = (root / "note.md").read_text(encoding="utf-8")
            assert "[[people/Alice.md]]" in body
            assert "[[topics/Alice.md]]" not in body
            assert "[[topics/Bob.md]]" in body
            await store.close()
        print("✓ test_retarget_exact_full_path_match passed")

    asyncio.run(run())


def test_retarget_short_and_no_ext_forms_ignored():
    """Short-form [[Alice]] and no-ext [[topics/Alice]] are NOT matched."""

    async def run():
        with tempfile.TemporaryDirectory() as tmp, temp_chdir(tmp):
            root = Path(tmp)
            body_in = "See [[Alice]] and [[topics/Alice]] but also [[topics/Alice.md]]."
            store = await _store_with(
                {
                    "note.md": body_in,
                    "people/Alice.md": "# Alice",
                },
            )
            payload = await WikilinkHandler.retarget_links(store, src="topics/Alice.md", dst="people/Alice.md")
            assert payload["links_changed"] == 1  # only the full-path form
            body = (root / "note.md").read_text(encoding="utf-8")
            assert "[[people/Alice.md]]" in body
            assert "[[Alice]]" in body  # short form untouched
            assert "[[topics/Alice]]" in body  # no-ext form untouched
            await store.close()
        print("✓ test_retarget_short_and_no_ext_forms_ignored passed")

    asyncio.run(run())


def test_retarget_anchor_preserved():
    """`[[topics/Alice.md#intro]]` → `[[people/Alice.md#intro]]` (anchor kept verbatim)."""

    async def run():
        with tempfile.TemporaryDirectory() as tmp, temp_chdir(tmp):
            root = Path(tmp)
            store = await _store_with({"note.md": "Jump to [[topics/Alice.md#intro]] please."})
            payload = await WikilinkHandler.retarget_links(store, src="topics/Alice.md", dst="people/Alice.md")
            assert payload["links_changed"] == 1
            body = (root / "note.md").read_text(encoding="utf-8")
            assert "[[people/Alice.md#intro]]" in body
            await store.close()
        print("✓ test_retarget_anchor_preserved passed")

    asyncio.run(run())


def test_retarget_alias_preserved():
    """`[[topics/Alice.md|Display Name]]` → `[[people/Alice.md|Display Name]]`."""

    async def run():
        with tempfile.TemporaryDirectory() as tmp, temp_chdir(tmp):
            root = Path(tmp)
            store = await _store_with({"note.md": "Meet [[topics/Alice.md|Alice the Architect]]."})
            payload = await WikilinkHandler.retarget_links(store, src="topics/Alice.md", dst="people/Alice.md")
            assert payload["links_changed"] == 1
            body = (root / "note.md").read_text(encoding="utf-8")
            assert "[[people/Alice.md|Alice the Architect]]" in body
            await store.close()
        print("✓ test_retarget_alias_preserved passed")

    asyncio.run(run())


def test_retarget_anchor_and_alias_together():
    """`[[A.md#h|disp]]` → `[[B.md#h|disp]]` keeps both suffixes."""

    async def run():
        with tempfile.TemporaryDirectory() as tmp, temp_chdir(tmp):
            root = Path(tmp)
            store = await _store_with({"note.md": "See [[topics/Alice.md#bio|her bio]] now."})
            payload = await WikilinkHandler.retarget_links(store, src="topics/Alice.md", dst="people/Alice.md")
            assert payload["links_changed"] == 1
            body = (root / "note.md").read_text(encoding="utf-8")
            assert "[[people/Alice.md#bio|her bio]]" in body
            await store.close()
        print("✓ test_retarget_anchor_and_alias_together passed")

    asyncio.run(run())


def test_retarget_image_marker_preserved():
    """`![[topics/diagram.md]]` (embed) keeps its `!` prefix on rewrite."""

    async def run():
        with tempfile.TemporaryDirectory() as tmp, temp_chdir(tmp):
            root = Path(tmp)
            store = await _store_with({"note.md": "Inline embed: ![[topics/diagram.md]] here."})
            payload = await WikilinkHandler.retarget_links(store, src="topics/diagram.md", dst="diagrams/diagram.md")
            assert payload["links_changed"] == 1
            body = (root / "note.md").read_text(encoding="utf-8")
            assert "![[diagrams/diagram.md]]" in body
            await store.close()
        print("✓ test_retarget_image_marker_preserved passed")

    asyncio.run(run())


def test_retarget_dataview_predicate_preserved():
    """Line-level + inline-bracketed Dataview predicates pass through outside ``[[..]]``."""

    async def run():
        with tempfile.TemporaryDirectory() as tmp, temp_chdir(tmp):
            root = Path(tmp)
            body_in = (
                "colleague:: [[topics/Alice.md]]\n" + "She is the [负责:: [[topics/Alice.md]]] for the migration.\n"
            )
            store = await _store_with({"note.md": body_in})
            payload = await WikilinkHandler.retarget_links(store, src="topics/Alice.md", dst="people/Alice.md")
            assert payload["links_changed"] == 2
            body = (root / "note.md").read_text(encoding="utf-8")
            assert "colleague:: [[people/Alice.md]]" in body
            assert "[负责:: [[people/Alice.md]]]" in body
            await store.close()
        print("✓ test_retarget_dataview_predicate_preserved passed")

    asyncio.run(run())


def test_retarget_multiple_files_aggregate_counts():
    """links_changed sums across files; by_file lists per-file counts."""

    async def run():
        with tempfile.TemporaryDirectory() as tmp, temp_chdir(tmp):
            store = await _store_with(
                {
                    "a.md": "[[topics/Alice.md]] then [[topics/Alice.md]]",
                    "sub/b.md": "[[topics/Alice.md]] and [[topics/Bob.md]]",
                },
            )
            payload = await WikilinkHandler.retarget_links(store, src="topics/Alice.md", dst="people/Alice.md")
            assert payload["links_changed"] == 3
            assert payload["files_touched"] == 2
            by_file = {row["path"]: row["count"] for row in payload["by_file"]}
            assert by_file == {"a.md": 2, "sub/b.md": 1}
            await store.close()
        print("✓ test_retarget_multiple_files_aggregate_counts passed")

    asyncio.run(run())


def test_retarget_dry_run_does_not_write():
    """dry_run=True reports counts but leaves files on disk unchanged."""

    async def run():
        with tempfile.TemporaryDirectory() as tmp, temp_chdir(tmp):
            root = Path(tmp)
            original = "See [[topics/Alice.md]]."
            store = await _store_with({"note.md": original})
            payload = await WikilinkHandler.retarget_links(
                store,
                src="topics/Alice.md",
                dst="people/Alice.md",
                dry_run=True,
            )
            assert payload["dry_run"] is True
            assert payload["links_changed"] == 1
            assert payload["files_touched"] == 1
            assert (root / "note.md").read_text(encoding="utf-8") == original
            await store.close()
        print("✓ test_retarget_dry_run_does_not_write passed")

    asyncio.run(run())


def test_retarget_scope_limits_sweep():
    """Files outside `scope` are not visited even if they contain matches."""

    async def run():
        with tempfile.TemporaryDirectory() as tmp, temp_chdir(tmp):
            root = Path(tmp)
            store = await _store_with(
                {
                    "in_scope/note.md": "See [[topics/Alice.md]].",
                    "outside/note.md": "Also [[topics/Alice.md]].",
                },
            )
            payload = await WikilinkHandler.retarget_links(
                store,
                src="topics/Alice.md",
                dst="people/Alice.md",
                scope="in_scope",
            )
            assert payload["links_changed"] == 1
            assert payload["files_touched"] == 1
            in_body = (root / "in_scope/note.md").read_text(encoding="utf-8")
            out_body = (root / "outside/note.md").read_text(encoding="utf-8")
            assert "[[people/Alice.md]]" in in_body
            assert "[[topics/Alice.md]]" in out_body  # untouched
            await store.close()
        print("✓ test_retarget_scope_limits_sweep passed")

    asyncio.run(run())


def test_retarget_src_eq_dst_is_noop():
    """src == dst: no scan needed, returns zero counts."""

    async def run():
        with tempfile.TemporaryDirectory() as tmp, temp_chdir(tmp):
            store = await _store_with({"note.md": "See [[topics/Alice.md]]."})
            payload = await WikilinkHandler.retarget_links(store, src="topics/Alice.md", dst="topics/Alice.md")
            assert payload["links_changed"] == 0
            assert payload["files_touched"] == 0
            await store.close()
        print("✓ test_retarget_src_eq_dst_is_noop passed")

    asyncio.run(run())


def test_retarget_empty_src_or_dst_errors():
    """Empty src or dst produces an error payload."""

    async def run():
        with tempfile.TemporaryDirectory() as tmp, temp_chdir(tmp):
            store = await _empty_store()
            for kwargs in ({"src": "", "dst": "people/Alice.md"}, {"src": "topics/Alice.md", "dst": ""}):
                payload = await WikilinkHandler.retarget_links(store, **kwargs)
                assert "error" in payload, f"expected error for {kwargs}"
            await store.close()
        print("✓ test_retarget_empty_src_or_dst_errors passed")

    asyncio.run(run())


def test_retarget_dst_with_forbidden_chars_errors():
    """``dst`` containing ``[ ] # |`` or newline lands as an error payload."""

    async def run():
        with tempfile.TemporaryDirectory() as tmp, temp_chdir(tmp):
            store = await _empty_store()
            for bad in ["people/Alice#anchor", "people/Alice|alias", "p/[A].md", "p/A]]"]:
                payload = await WikilinkHandler.retarget_links(store, src="topics/Alice.md", dst=bad)
                assert "error" in payload, f"expected error for dst={bad!r}, got {payload}"
            await store.close()
        print("✓ test_retarget_dst_with_forbidden_chars_errors passed")

    asyncio.run(run())


def test_retarget_absolute_path_rejected():
    """Absolute paths in src or dst are rejected (must be relative to the vault)."""

    async def run():
        with tempfile.TemporaryDirectory() as tmp, temp_chdir(tmp):
            store = await _empty_store()
            for kwargs in (
                {"src": "/abs/old.md", "dst": "people/Alice.md"},
                {"src": "topics/Alice.md", "dst": "/abs/new.md"},
            ):
                payload = await WikilinkHandler.retarget_links(store, **kwargs)
                assert "error" in payload, f"expected error for {kwargs}"
            await store.close()
        print("✓ test_retarget_absolute_path_rejected passed")

    asyncio.run(run())


def test_find_inbound_counts_references():
    """find_inbound reports per-file counts and excludes self-references."""

    async def run():
        with tempfile.TemporaryDirectory() as tmp, temp_chdir(tmp):
            store = await _store_with(
                {
                    "a.md": "[[topics/Alice.md]] and [[topics/Alice.md]] again",
                    "b.md": "see [[topics/Alice.md]]",
                    "topics/Alice.md": "self-ref [[topics/Alice.md]] should not count",
                },
            )
            payload = await WikilinkHandler.find_inbound(store, target="topics/Alice.md")
            assert payload["files_touched"] == 2
            assert payload["links_total"] == 3
            by_file = {row["path"]: row["count"] for row in payload["by_file"]}
            assert by_file == {"a.md": 2, "b.md": 1}
            await store.close()
        print("✓ test_find_inbound_counts_references passed")

    asyncio.run(run())


if __name__ == "__main__":
    print("\n=== wikilink helper tests ===")
    test_retarget_exact_full_path_match()
    test_retarget_short_and_no_ext_forms_ignored()
    test_retarget_anchor_preserved()
    test_retarget_alias_preserved()
    test_retarget_anchor_and_alias_together()
    test_retarget_image_marker_preserved()
    test_retarget_dataview_predicate_preserved()
    test_retarget_multiple_files_aggregate_counts()
    test_retarget_dry_run_does_not_write()
    test_retarget_scope_limits_sweep()
    test_retarget_src_eq_dst_is_noop()
    test_retarget_empty_src_or_dst_errors()
    test_retarget_dst_with_forbidden_chars_errors()
    test_retarget_absolute_path_rejected()
    test_find_inbound_counts_references()
    print("\n所有测试通过!")
