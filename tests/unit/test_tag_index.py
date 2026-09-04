"""Focused contracts for the FileNode-derived tag index."""

import asyncio
from pathlib import Path

import pytest

from reme.components.file_chunker import MarkdownFileChunker
from reme.components.file_store import LocalFileStore
from reme.components.tag_index import LocalTagIndex
from reme.config import resolve_app_config
from reme.schema import FileFrontMatter, FileNode


def _node(path: str, tags: object = None) -> FileNode:
    metadata = {} if tags is None else {"tags": tags}
    return FileNode(path=path, st_mtime=1.0, front_matter=FileFrontMatter(**metadata))


def test_tag_normalization_and_bidirectional_mutations() -> None:
    """Normalize FileNode tags and keep both lookup directions consistent."""

    async def run() -> None:
        index = LocalTagIndex(max_tags_per_file=3)
        await index.start()
        await index.upsert_nodes([_node("daily/a.md", ["Python", "PYTHON", "C++", ".NET", "ignored"])])

        assert await index.tags_for_path("daily/a.md") == ["python", "c++", ".net"]
        assert await index.paths_for_tags(["PYTHON"]) == ["daily/a.md"]
        assert index.tag_to_paths == {
            "python": {"daily/a.md"},
            "c++": {"daily/a.md"},
            ".net": {"daily/a.md"},
        }

        await index.upsert_nodes([_node("daily/a.md", ["ReMe"])])
        assert await index.tags_for_path("daily/a.md") == ["reme"]
        assert set(index.tag_to_paths) == {"reme"}

        await index.delete(["daily/a.md", "daily/missing.md"])
        assert await index.tags_for_path("daily/a.md") == []
        assert not index.tag_to_paths
        await index.close()

    asyncio.run(run())


def test_rebuild_is_atomic_and_supports_all_or_any_queries() -> None:
    """Publish complete rebuilds atomically and support intersection and union lookup."""

    async def run() -> None:
        index = LocalTagIndex()
        await index.rebuild(
            [
                _node("daily/a.md", ["python", "reme"]),
                _node("digest/b.md", ["python"]),
                _node("digest/untagged.md"),
            ],
        )

        assert await index.paths_for_tags(["python", "reme"]) == ["daily/a.md"]
        assert await index.paths_for_tags(["python", "reme"], match_all=False) == [
            "daily/a.md",
            "digest/b.md",
        ]
        assert "digest/untagged.md" not in index.path_to_tags

        before_paths = dict(index.path_to_tags)
        before_tags = {tag: set(paths) for tag, paths in index.tag_to_paths.items()}
        with pytest.raises(ValueError, match="Invalid workspace-relative"):
            await index.rebuild([_node("daily/new.md", ["new"]), _node("../escape.md", ["invalid"])])
        assert index.path_to_tags == before_paths
        assert index.tag_to_paths == before_tags

    asyncio.run(run())


def test_file_store_updates_tag_index_from_file_nodes(monkeypatch, tmp_path: Path) -> None:
    """Keep daily and digest tags aligned through file-store mutations."""

    async def run() -> None:
        monkeypatch.chdir(tmp_path)
        store = LocalFileStore(name="test", embedding_store="", tag_index="default")
        await store.start()
        assert isinstance(store.tag_index, LocalTagIndex)

        await store.upsert(
            [
                (_node("daily/a.md", ["Python"]), []),
                (_node("digest/b.md", ["Digest"]), []),
            ],
        )
        assert await store.tag_index.paths_for_tags(["python"]) == ["daily/a.md"]
        assert await store.tag_index.paths_for_tags(["digest"]) == ["digest/b.md"]

        await store.upsert([(_node("daily/a.md", ["ReMe"]), [])])
        assert await store.tag_index.paths_for_tags(["python"]) == []
        assert await store.tag_index.paths_for_tags(["reme"]) == ["daily/a.md"]

        await store.delete("daily/a.md")
        assert await store.tag_index.paths_for_tags(["reme"]) == []

        await store.clear()
        assert store.tag_index.path_to_tags == {}
        assert store.tag_index.tag_to_paths == {}
        await store.close()

    asyncio.run(run())


def test_existing_markdown_chunker_supplies_frontmatter_tags(monkeypatch, tmp_path: Path) -> None:
    """Use the FileNode produced by the existing chunker without reading frontmatter again."""

    async def run() -> None:
        monkeypatch.chdir(tmp_path)
        note = tmp_path / "daily" / "a.md"
        note.parent.mkdir()
        note.write_text("---\ntags: [Python, ReMe]\n---\nbody\n", encoding="utf-8")
        node, chunks = await MarkdownFileChunker().chunk(note)

        store = LocalFileStore(name="test", embedding_store="", tag_index="default")
        await store.start()
        await store.upsert([(node, chunks)])

        assert await store.tag_index.tags_for_path("daily/a.md") == ["python", "reme"]
        await store.close()

    asyncio.run(run())


def test_file_store_rebuilds_non_persistent_tag_index_from_graph(monkeypatch, tmp_path: Path) -> None:
    """Restore tag relationships from the persisted file graph on startup."""

    async def run() -> None:
        monkeypatch.chdir(tmp_path)
        first = LocalFileStore(name="test", embedding_store="", tag_index="default")
        await first.start()
        await first.upsert([(_node("daily/a.md", ["ReMe"]), [])])
        await first.close()

        assert not list((tmp_path / "metadata").glob("tag_index/**/*"))

        restored = LocalFileStore(name="test", embedding_store="", tag_index="default")
        await restored.start()
        assert await restored.tag_index.paths_for_tags(["reme"]) == ["daily/a.md"]
        await restored.close()

    asyncio.run(run())


def test_default_config_documents_optional_tag_index_without_enabling_it() -> None:
    """Document tag indexing in the default config without enabling another index or watcher."""

    config = resolve_app_config(config="default", log_config=False)

    assert config["jobs"]["index_update_loop"]["watch_dirs"] == ["daily_dir", "digest_dir"]
    assert "tag_index_loop" not in config["jobs"]
    assert "tag_index" not in config["components"]
    assert "tag_index" not in config["components"]["file_store"]["default"]

    default_yaml = Path("reme/config/default.yaml").read_text(encoding="utf-8")
    assert "#  tag_index:" in default_yaml
    assert "#      tag_index: default" in default_yaml
