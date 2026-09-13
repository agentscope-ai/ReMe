"""Focused contracts for the FileNode-derived tag index."""

# pylint: disable=protected-access

import asyncio
from pathlib import Path

import pytest

from reme.components.application_context import ApplicationContext
from reme.components.file_chunker import MarkdownFileChunker
from reme.components.file_store import LocalFileStore
from reme.components.tag_index import LocalTagIndex
from reme.config import resolve_app_config
from reme.enumeration import ComponentEnum
from reme.schema import FileChunk, FileFrontMatter, FileNode
from reme.steps.index.list_tags import ListTagsStep


def _node(path: str, tags: object = None, *, key: str = "memory_tags") -> FileNode:
    metadata = {} if tags is None else {key: tags}
    return FileNode(path=path, st_mtime=1.0, front_matter=FileFrontMatter(**metadata))


def _chunk(chunk_id: str, path: str, text: str) -> FileChunk:
    return FileChunk(id=chunk_id, path=path, text=text, start_line=1, end_line=1)


def _standalone_file_store_with_tag_index(**kwargs) -> LocalFileStore:
    return LocalFileStore(tag_index="default", **kwargs)


def test_tag_normalization_and_bidirectional_mutations() -> None:
    """Normalize FileNode tags and keep both lookup directions consistent."""

    async def run() -> None:
        index = LocalTagIndex(max_tags_per_file=3)
        await index.start()
        await index.upsert_nodes([_node("daily/a.md", ["Python Tag", "PYTHON TAG", "C++", ".NET", "ignored"])])

        assert await index.tags_for_path("daily/a.md") == ["python_tag", "c++", ".net"]
        assert await index.paths_for_tags(["PYTHON TAG"]) == ["daily/a.md"]
        assert index.tag_to_paths == {
            "python_tag": {"daily/a.md"},
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


def test_queries_are_not_truncated_by_per_file_tag_limit() -> None:
    """Apply the count limit to indexed files without dropping lookup conditions."""

    async def run() -> None:
        index = LocalTagIndex(max_tags_per_file=2, max_tag_length=8)
        await index.rebuild(
            [
                _node("daily/a.md", ["a", "b"]),
                _node("daily/c.md", ["c"]),
            ],
        )

        assert await index.paths_for_tags(["a", "b", "c"]) == []
        assert await index.paths_for_tags(["a", "b", "c"], match_all=False) == [
            "daily/a.md",
            "daily/c.md",
        ]
        assert index.normalize_query_tags([" A ", "B", "c c", "a", "too-long-tag", " "]) == ["a", "b", "c_c"]

    asyncio.run(run())


def test_list_tags_paginates_and_applies_default_sort_orders() -> None:
    """List only active tags with compact counts and deterministic pagination."""

    async def run() -> None:
        index = LocalTagIndex()
        await index.rebuild(
            [
                _node("daily/a.md", ["beta", "alpha"]),
                _node("daily/b.md", ["gamma", "beta"]),
                _node("daily/c.md", ["delta"]),
            ],
        )

        assert await index.list_tags(page_size=2) == {
            "total_tags": 4,
            "total_pages": 2,
            "page": 1,
            "range": (1, 2),
            "items": [("alpha", 1), ("beta", 2)],
        }
        assert await index.list_tags(page=2, page_size=2) == {
            "total_tags": 4,
            "total_pages": 2,
            "page": 2,
            "range": (3, 4),
            "items": [("delta", 1), ("gamma", 1)],
        }
        assert (await index.list_tags(order_by="file_count"))["items"] == [
            ("beta", 2),
            ("alpha", 1),
            ("delta", 1),
            ("gamma", 1),
        ]
        assert (await index.list_tags(order_by="file_count", order="asc"))["items"] == [
            ("alpha", 1),
            ("delta", 1),
            ("gamma", 1),
            ("beta", 2),
        ]
        assert (await index.list_tags(order="desc"))["items"] == [
            ("gamma", 1),
            ("delta", 1),
            ("beta", 2),
            ("alpha", 1),
        ]

        await index.delete(["daily/c.md"])
        result = await index.list_tags(page=3, page_size=2)
        assert result == {
            "total_tags": 3,
            "total_pages": 2,
            "page": 3,
            "range": (0, 0),
            "items": [],
        }

        await index.clear()
        assert await index.list_tags(page=9) == {
            "total_tags": 0,
            "total_pages": 0,
            "page": 9,
            "range": (0, 0),
            "items": [],
        }

        invalid_cases = [
            ({"page": 0}, "page must be a positive integer"),
            ({"page_size": True}, "page_size must be a positive integer"),
            ({"page_size": 1001}, "page_size must be less than or equal to 1000"),
            ({"order_by": "unknown"}, "order_by must be one of"),
            ({"order": "sideways"}, "order must be one of"),
        ]
        for kwargs, message in invalid_cases:
            with pytest.raises(ValueError, match=message):
                await index.list_tags(**kwargs)

        index.set_healthy(False)
        with pytest.raises(RuntimeError, match="tag index is unavailable"):
            await index.list_tags()

        store = LocalFileStore(name="test", embedding_store="", tag_index="")
        store.tag_index = LocalTagIndex()
        await store.tag_index.rebuild([_node("daily/a.md", ["ReMe"])])

        response = await ListTagsStep(file_store=store)(order_by="file_count")

        assert response.answer == {
            "total_tags": 1,
            "total_pages": 1,
            "page": 1,
            "range": (1, 1),
            "items": [("reme", 1)],
        }

    asyncio.run(run())

    config = resolve_app_config(config="default", log_config=False)
    job = config["jobs"]["list_tags"]
    assert job["steps"] == [{"backend": "list_tags_step"}]
    assert job["parameters"]["properties"]["page_size"]["default"] == 100
    assert "[tag, file_count]" in job["description"]
    assert "range" in job["description"]
    assert "empty page" in job["parameters"]["properties"]["page"]["description"]


@pytest.mark.asyncio
async def test_list_tags_resolves_named_file_store_from_runtime_context(tmp_path) -> None:
    """Injected tool context can route list_tags to the AutoTag Step's file store."""
    context = ApplicationContext(workspace_dir=str(tmp_path))
    default_store = LocalFileStore(name="default", embedding_store="", tag_index="")
    archive_store = LocalFileStore(name="archive", embedding_store="", tag_index="")
    default_store.tag_index = LocalTagIndex()
    archive_store.tag_index = LocalTagIndex()
    await default_store.tag_index.rebuild([_node("daily/default.md", ["default-tag"])])
    await archive_store.tag_index.rebuild([_node("daily/archive.md", ["archive-tag"])])
    context.components = {
        ComponentEnum.FILE_STORE: {
            "default": default_store,
            "archive": archive_store,
        },
    }

    response = await ListTagsStep(app_context=context)(file_store="archive")

    assert response.answer["items"] == [("archive-tag", 1)]


def test_configured_frontmatter_key_contract() -> None:
    """Validate, apply, and invalidate changes to the configured source key."""

    async def run() -> None:
        index = LocalTagIndex(tag_key="keywords")
        await index.rebuild(
            [
                _node("daily/a.md", ["ignored"]),
                _node("daily/b.md", ["Python"], key="keywords"),
            ],
        )

        assert index.tag_key == "keywords"
        assert await index.paths_for_tags(["python"]) == ["daily/b.md"]
        assert await index.paths_for_tags(["ignored"]) == []

    asyncio.run(run())
    invalid_cases = [
        ("", "tag_key must be a non-empty string", False),
        ("   ", "tag_key must be a non-empty string", False),
        (None, "tag_key must be a non-empty string", False),
        (123, "tag_key must be a non-empty string", False),
        ("name", "tag_key must not be a reserved frontmatter key", False),
        ("description", "tag_key must not be a reserved frontmatter key", False),
        ("kind", "tag_key must not be a reserved frontmatter key", False),
        ("session_id", "tag_key must not be a reserved frontmatter key", False),
        ("source_conversation", "tag_key must not be a reserved frontmatter key", False),
        ("source_resource", "tag_key must not be a reserved frontmatter key", False),
        ("status", "tag_key must not be a reserved frontmatter key", False),
        ("", "tag_key must be a non-empty string", True),
        ("name", "tag_key must not be a reserved frontmatter key", True),
        ("description", "tag_key must not be a reserved frontmatter key", True),
    ]
    for tag_key, message, at_runtime in invalid_cases:
        index = LocalTagIndex()
        with pytest.raises(ValueError, match=message):
            if at_runtime:
                index.tag_key = tag_key
            else:
                LocalTagIndex(tag_key=tag_key)
        if at_runtime:
            assert index.tag_key == "memory_tags"

    assert LocalTagIndex(tag_key="model_config").tag_key == "model_config"
    index = LocalTagIndex()
    index.tag_key = "keywords"
    assert index.tag_key == "keywords"
    assert not index.is_healthy


def test_file_store_tag_index_binding_only_defaults_for_default_name() -> None:
    """Support standalone defaults without silently substituting named dependencies."""
    default_store = LocalFileStore(name="default-store", embedding_store="", tag_index="default")
    custom_store = LocalFileStore(name="custom-store", embedding_store="", tag_index="custom")

    default_dependency = default_store.dependency_bindings["tag_index"]
    assert default_dependency.name == "default"
    assert default_dependency.default_factory is LocalTagIndex
    assert default_dependency.optional is False

    custom_dependency = custom_store.dependency_bindings["tag_index"]
    assert custom_dependency.name == "custom"
    assert custom_dependency.default_factory is None
    assert custom_dependency.optional is False


def test_file_store_without_tag_index_name_disables_tag_index() -> None:
    """Treat both an omitted tag-index setting and an explicit empty name as disabled."""
    omitted = LocalFileStore(name="omitted", embedding_store="")
    explicit = LocalFileStore(name="explicit", embedding_store="", tag_index="")

    assert omitted.tag_index is None
    assert omitted.tag_index_enabled is False
    assert explicit.tag_index is None
    assert explicit.tag_index_enabled is False


@pytest.mark.parametrize("tag_index_name", ["", "custom", "missing"])
def test_file_store_resolves_configured_tag_index(tag_index_name: str, tmp_path: Path) -> None:
    """Resolve tag indexes by configured component name and fail on missing names."""

    async def run() -> None:
        context = ApplicationContext(workspace_dir=str(tmp_path))
        index = LocalTagIndex(name="custom", tag_key="keywords")
        context.components = {ComponentEnum.TAG_INDEX: {"custom": index}}
        store = LocalFileStore(app_context=context, embedding_store="", tag_index=tag_index_name)

        if tag_index_name == "missing":
            with pytest.raises(ValueError, match="tag_index 'missing' not found"):
                await store.start()
            assert not store.is_started
        else:
            await store._resolve_bindings()
            assert store.tag_index is (index if tag_index_name else None)
            assert store.tag_index_enabled is bool(tag_index_name)

    asyncio.run(run())


def test_file_store_updates_tag_index_from_file_nodes(monkeypatch, tmp_path: Path) -> None:
    """Keep daily and digest tags aligned through file-store mutations."""

    disabled_store = LocalFileStore(name="test", embedding_store="", tag_index="")
    assert disabled_store.tag_index_enabled is False
    with pytest.raises(RuntimeError, match="tag index is not configured"):
        disabled_store.require_tag_index()

    async def run() -> None:
        monkeypatch.chdir(tmp_path)
        store = _standalone_file_store_with_tag_index(name="test", embedding_store="")
        await store.start()
        assert isinstance(store.tag_index, LocalTagIndex)
        assert store.tag_index_enabled is True
        assert store.require_tag_index() is store.tag_index

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


def test_tag_failures_do_not_block_other_indexes_and_retry_rebuild(monkeypatch, tmp_path: Path) -> None:
    """Keep core indexes writable and rebuild tags on the next mutation after a failed recovery."""

    async def run() -> None:
        monkeypatch.chdir(tmp_path)
        store = _standalone_file_store_with_tag_index(name="test", embedding_store="")
        await store.start()
        assert store.tag_index_enabled
        original_rebuild = store.tag_index.rebuild

        async def fail_incremental(_nodes) -> None:
            raise RuntimeError("incremental tag failure")

        async def fail_rebuild(_nodes) -> None:
            raise RuntimeError("tag rebuild failure")

        monkeypatch.setattr(store.tag_index, "upsert_nodes", fail_incremental)
        monkeypatch.setattr(store.tag_index, "rebuild", fail_rebuild)

        first_chunk = _chunk("chunk-a", "daily/a.md", "alpha memory")
        await store.upsert([(_node("daily/a.md", ["alpha"]), [first_chunk])])

        assert [node.path for node in await store.get_nodes()] == ["daily/a.md"]
        assert "chunk-a" in store.file_chunks
        assert "chunk-a" in store.keyword_index.document_ids
        assert store._tag_index_rebuild_required is True

        monkeypatch.setattr(store.tag_index, "rebuild", original_rebuild)
        second_chunk = _chunk("chunk-b", "digest/b.md", "beta memory")
        await store.upsert([(_node("digest/b.md", ["beta"]), [second_chunk])])

        assert store._tag_index_rebuild_required is False
        assert await store.tag_index.paths_for_tags(["alpha"]) == ["daily/a.md"]
        assert await store.tag_index.paths_for_tags(["beta"]) == ["digest/b.md"]
        assert {"chunk-a", "chunk-b"}.issubset(store.keyword_index.document_ids)
        await store.close()

    asyncio.run(run())


def test_failed_tag_reconciliation_makes_queries_fail_closed(monkeypatch, tmp_path: Path) -> None:
    """Never expose stale tag matches while reconciliation is pending."""

    async def run() -> None:
        monkeypatch.chdir(tmp_path)
        store = _standalone_file_store_with_tag_index(name="test", embedding_store="")
        await store.start()
        assert store.tag_index_enabled
        await store.upsert([(_node("daily/a.md", ["old"]), [])])

        async def fail(_items) -> None:
            raise RuntimeError("tag failure")

        monkeypatch.setattr(store.tag_index, "upsert_nodes", fail)
        monkeypatch.setattr(store.tag_index, "rebuild", fail)
        await store.upsert([(_node("daily/a.md", ["new"]), [])])

        assert store._tag_index_rebuild_required is True
        assert store.tag_index.is_healthy is False
        assert await store.tag_index.paths_for_tags(["old"]) == []
        assert await store.tag_index.paths_for_tags(["new"]) == []
        assert await store.tag_index.tags_for_path("daily/a.md") == []
        await store.close()

    asyncio.run(run())


def test_tag_rebuild_graph_read_failure_does_not_block_upsert(monkeypatch, tmp_path: Path) -> None:
    """Keep core indexes consistent if the optional tag repair cannot read the graph snapshot."""

    async def run() -> None:
        monkeypatch.chdir(tmp_path)
        store = _standalone_file_store_with_tag_index(name="test", embedding_store="")
        await store.start()
        assert store.tag_index_enabled
        assert store.file_graph is not None
        store._tag_index_rebuild_required = True
        original_get_nodes = store.file_graph.get_nodes

        async def fail_full_snapshot(paths=None):
            if paths is None:
                raise RuntimeError("graph snapshot failure")
            return await original_get_nodes(paths)

        monkeypatch.setattr(store.file_graph, "get_nodes", fail_full_snapshot)
        chunk = _chunk("chunk-a", "daily/a.md", "alpha memory")
        await store.upsert([(_node("daily/a.md", ["alpha"]), [chunk])])

        assert "chunk-a" in store.file_chunks
        assert "chunk-a" in store.keyword_index.document_ids
        assert store._tag_index_rebuild_required is True
        assert store.tag_index.is_healthy is False
        await store.close()

    asyncio.run(run())


def test_explicit_reindex_restores_tag_index(monkeypatch, tmp_path: Path) -> None:
    """The tag scope and all scope rebuild tags from the authoritative graph."""

    async def run() -> None:
        monkeypatch.chdir(tmp_path)
        store = _standalone_file_store_with_tag_index(name="test", embedding_store="")
        await store.start()
        assert store.tag_index_enabled
        await store.upsert(
            [
                (_node("daily/a.md", ["ReMe"]), []),
                (_node("daily/untagged.md"), []),
            ],
        )

        await store.tag_index.clear()
        assert await store.tag_index.paths_for_tags(["reme"]) == []
        assert await store.reindex("tag") == {"indexed": 1, "scope": "tag"}
        assert await store.tag_index.paths_for_tags(["reme"]) == ["daily/a.md"]

        await store.tag_index.clear()
        result = await store.reindex("all")
        assert result["tag"] == {"indexed": 1, "scope": "tag"}
        assert await store.tag_index.paths_for_tags(["reme"]) == ["daily/a.md"]
        await store.file_graph.upsert_nodes([_node("daily/a.md", ["new"], key="keywords")])

        store.tag_index.tag_key = "keywords"
        await store.reindex("tag")

        assert await store.tag_index.paths_for_tags(["old"]) == []
        assert await store.tag_index.paths_for_tags(["new"]) == ["daily/a.md"]
        await store.close()

    asyncio.run(run())


def test_tag_delete_failures_do_not_block_core_deletion(monkeypatch, tmp_path: Path) -> None:
    """Complete graph, chunk, and keyword deletion when tag deletion and recovery fail."""

    async def run() -> None:
        monkeypatch.chdir(tmp_path)
        store = _standalone_file_store_with_tag_index(name="test", embedding_store="")
        await store.start()
        assert store.tag_index_enabled
        chunk = _chunk("chunk-a", "daily/a.md", "alpha memory")
        await store.upsert([(_node("daily/a.md", ["alpha"]), [chunk])])

        async def fail_delete(_paths) -> None:
            raise RuntimeError("incremental tag delete failure")

        async def fail_rebuild(_nodes) -> None:
            raise RuntimeError("tag rebuild failure")

        monkeypatch.setattr(store.tag_index, "delete", fail_delete)
        monkeypatch.setattr(store.tag_index, "rebuild", fail_rebuild)
        await store.delete("daily/a.md")

        assert await store.get_nodes() == []
        assert "chunk-a" not in store.file_chunks
        assert "chunk-a" not in store.keyword_index.document_ids
        assert store._tag_index_rebuild_required is True
        await store.close()

    asyncio.run(run())


def test_existing_markdown_chunker_supplies_frontmatter_tags(monkeypatch, tmp_path: Path) -> None:
    """Use the FileNode produced by the existing chunker without reading frontmatter again."""

    async def run() -> None:
        monkeypatch.chdir(tmp_path)
        note = tmp_path / "daily" / "a.md"
        note.parent.mkdir()
        note.write_text("---\nmemory_tags: [Python, ReMe]\n---\nbody\n", encoding="utf-8")
        node, chunks = await MarkdownFileChunker().chunk(note)

        store = _standalone_file_store_with_tag_index(name="test", embedding_store="")
        await store.start()
        await store.upsert([(node, chunks)])

        assert await store.tag_index.tags_for_path("daily/a.md") == ["python", "reme"]
        await store.close()

    asyncio.run(run())


def test_file_store_rebuilds_non_persistent_tag_index_from_graph(monkeypatch, tmp_path: Path) -> None:
    """Restore tag relationships from the persisted file graph on startup."""

    async def run() -> None:
        monkeypatch.chdir(tmp_path)
        first = _standalone_file_store_with_tag_index(name="test", embedding_store="")
        await first.start()
        await first.upsert([(_node("daily/a.md", ["ReMe"]), [])])
        await first.close()

        assert not list((tmp_path / "metadata").glob("tag_index/**/*"))

        restored = _standalone_file_store_with_tag_index(name="test", embedding_store="")
        await restored.start()
        assert await restored.tag_index.paths_for_tags(["reme"]) == ["daily/a.md"]
        await restored.close()

    asyncio.run(run())


def test_default_config_enables_tag_index_with_explicit_key() -> None:
    """Keep auto-memory generation and file-store indexing on the same configured key."""

    config = resolve_app_config(config="default", log_config=False)

    assert config["jobs"]["index_update_loop"]["watch_dirs"] == ["daily_dir", "digest_dir"]
    assert "tag_index_loop" not in config["jobs"]
    assert config["components"]["tag_index"]["default"]["tag_key"] == "memory_tags"
    assert config["components"]["tag_index"]["default"]["max_tags_per_file"] == 3
    assert config["components"]["file_store"]["default"]["tag_index"] == "default"
    assert config["jobs"]["search"]["parameters"]["properties"]["tags"]["default"] == []
    assert config["jobs"]["auto_memory"]["steps"] == [
        {"backend": "auto_memory_step"},
        {"backend": "auto_tag_step"},
    ]
    assert config["jobs"]["auto_memory_cc"]["steps"] == [
        {"backend": "auto_memory_cc_step"},
        {"backend": "auto_tag_step"},
    ]
