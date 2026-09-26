"""Regression tests for chunker-sensitive file-store persistence."""

import asyncio

from reme.components.application_context import ApplicationContext
from reme.components.file_chunker import MarkdownFileChunker
from reme.components.file_store import LocalFileStore
from reme.enumeration import ComponentEnum
from reme.schema import FileChunk, FileNode
from reme.utils.jsonl_zst import write_jsonl_zst


def _store_for_chunker(workspace, *, chunk_byte_size: int) -> LocalFileStore:
    context = ApplicationContext(workspace_dir=str(workspace))
    chunker = MarkdownFileChunker(chunk_byte_size=chunk_byte_size, app_context=context)
    context.components[ComponentEnum.FILE_CHUNKER] = {"default": chunker}
    return LocalFileStore(
        name="default",
        embedding_store="",
        app_context=context,
    )


def test_chunk_store_path_changes_when_chunker_configuration_changes(tmp_path):
    first = _store_for_chunker(tmp_path, chunk_byte_size=1000)
    second = _store_for_chunker(tmp_path, chunk_byte_size=2000)

    assert first.chunks_path != second.chunks_path
    assert first.chunks_path.name.startswith("file_chunks_default_v1_")
    assert second.chunks_path.name.startswith("file_chunks_default_v1_")


def test_chunk_store_path_is_stable_for_equivalent_chunker_configuration(tmp_path):
    first = _store_for_chunker(tmp_path, chunk_byte_size=1000)
    second = _store_for_chunker(tmp_path, chunk_byte_size=1000)

    assert first.chunks_path == second.chunks_path


def test_changed_chunker_does_not_load_legacy_chunks_and_clears_stale_graph(tmp_path):
    class Graph:
        def __init__(self):
            self.nodes = [FileNode(path="note.md", st_mtime=1, chunk_ids=["legacy"])]
            self.cleared = False

        async def get_nodes(self):
            return self.nodes

        async def clear(self):
            self.cleared = True
            self.nodes = []

    async def run():
        store = _store_for_chunker(tmp_path, chunk_byte_size=2000)
        legacy_path = store.component_metadata_path / "file_chunks_default_v1.jsonl.zst"
        legacy_path.parent.mkdir(parents=True, exist_ok=True)
        write_jsonl_zst(legacy_path, [FileChunk(id="legacy", path="note.md", text="old chunk").model_dump_json()])

        graph = Graph()
        store.file_graph = graph
        store.keyword_index = None
        store.tag_index = None

        assert legacy_path != store.chunks_path
        await store.load()

        assert store.file_chunks == {}
        assert graph.cleared is True
        assert legacy_path.exists()

    asyncio.run(run())
