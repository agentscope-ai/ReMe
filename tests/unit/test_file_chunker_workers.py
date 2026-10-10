"""Worker admission, output compatibility, and real indexing/search regressions."""

import asyncio
import contextvars
import json
import os
import threading
from pathlib import Path
from typing import Callable
from unittest.mock import patch

import pytest
from httpx import ASGITransport, AsyncClient

from reme.application import Application
from reme.components.application_context import ApplicationContext
from reme.components.file_chunker import DefaultFileChunker, JsonFileChunker, JsonlFileChunker, MarkdownFileChunker
from reme.components.file_store import LocalFileStore
from reme.components.job import BaseJob, StreamJob
from reme.components.runtime_context import RuntimeContext
from reme.components.service import HttpService
from reme.enumeration import ChunkEnum, ComponentEnum
from reme.schema import FileChunk, FileNode, Response, StreamChunk
from reme.steps.index import UpdateIndexStep

# Worker tests intentionally inspect admission and teardown at the internal boundary.
# pylint: disable=protected-access


class _StoreSearchJob(BaseJob):
    """Use the actual keyword store behind the HTTP route and MCP tool."""

    def __init__(self, store: LocalFileStore) -> None:
        super().__init__(name="search", parameters={"type": "object", "properties": {"query": {"type": "string"}}})
        self.store = store

    async def __call__(self, query: str, **kwargs) -> Response:
        del kwargs
        chunks = await self.store.keyword_search(query, limit=10, search_filter={})
        return Response(answer=[chunk.text for chunk in chunks])


class _StoreSearchStreamJob(StreamJob):
    """Stream real keyword matches through the existing SSE endpoint."""

    def __init__(self, store: LocalFileStore) -> None:
        super().__init__(name="search_stream")
        self.store = store

    async def __call__(self, stream_queue: asyncio.Queue[StreamChunk], query: str, **kwargs) -> None:
        del kwargs
        chunks = await self.store.keyword_search(query, limit=10, search_filter={})
        for chunk in chunks:
            await stream_queue.put(StreamChunk(chunk=chunk.text))
            await asyncio.sleep(0)
        await stream_queue.put(StreamChunk(chunk_type=ChunkEnum.DONE, done=True))


@pytest.mark.asyncio
async def test_outputs_equal_pre_worker_fixtures(tmp_path: Path) -> None:
    """Compare every serialized field, including hashes, links, metadata and ranges."""
    context = ApplicationContext(workspace_dir=str(tmp_path))
    cases = [
        (DefaultFileChunker(chunk_byte_size=100), "plain.txt", "first\r\n" + "abc " * 100),
        (
            MarkdownFileChunker(chunk_byte_size=100, include_frontmatter_in_metadata=True),
            "note.md",
            "---\ntitle: Example\ntags: [one, two]\n---\n# First\n\nbody [[other.md]]\n\n## Second\n\n" + "text " * 70,
        ),
        (MarkdownFileChunker(chunk_byte_size=100, max_ast_sections=0), "fallback.md", "# First\n" + "text " * 100),
        (
            JsonFileChunker(chunk_chars=256),
            "data.json",
            json.dumps({"items": [{"name": "abc" * 35, "n": i} for i in range(6)]}, indent=2),
        ),
        (JsonFileChunker(chunk_chars=256), "bad.json", "{invalid\ntext\n"),
        (JsonlFileChunker(max_chars=64, max_overlap_chars=30), "events.jsonl", '{"value": "alpha"}\n' * 12),
    ]
    expected = json.loads((Path(__file__).parent / "fixtures" / "file_chunker_outputs.json").read_text())
    for chunker, name, text in cases:
        chunker.app_context = context
        path = tmp_path / name
        path.write_bytes(text.encode())
        node, chunks = await chunker.chunk(path)
        output = {"node": node.model_dump(mode="json"), "chunks": [c.model_dump(mode="json") for c in chunks]}
        assert output["node"].pop("st_mtime") == path.stat().st_mtime
        assert output == expected[name]


async def _wait_started(event: threading.Event) -> None:
    for _ in range(500):
        if event.is_set():
            return
        await asyncio.sleep(0.01)
    raise AssertionError("worker did not start")


@pytest.mark.asyncio
@pytest.mark.parametrize("chunker_class", [DefaultFileChunker, MarkdownFileChunker, JsonFileChunker, JsonlFileChunker])
async def test_parsing_leaves_event_loop_responsive(
    tmp_path: Path,
    chunker_class: type[DefaultFileChunker | MarkdownFileChunker | JsonFileChunker | JsonlFileChunker],
) -> None:
    """A parser can block while unrelated async requests keep progressing."""
    chunker = chunker_class()
    path = tmp_path / "source"
    path.write_text("{}")
    entered, release = threading.Event(), threading.Event()
    main_thread = threading.get_ident()
    original = chunker._chunk_sync

    def blocking_parse(file_path: Path, rel_path: str) -> tuple[FileNode, list[FileChunk]]:
        assert threading.get_ident() != main_thread
        entered.set()
        assert release.wait(5)
        return original(file_path, rel_path)

    with patch.object(chunker, "_chunk_sync", side_effect=blocking_parse):
        task = asyncio.create_task(chunker.chunk(path))
        try:
            await _wait_started(entered)
            for _ in range(10):
                await asyncio.sleep(0)
                assert not task.done()
        finally:
            release.set()
            await task
    assert chunker._worker is None


@pytest.mark.asyncio
async def test_cancellation_drains_worker_and_preserves_admission(tmp_path: Path) -> None:
    """Repeated cancellation cannot free admission while parsing is still running."""
    path = tmp_path / "source.txt"
    path.write_text("source")
    chunker = DefaultFileChunker()
    entered, release = threading.Event(), threading.Event()
    calls = []
    original = chunker._chunk_sync

    def blocking_parse(file_path: Path, rel_path: str) -> tuple[FileNode, list[FileChunk]]:
        calls.append(rel_path)
        entered.set()
        assert release.wait(5)
        return original(file_path, rel_path)

    async with chunker:
        with patch.object(chunker, "_chunk_sync", side_effect=blocking_parse):
            first = asyncio.create_task(chunker.chunk(path))
            await _wait_started(entered)
            first.cancel()
            second = asyncio.create_task(chunker.chunk(path))
            await asyncio.sleep(0.01)
            first.cancel()
            await asyncio.sleep(0.01)
            try:
                assert not first.done()
                assert len(calls) == 1
            finally:
                release.set()
                with pytest.raises(asyncio.CancelledError):
                    await first
                await second
    assert chunker._worker is None
    assert len(calls) == 2


@pytest.mark.asyncio
async def test_shutdown_rejects_queued_work_and_drains_despite_cancellation(tmp_path: Path) -> None:
    """Closing waits for running work, rejects queued parsing, and permits restart."""
    path = tmp_path / "source.txt"
    path.write_text("source")
    chunker = DefaultFileChunker()
    entered, release = threading.Event(), threading.Event()
    original = chunker._chunk_sync

    def blocking_parse(file_path: Path, rel_path: str) -> tuple[FileNode, list[FileChunk]]:
        entered.set()
        assert release.wait(5)
        return original(file_path, rel_path)

    await chunker.start()
    with patch.object(chunker, "_chunk_sync", side_effect=blocking_parse):
        running = asyncio.create_task(chunker.chunk(path))
        await _wait_started(entered)
        queued = asyncio.create_task(chunker.chunk(path))
        await asyncio.sleep(0)
        closing = asyncio.create_task(chunker.close())
        await asyncio.sleep(0.01)
        closing.cancel()
        try:
            assert not closing.done()
        finally:
            release.set()
            with pytest.raises(RuntimeError, match="closed"):
                await running
            with pytest.raises(RuntimeError, match="closed"):
                await queued
            with pytest.raises(asyncio.CancelledError):
                await closing
    assert chunker._worker is None
    assert not chunker.is_started
    async with chunker:
        assert (await chunker.chunk(path))[1][0].text == "source"


@pytest.mark.asyncio
@pytest.mark.parametrize("change", ["modify", "replace", "delete"])
async def test_source_changes_do_not_return_stale_chunks(tmp_path: Path, change: str) -> None:
    """Changes after reading cannot be mistaken for successfully parsed current content."""
    path = tmp_path / "source.txt"
    path.write_text("old source")
    chunker = DefaultFileChunker()
    original = chunker._chunk_sync
    entered, release = threading.Event(), threading.Event()

    def blocking_parse(file_path: Path, rel_path: str) -> tuple[FileNode, list[FileChunk]]:
        result = original(file_path, rel_path)
        entered.set()
        assert release.wait(5)
        return result

    with patch.object(chunker, "_chunk_sync", side_effect=blocking_parse):
        task = asyncio.create_task(chunker.chunk(path))
        await _wait_started(entered)
        try:
            if change == "modify":
                path.write_text("new source with different size")
            elif change == "replace":
                replacement = tmp_path / "replacement"
                replacement.write_text("replacement source")
                replacement.replace(path)
            else:
                path.unlink()
        finally:
            release.set()
        with pytest.raises((RuntimeError, FileNotFoundError)):
            await task


@pytest.mark.asyncio
async def test_change_before_event_loop_receives_result_is_rejected(tmp_path: Path) -> None:
    """Validate the worker snapshot again when its result reaches the calling loop."""
    path = tmp_path / "source.txt"
    path.write_text("old source")
    chunker = DefaultFileChunker()
    original = chunker._chunk_unchanged_file
    loop = asyncio.get_running_loop()
    changed = threading.Event()

    def change_on_loop() -> None:
        path.write_text("changed after the worker snapshot")
        changed.set()

    def change_after_parse(
        file_path: Path,
        rel_path: str,
        chunk: Callable[[Path, str], tuple[FileNode, list[FileChunk]]],
    ) -> tuple[tuple[FileNode, list[FileChunk]], os.stat_result]:
        result = original(file_path, rel_path, chunk)
        loop.call_soon_threadsafe(change_on_loop)
        assert changed.wait(5)
        return result

    with patch.object(chunker, "_chunk_unchanged_file", side_effect=change_after_parse):
        with pytest.raises(RuntimeError, match="File changed"):
            await chunker.chunk(path)


@pytest.mark.asyncio
async def test_queued_cancellation_does_not_submit_and_context_is_preserved(tmp_path: Path) -> None:
    """A cancelled waiter uses no worker slot; caller context is copied into the worker."""
    path = tmp_path / "source.txt"
    path.write_text("source")
    chunker = DefaultFileChunker()
    entered, release = threading.Event(), threading.Event()
    marker = contextvars.ContextVar("chunk_test_marker", default="unset")
    original = chunker._chunk_sync
    calls = []

    def blocking_parse(file_path: Path, rel_path: str) -> tuple[FileNode, list[FileChunk]]:
        calls.append(marker.get())
        entered.set()
        assert release.wait(5)
        return original(file_path, rel_path)

    token = marker.set("caller")
    try:
        async with chunker:
            with patch.object(chunker, "_chunk_sync", side_effect=blocking_parse):
                first = asyncio.create_task(chunker.chunk(path))
                await _wait_started(entered)
                queued = asyncio.create_task(chunker.chunk(path))
                await asyncio.sleep(0)
                queued.cancel()
                try:
                    with pytest.raises(asyncio.CancelledError):
                        await queued
                finally:
                    release.set()
                    await first
    finally:
        marker.reset(token)
    assert calls == ["caller"]


@pytest.mark.asyncio
async def test_markdown_renderer_is_serialized_across_components(tmp_path: Path) -> None:
    """Independent chunker instances cannot overlap renderer registry changes."""
    from mistletoe.markdown_renderer import MarkdownRenderer

    path = tmp_path / "source.md"
    path.write_text("# Title\n\nparagraph **bold** [[other.md]]\n")
    entered, release = threading.Event(), threading.Event()
    calls = []
    original_enter = MarkdownRenderer.__enter__

    def blocking_enter(renderer: MarkdownRenderer) -> MarkdownRenderer:
        calls.append(threading.get_ident())
        entered.set()
        assert release.wait(5)
        return original_enter(renderer)

    with patch.object(MarkdownRenderer, "__enter__", blocking_enter):
        first = asyncio.create_task(MarkdownFileChunker().chunk(path))
        await _wait_started(entered)
        second = asyncio.create_task(MarkdownFileChunker().chunk(path))
        try:
            await asyncio.sleep(0.05)
            assert len(calls) == 1
        finally:
            release.set()
            first_output, second_output = await asyncio.gather(first, second)
    assert first_output == second_output
    assert len(calls) == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", ["complete", "cancel", "change", "error"])
async def test_existing_content_searchable_during_indexing(
    tmp_path: Path,
    outcome: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exercise UpdateIndexStep and LocalFileStore while parsing is suspended."""
    monkeypatch.chdir(tmp_path)
    path = tmp_path / "source.txt"
    path.write_text("old searchable content")
    context = ApplicationContext(workspace_dir=str(tmp_path))
    chunker = DefaultFileChunker(app_context=context)
    context.components = {ComponentEnum.FILE_CHUNKER: {"default": chunker}}
    store = LocalFileStore(embedding_store="")
    async with store, chunker:
        await store.upsert([await chunker.chunk(path)])
        path.write_text("new searchable content")
        entered, release = threading.Event(), threading.Event()
        original = chunker._chunk_sync

        def blocking_parse(file_path: Path, rel_path: str) -> tuple[FileNode, list[FileChunk]]:
            result = original(file_path, rel_path)
            entered.set()
            assert release.wait(5)
            if outcome == "error":
                raise ValueError("parser failed")
            return result

        step = UpdateIndexStep(file_store=store, persist=False, app_context=context)
        app = Application(
            workspace_dir=str(tmp_path / "service"),
            service={"backend": "http", "web_enabled": False},
            components={},
            jobs={},
            enable_logo=False,
            log_to_console=False,
            log_to_file=False,
        )
        service = app.context.service
        assert isinstance(service, HttpService)
        service.build_service(app)
        service.add_job(_StoreSearchJob(store))
        service.add_job(_StoreSearchStreamJob(store))
        with patch.object(chunker, "_chunk_sync", side_effect=blocking_parse):
            updating = asyncio.create_task(step(RuntimeContext(changes=[{"change": "modified", "path": str(path)}])))
            await _wait_started(entered)
            try:
                nodes = await store.get_nodes()
                assert len(nodes) == 1
                matches = await store.keyword_search("searchable", limit=10, search_filter={})
                assert matches[0].text == "old searchable content"
                async with AsyncClient(transport=ASGITransport(app=service.service), base_url="http://test") as client:
                    response = await client.post("/search", json={"query": "searchable"})
                    assert response.status_code == 200
                    assert response.json()["answer"] == ["old searchable content"]
                    streaming = await client.post("/search_stream", json={"query": "searchable"})
                    assert streaming.status_code == 200
                    assert "old searchable content" in streaming.text
                    assert streaming.text.endswith("data:[DONE]\n\n")
                assert service.mcp_server is not None
                tool = await service.mcp_server.get_tool("search")
                assert tool is not None
                result = await tool.run({"query": "searchable"})
                assert "old searchable content" in str(result.content)
                if outcome == "cancel":
                    updating.cancel()
                elif outcome == "change":
                    path.write_text("latest source content")
            finally:
                release.set()
            if outcome == "cancel":
                with pytest.raises(asyncio.CancelledError):
                    await updating
            else:
                response = await updating
                assert response.success == (outcome == "complete")
        nodes = await store.get_nodes()
        chunks = [store.file_chunks[chunk_id] for chunk_id in nodes[0].chunk_ids]
        assert chunks[0].text == ("new searchable content" if outcome == "complete" else "old searchable content")
        if outcome != "complete":
            recovered = await step(RuntimeContext(changes=[{"change": "modified", "path": str(path)}]))
            assert recovered.success is True
            nodes = await store.get_nodes()
            assert store.file_chunks[nodes[0].chunk_ids[0]].text == path.read_text()
        assert not step._source_versions


@pytest.mark.asyncio
@pytest.mark.parametrize("change", ["modify", "delete", "replace", "cancel"])
async def test_changed_batched_file_is_not_published_and_valid_files_update(
    tmp_path: Path,
    change: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An earlier parsed file can change while the batch waits for the next worker."""
    monkeypatch.chdir(tmp_path)
    first_path, second_path = tmp_path / "first.txt", tmp_path / "second.txt"
    first_path.write_text("original first")
    second_path.write_text("original second")
    context = ApplicationContext(workspace_dir=str(tmp_path))
    chunker = DefaultFileChunker(app_context=context)
    context.components = {ComponentEnum.FILE_CHUNKER: {"default": chunker}}
    async with LocalFileStore(embedding_store="") as store, chunker:
        await store.upsert([await chunker.chunk(first_path), await chunker.chunk(second_path)])
        first_path.write_text("updated first")
        second_path.write_text("updated second")
        entered, release = threading.Event(), threading.Event()
        original = chunker.chunk

        async def blocking_second(file_path: Path) -> tuple[FileNode, list[FileChunk]]:
            if file_path == second_path:
                entered.set()
                while not release.is_set():
                    await asyncio.sleep(0.01)
            return await original(file_path)

        changes = [{"change": "modified", "path": str(path)} for path in (first_path, second_path)]
        step = UpdateIndexStep(file_store=store, persist=False, app_context=context)
        with patch.object(chunker, "chunk", side_effect=blocking_second):
            task = asyncio.create_task(step(RuntimeContext(changes=changes)))
            await _wait_started(entered)
            try:
                if change == "modify":
                    first_path.write_text("latest first")
                elif change == "delete":
                    first_path.unlink()
                elif change == "cancel":
                    task.cancel()
                else:
                    replacement = tmp_path / "replacement"
                    replacement.write_text("replacement first")
                    replacement.replace(first_path)
            finally:
                release.set()
            if change == "cancel":
                with pytest.raises(asyncio.CancelledError):
                    await task
            else:
                response = await task
                assert response.success is False
                assert [result["success"] for result in response.answer] == [False, True]
        assert not step._source_versions
        nodes = {node.path: node for node in await store.get_nodes()}
        assert store.file_chunks[nodes["first.txt"].chunk_ids[0]].text == "original first"
        assert store.file_chunks[nodes["second.txt"].chunk_ids[0]].text == (
            "original second" if change == "cancel" else "updated second"
        )
        recovered = await step(
            RuntimeContext(
                changes=[{"change": "deleted" if change == "delete" else "modified", "path": str(first_path)}],
            ),
        )
        assert recovered.success is True
        nodes = {node.path: node for node in await store.get_nodes()}
        if change == "delete":
            assert "first.txt" not in nodes
        else:
            assert store.file_chunks[nodes["first.txt"].chunk_ids[0]].text == first_path.read_text()
        assert not step._source_versions


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", ["change", "cancel"])
async def test_version_is_checked_after_waiting_for_store_lock(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    outcome: str,
) -> None:
    """A delayed mutation lock cannot admit a result that became stale while waiting."""
    monkeypatch.chdir(tmp_path)
    path = tmp_path / "source.txt"
    path.write_text("original source")
    context = ApplicationContext(workspace_dir=str(tmp_path))
    chunker = DefaultFileChunker(app_context=context)
    context.components = {ComponentEnum.FILE_CHUNKER: {"default": chunker}}
    async with LocalFileStore(embedding_store="") as store, chunker:
        await store.upsert([await chunker.chunk(path)])
        path.write_text("updated source")
        built = asyncio.Event()
        step = UpdateIndexStep(file_store=store, persist=False, app_context=context)
        original = step.build_item

        async def record_build(file_path: Path) -> tuple[FileNode, list[FileChunk]]:
            item = await original(file_path)
            built.set()
            return item

        with patch.object(step, "build_item", side_effect=record_build):
            async with store._maintenance_guard():
                task = asyncio.create_task(step(RuntimeContext(changes=[{"change": "modified", "path": str(path)}])))
                await asyncio.wait_for(built.wait(), 5)
                await asyncio.sleep(0)
                if outcome == "cancel":
                    task.cancel()
                    with pytest.raises(asyncio.CancelledError):
                        await task
                else:
                    path.write_text("latest source")
            if outcome == "change":
                response = await task
                assert response.success is False
        assert not step._source_versions
        nodes = await store.get_nodes()
        assert store.file_chunks[nodes[0].chunk_ids[0]].text == "original source"
        recovered = await step(RuntimeContext(changes=[{"change": "modified", "path": str(path)}]))
        assert recovered.success is True
        nodes = await store.get_nodes()
        assert store.file_chunks[nodes[0].chunk_ids[0]].text == path.read_text()
        assert not step._source_versions
