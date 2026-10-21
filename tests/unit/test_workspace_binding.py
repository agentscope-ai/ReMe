"""Phase 2: requests and components bind to the selected workspace."""

# pylint: disable=protected-access,missing-function-docstring

import asyncio
from pathlib import Path

import pytest

from reme.application import Application
from reme.components import ApplicationContext
from reme.components.base_component import BaseComponent
from reme.components.job import BaseJob, StreamJob
from reme.components.runtime_context import RuntimeContext
from reme.components.component_registry import R
from reme.enumeration import ChunkEnum, ComponentEnum
from reme.schema import FileChunk, FileNode
from reme.steps.base_step import BaseStep


def _app(tmp_path, **overrides):
    config = {
        "workspace_dir": {"alpha": str(tmp_path / "alpha"), "beta": str(tmp_path / "beta")},
        "enable_logo": False,
        "log_to_console": False,
        "log_to_file": False,
        "service": {"backend": "cli"},
        "components": {
            "tokenizer": {"default": {"backend": "regex"}},
            "file_store": {"default": {"backend": "local", "embedding_store": "", "tag_index": ""}},
            "keyword_index": {"default": {"backend": "bm25"}},
            "file_graph": {"default": {"backend": "local"}},
        },
        "jobs": {
            "write_note": {
                "backend": "base",
                "description": "write a note",
                "parameters": {"type": "object", "properties": {}},
                "steps": [{"backend": "write_step"}],
            },
            "read_note": {
                "backend": "base",
                "description": "read a note",
                "parameters": {"type": "object", "properties": {}},
                "steps": [{"backend": "read_step"}],
            },
        },
        **overrides,
    }
    return Application(**config)


class _WorkspaceReporterStep(BaseStep):
    """Report the bound workspace alias as answer and stream chunk."""

    async def execute(self):
        bound = self.context.workspace
        alias = bound.alias if bound is not None else ""
        assert "workspace" not in self.context.data, "workspace alias must not leak into step payloads"
        self.context.response.answer = alias
        if self.context.stream:
            await self.context.add_stream_string(alias, ChunkEnum.CONTENT)


def _reporter_context(tmp_path, job_cls):
    registry = R.copy()
    registry.register(_WorkspaceReporterStep, "workspace_reporter_step")
    context = ApplicationContext(
        registry=registry,
        workspace_dir={"alpha": str(tmp_path / "alpha"), "beta": str(tmp_path / "beta")},
    )
    job = job_cls(
        name="report",
        app_context=context,
        parameters={"type": "object", "properties": {}},
        steps=[{"backend": "workspace_reporter_step"}],
    )
    return job


# ----- Selection semantics ---------------------------------------------------


@pytest.mark.asyncio
async def test_default_binding_uses_first_workspace(tmp_path):
    app = _app(tmp_path)
    async with app:
        response = await app.run_job("write_note", path="daily/note.md", content="default-target")

    assert response.success, response.answer
    assert "default-target" in (tmp_path / "alpha" / "daily" / "note.md").read_text(encoding="utf-8")
    assert not (tmp_path / "beta" / "daily" / "note.md").exists()


@pytest.mark.asyncio
async def test_unknown_workspace_alias_rejected(tmp_path):
    app = _app(tmp_path)
    async with app:
        with pytest.raises(ValueError, match="Unknown workspace 'nope'"):
            await app.run_job("write_note", path="daily/note.md", content="x", workspace="nope")


@pytest.mark.asyncio
async def test_unknown_workspace_alias_rejected_for_stream_job(tmp_path):
    job = _reporter_context(tmp_path, StreamJob)
    await job.start()
    with pytest.raises(ValueError, match="Unknown workspace 'nope'"):
        async for _chunk in _stream(job, workspace="nope"):
            pass
    await job.close()


async def _stream(job, **kwargs):
    from reme.utils import execute_stream_task

    queue = asyncio.Queue()
    task = asyncio.create_task(job(stream_queue=queue, **kwargs))
    async for chunk in execute_stream_task(stream_queue=queue, task=task, output_format="chunk"):
        yield chunk


# ----- Concurrent and streaming isolation ------------------------------------


@pytest.mark.asyncio
async def test_concurrent_same_relative_path_stays_isolated(tmp_path):
    app = _app(tmp_path)
    async with app:
        await asyncio.gather(
            *(
                app.run_job("write_note", path="daily/shared.md", content=f"{alias}-{i}", workspace=alias)
                for i in range(5)
                for alias in ("alpha", "beta")
            )
        )
        alpha = (tmp_path / "alpha" / "daily" / "shared.md").read_text(encoding="utf-8")
        beta = (tmp_path / "beta" / "daily" / "shared.md").read_text(encoding="utf-8")
    assert "alpha-4" in alpha and "beta" not in alpha
    assert "beta-4" in beta and "alpha" not in beta


@pytest.mark.asyncio
async def test_stream_chunks_carry_caller_workspace(tmp_path):
    job = _reporter_context(tmp_path, StreamJob)
    await job.start()

    async def collect(alias):
        return [chunk async for chunk in _stream(job, workspace=alias)]

    chunks_alpha, chunks_beta = await asyncio.gather(collect("alpha"), collect("beta"))
    content_alpha = [c.chunk for c in chunks_alpha if c.chunk_type == ChunkEnum.CONTENT]
    content_beta = [c.chunk for c in chunks_beta if c.chunk_type == ChunkEnum.CONTENT]
    assert content_alpha == ["alpha"]
    assert content_beta == ["beta"]
    assert chunks_alpha[-1].done and chunks_beta[-1].done
    await job.close()


# ----- Component binding -----------------------------------------------------


@pytest.mark.asyncio
async def test_workspace_scoped_components_instantiate_per_alias(tmp_path):
    app = _app(tmp_path)
    async with app:
        store_alpha = app.context.get_component(ComponentEnum.FILE_STORE, "default", "alpha")
        store_beta = app.context.get_component(ComponentEnum.FILE_STORE, "default", "beta")

        assert store_alpha is not store_beta
        assert app.context.components[ComponentEnum.FILE_STORE]["default"] is store_alpha  # default alias entry
        assert store_alpha.workspace_path == tmp_path / "alpha"
        assert store_beta.workspace_path == tmp_path / "beta"
        assert store_alpha.chunks_path.parent == tmp_path / "alpha" / "metadata" / "file_store"

        # Same-alias dependency wiring: each store binds its own alias's index/graph.
        index_alpha = app.context.get_component(ComponentEnum.KEYWORD_INDEX, "default", "alpha")
        graph_beta = app.context.get_component(ComponentEnum.FILE_GRAPH, "default", "beta")
        assert store_alpha.keyword_index is index_alpha
        assert store_beta.file_graph is graph_beta

        with pytest.raises(KeyError, match="no instance for workspace alias 'nope'"):
            app.context.get_component(ComponentEnum.FILE_STORE, "default", "nope")


@pytest.mark.asyncio
async def test_store_chunks_and_search_stay_in_own_workspace(tmp_path):
    app = _app(tmp_path)
    async with app:
        for alias, text in (("alpha", "apple pie"), ("beta", "banana bread")):
            store = app.context.get_component(ComponentEnum.FILE_STORE, "default", alias)
            node = FileNode(path=f"daily/{alias}.md", st_mtime=0.0)
            chunk = FileChunk(path=f"daily/{alias}.md", text=text, start_line=0, end_line=0)
            await store.upsert([(node, [chunk])])

        store_alpha = app.context.get_component(ComponentEnum.FILE_STORE, "default", "alpha")
        store_beta = app.context.get_component(ComponentEnum.FILE_STORE, "default", "beta")
        hits_alpha = await store_alpha.keyword_search("apple", limit=5, search_filter={})
        hits_beta = await store_beta.keyword_search("apple", limit=5, search_filter={})
        assert [c.path for c in hits_alpha] == ["daily/alpha.md"]
        assert hits_beta == []


@pytest.mark.asyncio
async def test_store_state_persists_per_alias_across_restart(tmp_path):
    app = _app(tmp_path)
    async with app:
        store = app.context.get_component(ComponentEnum.FILE_STORE, "default", "alpha")
        await store.upsert([(FileNode(path="daily/a.md", st_mtime=0.0), [FileChunk(path="daily/a.md", text="alpha data", start_line=0, end_line=0)])])

    rebuilt = _app(tmp_path)
    async with rebuilt:
        store_alpha = rebuilt.context.get_component(ComponentEnum.FILE_STORE, "default", "alpha")
        store_beta = rebuilt.context.get_component(ComponentEnum.FILE_STORE, "default", "beta")
        nodes_alpha = await store_alpha.get_nodes()
        nodes_beta = await store_beta.get_nodes()
    assert [n.path for n in nodes_alpha] == ["daily/a.md"]
    assert nodes_beta == []


# ----- External backends and single-workspace parity -------------------------


@pytest.mark.asyncio
async def test_external_stateful_backend_rejected_in_multi_workspace(tmp_path):
    class FakeExternalGraph(BaseComponent):
        component_type = ComponentEnum.FILE_GRAPH
        supports_multi_workspace = False

    with R.preserve(allow_mutation=True):
        R.register(FakeExternalGraph, "fake_external")
        with pytest.raises(ValueError, match="does not support multi-workspace"):
            _app(
                tmp_path,
                components={
                    "file_graph": {"default": {"backend": "fake_external"}},
                },
            )

    # Single-workspace configurations may still use the same backend.
    with R.preserve(allow_mutation=True):
        R.register(FakeExternalGraph, "fake_external")
        app = Application(
            workspace_dir=str(tmp_path / "single"),
            enable_logo=False,
            log_to_console=False,
            log_to_file=False,
            service={"backend": "cli"},
            components={"file_graph": {"default": {"backend": "fake_external"}}},
        )
        async with app:
            assert app.context.components[ComponentEnum.FILE_GRAPH]["default"] is not None


@pytest.mark.asyncio
async def test_single_workspace_string_config_binds_default(tmp_path):
    app = Application(
        workspace_dir=str(tmp_path / "only"),
        enable_logo=False,
        log_to_console=False,
        log_to_file=False,
        service={"backend": "cli"},
        jobs={
            "write_note": {
                "backend": "base",
                "parameters": {"type": "object", "properties": {}},
                "steps": [{"backend": "write_step"}],
            },
        },
    )
    async with app:
        response = await app.run_job("write_note", path="daily/note.md", content="plain")
        assert response.success, response.answer
        assert (tmp_path / "only" / "daily" / "note.md").exists()
        # The explicit default alias resolves to the same workspace.
        response = await app.run_job("write_note", path="daily/other.md", content="plain", workspace="default")
        assert response.success
        assert (tmp_path / "only" / "daily" / "other.md").exists()
