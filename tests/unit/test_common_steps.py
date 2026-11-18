"""Tests for reme common steps with only local dependencies."""

# pylint: disable=protected-access

from contextlib import chdir as _temp_chdir
import asyncio
import tempfile
import warnings
from unittest.mock import AsyncMock

import pytest

from reme.components.agent_wrapper import BaseAgentWrapper
from reme.components.application_context import ApplicationContext
from reme.components.file_store import LocalFileStore
from reme.enumeration import ChunkEnum
from reme.schema import FileFrontMatter, FileLink, FileNode, StreamChunk, TokenUsage, TraverseGraph
from reme.steps.common.add import AddStep
from reme.steps.common.health_check import _file_graph_status
from reme.steps.common.llm_demo import LLMDemoStep
from reme.steps.common.python_execute import PythonExecuteStep
from reme.steps.common.stream_llm_demo import StreamLLMDemoStep
from reme.steps.index import traverse as traverse_mod
from reme.utils.local_backend import LocalProcessBackend

warnings.filterwarnings("ignore", category=DeprecationWarning, module="jieba")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="pkg_resources")


@pytest.fixture(autouse=True)
def python_process_backend(monkeypatch):
    """Test executor responses with real pipes; sandbox policy is tested separately."""
    monkeypatch.setattr(
        "reme.steps.common.python_execute.file_sandbox_for_step",
        lambda _step: LocalProcessBackend(),
    )


def _node(
    path: str,
    links: list[tuple[str, str | None]] | None = None,
    *,
    name: str = "",
    description: str = "",
) -> FileNode:
    """Build a FileNode with (target_path, target_anchor) outgoing edges."""
    return FileNode(
        path=path,
        st_mtime=1.0,
        links=[
            FileLink(source_path=path, target_path=target, target_anchor=anchor) for target, anchor in (links or [])
        ],
        front_matter=FileFrontMatter(name=name, description=description),
    )


async def _make_store(nodes: list[FileNode]) -> LocalFileStore:
    """LocalFileStore seeded with the given graph nodes (no files on disk)."""
    store = LocalFileStore(name="t", embedding_store="")
    await store.start()
    if nodes:
        await store.file_graph.upsert_nodes(nodes)
    return store


def _edges(step) -> list[dict]:
    return step.context.response.answer["edges"]


@pytest.mark.asyncio
async def test_add_step_coerces_numeric_inputs():
    """add accepts numeric strings as numbers, not string concatenation."""

    step = AddStep()
    resp = await step(a="1", b="2.5")
    assert resp.success is True
    assert resp.answer == "3.5"
    assert resp.metadata["result"] == 3.5


@pytest.mark.asyncio
async def test_add_step_rejects_invalid_inputs():
    """invalid add arguments should return a failed response instead of throwing or concatenating."""

    step = AddStep()
    resp = await step(a="one", b=2)
    assert resp.success is False
    assert "Invalid add arguments" in resp.answer


@pytest.mark.asyncio
async def test_python_execute_step_returns_printed_stdout(tmp_path):
    """python_execute returns stdout as answer and runs under workspace_dir."""

    app_context = ApplicationContext(workspace_dir=str(tmp_path))
    step = PythonExecuteStep(app_context=app_context)
    resp = await step(code="from pathlib import Path\nprint(Path.cwd().name)")
    assert resp.success is True
    assert resp.answer == f"{tmp_path.name}\n"
    assert resp.metadata["returncode"] == 0
    assert resp.metadata["stderr"] == ""


@pytest.mark.asyncio
async def test_python_execute_step_reports_stderr_on_failure():
    """python_execute captures traceback stderr instead of throwing."""

    step = PythonExecuteStep()
    resp = await step(code='raise RuntimeError("boom")')
    assert resp.success is False
    assert resp.metadata["returncode"] != 0
    assert "RuntimeError: boom" in resp.answer
    assert "RuntimeError: boom" in resp.metadata["stderr"]


@pytest.mark.asyncio
async def test_python_execute_step_times_out():
    """python_execute converts subprocess timeout into a failed response."""

    step = PythonExecuteStep()
    resp = await step(code="import time\ntime.sleep(1)", python_timeout=0.01)
    assert resp.success is False
    assert resp.answer == "Python execution timed out after 0.01s"
    assert resp.metadata["python_timeout"] == 0.01


@pytest.mark.asyncio
async def test_python_execute_step_accepts_legacy_timeout_name():
    """Configs written before the rename keep their timeout through the legacy key."""

    step = PythonExecuteStep()
    resp = await step(code="import time\ntime.sleep(1)", timeout=0.01)
    assert resp.success is False
    assert resp.answer == "Python execution timed out after 0.01s"
    assert resp.metadata["python_timeout"] == 0.01


@pytest.mark.asyncio
async def test_python_execute_step_rejects_invalid_timeout():
    """A non-positive python_timeout fails the call instead of running unbounded."""

    step = PythonExecuteStep()
    resp = await step(code="print(1)", python_timeout=0)
    assert resp.success is False
    assert resp.answer == "python_timeout must be a positive number"


class _FakeAgentWrapper(BaseAgentWrapper):
    """Capture reply kwargs without calling a real model."""

    def __init__(self):
        super().__init__()
        self.last_kwargs = None
        self.stream_chunks = []

    async def reply(self, inputs, **kwargs) -> dict:
        self.last_kwargs = kwargs
        return self._reply_result(session_id="session-1", result="ok", raw={})

    async def reply_stream(self, inputs, **kwargs):
        self.last_kwargs = kwargs
        for chunk in self.stream_chunks:
            yield chunk


@pytest.mark.asyncio
async def test_llm_demo_always_registers_add_tool():
    """LLM demo always passes the add job as a tool."""

    wrapper = _FakeAgentWrapper()
    step = LLMDemoStep()
    resp = await step(query="hello", agent_wrapper=wrapper)
    assert resp.success is True
    assert wrapper.last_kwargs["job_tools"] == ["add"]
    assert "job_tools" not in resp.metadata


@pytest.mark.asyncio
@pytest.mark.parametrize("step_class", [LLMDemoStep, StreamLLMDemoStep])
@pytest.mark.parametrize("status", ["completed", "failed", "interrupted", "waiting"])
async def test_nonstream_demo_forwards_outcome_and_native_details(step_class, status):
    """HTTP/tool results retain native details and never report incomplete calls as successes."""
    wrapper = _FakeAgentWrapper()
    reply = wrapper._reply_result(
        session_id="session-1",
        result="partial or final",
        status=status,
        error="provider failure" if status == "failed" else None,
        structured_output=False,
        usage=TokenUsage(input_tokens=10, output_tokens=2),
        raw={"native": "preserved"},
    )
    wrapper.reply = AsyncMock(return_value=reply)
    response = await step_class()(query="hello", agent_wrapper=wrapper)

    assert response.success is (status == "completed")
    assert response.answer == ("provider failure" if status == "failed" else "partial or final")
    assert all(response.metadata[key] == value for key, value in reply.items())


@pytest.mark.asyncio
async def test_stream_llm_demo_preserves_backend_chunk_metadata():
    """Forward retry flags, tool identities, usage, and completion status without rebuilding chunks."""
    wrapper = _FakeAgentWrapper()
    wrapper.stream_chunks = [
        StreamChunk(chunk_type=ChunkEnum.ERROR, chunk="retrying", metadata={"will_retry": True}),
        StreamChunk(chunk_type=ChunkEnum.TOOL_CALL, tool_call_id="call-1", tool_call_name="add", chunk={"a": 1}),
        StreamChunk(chunk_type=ChunkEnum.CONTENT, session_id="session-1", block_id="text-1", chunk="hello"),
        StreamChunk(chunk_type=ChunkEnum.USAGE, input_tokens=3, output_tokens=1),
        StreamChunk(
            chunk_type=ChunkEnum.REPLY_END,
            event="end",
            status="completed",
            chunk={
                key: value
                for key, value in wrapper._reply_result(session_id="session-1", result="hello", raw={}).items()
                if key != "events"
            },
        ),
    ]
    queue = asyncio.Queue()
    response = await StreamLLMDemoStep()(query="hello", stream_queue=queue, agent_wrapper=wrapper)
    forwarded = [queue.get_nowait() for _ in wrapper.stream_chunks]
    assert all(actual is expected for actual, expected in zip(forwarded, wrapper.stream_chunks))
    assert queue.empty()
    assert response.success
    assert response.answer == "hello"
    assert response.metadata["session_id"] == "session-1"


def test_file_graph_health_reports_neo4j_cached_counts():
    """Neo4j file graph health should not be reported as an empty local graph."""

    class FakeNeo4jGraph:
        """Minimal Neo4j graph stub with cached health counters."""

        is_started = True
        _driver = object()
        _uri = "bolt://example"
        _database = "neo4j"
        _n_nodes = 3
        _n_edges = 4
        _n_virtual = 1

    status = _file_graph_status(FakeNeo4jGraph())
    assert status["n_nodes"] == 3
    assert status["n_edges"] == 4
    assert status["n_virtual"] == 1


# ===========================================================================
# Direct unit tests: TraverseStep
# (LocalFileStore, no HTTP server — BFS over wikilink edges)
# ===========================================================================


@pytest.mark.asyncio
async def test_traverse_forward_depth_1():
    """depth=1 forward returns direct outbound neighbors."""

    with tempfile.TemporaryDirectory() as tmp, _temp_chdir(tmp):
        store = await _make_store(
            [
                _node("a.md", [("b.md", None), ("c.md", "intro")]),
                _node("b.md"),
                _node("c.md"),
            ],
        )
        step = traverse_mod.TraverseStep(file_store=store)
        await step(path="a.md", direction="forward", depth=1)
        results = _edges(step)
        paths = {r["target"] for r in results}
        assert paths == {"b.md", "c.md"}
        c_edge = next(r for r in results if r["target"] == "c.md")
        assert c_edge["target_anchor"] == "intro"
        assert c_edge["source"] == "a.md"
        assert c_edge["depth"] == 1
        await store.close()


@pytest.mark.asyncio
async def test_traverse_normalizes_windows_seed_path():
    """Windows-style seeds match the graph's portable POSIX path keys."""

    with tempfile.TemporaryDirectory() as tmp, _temp_chdir(tmp):
        store = await _make_store(
            [
                _node("topics/a.md", [("topics/b.md", None)]),
                _node("topics/b.md"),
            ],
        )
        step = traverse_mod.TraverseStep(file_store=store)
        await step(path=r"topics\a.md", direction="forward", depth=1)
        assert [edge["target"] for edge in _edges(step)] == ["topics/b.md"]
        await store.close()


@pytest.mark.asyncio
async def test_traverse_backward_returns_inlinks():
    """direction=backward walks inbound edges."""

    with tempfile.TemporaryDirectory() as tmp, _temp_chdir(tmp):
        store = await _make_store(
            [
                _node("a.md", [("b.md", None)]),
                _node("c.md", [("b.md", None)]),
                _node("b.md"),
            ],
        )
        step = traverse_mod.TraverseStep(file_store=store)
        await step(path="b.md", direction="backward", depth=1)
        results = _edges(step)
        assert {(r["source"], r["target"]) for r in results} == {
            ("a.md", "b.md"),
            ("c.md", "b.md"),
        }
        await store.close()


@pytest.mark.asyncio
async def test_traverse_depth_2_expands():
    """depth=2 traverses one hop beyond direct neighbors."""

    with tempfile.TemporaryDirectory() as tmp, _temp_chdir(tmp):
        store = await _make_store(
            [
                _node("a.md", [("b.md", None)]),
                _node("b.md", [("c.md", None)]),
                _node("c.md"),
            ],
        )
        step = traverse_mod.TraverseStep(file_store=store)
        await step(path="a.md", direction="forward", depth=2)
        results = _edges(step)
        depth_map = {r["target"]: r["depth"] for r in results}
        assert depth_map.get("b.md") == 1
        assert depth_map.get("c.md") == 2
        await store.close()


@pytest.mark.asyncio
async def test_traverse_short_seed_yields_empty():
    """A short (not relative to the workspace) seed isn't resolved anymore — BFS simply
    finds no edges from a path that doesn't match any graph node."""

    with tempfile.TemporaryDirectory() as tmp, _temp_chdir(tmp):
        store = await _make_store(
            [
                _node("topics/Bob.md"),
                _node("people/Bob.md"),
            ],
        )
        step = traverse_mod.TraverseStep(file_store=store)
        await step(path="Bob", direction="forward", depth=1)
        payload = _edges(step)
        # No error, just empty results because "Bob" isn't a graph key.
        assert payload == []
        await store.close()


@pytest.mark.asyncio
async def test_traverse_not_found_seed():
    """A seed not in the graph returns an empty list (no error)."""

    with tempfile.TemporaryDirectory() as tmp, _temp_chdir(tmp):
        store = await _make_store([_node("a.md")])
        step = traverse_mod.TraverseStep(file_store=store)
        await step(path="topics/ghost.md", direction="forward", depth=1)
        payload = _edges(step)
        assert payload == []
        await store.close()


@pytest.mark.asyncio
async def test_traverse_both_directions():
    """direction=both walks out- and in-bound; depth=1 returns one hop in each direction."""

    with tempfile.TemporaryDirectory() as tmp, _temp_chdir(tmp):
        store = await _make_store(
            [
                _node("upstream.md", [("center.md", None)]),
                _node("center.md", [("downstream.md", None)]),
                _node("downstream.md"),
            ],
        )
        step = traverse_mod.TraverseStep(file_store=store)
        await step(path="center.md", direction="both", depth=1)
        results = _edges(step)
        assert {(r["source"], r["target"]) for r in results} == {
            ("upstream.md", "center.md"),
            ("center.md", "downstream.md"),
        }
        await store.close()


@pytest.mark.asyncio
async def test_traverse_both_preserves_reciprocal_edge_directions():
    """Opposite wikilinks remain two distinct directed graph edges."""

    with tempfile.TemporaryDirectory() as tmp, _temp_chdir(tmp):
        store = await _make_store(
            [
                _node("a.md", [("b.md", None)]),
                _node("b.md", [("a.md", None)]),
            ],
        )
        step = traverse_mod.TraverseStep(file_store=store)
        response = await step(path="a.md", direction="both", depth=1)
        graph = TraverseGraph.model_validate(response.answer)

        assert {(edge.source, edge.target) for edge in graph.edges} == {
            ("a.md", "b.md"),
            ("b.md", "a.md"),
        }
        assert response.metadata == {}
        await store.close()


@pytest.mark.asyncio
async def test_traverse_returns_frontmatter_and_unresolved_nodes():
    """Graph nodes expose labels and distinguish indexed files from dangling targets."""

    with tempfile.TemporaryDirectory() as tmp, _temp_chdir(tmp):
        store = await _make_store(
            [
                _node(
                    "a.md",
                    [("missing.md", "details")],
                    name="Alpha",
                    description="Root note",
                ),
            ],
        )
        step = traverse_mod.TraverseStep(file_store=store)
        response = await step(path="a.md", direction="forward", depth=1)
        graph = TraverseGraph.model_validate(response.answer)
        nodes = {node.path: node for node in graph.nodes}

        assert graph.version == 1
        assert graph.seeds == ["a.md"]
        assert nodes["a.md"].model_dump() == {
            "id": "a.md",
            "path": "a.md",
            "name": "Alpha",
            "description": "Root note",
            "depth": 0,
            "indexed": True,
        }
        assert nodes["missing.md"].indexed is False
        assert nodes["missing.md"].depth == 1
        assert graph.edges[0].target_anchor == "details"
        await store.close()


@pytest.mark.asyncio
async def test_traverse_depth_zero_returns_only_seed_nodes():
    """A zero-hop traversal is valid and emits no edges."""

    with tempfile.TemporaryDirectory() as tmp, _temp_chdir(tmp):
        store = await _make_store([_node("a.md", [("b.md", None)]), _node("b.md")])
        step = traverse_mod.TraverseStep(file_store=store)
        response = await step(path="a.md", direction="forward", depth=0)
        graph = TraverseGraph.model_validate(response.answer)

        assert [node.path for node in graph.nodes] == ["a.md"]
        assert graph.edges == []
        await store.close()
