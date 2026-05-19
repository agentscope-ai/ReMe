"""Tests for ReMe memory error handling and raise_exception propagation."""

from types import SimpleNamespace

import pytest

import reme.reme as reme_module
from reme.core.runtime_context import RuntimeContext
from reme.core.schema import MemoryNode
from reme.memory.vector_tools.history.add_history import AddHistory
from reme.memory.vector_tools.history.read_history import ReadHistory
from reme.memory.vector_tools.history.retrieve_history import RetrieveHistory
from reme.reme import ReMe


class Recorder:
    """Stub that records constructor args for later inspection."""

    instances = []

    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        self.call_kwargs = None
        Recorder.instances.append(self)


class TopLevelAgent(Recorder):
    """Stub agent that returns a successful structured result."""

    async def call(self, **kwargs):
        """Simulate a successful agent call."""
        self.call_kwargs = kwargs
        return {"answer": "ok", "success": True}


def _make_reme() -> ReMe:
    """Create a ReMe instance with startup bypassed for unit testing."""
    reme = ReMe(enable_logo=False, log_to_console=False, enable_profile=False)
    reme._started = True  # pylint: disable=protected-access
    return reme


def _patch_summarize_dependencies(monkeypatch: pytest.MonkeyPatch) -> None:
    """Patch all summarize-path dependencies with stubs."""
    Recorder.instances = []
    monkeypatch.setattr(reme_module, "AddDraftAndRetrieveSimilarMemory", Recorder)
    monkeypatch.setattr(reme_module, "AddMemory", Recorder)
    monkeypatch.setattr(reme_module, "AddHistory", Recorder)
    monkeypatch.setattr(reme_module, "DelegateTask", Recorder)
    monkeypatch.setattr(reme_module, "PersonalSummarizer", Recorder)
    monkeypatch.setattr(reme_module, "ProceduralSummarizer", Recorder)
    monkeypatch.setattr(reme_module, "ToolSummarizer", Recorder)
    monkeypatch.setattr(reme_module, "ReMeSummarizer", TopLevelAgent)


def _patch_retrieve_dependencies(monkeypatch: pytest.MonkeyPatch) -> None:
    """Patch all retrieve-path dependencies with stubs."""
    Recorder.instances = []
    monkeypatch.setattr(reme_module, "RetrieveMemory", Recorder)
    monkeypatch.setattr(reme_module, "ReadHistory", Recorder)
    monkeypatch.setattr(reme_module, "DelegateTask", Recorder)
    monkeypatch.setattr(reme_module, "PersonalRetriever", Recorder)
    monkeypatch.setattr(reme_module, "ProceduralRetriever", Recorder)
    monkeypatch.setattr(reme_module, "ToolRetriever", Recorder)
    monkeypatch.setattr(reme_module, "ReMeRetriever", TopLevelAgent)


@pytest.mark.asyncio
@pytest.mark.parametrize("raise_exception", [False, True])
async def test_summarize_memory_propagates_raise_exception(
    monkeypatch: pytest.MonkeyPatch,
    raise_exception: bool,
):
    """Verify raise_exception is forwarded to every sub-agent in summarize."""
    _patch_summarize_dependencies(monkeypatch)
    reme = _make_reme()

    result = await reme.summarize_memory(
        messages=[{"role": "user", "content": "hi", "time_created": "2026-03-20 10:00:00"}],
        task_name="demo-task",
        raise_exception=raise_exception,
    )

    assert result == "ok"
    assert Recorder.instances
    assert all(instance.kwargs.get("raise_exception") is raise_exception for instance in Recorder.instances)


@pytest.mark.asyncio
async def test_retrieve_memory_binds_top_level_retriever_prompt_path(monkeypatch: pytest.MonkeyPatch):
    """Verify top-level ReMeRetriever does not accidentally inherit child retriever prompts."""
    _patch_retrieve_dependencies(monkeypatch)
    reme = _make_reme()

    result = await reme.retrieve_memory(
        query="hello",
        user_name="alice",
    )

    assert result == "ok"
    top_level = next(instance for instance in Recorder.instances if isinstance(instance, TopLevelAgent))
    assert str(top_level.kwargs["prompt_path"]).endswith("reme/memory/vector_based/reme_retriever.yaml")


@pytest.mark.asyncio
@pytest.mark.parametrize("raise_exception", [False, True])
async def test_retrieve_memory_propagates_raise_exception(
    monkeypatch: pytest.MonkeyPatch,
    raise_exception: bool,
):
    """Verify raise_exception is forwarded to every sub-agent in retrieve."""
    _patch_retrieve_dependencies(monkeypatch)
    reme = _make_reme()

    result = await reme.retrieve_memory(
        query="hello",
        task_name="demo-task",
        raise_exception=raise_exception,
    )

    assert result == "ok"
    assert Recorder.instances
    assert all(instance.kwargs.get("raise_exception") is raise_exception for instance in Recorder.instances)


@pytest.mark.asyncio
async def test_summarize_memory_raises_runtime_error_for_unstructured_result(monkeypatch: pytest.MonkeyPatch):
    """Verify RuntimeError is raised when the top-level summarizer returns a plain string."""
    Recorder.instances = []
    reme = _make_reme()

    monkeypatch.setattr(reme_module, "PersonalSummarizer", Recorder)
    monkeypatch.setattr(reme_module, "ProceduralSummarizer", Recorder)
    monkeypatch.setattr(reme_module, "ToolSummarizer", Recorder)
    monkeypatch.setattr(reme_module, "AddDraftAndRetrieveSimilarMemory", Recorder)
    monkeypatch.setattr(reme_module, "AddMemory", Recorder)
    monkeypatch.setattr(reme_module, "AddHistory", Recorder)
    monkeypatch.setattr(reme_module, "DelegateTask", Recorder)

    class FailingTopLevelAgent(Recorder):
        """Stub agent that returns a failure string instead of a dict."""

        async def call(self, **kwargs):
            """Simulate a failed agent call returning a plain error string."""
            self.call_kwargs = kwargs
            return "[ReMeSummarizer] failed: boom"

    monkeypatch.setattr(reme_module, "ReMeSummarizer", FailingTopLevelAgent)

    with pytest.raises(RuntimeError, match="summarize_memory failed before producing a structured result"):
        await reme.summarize_memory(
            messages=[{"role": "user", "content": "hi", "time_created": "2026-03-20 10:00:00"}],
            task_name="demo-task",
        )


@pytest.mark.asyncio
async def test_read_history_accepts_single_history_id_in_multiple_mode():
    """Verify multiple-mode history lookup accepts a single history_id string."""

    class FakeVectorStore:
        """Minimal vector store stub for ReadHistory tests."""

        async def get(self, vector_ids):
            """Return the requested history node."""
            assert vector_ids == ["history_123"]
            node = MemoryNode(
                memory_id="history_123",
                memory_type="history",
                memory_target="alice",
                content="Alice said hello.",
            )
            return [node.to_vector_node()]

    tool = ReadHistory(enable_multiple=True)
    tool._vector_store = FakeVectorStore()  # pylint: disable=protected-access
    tool.context = RuntimeContext(
        history_id="history_123",
        retrieved_nodes=[],
        service_context=SimpleNamespace(memory_target_type_mapping={"alice": "personal"}),
    )

    result = await tool.execute()

    assert "Historical Dialogue[history_123]" in result


@pytest.mark.asyncio
async def test_add_history_stores_parent_and_chunk_nodes():
    """Verify AddHistory stores a full parent node plus retrievable chunk nodes."""

    class FakeVectorStore:
        """Minimal vector store stub for AddHistory tests."""

        def __init__(self):
            self.deleted_ids = None
            self.inserted_nodes = []

        async def list(self, filters=None, limit=None, sort_key=None, reverse=True):
            return []

        async def delete(self, vector_ids):
            self.deleted_ids = vector_ids

        async def insert(self, nodes):
            self.inserted_nodes = nodes

    vector_store = FakeVectorStore()
    tool = AddHistory(turn_block_size=1, max_chunk_tokens=1000)
    tool._vector_store = vector_store  # pylint: disable=protected-access
    tool.context = RuntimeContext(
        description="A coffee chat.",
        messages=[
            {"role": "user", "content": "我想喝咖啡"},
            {"role": "assistant", "content": "好的，您喜欢什么类型的咖啡？"},
            {"role": "user", "content": "Latte with oat milk please."},
            {"role": "assistant", "content": "没问题。"},
        ],
        author="tester",
        service_context=SimpleNamespace(memory_target_type_mapping={"alice": "personal"}),
    )

    result = await tool.execute()

    assert "Successfully added history:" in result
    assert len(vector_store.inserted_nodes) == 3
    parent_node = vector_store.inserted_nodes[0]
    chunk_nodes = vector_store.inserted_nodes[1:]
    assert parent_node.metadata["node_kind"] == "history"
    assert parent_node.metadata["chunk_count"] == 2
    assert all(node.metadata["node_kind"] == "history_chunk" for node in chunk_nodes)
    assert all(node.metadata["history_id"] == parent_node.vector_id for node in chunk_nodes)


@pytest.mark.asyncio
async def test_retrieve_history_filters_by_optional_history_id():
    """Verify RetrieveHistory can search within a history group."""

    class FakeVectorStore:
        """Minimal vector store stub for RetrieveHistory tests."""

        def __init__(self):
            self.search_calls = []

        async def search(self, query, limit, filters):
            self.search_calls.append({"query": query, "limit": limit, "filters": filters})
            return [
                MemoryNode(
                    memory_id="history_123:chunk:0000",
                    memory_type="history",
                    content="user: 我想喝咖啡\nassistant: 好的",
                    ref_memory_id="history_123",
                    metadata={
                        "node_kind": "history_chunk",
                        "history_id": "history_123",
                        "chunk_index": 0,
                    },
                ).to_vector_node(),
            ]

    vector_store = FakeVectorStore()
    tool = RetrieveHistory(top_k=2, enable_multiple=False)
    tool._vector_store = vector_store  # pylint: disable=protected-access
    tool.context = RuntimeContext(
        query="咖啡",
        history_id="history_123",
        retrieved_nodes=[],
        service_context=SimpleNamespace(memory_target_type_mapping={"alice": "personal"}),
    )

    result = await tool.execute()

    assert vector_store.search_calls == [
        {
            "query": "咖啡",
            "limit": 2,
            "filters": {"node_kind": "history_chunk", "history_id": "history_123"},
        },
    ]
    assert "Historical Dialogue Chunk[history_123:chunk:0000]" in result
    assert "history_id=history_123" in result
