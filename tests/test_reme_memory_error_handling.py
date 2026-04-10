"""Tests for ReMe memory error handling and raise_exception propagation."""

import pytest

import reme.reme as reme_module
from reme import ReMe


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
