import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import reme.reme as reme_module
from reme import ReMe


class Recorder:
    instances = []

    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        Recorder.instances.append(self)


class TopLevelAgent(Recorder):
    async def call(self, **kwargs):
        self.call_kwargs = kwargs
        return {"answer": "ok", "success": True}


def _make_reme() -> ReMe:
    reme = ReMe(enable_logo=False, log_to_console=False, enable_profile=False)
    reme._started = True
    return reme


def _patch_summarize_dependencies(monkeypatch: pytest.MonkeyPatch) -> None:
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
    Recorder.instances = []
    reme = _make_reme()

    monkeypatch.setattr(
        reme_module,
        "PersonalSummarizer",
        lambda *args, **kwargs: Recorder(*args, **kwargs),
    )
    monkeypatch.setattr(
        reme_module,
        "ProceduralSummarizer",
        lambda *args, **kwargs: Recorder(*args, **kwargs),
    )
    monkeypatch.setattr(
        reme_module,
        "ToolSummarizer",
        lambda *args, **kwargs: Recorder(*args, **kwargs),
    )
    monkeypatch.setattr(reme_module, "AddDraftAndRetrieveSimilarMemory", Recorder)
    monkeypatch.setattr(reme_module, "AddMemory", Recorder)
    monkeypatch.setattr(reme_module, "AddHistory", Recorder)
    monkeypatch.setattr(reme_module, "DelegateTask", Recorder)

    class FailingTopLevelAgent(Recorder):
        async def call(self, **kwargs):
            self.call_kwargs = kwargs
            return "[ReMeSummarizer] failed: boom"

    monkeypatch.setattr(reme_module, "ReMeSummarizer", FailingTopLevelAgent)

    with pytest.raises(RuntimeError, match="summarize_memory failed before producing a structured result"):
        await reme.summarize_memory(
            messages=[{"role": "user", "content": "hi", "time_created": "2026-03-20 10:00:00"}],
            task_name="demo-task",
        )
