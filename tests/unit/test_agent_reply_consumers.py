"""Plugin answer steps consume the complete agent outcome in both return modes."""

# pylint: disable=protected-access

import asyncio
from importlib import import_module
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from reme.components.agent_wrapper import BaseAgentWrapper
from reme.components.application_context import ApplicationContext
from reme.enumeration import ChunkEnum
from reme.schema import StreamChunk
from reme.steps.index._dedup import _ToolContextDedupMixin


class _Agent(BaseAgentWrapper):
    """Emit a text delta followed by its authoritative final snapshot."""

    def __init__(self, status, **kwargs):
        super().__init__(**kwargs)
        self.result = self._reply_result(
            session_id="session",
            result="answer",
            raw={"signature": "signed"},
            status=status,
            error="provider failed" if status == "failed" else None,
        )
        self.chunks = [
            StreamChunk(chunk="answer", block_id="text", turn_id="turn"),
            StreamChunk(chunk="answer", event="end", block_id="text", turn_id="turn"),
            StreamChunk(
                chunk_type=ChunkEnum.REPLY_END,
                event="end",
                status=status,
                session_id="session",
                turn_id="turn",
                chunk={key: value for key, value in self.result.items() if key != "events"},
            ),
        ]

    def _add_tool_context(self, kwargs):
        if "tool_context_id" not in kwargs:
            return
        self.app_context.metadata.setdefault(_ToolContextDedupMixin.TOOL_CONTEXTS_KEY, {})[
            kwargs["tool_context_id"]
        ] = {"query": "question"}

    async def reply(self, inputs, **kwargs):
        """Return the same outcome as the stream."""
        self._add_tool_context(kwargs)
        return self.result

    async def reply_stream(self, inputs, **kwargs):
        """Yield original chunks, including IDs and snapshot operations."""
        self._add_tool_context(kwargs)
        for chunk in self.chunks:
            yield chunk


@pytest.mark.asyncio
@pytest.mark.parametrize(("plugin", "class_name"), [("lme", "LmeAgenticAnswerStep"), ("beam", "BeamAgenticAnswerStep")])
@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("status", ["completed", "failed", "interrupted", "waiting"])
async def test_answer_plugins_preserve_outcome_and_stream_chunks(
    tmp_path,
    monkeypatch,
    plugin,
    class_name,
    streaming,
    status,
):
    """Final text is not doubled, incomplete replies fail, and metadata survives forwarding."""
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2] / "plugins" / plugin / "src"))
    step_type = getattr(import_module(f"reme_{plugin}.agentic_answer"), class_name)
    context = ApplicationContext(workspace_dir=str(tmp_path))
    wrapper = _Agent(status, app_context=context)
    queue = asyncio.Queue() if streaming else None
    response = await step_type(app_context=context, agent_wrapper=wrapper)(query="question", stream_queue=queue)

    assert response.answer == ("provider failed" if status == "failed" else "answer")
    assert response.success is (status == "completed")
    assert response.metadata["status"] == status
    assert response.metadata["raw"] == wrapper.result["raw"]
    assert not context.metadata[_ToolContextDedupMixin.TOOL_CONTEXTS_KEY]
    if streaming:
        assert [queue.get_nowait() for _ in wrapper.chunks] == wrapper.chunks
        assert response.metadata["events"] == [chunk.model_dump(mode="json") for chunk in wrapper.chunks]


@pytest.mark.asyncio
@pytest.mark.parametrize("plugin", ["lme", "beam"])
async def test_stream_failure_releases_temporary_query_state(tmp_path, monkeypatch, plugin):
    """A failed collector/transport leaves no query dedup state alive."""
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2] / "plugins" / plugin / "src"))
    step_type = getattr(import_module(f"reme_{plugin}.agentic_answer"), f"{plugin.title()}AgenticAnswerStep")
    context = ApplicationContext(workspace_dir=str(tmp_path))
    wrapper = _Agent("completed", app_context=context)

    async def stream(_inputs, **kwargs):
        wrapper._add_tool_context(kwargs)
        yield wrapper.chunks[0]
        raise RuntimeError("transport disconnected")

    monkeypatch.setattr(wrapper, "reply_stream", stream)
    queue = asyncio.Queue()
    with pytest.raises(RuntimeError, match="transport disconnected"):
        await step_type(app_context=context, agent_wrapper=wrapper)(query="question", stream_queue=queue)
    assert not context.metadata[_ToolContextDedupMixin.TOOL_CONTEXTS_KEY]
    assert queue.get_nowait() == wrapper.chunks[0]


@pytest.mark.asyncio
@pytest.mark.parametrize("plugin", ["lme", "beam"])
@pytest.mark.parametrize("status", ["completed", "failed", "interrupted", "waiting"])
async def test_judges_do_not_score_incomplete_agent_replies(tmp_path, monkeypatch, plugin, status):
    """A parseable partial judgement cannot become a valid benchmark score."""
    if plugin == "beam":
        pytest.importorskip("scipy")
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2] / "plugins" / f"{plugin}-judge" / "src"))
    module = import_module(f"judge_{plugin}.llm_judge")
    context = ApplicationContext(workspace_dir=str(tmp_path))
    wrapper = _Agent(status, app_context=context)
    wrapper.reply = AsyncMock(return_value=wrapper.result)
    if plugin == "lme":
        step = module.LmeAnswerJudgeStep(app_context=context, agent_wrapper=wrapper)
        wrapper.result["result"] = "YES"
        args = {"query": "question", "agent_answer": "answer", "golden_answer": "answer"}
    else:
        step = module.BeamRubricJudgeStep(app_context=context, agent_wrapper=wrapper)
        wrapper.result["result"] = '{"score": 1.0, "reason": "correct"}'
        args = {"llm_response": "answer", "rubric": ["correct answer"]}
    if status == "completed":
        assert (await step(**args)).success
    else:
        with pytest.raises(RuntimeError, match="provider failed" if status == "failed" else status):
            await step(**args)


@pytest.mark.asyncio
@pytest.mark.parametrize("plugin", ["auto-fin", "daily_paper"])
@pytest.mark.parametrize("status", ["failed", "interrupted", "waiting"])
async def test_structured_consumers_reject_incomplete_agent_outputs(tmp_path, monkeypatch, plugin, status):
    """A valid-looking structured payload still requires a completed native invocation."""
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2] / "plugins" / plugin / "src"))
    context = ApplicationContext(workspace_dir=str(tmp_path))
    wrapper = _Agent(status, app_context=context)
    wrapper.reply = AsyncMock(return_value=wrapper.result)
    wrapper.result["structured_output"] = {"title": "report", "description": "summary", "body": "body"}
    error = "provider failed" if status == "failed" else status
    if plugin == "auto-fin":
        module = import_module("reme_auto_fin.research")
        schema = import_module("reme_auto_fin.schema").AutoFinReportOutput
        step = module.AutoFinResearchStep(app_context=context, agent_wrapper=wrapper, prompt_dict={"test": "question"})
        with pytest.raises(RuntimeError, match=error):
            await step._reply("test", schema)
    else:
        module = import_module("reme_daily_paper.base")
        schema = import_module("reme_daily_paper").DailyPaperMarkdownOutput
        with pytest.raises(RuntimeError, match=error):
            module.structured_output(wrapper.result, schema)
