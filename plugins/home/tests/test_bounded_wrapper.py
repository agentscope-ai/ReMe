"""Bounded ReMe agents use real Job tools, injection and persisted sessions."""

# pylint: disable=missing-function-docstring,missing-class-docstring,protected-access

import json
from importlib.metadata import EntryPoint
from types import SimpleNamespace

import pytest
from agentscope.message import HintBlock, TextBlock, ToolCallBlock
from agentscope.model import ChatResponse, ChatUsage
from agentscope.tool import FunctionTool, Toolkit
from reme_home.agent import BoundedAgentWrapper, BoundedReActAgent, RunTimeBudgetTimer
from test_bounded_agent import RecordingModel

from reme.components.agent_wrapper import AsAgentWrapper
from reme.components.application_context import ApplicationContext
from reme.components.job import BaseJob
from reme.enumeration import ComponentEnum
from reme.plugin import resolve_plugin_runtime
from reme.steps.base_step import BaseStep
from reme.utils import global_counter_get


class CaptureStep(BaseStep):
    async def execute(self):
        self.app_context.metadata.setdefault("test.calls", []).append(dict(self.context.data))
        self.context.response.answer = "captured"


class JobModel(RecordingModel):
    def __init__(self, arguments):
        super().__init__()
        self.arguments = arguments

    async def __call__(self, **kwargs):
        self.calls.append(kwargs)
        content = (
            [ToolCallBlock(id="capture-call", name="capture", input=json.dumps(self.arguments))]
            if len(self.calls) == 1
            else [TextBlock(text="finished")]
        )
        return ChatResponse(content=content, is_last=True, usage=ChatUsage(input_tokens=2, output_tokens=3, time=0))


@pytest.fixture(name="wrapper")
def bounded_wrapper(tmp_path):
    context = ApplicationContext(workspace_dir=str(tmp_path))
    result = BoundedAgentWrapper(
        app_context=context,
        name="investigator",
        as_llm="",
        time_budget_minutes=30,
        react_config={"max_iters": 2},
    )
    result.as_llm = SimpleNamespace(model=RecordingModel())
    return result


def test_plugin_registers_bounded_agent_only_when_enabled(monkeypatch):
    entry = EntryPoint(name="home", value="reme_home", group="reme.plugins")
    monkeypatch.setattr("reme.plugin.find_entry_points", lambda group, name: [entry])
    enabled = resolve_plugin_runtime({"plugins": ["home"]})
    assert enabled.registry.get(ComponentEnum.AGENT_WRAPPER, "home_bounded") is BoundedAgentWrapper
    assert resolve_plugin_runtime({}).registry.get(ComponentEnum.AGENT_WRAPPER, "home_bounded") is None
    assert issubclass(BoundedAgentWrapper, AsAgentWrapper)


@pytest.mark.asyncio
async def test_each_build_owns_timer_and_honors_call_overrides(wrapper):
    first, _ = await wrapper._build_agent("first", **wrapper._merged_kwargs({}))
    second, _ = await wrapper._build_agent("second", **wrapper._merged_kwargs({"time_budget_minutes": 5}))
    assert isinstance(first, BoundedReActAgent)
    assert first.run_time_budget_timer.time_budget_minutes == 30
    assert second.run_time_budget_timer.time_budget_minutes == 5
    assert first.run_time_budget_timer is not second.run_time_budget_timer
    assert first.react_config.max_iters == 2
    no_timer, _ = await wrapper._build_agent("third", **wrapper._merged_kwargs({"time_budget_minutes": None}))
    assert no_timer.run_time_budget_timer is None


@pytest.mark.asyncio
@pytest.mark.parametrize("conflict", [False, True])
async def test_real_job_tools_enforce_hidden_injected_arguments(wrapper, conflict):
    context = wrapper.app_context
    context.registry.register(CaptureStep, "test_capture")
    job = BaseJob(
        name="capture",
        app_context=context,
        parameters={
            "type": "object",
            "properties": {"visible": {"type": "string"}, "case_ids": {"type": "array"}},
            "required": ["visible", "case_ids"],
        },
        steps=[{"backend": "test_capture"}],
    )
    context.jobs["capture"] = job
    arguments = {"visible": "model value"}
    if conflict:
        arguments["case_ids"] = ["model-chosen"]
    model = JobModel(arguments)
    wrapper.as_llm.model = model
    await job.start()
    await wrapper.start()
    try:
        result = await wrapper.reply(
            "Run the capture job",
            job_tools=["capture"],
            injected_job_kwargs={"case_ids": ["host-chosen"]},
            tool_context_id="investigation-1",
        )
        schemas = json.dumps(model.calls[0]["tools"])
        assert "case_ids" not in schemas
        assert "tool_context_id" not in schemas
        if conflict:
            assert "test.calls" not in context.metadata
        else:
            assert context.metadata["test.calls"] == [
                {
                    "visible": "model value",
                    "case_ids": ["host-chosen"],
                    "tool_context_id": "investigation-1",
                },
            ]
        assert result["result"] == "finished"
        assert (wrapper.session_path / f"{result['session_id']}.jsonl").is_file()
        resumed = await wrapper.reply("Continue", resume=result["session_id"])
        assert resumed["session_id"] == result["session_id"]
    finally:
        await wrapper.close()
        await job.close()


@pytest.mark.asyncio
async def test_wrapper_preserves_observed_task_in_forced_finalization(wrapper):
    result = await wrapper.reply("Identify the original failure")
    assert result["result"] == "final report"
    messages = wrapper.as_llm.model.calls[-1]["messages"]
    assert "Original task: Identify the original failure" in messages[-1].get_text_content()
    assert result["session_id"]


@pytest.mark.asyncio
async def test_wrapper_streams_reme_chunks_and_persists_session(wrapper):
    chunks = [chunk async for chunk in wrapper.reply_stream("Investigate")]
    assert any(chunk.chunk == "final report" for chunk in chunks)
    session_id = chunks[-1].session_id
    assert session_id
    assert (wrapper.session_path / f"{session_id}.jsonl").is_file()


@pytest.mark.asyncio
@pytest.mark.parametrize("timed", [False, True])
async def test_multiround_usage_and_tools_continue_past_soft_time_limit(wrapper, monkeypatch, timed):
    now = [0.0]
    timer = RunTimeBudgetTimer(1, agent_label="investigator", clock=lambda: now[0])
    monkeypatch.setattr("reme_home.agent.wrapper.RunTimeBudgetTimer", lambda *args, **kwargs: timer)
    executions = []

    async def inspect_evidence():
        executions.append(now[0])
        return "evidence"

    class MultiRoundModel(RecordingModel):
        def __init__(self):
            super().__init__()
            self.time_reminders = []

        async def __call__(self, **kwargs):
            self.calls.append(kwargs)
            self.time_reminders.extend(
                block.hint
                for message in kwargs["messages"]
                for block in message.content
                if isinstance(block, HintBlock) and block.source == "home_run_time_guard"
            )
            now[0] += 65
            count = len(self.calls)
            content = (
                [ToolCallBlock(id=f"inspect-{count}", name="inspect_evidence", input="{}")]
                if count < 3
                else [TextBlock(text="finished")]
            )
            return ChatResponse(
                content=content,
                is_last=True,
                usage=ChatUsage(input_tokens=10, output_tokens=5, time=0),
            )

    model = MultiRoundModel()
    wrapper.as_llm.model = model
    options = {
        "react_config": {"max_iters": 4},
        "time_budget_minutes": 1 if timed else None,
        "toolkit": Toolkit(tools=[FunctionTool(inspect_evidence)]),
    }
    result = await wrapper.reply("Investigate", **options)
    assert result["usage"] == {"input_tokens": 30, "output_tokens": 15, "total_tokens": 45}
    assert result["last_message"]["finished_reason"] == "completed"
    assert len(executions) == 2
    assert executions[-1] > 60  # A tool really executes after the soft deadline.
    assert len(model.calls) == 3
    assert all(call["tools"] for call in model.calls)
    assert bool(model.time_reminders) is timed
    assert not timer.active
    metadata = wrapper.app_context.metadata
    assert global_counter_get(metadata, ["__token_counter", wrapper.name, "total_tokens"]) == 45
    resumed = await wrapper.reply("Continue", resume=result["session_id"], **options)
    assert resumed["usage"]["total_tokens"] == 15
    assert global_counter_get(metadata, ["__token_counter", wrapper.name, "total_tokens"]) == 60
