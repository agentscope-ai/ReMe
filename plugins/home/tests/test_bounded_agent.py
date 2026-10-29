"""Home budgets exercised through AgentScope's real reply and event loop."""

# pylint: disable=missing-function-docstring,missing-class-docstring,protected-access,unused-argument

import asyncio
from types import SimpleNamespace

import pytest
from agentscope.agent import Agent, ReActConfig
from agentscope.agent._config import ModelConfig
from agentscope.event import ModelCallEndEvent, ReplyEndEvent, ReplyEndReason
from agentscope.message import AssistantMsg, HintBlock, TextBlock, ThinkingBlock, ToolCallBlock, UserMsg
from agentscope.model import ChatResponse, ChatUsage, FinishedReason
from agentscope.tool import FunctionTool, Toolkit

from reme_home.agent import BoundedReActAgent, RunTimeBudgetTimer


class RecordingModel:
    model = "isolated-test"
    context_size = 10000
    formatter = SimpleNamespace(supported_input_media_types=[])

    def __init__(self, *, final=None, stream=False):
        self.calls = []
        self.final = final
        self.stream = stream

    async def count_tokens(self, messages, tools):
        return 10

    async def __call__(self, **kwargs):
        self.calls.append(kwargs)
        if kwargs["tool_choice"] is not None and kwargs["tool_choice"].mode == "none":
            if isinstance(self.final, BaseException):
                raise self.final
            response = self.final or ChatResponse(content=[TextBlock(text="final report")], is_last=True)
            if self.stream:

                async def chunks():
                    yield ChatResponse(content=[TextBlock(text="final ")], is_last=False)
                    yield response

                return chunks()
            return response
        return ChatResponse(
            content=[ThinkingBlock(thinking="investigating")],
            is_last=True,
            usage=ChatUsage(input_tokens=2, output_tokens=3, time=0),
        )


def hints(agent, source):
    return [
        block
        for message in agent.state.context
        for block in message.content
        if isinstance(block, HintBlock) and block.source == source
    ]


def make_agent(model=None, timer=None, max_iters=2):
    return BoundedReActAgent(
        name="investigator",
        system_prompt="Investigate the evidence.",
        model=model or RecordingModel(),
        react_config=ReActConfig(max_iters=max_iters, interruption_raise_cancelled_error=True),
        model_config=ModelConfig(max_retries=0),
        run_time_budget_timer=timer,
    )


@pytest.mark.parametrize("budget", [True, 0, -1, float("nan"), float("inf"), "10"])
def test_timer_rejects_invalid_budget(budget):
    with pytest.raises(ValueError, match="positive"):
        RunTimeBudgetTimer(budget, agent_label="diagnostic")


def test_timer_thresholds_restart_and_skipped_thresholds():
    now = [0.0]
    timer = RunTimeBudgetTimer(100, agent_label="diagnostic", clock=lambda: now[0])
    assert timer.next_reminder() is None
    timer.start()
    for percentage, persistent in [(59, None), (80, False), (90, True), (100, False), (110, True)]:
        now[0] = percentage * 60
        reminder = timer.next_reminder()
        if persistent is None:
            assert reminder is None
        else:
            assert reminder.persistent is persistent
            assert "does not terminate" in reminder.content
        assert timer.next_reminder() is None
    timer.record_context_compression()
    timer.stop()
    assert timer.next_reminder() is None
    timer.start()
    assert timer.context_compression_count == 0
    assert timer.next_reminder() is None


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [False, True])
async def test_native_loop_has_exactly_one_tool_free_finalization(stream):
    usage = ChatUsage(input_tokens=5, output_tokens=7, time=0, cache_input_tokens=2)
    model = RecordingModel(
        final=ChatResponse(content=[TextBlock(text="report")], is_last=True, usage=usage),
        stream=stream,
    )
    agent = make_agent(model)
    timer = RunTimeBudgetTimer(1, agent_label="diagnostic")
    agent.run_time_budget_timer = timer
    items = [
        item
        async for item in agent.reply_stream(UserMsg(name="caller", content="Find the cause"), yield_final_msg=True)
    ]
    assert len(model.calls) == 3
    assert agent.state.cur_iter == 3
    request = model.calls[-1]
    assert request["tools"] == []
    assert request["tool_choice"].mode == "none"
    assert "Find the cause" in request["messages"][-1].get_text_content()
    assert items[-1].get_text_content() == "report"
    # Reminders must preserve cumulative reply usage as well as per-call events.
    assert items[-1].usage.input_tokens == 9
    assert items[-1].usage.output_tokens == 13
    assert items[-1].usage.cache_input_tokens == 2
    calls = [item for item in items if isinstance(item, ModelCallEndEvent)]
    assert len(calls) == 3
    assert sum(item.input_tokens for item in calls) == 9
    assert sum(item.output_tokens for item in calls) == 13
    ends = [item for item in items if isinstance(item, ReplyEndEvent)]
    assert len(ends) == 1 and ends[0].finished_reason == ReplyEndReason.EXCEED_MAX_ITERS
    assert not timer.active


@pytest.mark.asyncio
async def test_finalization_discards_unsolicited_tools():
    executed = []

    def forbidden():
        executed.append(True)

    model = RecordingModel(
        final=ChatResponse(
            content=[TextBlock(text="report"), ToolCallBlock(id="call", name="forbidden", input="{}")],
            is_last=True,
        ),
    )
    agent = make_agent(model)
    agent.toolkit = Toolkit(tools=[FunctionTool(forbidden)])
    result = await agent.reply(UserMsg(name="caller", content="Investigate"))
    assert result.get_text_content() == "report"
    assert not executed
    assert not agent.state.get_unfinished_tool_calls(agent.name)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "failure",
    [
        RuntimeError("provider unavailable"),
        ChatResponse(content=[], is_last=True),
        ChatResponse(content=[], is_last=True, usage=ChatUsage(input_tokens=5, output_tokens=7, time=0)),
    ],
)
async def test_finalization_failure_returns_explicit_incomplete_output(failure):
    model = RecordingModel(final=failure)
    agent = make_agent(model)
    result = await agent.reply(UserMsg(name="caller", content="Investigate"))
    assert "incomplete prior model output" in result.get_text_content()
    assert len(model.calls) == 3
    final_usage = failure.usage if isinstance(failure, ChatResponse) else None
    assert result.usage.input_tokens == 4 + (final_usage.input_tokens if final_usage else 0)
    assert result.usage.output_tokens == 6 + (final_usage.output_tokens if final_usage else 0)


@pytest.mark.asyncio
async def test_cancellation_stops_timer_without_fallback():
    timer = RunTimeBudgetTimer(1, agent_label="diagnostic")
    agent = make_agent(RecordingModel(final=asyncio.CancelledError()), timer)
    with pytest.raises(asyncio.CancelledError):
        await agent.reply(UserMsg(name="caller", content="Investigate"))
    assert not timer.active
    assert not any("incomplete prior model output" in (msg.get_text_content() or "") for msg in agent.state.context)


@pytest.mark.asyncio
async def test_interrupted_response_preserves_native_interruption():
    model = RecordingModel(final=ChatResponse(content=[], is_last=True, finished_reason=FinishedReason.INTERRUPTED))
    with pytest.raises(asyncio.CancelledError):
        await make_agent(model).reply(UserMsg(name="caller", content="Investigate"))


@pytest.mark.asyncio
async def test_new_reply_resets_iteration_reminders():
    model = RecordingModel()
    agent = make_agent(model)
    for task in ("first", "second"):
        result = await agent.reply(UserMsg(name="caller", content=task))
        assert result.get_text_content() == "final report"
        assert agent._original_task_prompt == task
    assert len(model.calls) == 6
    reminders = hints(agent, "home_iteration_guard")
    assert len(reminders) == 4


@pytest.mark.asyncio
async def test_normal_reply_does_not_force_finalization():
    class ImmediateModel(RecordingModel):
        async def __call__(self, **kwargs):
            self.calls.append(kwargs)
            return ChatResponse(content=[TextBlock(text="already sufficient")], is_last=True)

    model = ImmediateModel()
    result = await make_agent(model).reply(UserMsg(name="caller", content="Investigate"))
    assert result.get_text_content() == "already sufficient"
    assert len(model.calls) == 1


@pytest.mark.asyncio
async def test_final_context_trims_old_evidence_but_preserves_system_and_task():
    class SmallContextModel(RecordingModel):
        async def count_tokens(self, messages, tools):
            return 10000 if len(messages) > 2 else 10

    model = SmallContextModel()
    agent = make_agent(model)
    agent.state.cur_iter = 2
    agent._original_task_prompt = "original task"
    messages = [UserMsg(name="system", content="system prompt"), UserMsg(name="caller", content="old evidence")]
    await agent._call_model(messages, tools=[{"name": "unused"}])
    assert len(model.calls[-1]["messages"]) == 2
    assert model.calls[-1]["messages"][0] is messages[0]
    assert "original task" in model.calls[-1]["messages"][-1].get_text_content()
    assert len(messages) == 2


@pytest.mark.asyncio
async def test_missing_stream_completion_returns_previous_text_without_retry():
    class IncompleteStreamModel(RecordingModel):
        async def __call__(self, **kwargs):
            self.calls.append(kwargs)

            async def chunks():
                yield ChatResponse(content=[TextBlock(text="partial")], is_last=False)

            return chunks()

    model = IncompleteStreamModel()
    agent = make_agent(model)
    agent.state.cur_iter = 2
    agent.state.context.append(AssistantMsg(name=agent.name, content="previous finding"))
    response = await agent._call_model([], tools=[])
    assert "incomplete prior model output" in response.content[0].text
    assert "previous finding" in response.content[0].text
    assert len(model.calls) == 1


@pytest.mark.asyncio
async def test_time_reminders_alternate_and_do_not_stop_investigation():
    now = [0.0]
    timer = RunTimeBudgetTimer(100, agent_label="diagnostic", clock=lambda: now[0])
    model = RecordingModel()
    agent = make_agent(model, timer, max_iters=100)
    timer.start()
    for percentage in (60, 75, 90, 100, 110):
        now[0] = percentage * 60
        _ = [item async for item in agent._reasoning()]
    persistent = hints(agent, "home_run_time_guard")
    assert len(persistent) == 2
    assert "(75%" in persistent[0].hint
    assert "(100%" in persistent[1].hint
    assert len(model.calls) == 5
    assert timer.active


@pytest.mark.asyncio
async def test_ephemeral_reminder_removed_when_reasoning_fails(monkeypatch):
    now = [0.0]
    timer = RunTimeBudgetTimer(1, agent_label="diagnostic", clock=lambda: now[0])
    agent = make_agent(timer=timer)
    timer.start()
    now[0] = 40

    async def fail(*args, **kwargs):
        raise RuntimeError("failed")
        yield  # pylint: disable=unreachable

    monkeypatch.setattr(Agent, "_reasoning", fail)
    with pytest.raises(RuntimeError):
        _ = [item async for item in agent._reasoning()]
    assert not hints(agent, "home_run_time_guard")


@pytest.mark.asyncio
async def test_only_successful_context_replacement_counts_as_compression(monkeypatch):
    timer = RunTimeBudgetTimer(1, agent_label="diagnostic")
    agent = make_agent(timer=timer)

    async def no_change(self):
        pass

    monkeypatch.setattr(Agent, "compress_context", no_change)
    await agent.compress_context()
    assert timer.context_compression_count == 0

    async def change(self):
        self.state.context = []

    monkeypatch.setattr(Agent, "compress_context", change)
    await agent.compress_context()
    assert timer.context_compression_count == 1


class CompressionFailureModel(RecordingModel):
    def __init__(self, failure, fail_after=2):
        super().__init__()
        self.failure = failure
        self.fail_after = fail_after

    async def count_tokens(self, messages, tools):
        return 9000 if len(self.calls) >= self.fail_after and len(messages) > 2 else 10

    async def generate_structured_output(self, **kwargs):
        raise self.failure


@pytest.mark.asyncio
async def test_final_report_survives_real_compression_failure():
    model = CompressionFailureModel(RuntimeError("compression provider unavailable"))
    timer = RunTimeBudgetTimer(1, agent_label="diagnostic")
    agent = make_agent(model, timer)
    agent.context_config.compression_fallback_to_truncation = False
    result = await agent.reply(UserMsg(name="caller", content="Investigate"))
    assert result.get_text_content() == "final report"
    assert len(model.calls) == 3
    assert model.calls[-1]["tools"] == []
    assert len(model.calls[-1]["messages"]) == 2
    assert timer.context_compression_count == 0
    assert not timer.active


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", [RuntimeError("compression failed"), asyncio.CancelledError()])
async def test_nonfinal_compression_errors_and_cancellation_propagate(failure):
    model = CompressionFailureModel(failure, fail_after=1 if isinstance(failure, RuntimeError) else 2)
    agent = make_agent(model)
    agent.context_config.compression_fallback_to_truncation = False
    with pytest.raises(type(failure)):
        await agent.reply(UserMsg(name="caller", content="Investigate"))
    assert len(model.calls) == model.fail_after
