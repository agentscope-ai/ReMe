"""Meta-ReMe cycle timing parity with fake clocks and isolated evaluation evidence."""

# pylint: disable=missing-function-docstring,missing-class-docstring,protected-access
# pylint: disable=redefined-outer-name,unused-argument

import asyncio
import copy
from types import SimpleNamespace

import pytest
from agentscope.agent import ModelConfig, ReActConfig
from agentscope.message import HintBlock, TextBlock, UserMsg
from agentscope.model import ChatResponse, ChatUsage
from agentscope.state import AgentState
from reme_home.agent import optimizer
from reme_home.agent.bounded import BoundedReActAgent
from reme_home.evaluation.artifacts import write_json
from test_bounded_agent import RecordingModel


def make_agent(timer, model=None):
    return optimizer.OptimizerAgent(
        name="optimizer",
        system_prompt="Improve general memory mechanisms.",
        model=model or RecordingModel(),
        validation_cycle_timer=timer,
        model_config=ModelConfig(max_retries=0),
        react_config=ReActConfig(max_iters=100),
    )


def cycle_hints(messages, source="meta_reme_validation_timer"):
    return [
        block
        for message in messages
        for block in message.content
        if isinstance(block, HintBlock) and block.source == source
    ]


@pytest.mark.parametrize("budget", [True, 0, -1, float("nan"), float("inf"), "10"])
def test_cycle_rejects_invalid_budget(budget):
    with pytest.raises(ValueError, match="positive"):
        optimizer.ValidationCycleTimer(budget)


@pytest.mark.asyncio
async def test_cycle_injects_dev_meta_reminders_and_resets():
    now = [0.0]
    timer = optimizer.ValidationCycleTimer(100, clock=lambda: now[0])

    class SnapshotModel(RecordingModel):
        async def __call__(self, **kwargs):
            return await super().__call__(**copy.deepcopy(kwargs))

    agent = make_agent(timer, SnapshotModel())
    timer.record_context_compression()
    timer.record_context_compression()
    delivered = []
    for minute in (49, 50, 74, 75, 90, 100, 109, 110, 120):
        now[0] = minute * 60
        _ = [item async for item in agent._reasoning()]
        reminders = [block.hint for block in cycle_hints(agent.model.calls[-1]["messages"])]
        if minute in (50, 75, 90, 100, 110, 120):
            delivered.append(reminders[-1])
        assert timer.next_reminder() is None
    for percentage, text in zip((50, 75, 90, 100, 110, 120), delivered):
        assert f"({percentage}% of its 100 minutes soft time budget)" in text
        assert "2 context compressions since the last scored validation" in text
    assert "start validation within 50 minutes" in delivered[0]
    assert "start validation within 10 minutes" in delivered[2]
    assert "starting validation as soon as possible" in delivered[3]
    persistent = cycle_hints(agent.state.context)
    assert len(persistent) == 3
    for percentage, message in zip((75, 100, 120), persistent):
        assert f"({percentage}%" in message.hint
    timer.reset()
    assert timer.context_compression_count == 0
    now[0] += 60
    assert timer.next_reminder() is None
    # Reset keeps historical persistent messages and restarts alternation.
    assert cycle_hints(agent.state.context) == persistent
    now[0] = 170 * 60
    assert timer.next_reminder().persistent is False


def test_cycle_skips_old_thresholds_and_formats_seconds():
    now = [0.0]
    timer = optimizer.ValidationCycleTimer(1, clock=lambda: now[0])
    now[0] = 59
    reminder = timer.next_reminder()
    assert "(90%" in reminder.content
    assert "within 1 second." in reminder.content
    now[0] = 101
    assert "(160%" in timer.next_reminder().content
    assert timer.next_reminder() is None


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", [RuntimeError("model unavailable"), asyncio.CancelledError()])
async def test_cycle_ephemeral_cleanup_on_error_and_cancellation(monkeypatch, failure):
    now = [0.0]
    timer = optimizer.ValidationCycleTimer(1, clock=lambda: now[0])
    agent = make_agent(timer)

    async def fail(*args, **kwargs):
        raise failure
        yield  # pylint: disable=unreachable

    monkeypatch.setattr(BoundedReActAgent, "_reasoning", fail)
    now[0] = 30
    with pytest.raises(type(failure)):
        _ = [item async for item in agent._reasoning()]
    assert not cycle_hints(agent.state.context)
    now[0] = 45
    with pytest.raises(type(failure)):
        _ = [item async for item in agent._reasoning()]
    assert len(cycle_hints(agent.state.context)) == 1  # The second reminder is persistent.


@pytest.mark.asyncio
async def test_cycle_counts_only_completed_context_replacements(monkeypatch):
    timer = optimizer.ValidationCycleTimer(1)
    agent = make_agent(timer)

    async def unchanged(self):
        pass

    monkeypatch.setattr(BoundedReActAgent, "compress_context", unchanged)
    await agent.compress_context()
    assert timer.context_compression_count == 0

    async def compressed(self):
        self.state.summary = "Retained previous evidence."
        self.state.context = []

    monkeypatch.setattr(BoundedReActAgent, "compress_context", compressed)
    await agent.compress_context()
    assert timer.context_compression_count == 1

    async def failed(self):
        raise RuntimeError("compression unavailable")

    monkeypatch.setattr(BoundedReActAgent, "compress_context", failed)
    with pytest.raises(RuntimeError, match="unavailable"):
        await agent.compress_context()
    assert timer.context_compression_count == 1


def write_history(home, started):
    write_json(
        home / "sampling/state.json",
        {
            "schema_version": 2,
            "round": len(started),
            "successful_round": 0,
            "started_validations": {key: index for index, key in enumerate(started, 1)},
            "processed_validations": [],
            "cases": {},
            "queries": {},
        },
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("timed", [False, True])
@pytest.mark.parametrize(
    "outcome,reset",
    [
        ("completed", True),
        ("success", True),
        ("candidate_failure", True),
        ("completed_infra_error", True),
        ("debug", False),
        ("rejected", False),
        ("failed", False),
        ("partial", False),
        ("not_started", False),
        ("preexisting", False),
        ("cancelled", False),
    ],
)
async def test_evaluation_reset_boundary_matches_dev_meta(tmp_path, outcome, reset, timed):
    home = tmp_path
    previous = ["init/initial/1", "candidate/previous/1"]
    key = "candidate/current/1"
    if outcome == "preexisting":
        previous.append(key)
    write_history(home, previous)
    now = [0.0]
    timer = optimizer.ValidationCycleTimer(1, clock=lambda: now[0])
    timer.record_context_compression()
    now[0] = 45
    timer.next_reminder()  # First reminder is ephemeral.
    state = AgentState()
    old = UserMsg(name="meta_reme_validation_timer", content="Previous persistent reminder.")
    state.context.append(old)

    class EvaluationJob:
        enabled = is_started = True

        async def __call__(self, **kwargs):
            assert kwargs["debug"] is (outcome == "debug")
            # Tool execution time remains part of the previous cycle until completion.
            now[0] = 90
            if outcome == "cancelled":
                raise asyncio.CancelledError()
            if outcome not in {"debug", "rejected", "not_started", "preexisting"}:
                write_history(home, [*previous, key])
            manifest = {
                "commit_sha": "1b2cdcc" + "0" * 33,
                "status": "failed" if outcome in {"failed", "partial"} else "completed",
                "mode": "debug" if outcome == "debug" else "scored",
                "dataset_kind": "training-set",
                "sandbox_started_at": None if outcome == "not_started" else "2026-10-31T00:00:00Z",
            }
            run = home / "evaluations" / key
            write_json(run / "manifest.json", manifest)
            status = outcome if outcome in {"success", "rejected", "failed", "partial"} else "completed"
            return SimpleNamespace(
                success=outcome not in {"completed_infra_error", "rejected", "failed", "partial"},
                answer={"status": status, "result_dir": str(run), "mean_query_score": 0.0},
                metadata={},
            )

    wrapper = SimpleNamespace(app_context=SimpleNamespace(jobs={"home_evaluate": EvaluationJob()}))
    tool = optimizer.evaluation_tool(wrapper, {}, home=home, timer=timer if timed else None)
    assert set(tool.input_schema["properties"]) == {"debug"}
    if outcome == "cancelled":
        with pytest.raises(asyncio.CancelledError):
            await tool._func(debug=False, _agent_state=state)
    else:
        await tool._func(debug=outcome == "debug", _agent_state=state)
    assert len(state.context) == 1  # The tool cannot insert anything before its result is written.
    pending = state.middle_context.get(optimizer.PENDING_CYCLE_TRANSITIONS, [])
    assert len(pending) == int(reset)
    agent = make_agent(None)
    agent.state = state
    _ = [item async for item in agent._reasoning()]
    transitions = cycle_hints(state.context, "meta_reme_validation_cycle_transition")
    assert len(transitions) == int(reset)
    assert state.context[0] is old
    if reset:
        assert transitions[0].hint == (
            "<system-reminder>Formal validation 2 completed successfully for commit 1b2cdcc. "
            "The previous cycle timer has been reset, and all timer reminders for cycle 2 are now expired. "
            "Remaining formal validations: 8. Start cycle 3.</system-reminder>"
        )
    if reset and timed:
        assert timer.context_compression_count == 0
        assert timer.next_reminder() is None
        now[0] += 30
        assert timer.next_reminder().persistent is False
    else:
        assert timer.context_compression_count == 1
        assert "(150%" in timer.next_reminder().content


def test_wrapper_owns_fresh_cycle_timers_and_supports_disabling():
    wrapper = optimizer.OptimizerAgentWrapper(name="optimizer", validation_cycle_time_budget_minutes=150)
    first, second = wrapper.create_cycle_timer(), wrapper.create_cycle_timer()
    assert first is not second
    assert first.time_budget_minutes == 150
    first.record_context_compression()
    assert second.context_compression_count == 0
    assert wrapper.create_cycle_timer({"validation_cycle_time_budget_minutes": None}) is None
    legacy = optimizer.OptimizerAgentWrapper(name="optimizer", time_budget_minutes=75)
    assert legacy.create_cycle_timer().time_budget_minutes == 75
    assert legacy.create_cycle_timer({"validation_cycle_time_budget_minutes": 30}).time_budget_minutes == 30


@pytest.mark.asyncio
async def test_temporary_and_persistent_reminders_preserve_total_reply_usage():
    now = [0.0]

    class AdvancingModel(RecordingModel):
        async def __call__(self, **kwargs):
            result = await super().__call__(**kwargs)
            now[0] = 30 if len(self.calls) == 1 else 45
            return result

    model = AdvancingModel(
        final=ChatResponse(
            content=[TextBlock(text="final report")],
            is_last=True,
            usage=ChatUsage(input_tokens=2, output_tokens=3, time=0),
        ),
    )
    agent = make_agent(optimizer.ValidationCycleTimer(1, clock=lambda: now[0]), model)
    agent.react_config.max_iters = 2
    reply = await agent.reply(UserMsg(name="user", content="Investigate"))
    assert len(model.calls) == 3
    assert reply.usage.input_tokens == 6
    assert reply.usage.output_tokens == 9
    retained = cycle_hints(agent.state.context)
    assert len(retained) == 1
    assert "(75%" in retained[0].hint
