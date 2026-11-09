"""Optional hint sampling at time, persistence and actual agent-query boundaries."""

# pylint: disable=missing-function-docstring,protected-access,redefined-outer-name

import asyncio
from datetime import datetime, timedelta
import json
from unittest.mock import AsyncMock, Mock

import pytest

from reme.components._pa_hints import ProactiveHints
from reme.schema.pa_proactive import PaProactiveConfig
from reme.steps.proactive import ProactiveAgentStep

from .pa_helpers import PA_CONFIG, bind_pa


def config(**kwargs):
    return PaProactiveConfig(**{**PA_CONFIG, **kwargs})


async def sample(env, settings=None):
    async with env.runtime.proactive_round():
        return await env.runtime.hints.select(settings or config(), env.now[0])


def model(env, monkeypatch, *, content="", advance=None):
    calls = []
    monkeypatch.setattr(ProactiveAgentStep, "_now", lambda _step: env.now[0])

    async def reply(query, **kwargs):
        if kwargs.get("fork_session") and kwargs.get("resume") == env.runtime.session_id:
            return {"session_id": "context-fork", "status": "completed", "result": "Main context"}
        calls.append(query)
        phase = env.runtime.current_round.phase
        if phase == "query_history" and advance is not None:
            env.now[0] = advance
        if phase == "query_message":
            bound = kwargs["injected_job_kwargs"]
            await env.runtime.get_chat_messages(round_id=bound["pa_round_id"])
            await env.runtime.send_chat_message(
                content,
                kind="normal",
                round_id=bound["pa_round_id"],
                session_id=bound["session_id"],
            )
        return {"session_id": "agent-hints", "status": "completed", "result": f"completed {phase}"}

    env.agent.reply = AsyncMock(side_effect=reply)
    return calls


def hint_positions(env, calls):
    step = ProactiveAgentStep(app_context=env.context)
    directions = [
        step.get_prompt(f"hint_{period}_{index}")
        for period in ("morning", "midday", "evening")
        for index in range(1, 7)
    ]
    return [index for index, query in enumerate(calls) if any(direction in query for direction in directions)]


def test_hints_default_on_and_probability_is_explicit():
    settings = config()
    assert settings.hints_enabled is True
    assert settings.hint_probabilities.model_dump() == {"morning": 1.0, "midday": 0.5, "evening": 0.5}
    assert config(hints_enabled=False).hints_enabled is False
    assert config(hint_probabilities={"midday": 0.25}).hint_probabilities.model_dump() == {
        "morning": 1.0,
        "midday": 0.25,
        "evening": 0.5,
    }


@pytest.mark.parametrize("value", [0, 1, "false", None])
def test_hint_switch_requires_a_boolean(value):
    with pytest.raises(ValueError):
        config(hints_enabled=value)


@pytest.mark.parametrize("value", [-0.1, 1.1, True, "0.5", None, float("nan"), float("inf")])
@pytest.mark.parametrize("period", ["morning", "midday", "evening"])
def test_probability_rejects_invalid_values(period, value):
    with pytest.raises(ValueError):
        config(hint_probabilities={period: value})


@pytest.mark.asyncio
@pytest.mark.parametrize("hour, minute, expected", [(9, 0, True), (12, 0, False), (22, 30, False)])
async def test_default_probability_depends_on_period(pa_workspace, monkeypatch, hour, minute, expected):
    env = pa_workspace
    env.now[0] = env.now[0].replace(hour=hour, minute=minute)
    monkeypatch.setattr("reme.components._pa_hints.random.random", lambda: 0.75)
    assert (await sample(env) is not None) is expected


@pytest.mark.asyncio
async def test_independent_probabilities_are_read_from_hot_config(pa_workspace, monkeypatch):
    env = pa_workspace
    settings = config(hint_probabilities={"morning": 0, "midday": 1, "evening": 0.25})
    env.runtime._write_json(env.runtime.proactive_config_path, settings.model_dump())
    monkeypatch.setattr("reme.components._pa_hints.random.random", lambda: 0.5)
    for hour, minute, expected in [(9, 0, False), (12, 0, True), (22, 30, False)]:
        env.now[0] = env.now[0].replace(hour=hour, minute=minute)
        assert (await sample(env, await env.runtime.get_proactive_config()) is not None) is expected


@pytest.mark.parametrize(
    "clock, expected",
    [
        ("05:59", None),
        ("06:00", "morning"),
        ("11:59", "morning"),
        ("12:00", "midday"),
        ("13:59", "midday"),
        ("14:00", None),
        ("18:00", None),
        ("22:29", None),
        ("22:30", "evening"),
        ("22:59", "evening"),
        ("23:00", None),
    ],
)
def test_periods_follow_local_time_and_active_hours(clock, expected):
    now = datetime.fromisoformat(f"2026-11-08T{clock}:00+08:00")
    assert ProactiveHints._period(config(start_time="0600"), now) == expected


@pytest.mark.parametrize(
    "start, end, clock, expected",
    [
        ("0900", "1900", "18:29", None),
        ("0900", "1900", "18:30", "evening"),
        ("2100", "0700", "21:00", "evening"),
        ("2100", "0700", "00:30", None),
        ("2100", "0700", "06:45", "morning"),
        ("0000", "0000", "21:00", "evening"),
        ("0000", "0000", "03:00", None),
    ],
)
def test_custom_and_overnight_hours_never_put_evening_hints_in_morning(start, end, clock, expected):
    now = datetime.fromisoformat(f"2026-11-08T{clock}:00+08:00")
    assert ProactiveHints._period(config(start_time=start, end_time=end), now) == expected


@pytest.mark.asyncio
async def test_miss_consumes_slot_across_wakeups_and_runtime_restart(pa_workspace, monkeypatch):
    env = pa_workspace
    draw = Mock(return_value=0.75)
    monkeypatch.setattr("reme.components._pa_hints.random.random", draw)
    assert await sample(env) is None
    path = env.runtime._metadata_file("hint_state.json")
    original = path.read_bytes()
    assert json.loads(original)["draws"]["midday"]["hint_index"] is None
    env.now[0] += timedelta(minutes=5)
    assert await sample(env) is None
    env.runtime = bind_pa(env.context, env.agent)
    env.runtime.session_id = "session-1"
    assert await sample(env) is None
    draw.assert_called_once_with()
    assert path.read_bytes() == original


@pytest.mark.asyncio
async def test_next_day_draw_avoids_previous_variant_and_periods_are_independent(pa_workspace, monkeypatch):
    env = pa_workspace
    monkeypatch.setattr("reme.components._pa_hints.random.choice", lambda choices: choices[0])
    settings = config(hint_probabilities={"midday": 1, "evening": 1})
    assert await sample(env, settings) == "hint_midday_1"
    env.now[0] += timedelta(days=1)
    assert await sample(env, settings) == "hint_midday_2"
    env.now[0] = env.now[0].replace(hour=22, minute=30)
    assert await sample(env, settings) == "hint_evening_1"


@pytest.mark.asyncio
async def test_concurrent_wakeups_cannot_sample_the_same_period_twice(pa_workspace, monkeypatch):
    draw = Mock(return_value=0)
    monkeypatch.setattr("reme.components._pa_hints.random.random", draw)
    results = await asyncio.gather(*(sample(pa_workspace) for _ in range(8)))
    assert sum(result is not None for result in results) == 1
    draw.assert_called_once_with()


@pytest.mark.asyncio
async def test_disabled_does_not_read_state_draw_randomness_or_consume_slot(pa_workspace, monkeypatch):
    env = pa_workspace
    path = env.runtime._metadata_file("hint_state.json")
    path.write_text("invalid JSON deliberately preserved")
    draw = Mock(side_effect=AssertionError("disabled hints must not sample"))
    monkeypatch.setattr("reme.components._pa_hints.random.random", draw)
    assert await sample(env, config(hints_enabled=False)) is None
    assert path.read_text() == "invalid JSON deliberately preserved"
    draw.assert_not_called()


@pytest.mark.asyncio
async def test_disabling_then_reenabling_preserves_the_consumed_slot(pa_workspace, monkeypatch):
    env = pa_workspace
    draw = Mock(return_value=0)
    monkeypatch.setattr("reme.components._pa_hints.random.random", draw)
    assert await sample(env) is not None
    assert await sample(env, config(hints_enabled=False)) is None
    assert await sample(env) is None
    draw.assert_called_once_with()


@pytest.mark.asyncio
async def test_no_current_or_superseded_chat_does_not_consume_hint(pa_workspace, monkeypatch):
    env = pa_workspace
    draw = Mock(side_effect=AssertionError("no target chat"))
    monkeypatch.setattr("reme.components._pa_hints.random.random", draw)
    env.runtime.session_id = None
    assert await sample(env) is None
    env.runtime.session_id = "session-1"
    async with env.runtime.proactive_round():
        env.runtime.session_id = "session-2"
        assert await env.runtime.hints.select(config(), env.now[0]) is None
    assert not env.runtime._metadata_file("hint_state.json").exists()
    draw.assert_not_called()


@pytest.mark.asyncio
async def test_hint_is_only_in_decision_query_and_model_can_still_choose_silence(pa_workspace, monkeypatch):
    env = pa_workspace
    env.runtime._write_json(
        env.runtime.proactive_config_path,
        config(hint_probabilities={"midday": 1, "evening": 1}).model_dump(),
    )
    monkeypatch.setattr("reme.components._pa_hints.random.choice", lambda choices: choices[0])
    calls = model(env, monkeypatch)
    response = await ProactiveAgentStep(app_context=env.context)()
    assert len(calls) == 7
    assert hint_positions(env, calls) == [4]
    hint = ProactiveAgentStep(app_context=env.context).get_prompt(response.metadata["proactive_hint"])
    assert "午间招呼" in hint and "早安" not in hint and "晚安" not in hint
    assert response.metadata["proactive_hint"] == "hint_midday_1"
    assert response.answer["status"] == "skipped"
    env.agent.append_session.assert_not_awaited()


@pytest.mark.asyncio
async def test_disabled_round_still_delivers_normal_information_and_reads_hot_config(pa_workspace, monkeypatch):
    env = pa_workspace
    settings = config(hints_enabled=False, hint_probabilities={"midday": 1})
    env.runtime._write_json(env.runtime.proactive_config_path, settings.model_dump())
    calls = model(env, monkeypatch, content="正常的新发现")
    first = await ProactiveAgentStep(app_context=env.context)()
    assert first.answer["status"] == "pending" and hint_positions(env, calls) == []
    env.agent.append_session.assert_not_awaited()
    env.runtime.reminders.idle = ("turn-1", asyncio.get_running_loop().time() - 11)
    await env.runtime.deliver_pending_reminder()
    env.agent.append_session.assert_awaited_once()
    assert not env.runtime._metadata_file("hint_state.json").exists()
    env.runtime._write_json(
        env.runtime.proactive_config_path,
        config(hint_probabilities={"midday": 1, "evening": 1}).model_dump(),
    )
    second = await ProactiveAgentStep(app_context=env.context)()
    assert "proactive_hint" in second.metadata and hint_positions(env, calls) == [11]


@pytest.mark.asyncio
async def test_hint_uses_time_after_tasks_finish_not_round_start(pa_workspace, monkeypatch):
    env = pa_workspace
    env.now[0] = env.now[0].replace(hour=11, minute=59)
    env.runtime._write_json(
        env.runtime.proactive_config_path,
        config(hint_probabilities={"midday": 1, "evening": 1}).model_dump(),
    )
    calls = model(env, monkeypatch, advance=env.now[0].replace(hour=12, minute=1))
    response = await ProactiveAgentStep(app_context=env.context)()
    assert response.metadata["proactive_hint"].startswith("hint_midday_")
    assert "12:01:00" in calls[4]


@pytest.mark.asyncio
@pytest.mark.parametrize("content", ["{", "[]", "x" * 4097])
async def test_invalid_hint_state_is_preserved_without_blocking_normal_delivery(pa_workspace, monkeypatch, content):
    env = pa_workspace
    path = env.runtime._metadata_file("hint_state.json")
    path.write_text(content)
    calls = model(env, monkeypatch, content="仍然传递有用信息")
    response = await ProactiveAgentStep(app_context=env.context)()
    assert response.answer["status"] == "pending" and "proactive_hint" not in response.metadata
    env.agent.append_session.assert_not_awaited()
    env.runtime.reminders.idle = ("turn-1", asyncio.get_running_loop().time() - 11)
    await env.runtime.deliver_pending_reminder()
    env.agent.append_session.assert_awaited_once()
    assert hint_positions(env, calls) == []
    assert path.read_text() == content


@pytest.mark.parametrize("language", ["en", "zh"])
def test_all_time_specific_hints_are_localized_with_one_evening_todo_direction(pa_workspace, language):
    step = ProactiveAgentStep(app_context=pa_workspace.context, language=language)
    todo_hints = []
    for period in ("morning", "midday", "evening"):
        for index in range(1, 7):
            key = f"hint_{period}_{index}"
            hint = step.get_prompt(key)
            assert hint
            if "todo" in hint.lower():
                todo_hints.append(key)
    assert todo_hints == ["hint_evening_2"]
