"""Daily special-message limits, final evening delivery and token-free scheduling gates."""

# pylint: disable=missing-function-docstring,protected-access

import asyncio
from datetime import datetime, timedelta
import json
from unittest.mock import AsyncMock

import pytest

from reme.schema.pa_proactive import PaProactiveConfig
from reme.steps.proactive import ProactiveAgentStep, CronAgentStep
from scripts.migrate_pa_reminder_kinds import migrate

from .pa_helpers import PA_CONFIG, bind_pa
from .test_pa_cron import stub_model, task


def settings(env, **overrides):
    config = PaProactiveConfig(
        **{**PA_CONFIG, "hint_probabilities": {"morning": 1, "midday": 1, "evening": 1}, **overrides},
    )
    env.runtime._write_json(env.runtime.proactive_config_path, config.model_dump())
    return config


async def send(env, content="新发现。", kind="normal", sample=False):
    async with env.runtime.proactive_round() as current:
        current.phase = "query_message"
        await env.runtime.get_chat_messages(round_id=current.id)
        if sample:
            await env.runtime.hints.select(await env.runtime.get_proactive_config(), env.now[0])
        return await env.runtime.send_chat_message(content, kind, round_id=current.id, session_id=current.session_id)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "hour,minute,kind,text",
    [(9, 0, "morning", "早安。"), (12, 0, "midday", "午安。"), (22, 30, "evening", "晚安。")],
)
async def test_special_messages_are_limited_across_rounds_and_restart(pa_workspace, hour, minute, kind, text):
    env = pa_workspace
    env.now[0] = env.now[0].replace(hour=hour, minute=minute)
    settings(env)
    first = await send(env, text, kind, sample=True)
    assert first["status"] == "pending" and first["kind"] == kind
    env.agent.append_session.assert_not_awaited()
    env.runtime.reminders.idle = ("turn-1", asyncio.get_running_loop().time() - 11)
    await env.runtime.deliver_pending_reminder()
    assert (await env.runtime.get_chat_messages())[0]["status"] == "sent"
    env.runtime = bind_pa(env.context, env.agent)
    env.runtime._now = lambda: env.now[0]
    env.runtime._initialize_files()
    env.runtime.session_id = "session-2"
    second = await send(env, text, kind)
    assert second["reason"] == ("day_closed" if kind == "evening" else "period_used")
    env.agent.append_session.assert_awaited_once()


@pytest.mark.asyncio
async def test_evening_pending_is_reserved_and_only_the_closing_message_is_allowed(pa_workspace):
    env = pa_workspace
    env.now[0] = env.now[0].replace(hour=22, minute=30)
    settings(env)
    env.snapshot.state.status = "active"
    closing = await send(env, "愿今晚轻松。", "evening", sample=True)
    assert closing["status"] == "pending"
    assert (await send(env))["reason"] == "day_closed"
    window = await env.runtime.reminders.window(await env.runtime.get_proactive_config())
    assert not window.allows(env.now[0])
    assert window.allows(env.now[0], f"reme-proactive-{closing['id']}")
    env.snapshot.state.status = "idle"
    await env.runtime.deliver_pending_reminder()
    env.runtime.reminders.idle = ("turn-1", asyncio.get_running_loop().time() - 11)
    await env.runtime.deliver_pending_reminder()
    assert (await env.runtime.get_chat_messages())[0]["status"] == "sent"
    env.agent.append_session.assert_awaited_once()


@pytest.mark.asyncio
async def test_expired_closing_never_reopens_the_day_after_uncertain_delivery(pa_workspace):
    env = pa_workspace
    env.now[0] = env.now[0].replace(hour=22, minute=30)
    settings(env)
    env.agent.append_session = AsyncMock(side_effect=TimeoutError("delivery acknowledgement lost"))
    assert (await send(env, "晚安。", "evening", sample=True))["status"] == "pending"
    env.now[0] += timedelta(minutes=6)
    await env.runtime.deliver_pending_reminder()
    assert (await env.runtime.get_chat_messages())[0]["status"] == "discarded"
    assert await env.runtime.proactive_skip_reason() == "day_closed"
    assert (await send(env))["reason"] == "day_closed"


@pytest.mark.asyncio
async def test_new_opening_releases_evening_close_and_old_messages_expire(pa_workspace):
    env = pa_workspace
    env.now[0] = env.now[0].replace(hour=22, minute=59, second=59)
    settings(env)
    closing = await send(env, "晚安。", "evening", sample=True)
    assert closing["expires_at"] == "2026-11-08T23:00:00+08:00"
    assert await env.runtime.proactive_skip_reason() == "day_closed"
    env.now[0] = env.now[0].replace(hour=23)
    assert await env.runtime.proactive_skip_reason() == "outside_active_hours"
    env.now[0] = (env.now[0] + timedelta(days=1)).replace(hour=9, minute=0, second=0)
    assert await env.runtime.proactive_skip_reason() is None
    assert (await send(env))["status"] == "pending"
    await env.runtime.deliver_pending_reminder()
    env.runtime.reminders.idle = ("turn-1", asyncio.get_running_loop().time() - 11)
    await env.runtime.deliver_pending_reminder()
    assert (await env.runtime.get_chat_messages())[0]["status"] == "sent"


@pytest.mark.asyncio
async def test_special_kind_requires_sample_and_greetings_cannot_bypass_it(pa_workspace):
    env = pa_workspace
    settings(env)
    for kind, content in [("normal", "主人，晚安。"), ("normal", "Good morning!"), ("midday", "午安。")]:
        with pytest.raises(ValueError):
            await send(env, content, kind)
    assert not env.runtime.chat_inbox_path.read_text()


@pytest.mark.asyncio
async def test_concurrent_special_decisions_use_one_daily_slot(pa_workspace):
    env = pa_workspace
    settings(env)
    results = await asyncio.gather(*(send(env, "午安。", "midday", sample=True) for _ in range(8)))
    assert sum(result["status"] == "pending" for result in results) == 1
    env.agent.append_session.assert_not_awaited()
    env.runtime.reminders.idle = ("turn-1", asyncio.get_running_loop().time() - 11)
    await env.runtime.deliver_pending_reminder()
    env.agent.append_session.assert_awaited_once()


@pytest.mark.asyncio
async def test_evening_round_stops_after_decision_without_more_model_queries(pa_workspace, monkeypatch):
    env = pa_workspace
    env.now[0] = env.now[0].replace(hour=22, minute=30)
    settings(env)
    monkeypatch.setattr(ProactiveAgentStep, "_now", lambda _self: env.now[0])
    calls = []

    async def reply(query, **kwargs):
        calls.append(query)
        if env.runtime.current_round.phase == "query_message":
            bound = kwargs["injected_job_kwargs"]
            await env.runtime.get_chat_messages(round_id=bound["pa_round_id"])
            await env.runtime.send_chat_message(
                "晚安。",
                "evening",
                round_id=bound["pa_round_id"],
                session_id=bound["session_id"],
            )
        return {"session_id": "proactive", "status": "completed", "result": "checkpoint"}

    env.agent.reply = AsyncMock(side_effect=reply)
    result = await ProactiveAgentStep(app_context=env.context)()
    assert result.answer["status"] == "pending" and result.answer["reason"] == "day_closed"
    assert len(calls) == 6 and env.runtime.proactive_context_path.read_text()
    assert (await ProactiveAgentStep(app_context=env.context)()).answer["reason"] == "day_closed"
    assert len(calls) == 6
    env.runtime.reminders.idle = ("turn-1", asyncio.get_running_loop().time() - 11)
    await env.runtime.deliver_pending_reminder()
    assert (await env.runtime.get_chat_messages())[0]["status"] == "sent"


@pytest.mark.asyncio
async def test_round_does_not_start_another_query_after_deadline(pa_workspace, monkeypatch):
    env = pa_workspace
    env.now[0] = env.now[0].replace(hour=22, minute=59)
    monkeypatch.setattr(ProactiveAgentStep, "_now", lambda _self: env.now[0])

    async def reply(*_args, **_kwargs):
        env.now[0] = env.now[0].replace(hour=23)
        return {"session_id": "proactive", "status": "completed", "result": "retained"}

    env.agent.reply = AsyncMock(side_effect=reply)
    result = await ProactiveAgentStep(app_context=env.context)()
    assert result.answer["reason"] == "outside_active_hours"
    env.agent.reply.assert_awaited_once()
    assert "retained" in env.runtime.proactive_context_path.read_text()


@pytest.mark.asyncio
async def test_slow_hint_sampling_cannot_start_a_query_after_deadline(pa_workspace, monkeypatch):
    env = pa_workspace
    env.now[0] = env.now[0].replace(hour=22, minute=59)
    settings(env)
    monkeypatch.setattr(ProactiveAgentStep, "_now", lambda _self: env.now[0])

    async def sample(*_args):
        env.now[0] = env.now[0].replace(hour=23)
        return None

    env.agent.reply = AsyncMock(return_value={"session_id": "proactive", "status": "completed", "result": "retained"})
    monkeypatch.setattr(env.runtime.hints, "select", sample)
    result = await ProactiveAgentStep(app_context=env.context)()
    assert result.answer["reason"] == "outside_active_hours"
    assert env.agent.reply.await_count == 5


@pytest.mark.asyncio
@pytest.mark.parametrize("role", ["proactive", "task"])
async def test_active_query_is_cancelled_at_the_window_deadline(pa_workspace, monkeypatch, role):
    env = pa_workspace
    env.now[0] = env.now[0].replace(hour=22, minute=59, second=59, microsecond=980000)
    step = ProactiveAgentStep if role == "proactive" else CronAgentStep
    monkeypatch.setattr(step, "_now", lambda _self: env.now[0])
    stopped = asyncio.Event()

    async def reply(*_args, **_kwargs):
        try:
            await asyncio.Event().wait()
        finally:
            stopped.set()

    env.agent.reply = AsyncMock(side_effect=reply)
    result = await step(app_context=env.context)(**({"task": task()} if role == "task" else {}))
    assert result.answer["reason"] == "outside_active_hours" and stopped.is_set()
    env.agent.reply.assert_awaited_once()


@pytest.mark.parametrize(
    "clock,start,end,expected",
    [
        ("2026-11-09T00:30:00+08:00", "2100", "0700", ("2026-11-08T21:00:00+08:00", "2026-11-09T07:00:00+08:00")),
        ("2026-11-08T09:00:00+08:00", "0900", "2300", ("2026-11-08T09:00:00+08:00", "2026-11-08T23:00:00+08:00")),
        ("2026-11-08T09:00:00+08:00", "0900", "0900", ("2026-11-08T09:00:00+08:00", "2026-11-09T09:00:00+08:00")),
    ],
)
def test_window_deadlines_follow_local_openings(clock, start, end, expected):
    config = PaProactiveConfig(**{**PA_CONFIG, "start_time": start, "end_time": end})
    assert tuple(value.isoformat() for value in config.active_window(datetime.fromisoformat(clock))) == expected


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "hour,minute,follow,expected",
    [(8, 59, True, 0), (9, 0, True, 4), (22, 59, True, 4), (23, 0, True, 0), (23, 0, False, 4)],
)
async def test_tasks_follow_hours_by_default_with_explicit_opt_out(
    pa_workspace,
    monkeypatch,
    hour,
    minute,
    follow,
    expected,
):
    env = pa_workspace
    env.now[0] = env.now[0].replace(hour=hour, minute=minute)
    settings(env, tasks_follow_active_hours=follow)
    stub_model(env, monkeypatch)
    result = await CronAgentStep(app_context=env.context)(task=task())
    assert env.agent.reply.await_count == expected
    if not expected:
        assert result.answer["reason"] == "outside_active_hours"
        assert not env.runtime.cron_runs_path.read_text()


@pytest.mark.asyncio
async def test_task_rechecks_hot_config_before_summary_query(pa_workspace, monkeypatch):
    env = pa_workspace
    stub_model(env, monkeypatch)
    original = env.agent.reply.side_effect

    async def reply(*args, **kwargs):
        result = await original(*args, **kwargs)
        settings(env, end_time="1200")
        return result

    env.agent.reply.side_effect = reply
    result = await CronAgentStep(app_context=env.context)(task=task())
    assert result.answer["status"] == "interrupted"
    env.agent.reply.assert_awaited_once()
    assert not env.runtime.tasks.paths("one")


@pytest.mark.parametrize("value", [None, 0, 1, "true", "false"])
def test_task_hours_switch_is_strict(value):
    with pytest.raises(ValueError):
        PaProactiveConfig(**{**PA_CONFIG, "tasks_follow_active_hours": value})


def test_task_hours_switch_defaults_on():
    assert PaProactiveConfig(**PA_CONFIG).tasks_follow_active_hours is True


@pytest.mark.asyncio
@pytest.mark.parametrize("role", ["proactive", "task"])
async def test_provider_timeout_is_not_mistaken_for_closing_time(pa_workspace, monkeypatch, role):
    env = pa_workspace
    step = ProactiveAgentStep if role == "proactive" else CronAgentStep
    monkeypatch.setattr(step, "_now", lambda _self: env.now[0])
    env.agent.reply = AsyncMock(side_effect=TimeoutError("provider timeout"))
    with pytest.raises(TimeoutError, match="provider timeout"):
        await step(app_context=env.context)(**({"task": task()} if role == "task" else {}))


def test_explicit_journal_migration_preserves_original_and_is_idempotent(pa_workspace):
    env = pa_workspace

    record = {
        "id": "old",
        "content": "历史提醒。",
        "status": "sent",
        "created_at": env.now[0].isoformat(),
        "updated_at": env.now[0].isoformat(),
        "expires_at": env.now[0].isoformat(),
    }
    original = (json.dumps(record, ensure_ascii=False) + "\n").encode()
    env.runtime.chat_inbox_path.write_bytes(original)
    assert migrate(env.runtime.chat_inbox_path) == 1
    assert env.runtime.reminders.read()[0].kind == "normal"
    assert next(env.runtime.chat_inbox_path.parent.glob("*.bak")).read_bytes() == original
    assert migrate(env.runtime.chat_inbox_path) == 0


def test_migration_validates_every_row_before_writing(pa_workspace):
    path = pa_workspace.runtime.chat_inbox_path
    path.write_text('{"id":"broken"}\n')
    original = path.read_bytes()
    with pytest.raises(ValueError):
        migrate(path)
    assert path.read_bytes() == original
    assert not list(path.parent.glob("*.bak"))


@pytest.mark.asyncio
async def test_migration_recovers_saved_hint_kind_without_changing_content(pa_workspace):
    env = pa_workspace
    settings(env)
    sent = await send(env, "午安。", "midday", sample=True)
    row = {key: value for key, value in sent.items() if key not in {"kind", "sent"}}
    env.runtime.chat_inbox_path.write_text(json.dumps(row, ensure_ascii=False) + "\n")
    assert migrate(env.runtime.chat_inbox_path) == 1
    restored = env.runtime.reminders.read()[0]
    assert restored.kind == "midday" and restored.content == "午安。"


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", [None, [], "unknown", "evening"])
async def test_empty_decisions_require_the_new_normal_kind(pa_workspace, kind):
    with pytest.raises(ValueError):
        await send(pa_workspace, "", kind)


@pytest.mark.asyncio
async def test_pausing_without_pending_message_publishes_closed_display_window(pa_workspace):
    env = pa_workspace
    await env.runtime.proactive_skip_reason()
    assert env.runtime.notifications.enabled is True and env.runtime.reminders.pending is None
    settings(env, enabled=False)
    assert await env.runtime.proactive_skip_reason() == "disabled"
    assert env.runtime.notifications.enabled is False
