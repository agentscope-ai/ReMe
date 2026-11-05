"""Proactive rounds at mocked model boundaries, with real workspace files and jobs."""

# pylint: disable=missing-function-docstring,protected-access,redefined-outer-name

import asyncio
from datetime import datetime
import json
from pathlib import Path
from unittest.mock import AsyncMock
from zoneinfo import ZoneInfo

import frontmatter
import pytest

from reme.components.agent_wrapper import CodexAgentWrapper
from reme.components.application_context import ApplicationContext
from reme.components.file_catalog import LocalFileCatalog
from reme.components.job import BackgroundJob, BaseJob
from reme.config import resolve_app_config
from reme.schema import Response, SessionMessage, SessionSnapshot, SessionStatus
from reme.steps.proactive import FindInterestStep, PaLoopStep, PushInterestStep, SetNextWakeupStep
from reme.steps.proactive._interest import _InterestStep
from reme.steps.proactive.loop import WAKEUP_KEY


@pytest.fixture
def runtime(tmp_path, monkeypatch):
    context = ApplicationContext(workspace_dir=str(tmp_path), daily_dir="notes", language="zh")
    agent = CodexAgentWrapper(app_context=context, app_server="")
    agent.reply = AsyncMock(return_value={"status": "completed"})
    snapshot = SessionSnapshot(
        session_id="session-1",
        state=SessionStatus(session_id="session-1", status="idle"),
        path=str(tmp_path / ".codex/sessions/current.jsonl"),
        last_turn_id="turn-1",
    )
    agent.read_session = AsyncMock(return_value=snapshot)
    agent.append_session = AsyncMock(
        return_value=SessionMessage(
            id="delivered",
            role="assistant",
            content="提醒",
            created_at="2026-11-05T14:30:13+08:00",
        ),
    )
    context.components["agent_wrapper"] = {"default": agent}
    context.metadata["pa"] = {"session_id": "session-1", "lock": asyncio.Lock()}
    memory = tmp_path / "personal/REME_MEMORY.md"
    memory.parent.mkdir()
    memory.write_text("用户关注 ReMe 和记忆研究。", encoding="utf-8")
    instant = datetime(2026, 11, 5, 14, 30, 12, tzinfo=ZoneInfo("Asia/Shanghai"))
    monkeypatch.setattr(_InterestStep, "_now", lambda self: instant)
    return context, agent


def note(context, name="2026-11-04/interest_090000.md", **metadata):
    path = Path(context.app_config.workspace_dir) / context.app_config.daily_dir / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(frontmatter.dumps(frontmatter.Post("值得关注的研究发现。", **metadata)), encoding="utf-8")
    return path


def change(path, **metadata):
    post = frontmatter.loads(path.read_text(encoding="utf-8"))
    post.metadata.update(metadata)
    path.write_text(frontmatter.dumps(post), encoding="utf-8")


@pytest.mark.asyncio
@pytest.mark.parametrize("step_cls", [FindInterestStep, PushInterestStep])
async def test_missing_memory_fails_before_model_or_ack(runtime, step_cls):
    context, agent = runtime
    (Path(context.app_config.workspace_dir) / "personal/REME_MEMORY.md").unlink()
    with pytest.raises(FileNotFoundError):
        await step_cls(app_context=context)()
    agent.reply.assert_not_awaited()
    assert not (Path(context.app_config.workspace_dir) / "metadata").exists()


@pytest.mark.asyncio
async def test_find_agent_acknowledges_after_saving_and_code_only_fills_metadata(runtime):
    context, agent = runtime
    workspace = Path(context.app_config.workspace_dir)
    catalog = LocalFileCatalog(
        app_context=context,
        watch_dirs=["resource_dir"],
        watch_filenames=["metadata.json"],
        ttl_days=7,
    )
    context.components["file_catalog"] = {"resource": catalog}
    resource = workspace / "resource/2026-11-05/package/metadata.json"
    resource.parent.mkdir(parents=True)
    resource.write_text('{"description": "memory research", "files": []}', encoding="utf-8")
    config = resolve_app_config(config="proactive")
    for name in ("resource_changes", "resource_processed"):
        job = BaseJob(name=name, app_context=context, **config["jobs"][name])
        context.jobs[name] = job
        await job.start()

    async def discover(inputs, **kwargs):
        details = json.loads(inputs)
        assert kwargs["job_tools"] == ["resource_changes", "resource_processed"]
        assert "resume" not in kwargs
        paths = (await context.jobs["resource_changes"]()).answer
        assert paths == [resource.relative_to(workspace).as_posix()]
        target = workspace / details["output_path"]
        target.parent.mkdir(parents=True)
        target.write_text(frontmatter.dumps(frontmatter.Post("研究发现。", sources=paths)), encoding="utf-8")
        assert (await context.jobs["resource_processed"](paths=paths)).success
        assert not frontmatter.loads(target.read_text()).get("status")
        return {"status": "completed"}

    agent.reply.side_effect = discover
    result = await FindInterestStep(app_context=context)()
    assert result.answer == {"created": ["notes/2026-11-05/interest_143012.md"]}
    post = frontmatter.loads((workspace / result.answer["created"][0]).read_text())
    assert post["status"] == "unsent" and post["sources"] == [resource.relative_to(workspace).as_posix()]
    assert post["created_at"] == post["updated_at"] == "2026-11-05T14:30:12+08:00"
    assert await catalog.changes() == []
    assert all(job.is_started for job in context.jobs.values())
    for job in context.jobs.values():
        await job.close()


@pytest.mark.asyncio
async def test_no_findings_is_a_valid_noop_and_existing_filename_is_not_overwritten(runtime):
    context, agent = runtime
    existing = note(context, "2026-11-05/interest_143012.md", status="sent")
    await FindInterestStep(app_context=context)()
    details = json.loads(agent.reply.call_args.args[0])
    assert details["output_path"] == "notes/2026-11-05/interest_143012_1.md"
    assert frontmatter.loads(existing.read_text())["status"] == "sent"
    assert not existing.with_name("interest_143012_1.md").exists()


@pytest.mark.asyncio
async def test_interrupted_note_initialization_is_completed_before_delivery(runtime):
    context, agent = runtime
    path = note(context)

    async def choose(inputs, **kwargs):
        details = json.loads(inputs)
        assert details["current_session_file"].endswith(".codex/sessions/current.jsonl")
        assert details["last_turn_id"] == "turn-1"
        assert details["interests"][0]["status"] == "unsent"
        assert kwargs["job_tools"] == ["set_next_wakeup"] and "resume" not in kwargs
        change(path, status="ready", message="主人，有一项相关研究值得看看。")
        await SetNextWakeupStep(app_context=context)(minutes=12)
        return {"status": "completed"}

    agent.reply.side_effect = choose
    result = await PushInterestStep(app_context=context)()
    assert result.answer == {"sent": True}
    post = frontmatter.loads(path.read_text())
    assert post["status"] == "sent" and post["target_session_id"] == "session-1"
    assert post["context_turn_id"] == "turn-1" and post["sent_at"]
    assert context.metadata[WAKEUP_KEY] == 12
    agent.append_session.assert_awaited_once_with(
        "session-1",
        post["message"],
        message_id=post["message_id"],
        timeout=1,
        expected_turn_id="turn-1",
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("status", ["idle", "active", "error", "not_loaded"])
async def test_delivery_respects_activity_and_restores_unloaded_sessions_through_owner(runtime, status):
    context, agent = runtime
    path = note(context)
    agent.read_session.return_value.state.status = status

    async def choose(*_args, **_kwargs):
        change(path, status="ready", message="提醒")
        return {"status": "completed"}

    agent.reply.side_effect = choose
    await PushInterestStep(app_context=context)()
    expected = "sent" if status in {"idle", "not_loaded"} else "unsent"
    assert frontmatter.loads(path.read_text())["status"] == expected
    assert agent.append_session.await_count == (expected == "sent")


@pytest.mark.asyncio
async def test_switching_session_during_decision_defers_delivery(runtime):
    context, agent = runtime
    path = note(context)

    async def choose(*_args, **_kwargs):
        change(path, status="ready", message="提醒")
        context.metadata["pa"]["session_id"] = "session-2"
        return {"status": "completed"}

    agent.reply.side_effect = choose
    await PushInterestStep(app_context=context)()
    assert frontmatter.loads(path.read_text())["status"] == "unsent"
    agent.append_session.assert_not_awaited()


@pytest.mark.asyncio
async def test_only_one_ready_and_terminal_notes_are_enforced(runtime):
    context, agent = runtime
    first, second = note(context), note(context, "2026-11-05/interest_100000.md")

    async def choose(*_args, **_kwargs):
        for path in (first, second):
            change(path, status="ready", message="提醒")
        return {"status": "completed"}

    agent.reply.side_effect = choose
    with pytest.raises(ValueError, match="Only one"):
        await PushInterestStep(app_context=context)()
    agent.append_session.assert_not_awaited()
    change(first, status="sent")
    change(second, status="unsent")
    with pytest.raises(ValueError, match="terminal"):
        await PushInterestStep(app_context=context)()


@pytest.mark.asyncio
async def test_failed_delivery_is_recovered_before_regeneration_even_after_session_switch(runtime):
    context, agent = runtime
    path = note(context)

    async def choose(*_args, **_kwargs):
        change(path, status="ready", message="提醒")
        return {"status": "completed"}

    agent.reply.side_effect = choose
    agent.append_session.side_effect = TimeoutError("publication interrupted")
    with pytest.raises(TimeoutError):
        await PushInterestStep(app_context=context)()
    pending = frontmatter.loads(path.read_text())
    assert pending["status"] == "ready" and pending["last_error"] and pending["delivery_attempted_at"]
    agent.read_session.return_value.messages = [
        SessionMessage(
            id=f"reme-proactive-{pending['message_id']}",
            role="assistant",
            content="提醒",
            source="proactive",
            created_at="2026-11-05T14:30:13+08:00",
        ),
    ]
    context.metadata["pa"]["session_id"] = "session-2"
    await PushInterestStep(app_context=context)()
    assert frontmatter.loads(path.read_text())["status"] == "sent"
    assert agent.reply.await_count == agent.append_session.await_count == 1


@pytest.mark.asyncio
async def test_ready_retry_reuses_target_id_and_message(runtime):
    context, agent = runtime
    path = note(
        context,
        status="ready",
        id="stable",
        message_id="delivery-1",
        message="同一条提醒",
        target_session_id="session-1",
        context_turn_id="turn-1",
        delivery_attempted_at="earlier",
    )
    await PushInterestStep(app_context=context)()
    agent.reply.assert_not_awaited()
    assert frontmatter.loads(path.read_text())["status"] == "sent"
    assert agent.append_session.call_args.kwargs["message_id"] == "delivery-1"
    assert agent.append_session.call_args.args == ("session-1", "同一条提醒")


@pytest.mark.asyncio
async def test_stale_delivery_is_replanned_with_a_new_message_id(runtime):
    from reme.components.codex_app_server import SessionChangedError

    context, agent = runtime
    path = note(context)

    async def choose(*_args, **_kwargs):
        change(path, status="ready", message=f"提醒 {agent.reply.await_count}")
        return {"status": "completed"}

    agent.reply.side_effect = choose
    agent.append_session.side_effect = SessionChangedError("new user turn")
    await PushInterestStep(app_context=context)()
    first = frontmatter.loads(path.read_text())
    assert first["status"] == "unsent" and not first["delivery_attempted_at"]
    agent.append_session.side_effect = None
    await PushInterestStep(app_context=context)()
    second = frontmatter.loads(path.read_text())
    assert second["status"] == "sent"
    assert first["message_id"] != second["message_id"]
    assert first["message"] != second["message"]


@pytest.mark.asyncio
async def test_no_session_or_candidates_skips_the_push_agent(runtime):
    context, agent = runtime
    await PushInterestStep(app_context=context)()
    note(context)
    context.metadata["pa"]["session_id"] = None
    await PushInterestStep(app_context=context)()
    agent.reply.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("mutation", ["edit", "remove", "extra"])
async def test_find_preserves_existing_notes_and_only_creates_its_output_path(runtime, mutation):
    context, agent = runtime
    existing = note(context)

    async def discover(*_args, **_kwargs):
        if mutation == "edit":
            existing.write_text("Replaced existing findings.")
        elif mutation == "remove":
            existing.unlink()
        else:
            note(context, "2026-11-05/interest_120000.md")
        return {"status": "completed"}

    agent.reply.side_effect = discover
    with pytest.raises(ValueError, match="Find agent"):
        await FindInterestStep(app_context=context)()


@pytest.mark.asyncio
@pytest.mark.parametrize("mutation", ["metadata", "extra"])
async def test_push_rejects_added_fields_and_new_notes_before_delivery(runtime, mutation):
    context, agent = runtime
    existing = note(context)

    async def choose(*_args, **_kwargs):
        change(existing, status="ready", message="提醒")
        if mutation == "metadata":
            change(existing, target_session_id="agent-selected-session")
        else:
            note(context, "2026-11-05/interest_120000.md")
        return {"status": "completed"}

    agent.reply.side_effect = choose
    with pytest.raises(ValueError, match="Push agent"):
        await PushInterestStep(app_context=context)()
    agent.append_session.assert_not_awaited()


@pytest.mark.asyncio
async def test_delivery_recovery_preserves_concurrent_note_edits(runtime):
    context, agent = runtime
    path = note(
        context,
        status="ready",
        message_id="delivery-1",
        message="原来的提醒",
        target_session_id="session-1",
        context_turn_id="turn-1",
        delivery_attempted_at="earlier",
    )
    snapshot = agent.read_session.return_value
    snapshot.messages = [
        SessionMessage(id="reme-proactive-delivery-1", role="assistant", content="原来的提醒", source="proactive"),
    ]

    async def recover(_session_id):
        change(path, message="用户修改后的提醒")
        return snapshot

    agent.read_session.side_effect = recover
    with pytest.raises(ValueError, match="Interest changed"):
        await PushInterestStep(app_context=context)()
    current = frontmatter.loads(path.read_text())
    assert current["status"] == "ready" and current["message"] == "用户修改后的提醒"
    agent.append_session.assert_not_awaited()


@pytest.mark.asyncio
async def test_round_runs_both_roles_and_revisits_old_candidates_without_new_files(runtime):
    context, agent = runtime
    path = note(context)
    calls = []

    async def respond(inputs, **kwargs):
        details = json.loads(inputs)
        calls.append(kwargs["job_tools"])
        if "interests" in details:
            change(path, status="ready", message="主人，这篇历史候选现在值得看看。")
        return {"status": "completed"}

    agent.reply.side_effect = respond
    job = BaseJob(
        name="pa_proactive",
        app_context=context,
        **resolve_app_config(config="proactive")["jobs"]["pa_proactive"],
    )
    context.jobs[job.name] = job
    async with job:
        result = await job(stop_event=asyncio.Event())
    assert result.success and result.answer == {"sent": True}
    assert calls == [["resource_changes", "resource_processed"], ["set_next_wakeup"]]
    assert frontmatter.loads(path.read_text())["status"] == "sent"
    assert context.job_history[job.name][0].status == "succeeded"


@pytest.mark.asyncio
async def test_interest_scan_ignores_escape_symlinks_and_invalid_dates(runtime, tmp_path):
    context, _ = runtime
    outside = tmp_path.parent / f"{tmp_path.name}-outside.md"
    outside.write_text("outside")
    target = note(context)
    target.unlink()
    target.symlink_to(outside)
    note(context, "2026-99-99/interest_143012.md")
    assert await FindInterestStep(app_context=context)._notes() == {}
    assert outside.read_text() == "outside"


@pytest.mark.asyncio
@pytest.mark.parametrize("minutes", [0, -1, True, "5", None, float("inf"), float("nan")])
async def test_wakeup_rejects_invalid_delays(runtime, minutes):
    context, _ = runtime
    with pytest.raises(ValueError, match="positive finite"):
        await SetNextWakeupStep(app_context=context)(minutes=minutes)
    assert WAKEUP_KEY not in context.metadata


@pytest.mark.asyncio
async def test_loop_consumes_override_once_and_resets_on_failure_and_restart(runtime, monkeypatch):
    context, _ = runtime
    context.metadata[WAKEUP_KEY] = 99
    stop, delays = asyncio.Event(), []
    step = PaLoopStep(app_context=context)

    async def wait(waiter, *, timeout):
        waiter.close()
        delays.append(timeout)
        if len(delays) == 4:
            stop.set()
        else:
            raise asyncio.TimeoutError

    async def run(*_args, **kwargs):
        assert kwargs["stop_event"] is stop
        if len(delays) == 1:
            await SetNextWakeupStep(app_context=context)(minutes=2)
        if len(delays) == 3:
            await SetNextWakeupStep(app_context=context)(minutes=20)
            return Response(success=False)
        return Response()

    step.run_job = AsyncMock(side_effect=run)
    monkeypatch.setattr("reme.steps.proactive.loop.asyncio.wait_for", wait)
    await step(stop_event=stop)
    assert delays == [300, 120, 300, 300]
    assert step.run_job.await_count == 3 and WAKEUP_KEY not in context.metadata


@pytest.mark.asyncio
async def test_background_close_interrupts_initial_wait(runtime):
    context, _ = runtime
    job = BackgroundJob(name="pa_loop", app_context=context, steps=[{"backend": "pa_loop_step"}])
    context.jobs[job.name] = job
    await job.start()
    await asyncio.sleep(0)
    await asyncio.wait_for(job.close(), timeout=1)
    assert WAKEUP_KEY not in context.metadata
    assert "pa_proactive" not in context.job_history


def test_proactive_jobs_are_registered_and_background_tools_stay_private():
    jobs = resolve_app_config(config="proactive")["jobs"]
    assert jobs["pa_loop"]["backend"] == "background"
    assert not jobs["pa_proactive"]["enable_serve"]
    assert not jobs["set_next_wakeup"]["enable_serve"]
    assert jobs["pa_proactive"]["track_execution"]
