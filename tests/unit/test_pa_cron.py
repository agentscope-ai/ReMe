"""Persistent cron sessions, one-query execution and recoverable result submission."""

# pylint: disable=missing-function-docstring,protected-access

import asyncio
import json
from datetime import timedelta
from unittest.mock import AsyncMock

import frontmatter
import pytest

from reme.components.job import BaseJob, CronJob
from reme.components.runtime_context import RuntimeContext
from reme.schema.pa_task import PaTask
from reme.steps.proactive import CronAgentStep, CronScheduleStep
from reme.steps.proactive.tools import ReadCronResultsStep


from .pa_helpers import configure, snapshot, stub_model, task, wait_until


@pytest.mark.asyncio
@pytest.mark.parametrize("language", ["en", "zh"])
async def test_persistent_session_gets_one_timed_query_per_run(pa_workspace, monkeypatch, language):
    env = pa_workspace
    previous = snapshot(env)
    captured = stub_model(env, monkeypatch)
    step = CronAgentStep(app_context=env.context, language=language)
    first = await step(task=task())
    run_id = first.answer["started_at"]
    first_query = env.agent.reply.call_args.args[0]
    assert str(previous) in first_query and env.now[0].isoformat() in first_query
    assert captured["query"] == task()["query"]
    env.now[0] += timedelta(minutes=5)
    second = await step(task=task())
    assert env.agent.create_session.await_count == 1 and env.agent.reply.await_count == 2
    calls = env.agent.reply.call_args_list
    assert calls[0].kwargs["resume"] == calls[1].kwargs["resume"] == "worker-session"
    assert calls[0].kwargs["injected_job_kwargs"] == calls[1].kwargs["injected_job_kwargs"]
    assert "cron_run_id" not in calls[0].kwargs["injected_job_kwargs"]
    assert env.now[0].isoformat() in calls[1].args[0] and calls[1].args[0] != first_query
    assert second.answer["started_at"] != run_id and second.answer["path"] != first.answer["path"]
    post = frontmatter.load(env.runtime.workspace_path / first.answer["path"])
    assert post["task_id"] == "one" and post["started_at"] == run_id
    assert post["status"] == "unread" and post["user_read_at"] is None
    assert env.runtime.tasks.state(PaTask.model_validate(task())).session_id == "worker-session"
    assert not list(env.runtime.tasks.task_path("one").glob("handoff*"))


@pytest.mark.asyncio
async def test_cold_start_reuses_persisted_session(pa_workspace, monkeypatch):
    from reme.components.pa.tasks import TaskFiles

    env = pa_workspace
    stub_model(env, monkeypatch)
    await CronAgentStep(app_context=env.context)(task=task())
    env.runtime.tasks = TaskFiles(env.runtime)
    env.now[0] += timedelta(minutes=5)
    await CronAgentStep(app_context=env.context)(task=task())
    env.agent.create_session.assert_awaited_once()
    assert env.agent.reply.call_args.kwargs["resume"] == "worker-session"


@pytest.mark.asyncio
async def test_missing_session_recreates_using_latest_valid_result(pa_workspace, monkeypatch):
    env = pa_workspace
    stub_model(env, monkeypatch)
    first = await CronAgentStep(app_context=env.context)(task=task())
    env.agent.session_exists = AsyncMock(return_value=False)
    env.agent.create_session = AsyncMock(return_value="replacement")
    await CronAgentStep(app_context=env.context)(task=task())
    assert env.agent.reply.call_args.kwargs["resume"] == "replacement"
    assert first.answer["path"] in env.agent.reply.call_args.args[0]


@pytest.mark.asyncio
async def test_session_transport_errors_do_not_discard_history(pa_workspace, monkeypatch):
    env = pa_workspace
    stub_model(env, monkeypatch)
    await CronAgentStep(app_context=env.context)(task=task())
    env.agent.session_exists = AsyncMock(side_effect=ConnectionError("offline"))
    with pytest.raises(ConnectionError):
        await CronAgentStep(app_context=env.context)(task=task())
    env.agent.create_session.assert_awaited_once()
    assert env.runtime.tasks.state(PaTask.model_validate(task())).session_id == "worker-session"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "answer",
    ["NO_UPDATE", "NO_UPDATE\n无新发现。", "无新发现。\nNO_UPDATE", "本轮返回 `NO_UPDATE`，保持安静。"],
)
async def test_no_update_completes_without_publication_or_handoff(pa_workspace, answer):
    env = pa_workspace
    configure(env, [task()])
    env.agent.reply = AsyncMock(return_value={"status": "completed", "result": answer})
    result = await CronAgentStep(app_context=env.context)(task=task())
    assert result.answer["reason"] == "no_new_information" and result.answer["path"] is None
    env.agent.reply.assert_awaited_once()
    assert env.runtime.tasks.paths("one") == [] and env.runtime.reminders.read() == []


@pytest.mark.asyncio
async def test_no_update_with_a_summary_rejects_the_conflict_without_deleting_it(pa_workspace):
    env = pa_workspace
    configure(env, [task()])

    async def reply(_query, **kwargs):
        state = env.runtime.tasks.execution(kwargs["injected_job_kwargs"]["cron_task_id"]).state
        env.runtime.tasks.staging_path(state.run).write_text("A saved finding", encoding="utf-8")
        return {"status": "completed", "result": "NO_UPDATE\nNothing new."}

    env.agent.reply = reply
    with pytest.raises(ValueError, match="must not accompany"):
        await CronAgentStep(app_context=env.context)(task=task())
    state = env.runtime.tasks.state(PaTask.model_validate(task()))
    assert env.runtime.tasks.staging_path(state.run).read_text() == "A saved finding"
    assert env.runtime.tasks.paths("one") == [] and env.runtime.reminders.read() == []


@pytest.mark.asyncio
async def test_incomplete_reply_cannot_be_accepted_as_no_update(pa_workspace):
    env = pa_workspace
    configure(env, [task()])
    env.agent.reply = AsyncMock(return_value={"status": "failed", "result": "NO_UPDATE", "error": "Model failed"})
    with pytest.raises(RuntimeError, match="Model failed"):
        await CronAgentStep(app_context=env.context)(task=task())


@pytest.mark.asyncio
async def test_reply_without_submission_is_not_success(pa_workspace):
    env = pa_workspace
    configure(env, [task()])
    env.agent.reply = AsyncMock(return_value={"status": "completed", "session_id": "worker-session", "result": "done"})
    with pytest.raises(ValueError, match="write its summary document"):
        await CronAgentStep(app_context=env.context)(task=task())
    assert not env.runtime.tasks.paths("one")


@pytest.mark.asyncio
async def test_submission_intent_recovers_without_model_after_crash(pa_workspace, monkeypatch):
    env = pa_workspace
    stub_model(env, monkeypatch)
    publish = env.runtime.tasks._publish
    monkeypatch.setattr(env.runtime.tasks, "_publish", lambda *_: (_ for _ in ()).throw(OSError("disk full")))
    with pytest.raises(OSError, match="disk full"):
        await CronAgentStep(app_context=env.context)(task=task())
    state = env.runtime.tasks.state(PaTask.model_validate(task()))
    assert state.completion.summary and state.run.result_path is None
    monkeypatch.setattr(env.runtime.tasks, "_publish", publish)
    env.agent.reply = AsyncMock(side_effect=AssertionError("Do not repeat research"))
    result = await CronAgentStep(app_context=env.context)(task=task())
    assert result.answer["status"] == "succeeded" and len(env.runtime.tasks.paths("one")) == 1
    assert result.answer["started_at"] == state.run.started_at.isoformat()
    env.agent.reply.assert_not_awaited()


@pytest.mark.asyncio
async def test_crash_after_publication_preserves_read_status_on_recovery(pa_workspace, monkeypatch):
    env = pa_workspace
    stub_model(env, monkeypatch)
    save = env.runtime.tasks.save
    failed = False

    async def fail(state):
        nonlocal failed
        if state.run.result_path and not failed:
            failed = True
            raise OSError("checkpoint failed")
        await save(state)

    monkeypatch.setattr(env.runtime.tasks, "save", fail)
    with pytest.raises(OSError, match="checkpoint failed"):
        await CronAgentStep(app_context=env.context)(task=task())
    path = env.runtime.tasks.paths("one")[0]
    post = frontmatter.load(path)
    post["status"], post["user_read_at"] = "read", "read-at"
    path.write_text(frontmatter.dumps(post))
    original = path.read_bytes()
    monkeypatch.setattr(env.runtime.tasks, "save", save)
    env.agent.reply = AsyncMock(side_effect=AssertionError("Do not repeat completed work"))
    result = await CronAgentStep(app_context=env.context)(task=task())
    assert result.answer["status"] == "succeeded" and path.read_bytes() == original
    assert len(env.runtime.tasks.paths("one")) == 1
    env.agent.reply.assert_not_awaited()


@pytest.mark.asyncio
async def test_changed_definition_starts_new_run_in_same_session(pa_workspace, monkeypatch):
    env = pa_workspace
    stub_model(env, monkeypatch)
    first = await CronAgentStep(app_context=env.context)(task=task())
    revised = task(query="主人修改后的任务")
    configure(env, [revised])
    second = await CronAgentStep(app_context=env.context)(task=revised)
    assert revised["query"] in env.agent.reply.call_args.args[0]
    assert first.answer["started_at"] != second.answer["started_at"]
    env.agent.create_session.assert_awaited_once()


@pytest.mark.asyncio
async def test_task_state_symlink_rejected_before_native_calls(pa_workspace, monkeypatch):
    env = pa_workspace
    path = env.runtime.tasks.state_path("one")
    path.parent.mkdir(parents=True)
    target = env.runtime.workspace_path / "private.txt"
    target.write_text("secret")
    path.symlink_to(target)
    stub_model(env, monkeypatch)
    with pytest.raises(ValueError, match="without symlinks"):
        await CronAgentStep(app_context=env.context)(task=task())
    env.agent.create_session.assert_not_awaited()
    env.agent.reply.assert_not_awaited()
    assert target.read_text() == "secret"


@pytest.mark.asyncio
async def test_oversized_submission_does_not_publish(pa_workspace, monkeypatch):
    env = pa_workspace
    stub_model(env, monkeypatch, content="x" * 2048)
    monkeypatch.setattr("reme.components.pa.files.DEFAULT_MAX_FILE_BYTES", 1024)
    with pytest.raises(ValueError, match="read limit"):
        await CronAgentStep(app_context=env.context)(task=task())
    assert not env.runtime.tasks.paths("one")


@pytest.mark.parametrize(
    "overrides",
    [{"id": "../escape"}, {"cron": "* * * * * *"}, {"query": " "}, {"enabled": 1}],
)
def test_task_definition_rejects_invalid_fields(overrides):
    with pytest.raises(ValueError):
        PaTask.model_validate(task(**overrides))


def test_definition_loading_preserves_user_files_and_rejects_duplicates(pa_workspace):
    env = pa_workspace
    configure(env, [])
    env.runtime.default_tasks = [PaTask.model_validate(task())]
    env.runtime.tasks.initialize()
    assert env.runtime.tasks.load() == []
    configure(env, [task(), task()])
    original = env.runtime.cron_path.read_bytes()
    with pytest.raises(ValueError, match="duplicate"):
        env.runtime.tasks.load()
    assert env.runtime.cron_path.read_bytes() == original


@pytest.mark.asyncio
async def test_existing_snapshot_is_never_overwritten(pa_workspace, monkeypatch):
    env = pa_workspace
    path = snapshot(env)
    original = path.read_bytes()
    stub_model(env, monkeypatch)
    result = await CronAgentStep(app_context=env.context)(task=task())
    assert path.read_bytes() == original and env.agent.reply.await_count == 1
    assert result.answer["path"] != env.runtime.to_workspace_relative(path)
    assert len(env.runtime.tasks.paths("one")) == 2


@pytest.mark.asyncio
async def test_final_record_failure_clears_live_task_activity(pa_workspace, monkeypatch):
    env = pa_workspace
    env.runtime.event_stream._closed = False
    stub_model(env, monkeypatch)
    record = env.runtime.tasks.record

    async def fail_final(run):
        if run.status != "running":
            raise OSError("disk full")
        await record(run)

    monkeypatch.setattr(env.runtime.tasks, "record", fail_final)
    with pytest.raises(OSError, match="disk full"):
        await CronAgentStep(app_context=env.context)(task=task())
    assert env.runtime.event_stream._task_activities == {}


@pytest.mark.asyncio
async def test_task_cancellation_records_interruption(pa_workspace):
    env = pa_workspace
    env.runtime.event_stream._closed = False
    entered = asyncio.Event()

    async def reply(*_args, **_kwargs):
        entered.set()
        await asyncio.Event().wait()

    env.agent.reply = AsyncMock(side_effect=reply)
    pending = asyncio.create_task(CronAgentStep(app_context=env.context)(task=task()))
    await entered.wait()
    assert len(env.runtime.event_stream._task_activities) == 1
    pending.cancel()
    with pytest.raises(asyncio.CancelledError):
        await pending
    rows = [json.loads(line) for line in env.runtime.cron_runs_path.read_text().splitlines()]
    assert rows[-1]["status"] == "interrupted" and not env.runtime.tasks.paths("one")
    assert env.runtime.event_stream._task_activities == {}


@pytest.mark.asyncio
async def test_consume_latest_three_per_task_checkpoints_then_marks_older_superseded(
    pa_workspace,
):
    env = pa_workspace
    configure(env, [task(), task("two", enabled=False)])
    old = snapshot(env, day="2026-11-07", time="235900")
    selected = [snapshot(env, time=time) for time in ("110000", "111000", "112000")]
    paused = snapshot(env, identifier="two")
    deleted = snapshot(env, identifier="deleted")
    async with env.runtime.proactive_round() as current:
        result = await ReadCronResultsStep(app_context=env.context)(pa_round_id=current.id)
    assert len(result.answer) == 4
    assert [row["path"] for row in result.answer[:3]] == [
        env.runtime.to_workspace_relative(path) for path in reversed(selected)
    ]
    assert all(frontmatter.load(path)["status"] == "read" for path in [*selected, paused])
    assert frontmatter.load(old)["status"] == "superseded"
    assert frontmatter.load(deleted)["status"] == "unread"
    context = env.runtime.proactive_context_path.read_text()
    assert "task_results" in context and "累计发现" in context
    assert await env.runtime.tasks.consume(3, "next") == []


@pytest.mark.asyncio
async def test_checkpoint_failure_keeps_all_documents_unread(pa_workspace, monkeypatch):
    env = pa_workspace
    configure(env, [task()])
    path = snapshot(env)
    monkeypatch.setattr(env.runtime, "checkpoint_proactive", AsyncMock(side_effect=OSError("disk full")))
    with pytest.raises(OSError, match="disk full"):
        await env.runtime.tasks.consume(3, "round")
    assert frontmatter.load(path)["status"] == "unread"


@pytest.mark.asyncio
async def test_new_publication_after_read_is_not_acknowledged_as_old_input(pa_workspace, monkeypatch):
    env = pa_workspace
    configure(env, [task()])
    first = snapshot(env, time="110000")
    original = env.runtime.checkpoint_proactive
    newer = None

    async def checkpoint(*args):
        nonlocal newer
        await original(*args)
        newer = snapshot(env, time="120000")

    monkeypatch.setattr(env.runtime, "checkpoint_proactive", checkpoint)
    result = await env.runtime.tasks.consume(3, "round")
    assert len(result) == 1 and frontmatter.load(first)["status"] == "read"
    assert frontmatter.load(newer)["status"] == "unread"


@pytest.mark.asyncio
async def test_read_requires_live_round_and_permissions(pa_workspace):
    env = pa_workspace
    configure(env, [task()])
    path = snapshot(env)
    with pytest.raises(RuntimeError, match="live proactive"):
        await ReadCronResultsStep(app_context=env.context)(pa_round_id="fake")
    async with env.runtime.proactive_round() as current:
        with pytest.raises(PermissionError):
            await ReadCronResultsStep(app_context=env.context)(pa_round_id=current.id, _allowed_paths=[])
    assert frontmatter.load(path)["status"] == "unread"


@pytest.mark.parametrize("kind", ["root", "date", "task", "file"])
def test_result_symlinks_fail_closed(pa_workspace, tmp_path, kind):
    env = pa_workspace
    external = tmp_path / "external"
    external.mkdir()
    path = env.runtime.tasks.task_path("one") / "2026-11-08/120000.md"
    part = {
        "root": env.runtime.tasks.root_path,
        "date": path.parent,
        "task": path.parent.parent,
        "file": path,
    }[kind]
    part.parent.mkdir(parents=True, exist_ok=True)
    part.symlink_to(
        external if kind != "file" else external / "note.md",
        target_is_directory=kind != "file",
    )
    if kind == "file":
        (external / "note.md").write_text("private")
    with pytest.raises(ValueError, match="without symlinks"):
        env.runtime.tasks.paths("one")


@pytest.mark.asyncio
async def test_scheduler_reconciles_edits_and_keeps_valid_schedule_on_invalid_json(pa_workspace, monkeypatch):
    env = pa_workspace
    configure(env, [task()])
    env.context.jobs["cron_agent"] = BaseJob(app_context=env.context, steps=[{"backend": "cron_agent_step"}])
    created = []

    class ObservedCron(CronJob):
        """Observe actual framework start/close behavior without running model calls."""

        async def _start(self):
            await super()._start()
            created.append(self)

    monkeypatch.setattr("reme.steps.proactive.background.cron_schedule.CronJob", ObservedCron)
    stop = asyncio.Event()
    pending = asyncio.create_task(CronScheduleStep(app_context=env.context)(RuntimeContext(stop_event=stop)))
    try:
        await wait_until(lambda: len(created) == 1)
        first = created[0]
        env.runtime.cron_path.write_text("[")
        await asyncio.sleep(1.05)
        assert first.is_started and len(created) == 1
        configure(env, [task(query="新需求", cron="0 * * * *"), task("two")])
        await wait_until(lambda: len(created) == 3)
        assert not first.is_started
        assert created[1].cron_expr == "0 * * * *" and created[1].kwargs["task"]["query"] == "新需求"
        configure(env, [task("two", enabled=False)])
        await wait_until(lambda: all(not job.is_started for job in created))
    finally:
        stop.set()
        await pending
    assert all(job._task is None for job in created)


@pytest.mark.asyncio
async def test_real_cron_can_opt_out_of_active_hours_and_closes_on_cancellation(pa_workspace, monkeypatch):
    env = pa_workspace
    env.now[0] = env.now[0].replace(hour=2)
    config = await env.runtime.get_proactive_config()
    env.runtime.files.write_json(
        env.runtime.proactive_config_path,
        {**config.model_dump(), "tasks_follow_active_hours": False},
    )
    configure(env, [task(), task("two")])
    stub_model(env, monkeypatch)
    env.context.jobs["cron_agent"] = BaseJob(
        app_context=env.context,
        steps=[{"backend": "cron_agent_step", "job_tools": ["screen_capture"]}],
    )
    fired = set()
    jobs = []

    class OnceCron(CronJob):
        """Trigger once promptly, then wait so cleanup is observable."""

        def _next_fire_delay(self):
            if self.name not in fired:
                fired.add(self.name)
                jobs.append(self)
                return 0.001
            return 3600

    monkeypatch.setattr("reme.steps.proactive.background.cron_schedule.CronJob", OnceCron)
    pending = asyncio.create_task(
        CronScheduleStep(app_context=env.context)(RuntimeContext(stop_event=asyncio.Event())),
    )
    try:
        await wait_until(lambda: all(len(env.runtime.tasks.paths(identifier)) == 1 for identifier in ("one", "two")))
        await wait_until(lambda: env.agent.reply.await_count == 2)
        assert len(jobs) == 2 and all(job.is_started for job in jobs)
    finally:
        pending.cancel()
        with pytest.raises(asyncio.CancelledError):
            await pending
    assert all(not job.is_started and job._task is None for job in jobs)


@pytest.mark.asyncio
async def test_result_tool_returns_paths_and_keeps_full_recovery_inputs(pa_workspace):
    env = pa_workspace
    configure(env, [task()])
    content = "完整证据。" * 10000
    path = snapshot(env, content=content)
    result = await env.runtime.tasks.consume(3, "round")
    assert result == [{"task_id": "one", "path": env.runtime.to_workspace_relative(path)}]
    recovery = [json.loads(line) for line in env.runtime.proactive_context_path.read_text().splitlines()]
    assert content in recovery[-1]["value"][0]["content"]


@pytest.mark.asyncio
async def test_scheduler_closes_remaining_jobs_after_one_close_fails(pa_workspace, monkeypatch):
    env = pa_workspace
    configure(env, [task(), task("two")])
    env.context.jobs["cron_agent"] = BaseJob(app_context=env.context, steps=[{"backend": "cron_agent_step"}])
    jobs = []

    class FailingCloseCron(CronJob):
        """Fail one close after releasing that job, leaving peer cleanup observable."""

        async def _start(self):
            await super()._start()
            jobs.append(self)

        async def close(self):
            await super().close()
            if self.name == "cron_one":
                raise RuntimeError("close failed")

    monkeypatch.setattr("reme.steps.proactive.background.cron_schedule.CronJob", FailingCloseCron)
    stop = asyncio.Event()
    pending = asyncio.create_task(CronScheduleStep(app_context=env.context)(RuntimeContext(stop_event=stop)))
    try:
        await wait_until(lambda: len(jobs) == 2)
        stop.set()
        with pytest.raises(RuntimeError, match="close failed"):
            await pending
        assert all(not job.is_started for job in jobs)
    finally:
        stop.set()
        for job in jobs:
            await CronJob.close(job)


@pytest.mark.asyncio
async def test_stopped_task_does_not_start_a_model_turn(pa_workspace):
    env = pa_workspace
    stop = asyncio.Event()
    stop.set()
    env.agent.reply = AsyncMock()
    with pytest.raises(asyncio.CancelledError):
        await CronAgentStep(app_context=env.context)(RuntimeContext(stop_event=stop), task=task())
    env.agent.reply.assert_not_awaited()
    rows = [json.loads(line) for line in env.runtime.cron_runs_path.read_text().splitlines()]
    assert rows[-1]["status"] == "interrupted"


@pytest.mark.parametrize(
    "field,value",
    [
        ("message", "fixed"),
        ("delivery", "immediate"),
        ("source_session_id", "main"),
        ("created_at", "2026-11-09T12:00:00+08:00"),
        ("preflight", "screen_capture"),
    ],
)
def test_removed_task_fields_are_rejected(field, value):
    with pytest.raises(ValueError, match="Extra inputs"):
        PaTask.model_validate(task(**{field: value}))


@pytest.mark.parametrize("field", ["id", "cron", "query", "enabled"])
def test_all_four_task_fields_are_required(field):
    definition = task()
    del definition[field]
    with pytest.raises(ValueError, match="Field required"):
        PaTask.model_validate(definition)
