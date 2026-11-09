"""Independent schedules, four-query tasks and recoverable file-native results."""

# pylint: disable=missing-function-docstring,protected-access

import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock

import frontmatter
import pytest

from reme.components.job import BaseJob, CronJob
from reme.components.runtime_context import RuntimeContext
from reme.schema.pa_task import PaTask
from reme.steps.proactive import CronAgentStep, CronScheduleStep
from reme.steps.proactive.tools import ReadCronResultsStep


def task(identifier="one", **overrides):
    return {
        "id": identifier,
        "cron": "*/5 * * * *",
        "query": "观察屏幕",
        "enabled": True,
        **overrides,
    }


def configure(env, definitions):
    env.runtime._write_json(env.runtime.cron_path, definitions)


def snapshot(
    env,
    identifier="one",
    time="120000",
    day="2026-11-08",
    status="unread",
    content="累计发现",
):
    path = env.runtime.tasks.task_path(identifier) / day / f"{time}.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(frontmatter.dumps(frontmatter.Post(content, task_id=identifier, status=status)))
    return path


def stub_model(
    env,
    monkeypatch,
    *,
    content="此前总结。\n\n本次发现。",
    fail_at=None,
    no_update=False,
    handoff="进展与来源。\n下一步调查。",
    message=None,
):
    captured = {}
    summaries = {}
    original = CronAgentStep.prompt_format
    monkeypatch.setattr(CronAgentStep, "_now", lambda _self: env.now[0])

    def prompt(step, name, **kwargs):
        rendered = original(step, name, **kwargs)
        if name == "query_summary":
            captured.update(kwargs)
            summaries[rendered] = kwargs
        elif name == "query_message":
            summaries[rendered] = {"message": True}
        elif name == "query_handoff":
            summaries[rendered] = {"handoff": True}
        return rendered

    monkeypatch.setattr(CronAgentStep, "prompt_format", prompt)

    async def reply(_query, **_kwargs):
        handoff_query = any(_query.startswith(text) and details.get("handoff") for text, details in summaries.items())
        message_query = any(_query == text and details.get("message") for text, details in summaries.items())
        stage = 4 if handoff_query else 3 if message_query else 2 if _query in summaries else 1
        if stage == fail_at:
            raise RuntimeError("provider failed")
        if stage == 2 and content is not None and not no_update:
            path = Path(summaries[_query]["staging_path"])
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
        if stage == 3 and message is not None:
            await message(_kwargs["injected_job_kwargs"])
        result = handoff if stage == 4 else "NO_UPDATE" if stage == 2 and no_update else "finished"
        return {
            "session_id": ("handoff-fork" if _kwargs.get("fork_session") else "task-session"),
            "status": "completed",
            "result": result,
        }

    env.agent.reply = AsyncMock(side_effect=reply)
    return captured


async def wait_until(predicate):
    async with asyncio.timeout(3):
        while not predicate():
            await asyncio.sleep(0.005)


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
@pytest.mark.parametrize("language", ["en", "zh"])
@pytest.mark.parametrize("previous_day", ["2026-11-07", "2026-11-08"])
async def test_task_summary_and_handoff_share_session_with_loaded_memory(
    pa_workspace,
    monkeypatch,
    language,
    previous_day,
):
    env = pa_workspace
    env.context.app_config.daily_dir = "notes"
    memory_path = env.runtime.workspace_path / "notes/personal/MEMORY.md"
    memory_path.parent.mkdir(parents=True)
    memory = "# 当前偏好\n只关注有来源的新信息，保留 {原始内容}。\n"
    memory_path.write_text(memory, encoding="utf-8")
    old = snapshot(env, day="2026-11-06", time="235900", status="read")
    newer = snapshot(env, day=previous_day, time="110000")
    env.runtime.tasks.handoff_path("one").write_text("上次查到的来源和待确认事项。", encoding="utf-8")
    captured = stub_model(env, monkeypatch)
    step = CronAgentStep(app_context=env.context, language=language, job_tools=["screen_capture"])
    result = await step(task=task())
    calls = env.agent.reply.await_args_list
    assert len(calls) == 4 and calls[0].args[0] == step.prompt_format("query_execute", query=task()["query"])
    assert "resume" not in calls[0].kwargs and calls[1].kwargs["resume"] == "task-session"
    assert all(call.kwargs["resume"] == "task-session" for call in calls[1:])
    assert all(not call.kwargs.get("fork_session") for call in calls)
    assert all(
        call.kwargs["job_tools"]
        == [
            "screen_capture",
            "get_chat_messages",
            "send_chat_message_immediate",
        ]
        for call in calls
    )
    assert all(call.kwargs["injected_job_kwargs"] == calls[0].kwargs["injected_job_kwargs"] for call in calls)
    prompt = calls[0].kwargs["system_prompt"]
    assert memory in prompt and "上次查到的来源和待确认事项。" in prompt
    assert str(env.runtime.tasks.recovery_path("one")) not in prompt
    assert all(call.kwargs["system_prompt"] == prompt for call in calls)
    assert memory_path.read_text(encoding="utf-8") == memory
    assert all(call.kwargs["activity_source"] == "task" for call in calls)
    assert calls[0].kwargs["activity_run_id"] == calls[1].kwargs["activity_run_id"]
    assert all(call.kwargs["activity_task_id"] == "one" for call in calls)
    assert captured["previous_path"] == str(newer)
    assert str(old) not in calls[1].args[0]
    path = env.runtime.workspace_path / result.answer["path"]
    assert path == env.runtime.tasks.task_path("one") / "2026-11-08/120000.md"
    post = frontmatter.load(path)
    assert post["status"] == "unread" and post["task_id"] == "one" and "本次发现" in post.content
    assert frontmatter.load(newer)["status"] == "unread"
    assert post["user_read_at"] is None
    handoff = frontmatter.load(env.runtime.tasks.handoff_path("one"))
    assert handoff["run_id"] == post["run_id"] and "下一步" in handoff.content
    assert not env.runtime.tasks.recovery_path("one").exists()
    assert not list(env.runtime.workspace_path.glob(".reme-task-*"))
    rows = [json.loads(line) for line in env.runtime.cron_runs_path.read_text().splitlines()]
    assert [row["status"] for row in rows] == ["running", "succeeded"]
    assert rows[0]["run_id"] == rows[1]["run_id"] == post["run_id"]
    assert rows[1]["result_path"] == result.answer["path"]
    assert rows[1]["agent_session_id"] == "task-session"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "content,fail_at",
    [
        (None, None),
        (" ", None),
        ("---\nstatus: read\n---\n", None),
        (None, 1),
        (None, 2),
    ],
)
async def test_failed_tasks_never_publish_unread_documents(pa_workspace, monkeypatch, content, fail_at):
    env = pa_workspace
    stub_model(env, monkeypatch, content=content, fail_at=fail_at)
    with pytest.raises((ValueError, OSError, RuntimeError)):
        await CronAgentStep(app_context=env.context)(task=task())
    assert not env.runtime.tasks.paths("one")
    assert not list(env.runtime.workspace_path.glob(".reme-task-*"))
    rows = [json.loads(line) for line in env.runtime.cron_runs_path.read_text().splitlines()]
    assert rows[-1]["status"] == "failed" and rows[-1]["error"]


@pytest.mark.asyncio
async def test_existing_snapshot_is_never_overwritten(pa_workspace, monkeypatch):
    env = pa_workspace
    path = snapshot(env)
    original = path.read_bytes()
    stub_model(env, monkeypatch)
    result = await CronAgentStep(app_context=env.context)(task=task())
    assert path.read_bytes() == original and env.agent.reply.await_count == 4
    assert result.answer["path"] != env.runtime.to_workspace_relative(path)
    assert len(env.runtime.tasks.paths("one")) == 2


@pytest.mark.asyncio
async def test_final_record_failure_clears_live_task_activity(pa_workspace, monkeypatch):
    env = pa_workspace
    env.runtime._closed = False
    stub_model(env, monkeypatch)
    record = env.runtime.tasks.record

    async def fail_final(run):
        if run.status != "running":
            raise OSError("disk full")
        await record(run)

    monkeypatch.setattr(env.runtime.tasks, "record", fail_final)
    with pytest.raises(OSError, match="disk full"):
        await CronAgentStep(app_context=env.context)(task=task())
    assert env.runtime._task_activities == {}


@pytest.mark.asyncio
async def test_task_cancellation_records_interruption(pa_workspace):
    env = pa_workspace
    env.runtime._closed = False
    entered = asyncio.Event()

    async def reply(*_args, **_kwargs):
        entered.set()
        await asyncio.Event().wait()

    env.agent.reply = AsyncMock(side_effect=reply)
    pending = asyncio.create_task(CronAgentStep(app_context=env.context)(task=task()))
    await entered.wait()
    assert len(env.runtime._task_activities) == 1
    pending.cancel()
    with pytest.raises(asyncio.CancelledError):
        await pending
    rows = [json.loads(line) for line in env.runtime.cron_runs_path.read_text().splitlines()]
    assert rows[-1]["status"] == "interrupted" and not env.runtime.tasks.paths("one")
    assert env.runtime._task_activities == {}


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

    monkeypatch.setattr("reme.steps.proactive.cron_schedule.CronJob", ObservedCron)
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
    env.runtime._write_json(
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

    monkeypatch.setattr("reme.steps.proactive.cron_schedule.CronJob", OnceCron)
    pending = asyncio.create_task(
        CronScheduleStep(app_context=env.context)(RuntimeContext(stop_event=asyncio.Event())),
    )
    try:
        await wait_until(lambda: len(list(env.runtime.tasks.root_path.glob("*/*/*.md"))) == 2)
        await wait_until(lambda: env.agent.reply.await_count == 8)
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
async def test_publication_rejects_a_summary_that_exceeds_limit_after_metadata(pa_workspace, monkeypatch):
    env = pa_workspace
    monkeypatch.setattr("reme.components._pa_tasks.DEFAULT_MAX_FILE_BYTES", 512)
    staging = env.runtime.workspace_path / "summary.md"
    staging.write_text("x" * 512)
    target = env.runtime.tasks.task_path("one") / "2026-11-08/120000.md"
    with pytest.raises(ValueError, match="read limit"):
        await env.runtime.tasks.publish(staging, target, "one", "run")
    assert not env.runtime.tasks.paths("one")


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

    monkeypatch.setattr("reme.steps.proactive.cron_schedule.CronJob", FailingCloseCron)
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


@pytest.mark.asyncio
async def test_no_update_skips_message_query_but_preserves_handoff(pa_workspace, monkeypatch):
    env = pa_workspace
    captured = stub_model(env, monkeypatch, no_update=True)
    step = CronAgentStep(app_context=env.context)
    result = await step(task=task())
    assert captured["previous_path"] == step.get_prompt("no_previous")
    assert result.answer["status"] == "succeeded" and result.answer["path"] is None
    assert result.answer["reason"] == "no_new_information"
    assert env.agent.reply.await_count == 3 and not env.runtime.tasks.paths("one")
    assert env.runtime.tasks.handoff("one").content and not env.runtime.tasks.recovery_path("one").exists()
    assert env.runtime.reminders.read() == []


@pytest.mark.asyncio
async def test_handoff_failure_recovers_without_republishing_or_consumption_reset(pa_workspace, monkeypatch):
    env = pa_workspace
    configure(env, [task()])
    stub_model(env, monkeypatch, fail_at=4)
    with pytest.raises(RuntimeError, match="provider failed"):
        await CronAgentStep(app_context=env.context)(task=task())
    path = env.runtime.tasks.paths("one")[0]
    run_id = frontmatter.load(path)["run_id"]
    await env.runtime.tasks.consume(3, "proactive-round")
    original = path.read_bytes()
    stub_model(env, monkeypatch)
    result = await CronAgentStep(app_context=env.context)(task=task())
    env.agent.reply.assert_awaited_once()
    assert env.agent.reply.await_args.kwargs["resume"] == "task-session"
    assert result.answer["path"] == env.runtime.to_workspace_relative(path)
    assert env.runtime.tasks.handoff("one")["run_id"] == run_id
    assert path.read_bytes() == original and len(env.runtime.tasks.paths("one")) == 1
    assert not env.runtime.tasks.recovery_path("one").exists()


@pytest.mark.asyncio
async def test_summary_failure_recovers_original_query_context(pa_workspace, monkeypatch):
    env = pa_workspace
    stub_model(env, monkeypatch, fail_at=2)
    with pytest.raises(RuntimeError, match="provider failed"):
        await CronAgentStep(app_context=env.context)(task=task())
    first = env.agent.reply
    memory = "主人最新纠正：只关注已经核实的新消息。\n"
    (env.runtime.workspace_path / "daily/personal/MEMORY.md").write_text(memory, encoding="utf-8")
    stub_model(env, monkeypatch)
    await CronAgentStep(app_context=env.context)(task=task())
    assert first.await_count == 2 and env.agent.reply.await_count == 3
    execution_query = CronAgentStep(app_context=env.context).prompt_format("query_execute", query=task()["query"])
    assert all(call.args[0] != execution_query for call in env.agent.reply.await_args_list)
    assert all(call.kwargs["resume"] == "task-session" for call in env.agent.reply.await_args_list)
    assert all(not call.kwargs.get("fork_session") for call in env.agent.reply.await_args_list)
    assert all(memory in call.kwargs["system_prompt"] for call in env.agent.reply.await_args_list)
    assert all(
        str(env.runtime.tasks.recovery_path("one")) not in call.kwargs["system_prompt"]
        for call in env.agent.reply.await_args_list
    )


@pytest.mark.asyncio
async def test_recovery_handles_crash_between_publication_and_phase_checkpoint(pa_workspace, monkeypatch):
    env = pa_workspace
    stub_model(env, monkeypatch)
    checkpoint = env.runtime.tasks.checkpoint

    async def fail_handoff(saved):
        if saved.run.phase == "message":
            raise OSError("checkpoint failed")
        await checkpoint(saved)

    monkeypatch.setattr(env.runtime.tasks, "checkpoint", fail_handoff)
    with pytest.raises(OSError, match="checkpoint failed"):
        await CronAgentStep(app_context=env.context)(task=task())
    path = env.runtime.tasks.paths("one")[0]
    original = path.read_bytes()
    monkeypatch.setattr(env.runtime.tasks, "checkpoint", checkpoint)
    stub_model(env, monkeypatch)
    await CronAgentStep(app_context=env.context)(task=task())
    assert env.agent.reply.await_count == 2
    assert path.read_bytes() == original and len(env.runtime.tasks.paths("one")) == 1


@pytest.mark.asyncio
async def test_saved_handoff_recovers_without_another_model_call(pa_workspace, monkeypatch):
    env = pa_workspace
    stub_model(env, monkeypatch)
    clear = env.runtime.tasks.clear_recovery
    monkeypatch.setattr(
        env.runtime.tasks,
        "clear_recovery",
        AsyncMock(side_effect=OSError("cleanup failed")),
    )
    with pytest.raises(OSError, match="cleanup failed"):
        await CronAgentStep(app_context=env.context)(task=task())
    previous = env.runtime.tasks.handoff_path("one").read_bytes()
    monkeypatch.setattr(env.runtime.tasks, "clear_recovery", clear)
    env.agent.reply = AsyncMock()
    result = await CronAgentStep(app_context=env.context)(task=task())
    env.agent.reply.assert_not_awaited()
    assert result.answer["status"] == "succeeded"
    assert env.runtime.tasks.handoff_path("one").read_bytes() == previous


@pytest.mark.asyncio
async def test_new_task_definition_does_not_resume_old_failed_query(pa_workspace, monkeypatch):
    env = pa_workspace
    stub_model(env, monkeypatch, fail_at=2)
    with pytest.raises(RuntimeError):
        await CronAgentStep(app_context=env.context)(task=task())
    stub_model(env, monkeypatch)
    step = CronAgentStep(app_context=env.context)
    await step(task=task(query="主人修改后的任务"))
    assert env.agent.reply.await_args_list[0].args[0] == step.prompt_format("query_execute", query="主人修改后的任务")
    assert "resume" not in env.agent.reply.await_args_list[0].kwargs


@pytest.mark.asyncio
@pytest.mark.parametrize("filename", ["handoff.md", "recovery.json"])
async def test_task_private_files_reject_symlinks_before_model_calls(pa_workspace, monkeypatch, filename):
    env = pa_workspace
    root = env.runtime.tasks.task_path("one")
    root.mkdir(parents=True)
    target = env.runtime.workspace_path / "private.txt"
    target.write_text("secret")
    (root / filename).symlink_to(target)
    stub_model(env, monkeypatch)
    with pytest.raises(ValueError, match="without symlinks"):
        await CronAgentStep(app_context=env.context)(task=task())
    env.agent.reply.assert_not_awaited()
    assert target.read_text() == "secret"


@pytest.mark.asyncio
async def test_no_update_marker_cannot_hide_a_written_result(pa_workspace, monkeypatch):
    env = pa_workspace
    stub_model(env, monkeypatch)
    original = env.agent.reply.side_effect
    count = 0

    async def reply(*args, **kwargs):
        nonlocal count
        count += 1
        result = await original(*args, **kwargs)
        if count == 2:
            result["result"] = "NO_UPDATE"
        return result

    env.agent.reply.side_effect = reply
    with pytest.raises(ValueError, match="must not accompany"):
        await CronAgentStep(app_context=env.context)(task=task())
    assert not env.runtime.tasks.paths("one")


@pytest.mark.asyncio
async def test_empty_handoff_preserves_previous_handoff_and_published_result(pa_workspace, monkeypatch):
    env = pa_workspace
    path = env.runtime.tasks.handoff_path("one")
    path.parent.mkdir(parents=True)
    path.write_text("上次有效交接")
    stub_model(env, monkeypatch, handoff=" ")
    with pytest.raises(ValueError, match="nonempty"):
        await CronAgentStep(app_context=env.context)(task=task())
    assert path.read_text() == "上次有效交接"
    assert len(env.runtime.tasks.paths("one")) == 1 and env.runtime.tasks.recovery_path("one").exists()
