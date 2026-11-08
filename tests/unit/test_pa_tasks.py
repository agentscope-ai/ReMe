"""Independent cron schedules, two-query publication and recoverable snapshot consumption."""

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
from reme.schema import Response
from reme.schema.screen_capture import CapturedDisplay
from reme.steps.proactive import TaskAgentStep, TaskScheduleStep
from reme.steps.proactive.tools import ReadTaskResultsStep


def task(identifier="one", **overrides):
    return {"id": identifier, "cron": "*/5 * * * *", "query": "观察屏幕", "enabled": True, **overrides}


def configure(env, definitions):
    env.runtime._write_json(env.runtime.tasks_path, definitions)


def snapshot(env, identifier="one", time="120000", day="2026-11-08", status="unread", content="累计发现"):
    path = env.runtime.tasks.daily_path / day / "cron" / identifier / f"{time}.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(frontmatter.dumps(frontmatter.Post(content, task_id=identifier, status=status)))
    return path


def stub_model(env, monkeypatch, *, content="此前总结。\n\n本次发现。", fail_at=None):
    captured = {}
    summaries = {}
    original = TaskAgentStep.prompt_format
    monkeypatch.setattr(TaskAgentStep, "_now", lambda _self: env.now[0])

    def prompt(step, name, **kwargs):
        rendered = original(step, name, **kwargs)
        if name == "query_summary":
            captured.update(kwargs)
            summaries[rendered] = kwargs
        return rendered

    monkeypatch.setattr(TaskAgentStep, "prompt_format", prompt)

    async def reply(_query, **kwargs):
        stage = 2 if kwargs.get("resume") else 1
        if stage == fail_at:
            raise RuntimeError("provider failed")
        if stage == 2 and content is not None:
            path = Path(summaries[_query]["staging_path"])
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
        return {"session_id": "task-session", "status": "completed", "result": "finished"}

    env.agent.reply = AsyncMock(side_effect=reply)
    return captured


async def wait_until(predicate):
    async with asyncio.timeout(3):
        while not predicate():
            await asyncio.sleep(0.005)


@pytest.mark.parametrize("overrides", [{"id": "../escape"}, {"cron": "* * * * * *"}, {"query": " "}, {"enabled": 1}])
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
    original = env.runtime.tasks_path.read_bytes()
    with pytest.raises(ValueError, match="duplicate"):
        env.runtime.tasks.load()
    assert env.runtime.tasks_path.read_bytes() == original


@pytest.mark.asyncio
async def test_two_queries_share_session_and_publish_latest_cross_day_summary(pa_workspace, monkeypatch):
    env = pa_workspace
    env.context.app_config.daily_dir = "notes"
    old = snapshot(env, day="2026-11-07", time="235900", status="read")
    newer = snapshot(env, time="110000")
    captured = stub_model(env, monkeypatch)
    result = await TaskAgentStep(app_context=env.context, job_tools=["screen_capture"])(task=task())
    calls = env.agent.reply.await_args_list
    assert len(calls) == 2 and calls[0].args[0] == task()["query"]
    assert "resume" not in calls[0].kwargs and calls[1].kwargs["resume"] == "task-session"
    assert all(call.kwargs["job_tools"] == ["screen_capture"] for call in calls)
    assert all(call.kwargs["activity_source"] == "task" for call in calls)
    assert calls[0].kwargs["activity_run_id"] == calls[1].kwargs["activity_run_id"]
    assert all(call.kwargs["activity_task_id"] == "one" for call in calls)
    assert captured["previous_paths"].splitlines() == [str(newer), str(old)]
    path = env.runtime.workspace_path / result.answer["path"]
    assert path == env.runtime.tasks.daily_path / "2026-11-08/cron/one/120000.md"
    post = frontmatter.load(path)
    assert post["status"] == "unread" and post["task_id"] == "one" and "本次发现" in post.content
    assert frontmatter.load(newer)["status"] == "unread"
    assert not list(env.runtime.workspace_path.glob(".reme-task-*"))
    rows = [json.loads(line) for line in env.runtime.task_runs_path.read_text().splitlines()]
    assert [row["status"] for row in rows] == ["running", "succeeded"]
    assert rows[0]["run_id"] == rows[1]["run_id"] == post["run_id"]
    assert rows[1]["result_path"] == result.answer["path"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "content,fail_at",
    [(None, None), (" ", None), ("---\nstatus: read\n---\n", None), (None, 1), (None, 2)],
)
async def test_failed_tasks_never_publish_unread_documents(pa_workspace, monkeypatch, content, fail_at):
    env = pa_workspace
    stub_model(env, monkeypatch, content=content, fail_at=fail_at)
    with pytest.raises((ValueError, OSError, RuntimeError)):
        await TaskAgentStep(app_context=env.context)(task=task())
    assert not env.runtime.tasks.paths("one")
    assert not list(env.runtime.workspace_path.glob(".reme-task-*"))
    rows = [json.loads(line) for line in env.runtime.task_runs_path.read_text().splitlines()]
    assert rows[-1]["status"] == "failed" and rows[-1]["error"]


@pytest.mark.asyncio
async def test_existing_snapshot_is_never_overwritten(pa_workspace, monkeypatch):
    env = pa_workspace
    path = snapshot(env)
    original = path.read_bytes()
    stub_model(env, monkeypatch)
    with pytest.raises(FileExistsError):
        await TaskAgentStep(app_context=env.context)(task=task())
    assert path.read_bytes() == original and env.agent.reply.await_count == 1


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
        await TaskAgentStep(app_context=env.context)(task=task())
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
    pending = asyncio.create_task(TaskAgentStep(app_context=env.context)(task=task()))
    await entered.wait()
    assert len(env.runtime._task_activities) == 1
    pending.cancel()
    with pytest.raises(asyncio.CancelledError):
        await pending
    rows = [json.loads(line) for line in env.runtime.task_runs_path.read_text().splitlines()]
    assert rows[-1]["status"] == "interrupted" and not env.runtime.tasks.paths("one")
    assert env.runtime._task_activities == {}


@pytest.mark.asyncio
async def test_consume_latest_three_per_task_checkpoints_then_marks_older_superseded(pa_workspace):
    env = pa_workspace
    configure(env, [task(), task("two", enabled=False)])
    old = snapshot(env, day="2026-11-07", time="235900")
    selected = [snapshot(env, time=time) for time in ("110000", "111000", "112000")]
    paused = snapshot(env, identifier="two")
    deleted = snapshot(env, identifier="deleted")
    async with env.runtime.proactive_round() as current:
        result = await ReadTaskResultsStep(app_context=env.context)(pa_round_id=current.id)
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
        await ReadTaskResultsStep(app_context=env.context)(pa_round_id="fake")
    async with env.runtime.proactive_round() as current:
        with pytest.raises(PermissionError):
            await ReadTaskResultsStep(app_context=env.context)(pa_round_id=current.id, _allowed_paths=[])
    assert frontmatter.load(path)["status"] == "unread"


@pytest.mark.parametrize("kind", ["daily", "date", "task", "file"])
def test_result_symlinks_fail_closed(pa_workspace, tmp_path, kind):
    env = pa_workspace
    external = tmp_path / "external"
    external.mkdir()
    path = env.runtime.tasks.daily_path / "2026-11-08/cron/one/120000.md"
    part = {"daily": env.runtime.tasks.daily_path, "date": path.parents[2], "task": path.parent, "file": path}[kind]
    part.parent.mkdir(parents=True, exist_ok=True)
    part.symlink_to(external if kind != "file" else external / "note.md", target_is_directory=kind != "file")
    if kind == "file":
        (external / "note.md").write_text("private")
    with pytest.raises(ValueError, match="without symlinks"):
        env.runtime.tasks.paths("one")


@pytest.mark.asyncio
async def test_scheduler_reconciles_edits_and_keeps_valid_schedule_on_invalid_json(pa_workspace, monkeypatch):
    env = pa_workspace
    configure(env, [task()])
    env.context.jobs["task_agent"] = BaseJob(app_context=env.context, steps=[{"backend": "task_agent_step"}])
    created = []

    class ObservedCron(CronJob):
        """Observe actual framework start/close behavior without running model calls."""

        async def _start(self):
            await super()._start()
            created.append(self)

    monkeypatch.setattr("reme.steps.proactive.task_schedule.CronJob", ObservedCron)
    stop = asyncio.Event()
    pending = asyncio.create_task(TaskScheduleStep(app_context=env.context)(RuntimeContext(stop_event=stop)))
    try:
        await wait_until(lambda: len(created) == 1)
        first = created[0]
        env.runtime.tasks_path.write_text("[")
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
    env.context.jobs["task_agent"] = BaseJob(
        app_context=env.context,
        steps=[{"backend": "task_agent_step", "job_tools": ["screen_capture"]}],
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

    monkeypatch.setattr("reme.steps.proactive.task_schedule.CronJob", OnceCron)
    pending = asyncio.create_task(
        TaskScheduleStep(app_context=env.context)(RuntimeContext(stop_event=asyncio.Event())),
    )
    try:
        await wait_until(lambda: len(list(env.runtime.tasks.daily_path.glob("*/cron/*/*.md"))) == 2)
        assert env.agent.reply.await_count == 4
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
    stub_model(env, monkeypatch, content="x" * 512)
    with pytest.raises(ValueError, match="read limit"):
        await TaskAgentStep(app_context=env.context)(task=task())
    assert not env.runtime.tasks.paths("one")


@pytest.mark.asyncio
async def test_scheduler_closes_remaining_jobs_after_one_close_fails(pa_workspace, monkeypatch):
    env = pa_workspace
    configure(env, [task(), task("two")])
    env.context.jobs["task_agent"] = BaseJob(app_context=env.context, steps=[{"backend": "task_agent_step"}])
    jobs = []

    class FailingCloseCron(CronJob):
        """Fail one close after releasing that job, leaving peer cleanup observable."""

        async def _start(self):
            await super()._start()
            jobs.append(self)

        async def close(self):
            await super().close()
            if self.name == "task_one":
                raise RuntimeError("close failed")

    monkeypatch.setattr("reme.steps.proactive.task_schedule.CronJob", FailingCloseCron)
    stop = asyncio.Event()
    pending = asyncio.create_task(TaskScheduleStep(app_context=env.context)(RuntimeContext(stop_event=stop)))
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
        await TaskAgentStep(app_context=env.context)(RuntimeContext(stop_event=stop), task=task())
    env.agent.reply.assert_not_awaited()
    rows = [json.loads(line) for line in env.runtime.task_runs_path.read_text().splitlines()]
    assert rows[-1]["status"] == "interrupted"


def captured_frame(env, name="new.png"):
    from PIL import Image

    path = env.runtime.workspace_path / "resource/screen_capture" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (64, 64), "white").save(path)
    return CapturedDisplay(
        display_id="stable-display",
        stable_id=True,
        bounds=(0, 0, 64, 64),
        pixel_size=(64, 64),
        path=env.runtime.to_workspace_relative(path),
        scope="task:one",
        captured_at=env.now[0],
    ).model_dump(mode="json")


@pytest.mark.asyncio
async def test_unchanged_screen_preflight_skips_both_model_queries_and_unread_summary(pa_workspace):
    env = pa_workspace
    env.agent.reply = AsyncMock()
    capture = AsyncMock(return_value=Response(answer={"context": "与上次一致", "paths": []}))
    env.context.jobs["screen_capture"] = capture
    result = await TaskAgentStep(app_context=env.context)(task=task(preflight="screen_capture"))
    assert result.answer == {"task_id": "one", "status": "skipped", "reason": "screen_unchanged"}
    env.agent.reply.assert_not_awaited()
    assert not env.runtime.tasks.paths("one")
    rows = [json.loads(line) for line in env.runtime.task_runs_path.read_text().splitlines()]
    assert [row["status"] for row in rows] == ["running", "skipped"]
    assert rows[-1]["reason"] == "screen_unchanged"


@pytest.mark.asyncio
async def test_changed_preflight_keeps_original_query_and_reuses_normal_tool_result(pa_workspace, monkeypatch):
    env = pa_workspace
    frame = captured_frame(env)
    payload = {"context": "changed", "paths": [frame["path"]]}
    capture = AsyncMock(return_value=Response(answer=payload, metadata={"screen_capture": [frame]}))
    env.context.jobs["screen_capture"] = capture
    stub_model(env, monkeypatch)
    result = await TaskAgentStep(app_context=env.context, job_tools=["screen_capture"])(
        task=task(preflight="screen_capture"),
    )
    calls = env.agent.reply.await_args_list
    assert calls[0].args[0] == task()["query"]
    assert calls[0].kwargs["job_tools"] == ["screen_capture"]
    assert calls[0].kwargs["injected_job_kwargs"]["_screen_preflight"] == {"result": payload, "frames": [frame]}
    assert calls[0].kwargs["injected_job_kwargs"] == calls[1].kwargs["injected_job_kwargs"]
    assert capture.await_args.kwargs["_screen_baseline"] == []
    post = frontmatter.load(env.runtime.workspace_path / result.answer["path"])
    assert post["screen_capture"] == [frame]
    assert env.runtime.tasks.screen_baseline("one") == [frame]


@pytest.mark.asyncio
async def test_model_failure_keeps_previous_successful_screen_baseline(pa_workspace, monkeypatch):
    env = pa_workspace
    old, new = captured_frame(env, "old.png"), captured_frame(env)
    previous = snapshot(env, time="110000")
    previous.write_text(frontmatter.dumps(frontmatter.Post("previous", task_id="one", screen_capture=[old])))
    before = previous.read_bytes()
    env.context.jobs["screen_capture"] = AsyncMock(
        return_value=Response(
            answer={"context": "changed", "paths": [new["path"]]},
            metadata={"screen_capture": [new]},
        ),
    )
    stub_model(env, monkeypatch, fail_at=1)
    with pytest.raises(RuntimeError, match="provider failed"):
        await TaskAgentStep(app_context=env.context)(task=task(preflight="screen_capture"))
    assert previous.read_bytes() == before
    assert env.runtime.tasks.screen_baseline("one") == [old]
    assert env.context.jobs["screen_capture"].await_args.kwargs["_screen_baseline"] == [old]


@pytest.mark.asyncio
async def test_screen_capture_failure_is_not_misreported_as_unchanged(pa_workspace):
    env = pa_workspace
    env.agent.reply = AsyncMock()
    env.context.jobs["screen_capture"] = AsyncMock(return_value=Response(success=False, answer="capture timed out"))
    with pytest.raises(RuntimeError, match="timed out"):
        await TaskAgentStep(app_context=env.context)(task=task(preflight="screen_capture"))
    env.agent.reply.assert_not_awaited()
    rows = [json.loads(line) for line in env.runtime.task_runs_path.read_text().splitlines()]
    assert rows[-1]["status"] == "failed" and rows[-1]["reason"] is None
