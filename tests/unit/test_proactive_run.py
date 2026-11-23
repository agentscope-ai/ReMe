"""Single-turn file decisions and delivery in automatic proactive rounds."""

# pylint: disable=missing-function-docstring,protected-access

import asyncio
from datetime import datetime, timedelta
from os.path import relpath
from pathlib import Path
from unittest.mock import AsyncMock, Mock

import frontmatter
import pytest

from reme.config import resolve_app_config
from reme.core.registry import R
from reme.enumeration import ChunkEnum, ComponentEnum
from reme.schema import Event, StreamChunk
from reme.steps.proactive.agent import ProactiveAgentStep
from reme.steps.proactive.tools.manage_proactive_agent import ManageProactiveAgentStep
from .test_proactive_runtime import create_agent, running_runtime


async def queue_cron_trigger(runtime, name, monkeypatch):
    agent = await runtime.definition(name)
    execution = runtime.execution(name)
    if execution.schedule is None:
        await runtime._schedule_cron(agent)
    trigger = execution.schedule[1]
    monkeypatch.setattr(runtime, "now", lambda: trigger)
    await runtime._schedule_cron(agent)
    return trigger


async def create_round_task(runtime, monkeypatch, agent_type):
    if agent_type == "cron":
        await runtime.create({"agent_name": "monitor", "task_description": "Monitor", "cron": "0 * * * *"})
        await queue_cron_trigger(runtime, "monitor", monkeypatch)
    else:
        await create_agent(runtime)


SKIP = "---\nreminder_priority: skip\nreminder_message: No new information\n---\n"
REMIND = "---\nreminder_priority: {priority}\nreminder_message: Verified result\n---{body}"


def install_round(
    runtime,
    wrapper,
    monkeypatch,
    paths,
    *,
    text=REMIND.format(priority="P1", body="\nNew progress\n"),
    failed=False,
    error=None,
    reply_text="Finished",
):
    calls = []

    async def generate(query, **options):
        calls.append((query, options))
        assert "output_schema" not in options
        if text is not None:
            runtime.files.write(paths["record_path"], text)
        if error is not None:
            raise error
        result = {
            "session_id": options["resume"],
            "status": "failed" if failed else "completed",
            "error": "Model failed" if failed else None,
            "result": reply_text,
            "structured_output": {"notifications": [{"message": "Ignore this"}]},
            "usage": None,
            "raw": {},
        }
        yield StreamChunk(chunk_type=ChunkEnum.REPLY_END, status=result["status"], chunk=result)

    monkeypatch.setattr(wrapper, "reply_stream", generate)
    return calls


@pytest.mark.parametrize("profile", ["proactive", "proactive_cc", "demo"])
def test_profiles_use_the_same_proactive_workflow(profile):
    job = resolve_app_config(config=profile, log_config=False)["jobs"]["run_proactive_agent"]
    (step,) = job["steps"]
    assert R.get(ComponentEnum.STEP, step["backend"]) is ProactiveAgentStep
    assert job["enable_serve"] is False
    assert step["recent_history_hours"] == 24


@pytest.mark.parametrize(
    "agent_type,language,priority,body",
    [
        ("proactive", "zh", "P1", "\n\n    indented code\n\n"),
        ("proactive", "en", "P2", "\n"),
        ("cron", "zh", "P1", "\n"),
        ("cron", "en", "P2", "\n"),
    ],
)
@pytest.mark.asyncio
async def test_round_publishes_and_acknowledges_the_same_note(
    tmp_path,
    monkeypatch,
    agent_type,
    language,
    priority,
    body,
):
    async with running_runtime(tmp_path, language=language) as (runtime, context, wrapper):
        await create_round_task(runtime, monkeypatch, agent_type)
        scheduled = runtime.now() if agent_type == "cron" else None
        now = scheduled + timedelta(minutes=7) if scheduled else runtime.now()
        monkeypatch.setattr(runtime, "now", lambda: now)
        window_start = now + timedelta(hours=1)
        await runtime.update(
            "monitor",
            {
                "notify_window": {
                    "start": f"{window_start:%H:%M}",
                    "end": f"{window_start + timedelta(minutes=1):%H:%M}",
                },
            },
        )
        step = ProactiveAgentStep(app_context=context)
        paths = step._paths(await runtime.definition("monitor"), now)
        path = paths["record_path"]
        queries = install_round(runtime, wrapper, monkeypatch, paths, text=REMIND.format(priority=priority, body=body))

        result = await step(agent_name="monitor")
        assert result.success and result.answer["processed"] == 1
        assert result.answer["notifications"][0]["message"] == "Verified result"
        ((query, options),) = queries
        assert options["sandbox"] == "workspace-write"
        if language == "zh":
            assert "## 任务要求\nMonitor" in query
            assert "## 2. 查询历史" in query and "## 3. 记录与提醒" in query
            assert "## 本轮事件" in query
            if scheduled is not None:
                assert f"定时触发，计划时间 {scheduled.isoformat()}" in query
                assert f"当前时间：{now.isoformat()}" in query
        else:
            assert "## Task requirements\nMonitor" in query
            assert "## 2. Search past records" in query and "## 3. Record and remind" in query
            assert "## Events for this run" in query
            if scheduled is not None:
                assert f"Scheduled trigger, planned time {scheduled.isoformat()}" in query
                assert f"Current time: {now.isoformat()}" in query
        assert ("Finished" in query) == (agent_type == "proactive")
        assert list((tmp_path / "daily").rglob("*.md")) == [path]

        (ref,) = runtime.outbox.records
        related = [path.relative_to(tmp_path).as_posix()]
        item = runtime.outbox.message(ref)
        assert item["message"] == "Verified result"
        assert item["priority"] == priority
        assert related == [item["id"]]
        assert item["delivered_at"] and item["id"] == ref == related[0]
        runtime.notifications.last_activity_at = now - timedelta(hours=1)
        run_job = AsyncMock()
        await runtime.notifications.tick(run_job)
        run_job.assert_not_awaited()
        assert runtime.notifications.worker is None

        await runtime.notifications.consume([ref])
        saved = path.read_text()
        assert saved.endswith(f"---{body}")
        metadata = frontmatter.loads(saved).metadata
        assert set(metadata) == {"reminder_priority", "reminder_message", "delivered_at", "read_at"}
        assert metadata["delivered_at"] == item["delivered_at"] and metadata["read_at"]

    async with running_runtime(tmp_path) as (runtime, _, _):
        item = await runtime.outbox.read("monitor", ref)
        assert [item["id"]] == related and item["read_at"]


@pytest.mark.parametrize("error", [OSError("interrupted after commit"), asyncio.CancelledError()])
@pytest.mark.asyncio
async def test_delivery_recovers_from_a_committed_input_receipt(tmp_path, monkeypatch, error):
    async with running_runtime(tmp_path, language="zh") as (runtime, context, wrapper):
        await create_agent(runtime)
        now = runtime.now()
        monkeypatch.setattr(runtime, "now", lambda: now)
        step = ProactiveAgentStep(app_context=context)
        paths = step._paths(await runtime.definition("monitor"), now)
        path = paths["record_path"]
        draft = REMIND.format(priority="P1", body="\nDraft body\n")
        install_round(runtime, wrapper, monkeypatch, paths, text=draft)
        ack = runtime.inbox.ack

        async def interrupt(*args, **kwargs):
            await ack(*args, **kwargs)
            raise error

        monkeypatch.setattr(runtime.inbox, "ack", interrupt)
        with pytest.raises(type(error)):
            await step(agent_name="monitor")
        assert path.read_text() == draft
        assert not runtime.notifications.pending

    async with running_runtime(tmp_path) as (runtime, _, _):
        ref = path.relative_to(tmp_path).as_posix()
        assert runtime.notifications.pending == {ref}
        assert (await runtime.status("monitor"))["pending_events"] == 0
        document = frontmatter.loads(path.read_text())
        assert document["delivered_at"] and document.content == "Draft body"


@pytest.mark.asyncio
async def test_cron_can_finish_with_a_skip_record(tmp_path, monkeypatch):
    async with running_runtime(tmp_path, language="zh") as (runtime, context, wrapper):
        await create_round_task(runtime, monkeypatch, "cron")
        now = runtime.now()
        monkeypatch.setattr(runtime, "now", lambda: now)
        step = ProactiveAgentStep(app_context=context, recent_history_hours=48)
        paths = step._paths(await runtime.definition("monitor"), now)

        old_record = paths["daily_dir"] / "2000-01-01" / "monitor_100000.md"
        runtime.files.write(old_record, SKIP + "private old body")
        record = paths["record_path"]
        queries = install_round(runtime, wrapper, monkeypatch, paths, text=SKIP)

        result = await step(agent_name="monitor")
        assert result.success and result.answer["processed"] == 1
        assert result.answer["notifications"] == []
        assert record.read_text() == SKIP
        assert (await runtime.status("monitor"))["pending_events"] == 0
        ((query, _),) = queries
        assert "留到第 3 步处理" in query
        assert "优先核对最近 48 小时" in query
        assert old_record.name not in query and "private old body" not in query
        assert not (await runtime.outbox.list("monitor"))["items"]


@pytest.mark.parametrize(
    "draft,error",
    [
        (SKIP.replace("skip", "P3"), ValueError),
        (SKIP.replace("No new information", "'  '"), ValueError),
        (SKIP.replace("---\n", "---\ndelivered_at: 2026-11-22T08:00:00+08:00\n", 1), ValueError),
        (REMIND.format(priority="P1", body="\nNew result\n"), OSError),
    ],
)
@pytest.mark.asyncio
async def test_failed_round_keeps_the_input_and_removes_only_its_draft(tmp_path, monkeypatch, draft, error):
    async with running_runtime(tmp_path, language="zh") as (runtime, context, wrapper):
        await create_agent(runtime)
        now = runtime.now()
        monkeypatch.setattr(runtime, "now", lambda: now)
        step = ProactiveAgentStep(app_context=context)
        paths = step._paths(await runtime.definition("monitor"), now)
        path = paths["record_path"]
        install_round(runtime, wrapper, monkeypatch, paths, text=draft)
        if error is OSError:
            monkeypatch.setattr(runtime.inbox, "ack", AsyncMock(side_effect=OSError("receipt unavailable")))
        with pytest.raises(error):
            await step(agent_name="monitor")
        assert not path.exists()
        assert (await runtime.status("monitor"))["pending_events"] == 1


@pytest.mark.parametrize("draft", [None, SKIP, SKIP + "Normal progress"])
@pytest.mark.asyncio
async def test_event_round_can_finish_without_an_inbox_message(tmp_path, monkeypatch, draft):
    async with running_runtime(tmp_path, language="zh") as (runtime, context, wrapper):
        await create_agent(runtime)
        now = runtime.now()
        monkeypatch.setattr(runtime, "now", lambda: now)
        step = ProactiveAgentStep(app_context=context)
        paths = step._paths(await runtime.definition("monitor"), now)
        path = paths["record_path"]
        reply_text = "NO_KEY_INFORMATION" if draft is None else None
        queries = install_round(runtime, wrapper, monkeypatch, paths, text=draft, reply_text=reply_text)

        result = await step(agent_name="monitor")
        assert result.success and result.answer["processed"] == 1
        assert result.answer["notifications"] == []
        assert (await runtime.status("monitor"))["pending_events"] == 0
        if draft is None:
            assert "只回复「NO_KEY_INFORMATION」" in queries[0][0]
            assert not path.exists()
        else:
            assert path.read_text() == draft


@pytest.mark.parametrize(
    "agent_type,reply_text,model_error,expected_error",
    [
        ("proactive", "还没有写入结果", None, FileNotFoundError),
        ("proactive", "NO_KEY_INFORMATION, but writing failed", None, FileNotFoundError),
        ("proactive", None, RuntimeError("model failed"), RuntimeError),
        ("proactive", None, TimeoutError("model timed out"), TimeoutError),
        ("cron", "NO_KEY_INFORMATION", None, FileNotFoundError),
    ],
)
@pytest.mark.asyncio
async def test_missing_record_requires_an_explicit_successful_event_skip(
    tmp_path,
    monkeypatch,
    agent_type,
    reply_text,
    model_error,
    expected_error,
):
    async with running_runtime(tmp_path, language="zh") as (runtime, context, wrapper):
        await create_round_task(runtime, monkeypatch, agent_type)
        now = runtime.now()
        monkeypatch.setattr(runtime, "now", lambda: now)
        step = ProactiveAgentStep(app_context=context)
        paths = step._paths(await runtime.definition("monitor"), now)
        install_round(runtime, wrapper, monkeypatch, paths, text=None, error=model_error, reply_text=reply_text)

        with pytest.raises(expected_error):
            await step(agent_name="monitor")
        assert (await runtime.status("monitor"))["pending_events"] == 1


@pytest.mark.parametrize("field", ["body", "message"])
@pytest.mark.asyncio
async def test_broken_local_links_warn_without_rejecting_the_record(tmp_path, monkeypatch, field):
    workspace = tmp_path / "workspace"
    async with running_runtime(workspace) as (runtime, context, wrapper):
        await create_agent(runtime)
        now = runtime.now()
        monkeypatch.setattr(runtime, "now", lambda: now)
        step = ProactiveAgentStep(app_context=context)
        step.logger = Mock()
        paths = step._paths(await runtime.definition("monitor"), now)
        path = paths["record_path"]
        runtime.files.write(path.with_name("proof (confirmed).md"), "Evidence\n")
        outside = tmp_path / "outside.md"
        outside.write_text("Outside workspace\n")
        outside_path = Path(relpath(outside, path.parent)).as_posix()
        record = path.name
        wrong_depth = Path(relpath(paths["record_path"], path.parent.parent)).as_posix()
        body = (
            f"[Record]({record}#details)\n"
            "[Evidence][proof] [Encoded evidence](proof%20%28confirmed%29.md)\n\n"
            "[proof]: <proof (confirmed).md>\n\n"
            "[This section](#details)\n"
            "[Web](https://example.com/news)\n"
            "`[Example](not-real-inline.md)`\n"
            "```markdown\n[Example](not-real-fenced.md)\n```\n"
            "~~~markdown\n[Example](not-real-tilde.md)\n~~~\n"
            "[[legacy-missing.md|Legacy]]\n"
            "[Missing](missing.md) [Repeated](missing.md)\n"
            f"[Wrong depth]({wrong_depth})\n"
            f"[Outside]({outside_path})\n"
        )
        document = frontmatter.Post(
            body if field == "body" else "Progress",
            reminder_priority="P1",
            reminder_message=body if field == "message" else "New result",
        )
        install_round(runtime, wrapper, monkeypatch, paths, text=frontmatter.dumps(document))
        response = await step(agent_name="monitor")

        assert response.success and response.answer["processed"] == 1
        assert len(response.answer["notifications"]) == 1
        warnings = [call.args[0] for call in step.logger.warning.call_args_list]
        assert len(warnings) == 3
        assert all(f"source={path}" in warning for warning in warnings)
        assert any("target='missing.md': target does not exist" in warning for warning in warnings)
        assert any(f"target={wrong_depth!r}: target does not exist" in warning for warning in warnings)
        assert any(
            f"target={outside_path!r}" in warning and "stay inside the workspace" in warning for warning in warnings
        )
        saved = frontmatter.loads(path.read_text())
        assert saved.content == document.content.strip()
        assert saved["reminder_message"] == document["reminder_message"]
        assert outside.read_text() == "Outside workspace\n"
        logs = "\n".join(call.args[0] for call in step.logger.info.call_args_list)
        assert "result=validated notifications=1" in logs and f"path={path}" in logs


@pytest.mark.asyncio
async def test_queued_observations_run_before_cron_trigger_after_switch(tmp_path, monkeypatch):
    async with running_runtime(tmp_path) as (runtime, context, wrapper):
        await create_agent(runtime)
        attachment = "resource/alert.txt"
        runtime.files.write(runtime.files.path(attachment), "Evidence")
        await runtime.append_events(
            "monitor",
            [
                Event(
                    id="observation",
                    event_ts=runtime.now(),
                    source="experiment",
                    description="Alarm from the experiment",
                    paths=[attachment],
                ),
            ],
        )
        await ManageProactiveAgentStep(app_context=context)(action="update", agent_name="monitor", cron="* * * * *")
        now = await queue_cron_trigger(runtime, "monitor", monkeypatch)
        step = ProactiveAgentStep(app_context=context)
        paths = step._paths(await runtime.definition("monitor"), now)
        calls = install_round(runtime, wrapper, monkeypatch, paths, text=None, reply_text="NO_KEY_INFORMATION")

        response = await step(agent_name="monitor")

        assert response.success and response.answer["processed"] == 2
        assert response.metadata["reme.agent_run"]["agent_type"] == "proactive"
        query = calls[0][0]
        assert "Alarm from the experiment" in query and attachment in query
        snapshot = await runtime.inbox.snapshot("monitor")
        assert snapshot["receipt"].item_ids == ["event-1", "observation"]
        assert await runtime.inbox.pending_ids("monitor") == {f"cron-monitor-{int(now.timestamp())}"}


@pytest.mark.parametrize("language", ["en", "zh"])
@pytest.mark.asyncio
async def test_backlogged_cron_processes_each_trigger_with_current_time(tmp_path, monkeypatch, language):
    async with running_runtime(tmp_path, language=language) as (runtime, context, wrapper):
        await runtime.create({"agent_name": "timer", "task_description": "Summarize prior hour", "cron": "0 * * * *"})
        first = await queue_cron_trigger(runtime, "timer", monkeypatch)
        second = await queue_cron_trigger(runtime, "timer", monkeypatch)
        now = second + timedelta(minutes=30)
        monkeypatch.setattr(runtime, "now", lambda: now)
        await ManageProactiveAgentStep(app_context=context)(action="update", agent_name="timer", cron=None)
        agent = await runtime.definition("timer")

        for due in (first, second):
            with monkeypatch.context() as round_patch:
                step = ProactiveAgentStep(app_context=context)
                paths = step._paths(agent, now)
                calls = install_round(runtime, wrapper, round_patch, paths, text=SKIP)
                response = await step(agent_name="timer")

            assert response.answer["processed"] == 1
            query = calls[0][0]
            assert now.isoformat() in query
            assert due.isoformat() in query
            receipt = (await runtime.inbox.snapshot("timer"))["receipt"]
            assert receipt.item_ids == [f"cron-timer-{int(due.timestamp())}"]


@pytest.mark.parametrize("switch_to_event", [False, True])
@pytest.mark.asyncio
async def test_cron_skip_record_keeps_trigger_pending_after_model_failure(tmp_path, monkeypatch, switch_to_event):
    async with running_runtime(tmp_path) as (runtime, context, wrapper):
        await runtime.create({"agent_name": "timer", "task_description": "Save progress", "cron": "* * * * *"})
        trigger = await queue_cron_trigger(runtime, "timer", monkeypatch)
        if switch_to_event:
            await queue_cron_trigger(runtime, "timer", monkeypatch)
            await ManageProactiveAgentStep(app_context=context)(action="update", agent_name="timer", cron=None)
            await runtime.append_events(
                "timer",
                [Event(event_ts=runtime.now(), source="experiment", description="Later observation")],
            )
        pending = await runtime.inbox.pending_ids("timer")

        step = ProactiveAgentStep(app_context=context)
        paths = step._paths(await runtime.definition("timer"), runtime.now())
        calls = install_round(runtime, wrapper, monkeypatch, paths, text=None, failed=True)
        with pytest.raises(RuntimeError, match="Model failed"):
            await step(agent_name="timer")

        snapshot = await runtime.inbox.snapshot("timer")
        assert await runtime.inbox.pending_ids("timer") == pending
        assert snapshot["receipt"] is None and snapshot["in_flight_count"] == 0
        assert runtime.execution("timer").retry_items == [f"cron-timer-{int(trigger.timestamp())}"]
        assert "## Events for this run" in calls[0][0]
        assert not paths["record_path"].exists()


@pytest.mark.asyncio
async def test_failed_turn_accepts_a_valid_record(tmp_path, monkeypatch):
    async with running_runtime(tmp_path) as (runtime, context, wrapper):
        await create_agent(runtime)
        now = runtime.now()
        monkeypatch.setattr(runtime, "now", lambda: now)
        step = ProactiveAgentStep(app_context=context)
        paths = step._paths(await runtime.definition("monitor"), now)
        calls = install_round(runtime, wrapper, monkeypatch, paths, failed=True)

        response = await step(agent_name="monitor")
        assert response.success
        assert len(calls) == 1 and paths["record_path"].exists()
        assert (await runtime.status("monitor"))["pending_events"] == 0
        await runtime.outbox.sync()
        assert len(runtime.outbox.records) == 1


@pytest.mark.asyncio
async def test_timeout_accepts_a_valid_record(tmp_path, monkeypatch):
    async with running_runtime(tmp_path) as (runtime, context, wrapper):
        await create_agent(runtime)
        now = runtime.now()
        monkeypatch.setattr(runtime, "now", lambda: now)
        step = ProactiveAgentStep(app_context=context, timeout_seconds=0.5)
        paths = step._paths(await runtime.definition("monitor"), now)
        install_round(runtime, wrapper, monkeypatch, paths)
        original = wrapper.reply_stream

        async def pause(query, **kwargs):
            async for chunk in original(query, **kwargs):
                yield chunk
            await asyncio.Event().wait()

        monkeypatch.setattr(wrapper, "reply_stream", pause)
        response = await step(agent_name="monitor")
        assert response.success
        document = frontmatter.loads(runtime.files.read(paths["record_path"]))
        assert document["reminder_priority"] == "P1" and document["delivered_at"]
        snapshot = await runtime.inbox.snapshot("monitor")
        assert snapshot["pending_count"] == 0 and snapshot["in_flight_count"] == 0
        await runtime.outbox.sync()
        assert len(runtime.outbox.records) == 1


@pytest.mark.asyncio
async def test_cancelled_round_removes_only_its_uncommitted_record(tmp_path, monkeypatch):
    async with running_runtime(tmp_path) as (runtime, context, wrapper):
        await create_agent(runtime)
        now = runtime.now()
        monkeypatch.setattr(runtime, "now", lambda: now)
        step = ProactiveAgentStep(app_context=context)
        paths = step._paths(await runtime.definition("monitor"), now)
        path = paths["record_path"]
        earlier = path.with_name("monitor_000000_1.md")
        runtime.files.write(earlier, SKIP)
        ready = asyncio.Event()

        async def generate(*_args, **_kwargs):
            runtime.files.write(path, SKIP)
            ready.set()
            await asyncio.Event().wait()
            yield StreamChunk(chunk_type=ChunkEnum.CONTENT, chunk="unreachable")

        monkeypatch.setattr(wrapper, "reply_stream", generate)
        task = asyncio.create_task(step(agent_name="monitor"))
        try:
            await asyncio.wait_for(ready.wait(), 2)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        finally:
            if not task.done():
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
        assert not path.exists()
        assert earlier.read_text() == SKIP
        assert (await runtime.inbox.snapshot("monitor"))["pending_count"] == 1


@pytest.mark.asyncio
async def test_same_second_records_get_unique_paths(tmp_path):
    async with running_runtime(tmp_path) as (runtime, context, _):
        await create_agent(runtime)
        agent = await runtime.definition("monitor")
        step = ProactiveAgentStep(app_context=context)
        now = datetime.fromisoformat("2026-01-01T00:00:00+08:00")
        directory = tmp_path / "daily" / "2026-01-01"
        for name in ("monitor_000000.md", "monitor_000000_1.md"):
            path = step._paths(agent, now)["record_path"]
            assert path == directory / name
            runtime.files.write(path, SKIP)
        assert step._paths(agent, now)["record_path"] == directory / "monitor_000000_2.md"
