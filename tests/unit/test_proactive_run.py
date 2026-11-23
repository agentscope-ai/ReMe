"""File decisions, repair and delivery in automatic proactive rounds."""

# pylint: disable=missing-function-docstring,protected-access

import asyncio
from datetime import datetime, timedelta
from os.path import relpath
from pathlib import Path
from unittest.mock import AsyncMock, Mock

import frontmatter
import pytest

from reme.enumeration import ChunkEnum
from reme.schema import Event, StreamChunk
from reme.steps.proactive.agent import ProactiveAgentStep
from reme.steps.proactive.tools.manage_proactive_agent import ManageProactiveAgentStep
from .test_proactive_runtime import create_agent, running_manager


async def queue_cron_trigger(runtime, name, monkeypatch):
    agent = await runtime.definition(name)
    execution = runtime.runtimes[name]
    if execution.schedule is None:
        await runtime.runtimes[agent.agent_name]._schedule_cron(agent)
    trigger = execution.schedule[1]
    monkeypatch.setattr(runtime, "now", lambda: trigger)
    await runtime.runtimes[agent.agent_name]._schedule_cron(agent)
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
    repaired_metadata=None,
):
    calls = []

    async def generate(query, **options):
        calls.append((query, options))
        repairing = "output_schema" in options
        if repairing:
            assert options["sandbox"] == "read-only"
            assert options["job_tools"] == []
        elif text is not None:
            runtime.files.write(paths["record_path"], text)
        if error is not None:
            raise error
        result = {
            "session_id": options["resume"],
            "status": "failed" if failed else "completed",
            "error": "Model failed" if failed else None,
            "result": reply_text,
            "structured_output": repaired_metadata if repairing else {"notifications": [{"message": "Ignore this"}]},
            "usage": None,
            "raw": {},
        }
        yield StreamChunk(chunk_type=ChunkEnum.REPLY_END, status=result["status"], chunk=result)

    monkeypatch.setattr(wrapper, "reply_stream", generate)
    return calls


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
    async with running_manager(tmp_path, language=language) as (runtime, context, wrapper):
        await create_round_task(runtime, monkeypatch, agent_type)
        now = runtime.now()
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
        assert runtime.files.read(paths["task_path"]) in query
        assert ("Finished" in query) == (agent_type == "proactive")
        assert list((tmp_path / "daily").rglob("*.md")) == [path]

        (ref,) = runtime.outbox.entries
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
        assert set(metadata) == {"reminder_priority", "reminder_message"}
        assert (await runtime.outbox.read("monitor", ref))["read_at"]

    async with running_manager(tmp_path) as (runtime, _, _):
        item = await runtime.outbox.read("monitor", ref)
        assert [item["id"]] == related and item["read_at"]


@pytest.mark.parametrize(
    "agent_type,language,invalid_message,error",
    [
        ("proactive", "zh", "Updated palette\n  to blue #20774D\n  and white", "while parsing"),
        ("cron", "en", "", "reminder_message"),
    ],
)
@pytest.mark.parametrize("edit_during_repair", [False, True])
@pytest.mark.asyncio
async def test_record_format_is_repaired_in_the_same_session(
    tmp_path,
    monkeypatch,
    agent_type,
    language,
    invalid_message,
    error,
    edit_during_repair,
):
    async with running_manager(tmp_path, language=language) as (runtime, context, wrapper):
        await create_round_task(runtime, monkeypatch, agent_type)
        now = runtime.now()
        monkeypatch.setattr(runtime, "now", lambda: now)
        step = ProactiveAgentStep(app_context=context)
        paths = step._paths(await runtime.definition("monitor"), now)
        path = paths["record_path"]
        body = "\n# Progress\n"
        repaired = REMIND.format(priority="P2", body=body).replace("Verified result", "'Palette #20774D'")
        draft = repaired.replace("'Palette #20774D'", invalid_message)
        calls = install_round(
            runtime,
            wrapper,
            monkeypatch,
            paths,
            text=draft,
            repaired_metadata={"reminder_priority": "P2", "reminder_message": "Palette #20774D"},
        )
        original = wrapper.reply_stream
        user_edit = repaired + "User's own edit.\n"

        async def stream(query, **options):
            attempt = len(calls)
            assert (await runtime.inbox.snapshot("monitor"))["receipt"] is None
            assert not runtime.outbox.entries
            if attempt and edit_during_repair:
                runtime.files.write(path, user_edit)
            yield StreamChunk(
                chunk_type=ChunkEnum.TOOL_CALL,
                turn_id=f"turn-{attempt}",
                tool_call_id="write",
                tool_call_name="apply_patch",
            )
            yield StreamChunk(chunk_type=ChunkEnum.USAGE, chunk={"output_tokens": 5})
            async for chunk in original(query, **options):
                yield chunk

        monkeypatch.setattr(wrapper, "reply_stream", stream)
        saved_calls = AsyncMock(return_value=[{"id": "write", "name": "apply_patch"}])
        monkeypatch.setattr(wrapper, "get_tool_calls", saved_calls)
        if edit_during_repair:
            with pytest.raises(RuntimeError, match="Agent output changed outside this round"):
                await step(agent_name="monitor")
            assert path.read_text() == user_edit
            assert (await runtime.status("monitor"))["pending_events"] == 1
            assert not runtime.outbox.entries
            return
        result = await step(agent_name="monitor")

        assert result.success and result.answer["processed"] == 1
        assert result.answer["notifications"][0]["message"] == "Palette #20774D"
        assert len(runtime.outbox.entries) == len(wrapper.created) == 1
        assert (await runtime.status("monitor"))["pending_events"] == 0
        assert path.read_text().endswith(f"---{body}")
        (_, initial), (repair, followup) = calls
        assert initial["resume"] == followup["resume"] == "session-1"
        assert path.relative_to(tmp_path).as_posix() in repair
        assert error in repair
        assert followup["output_schema"]["required"] == ["reminder_priority", "reminder_message"]
        assert [call.args for call in saved_calls.await_args_list] == [("session-1", "turn-0"), ("session-1", "turn-1")]
        stats = result.metadata["reme.agent_run"]
        assert stats["tool_calls"] == 2 and stats["usage"]["output_tokens"] == 10


@pytest.mark.parametrize("error", [OSError("interrupted after commit"), asyncio.CancelledError()])
@pytest.mark.asyncio
async def test_delivery_recovers_from_a_committed_input_receipt(tmp_path, monkeypatch, error):
    async with running_manager(tmp_path, language="zh") as (runtime, context, wrapper):
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

    async with running_manager(tmp_path) as (runtime, _, _):
        ref = path.relative_to(tmp_path).as_posix()
        assert runtime.notifications.pending == {ref}
        assert (await runtime.status("monitor"))["pending_events"] == 0
        document = frontmatter.loads(path.read_text())
        assert runtime.outbox.entries[ref].reminder.delivered_at and document.content == "Draft body"
        assert path.read_text() == draft


@pytest.mark.asyncio
async def test_failed_record_with_delivery_frontmatter_never_enters_the_inbox(tmp_path, monkeypatch):
    async with running_manager(tmp_path) as (runtime, context, wrapper):
        await create_agent(runtime)
        now = runtime.now()
        monkeypatch.setattr(runtime, "now", lambda: now)
        step = ProactiveAgentStep(app_context=context)
        paths = step._paths(await runtime.definition("monitor"), now)
        text = REMIND.format(priority="P0", body="\nCopied result\n")
        text = text.replace("---\n", f"---\ndelivered_at: '{now.isoformat()}'\n", 1)
        install_round(runtime, wrapper, monkeypatch, paths, text=text, failed=True)
        reply_stream = wrapper.reply_stream

        async def generate(query, **options):
            async for chunk in reply_stream(query, **options):
                assert not (await runtime.list_inputs("Remy"))["items"]
                yield chunk

        monkeypatch.setattr(wrapper, "reply_stream", generate)
        with pytest.raises(RuntimeError, match="Model failed"):
            await step(agent_name="monitor")
        assert paths["record_path"].read_text() == text
        snapshot = await runtime.inbox.snapshot("monitor")
        assert snapshot["pending_count"] == 1 and snapshot["receipt"] is None

    async with running_manager(tmp_path) as (runtime, _, _):
        assert not (await runtime.list_inputs("Remy"))["items"]


@pytest.mark.asyncio
async def test_cron_can_finish_with_a_skip_record(tmp_path, monkeypatch):
    async with running_manager(tmp_path, language="zh") as (runtime, context, wrapper):
        await create_round_task(runtime, monkeypatch, "cron")
        now = runtime.now()
        monkeypatch.setattr(runtime, "now", lambda: now)
        step = ProactiveAgentStep(app_context=context)
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
async def test_failed_round_keeps_the_input_and_record(tmp_path, monkeypatch, draft, error):
    async with running_manager(tmp_path, language="zh") as (runtime, context, wrapper):
        await create_agent(runtime)
        now = runtime.now()
        monkeypatch.setattr(runtime, "now", lambda: now)
        step = ProactiveAgentStep(app_context=context)
        paths = step._paths(await runtime.definition("monitor"), now)
        path = paths["record_path"]
        queries = install_round(runtime, wrapper, monkeypatch, paths, text=draft)
        expected = draft
        if error is OSError:
            expected += "User edit while the model is still streaming.\n"
            reply_stream = wrapper.reply_stream

            async def edit_before_turn_ends(query, **options):
                async for chunk in reply_stream(query, **options):
                    runtime.files.write(path, expected)
                    yield chunk

            monkeypatch.setattr(wrapper, "reply_stream", edit_before_turn_ends)
            monkeypatch.setattr(runtime.inbox, "ack", AsyncMock(side_effect=OSError("receipt unavailable")))
        with pytest.raises(error):
            await step(agent_name="monitor")
        assert len(queries) == (1 if error is OSError else 2)
        assert path.read_text() == expected
        assert not runtime.outbox.entries
        assert (await runtime.status("monitor"))["pending_events"] == 1


@pytest.mark.parametrize("draft", [None, SKIP, SKIP + "Normal progress"])
@pytest.mark.asyncio
async def test_event_round_can_finish_without_an_inbox_message(tmp_path, monkeypatch, draft):
    async with running_manager(tmp_path, language="zh") as (runtime, context, wrapper):
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
    async with running_manager(tmp_path, language="zh") as (runtime, context, wrapper):
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
    async with running_manager(workspace) as (runtime, context, wrapper):
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
    async with running_manager(tmp_path) as (runtime, context, wrapper):
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
    async with running_manager(tmp_path, language=language) as (runtime, context, wrapper):
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
    async with running_manager(tmp_path) as (runtime, context, wrapper):
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
        install_round(runtime, wrapper, monkeypatch, paths, text=None, failed=True)
        with pytest.raises(RuntimeError, match="Model failed"):
            await step(agent_name="timer")

        snapshot = await runtime.inbox.snapshot("timer")
        assert await runtime.inbox.pending_ids("timer") == pending
        assert snapshot["receipt"] is None and snapshot["in_flight_count"] == 0
        assert runtime.runtimes["timer"].retry_items == [f"cron-timer-{int(trigger.timestamp())}"]
        assert not paths["record_path"].exists()


@pytest.mark.asyncio
async def test_failed_turn_accepts_a_valid_record(tmp_path, monkeypatch):
    async with running_manager(tmp_path) as (runtime, context, wrapper):
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
        assert len(runtime.outbox.entries) == 1


@pytest.mark.asyncio
async def test_timeout_accepts_a_valid_record(tmp_path, monkeypatch):
    async with running_manager(tmp_path) as (runtime, context, wrapper):
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
        assert document["reminder_priority"] == "P1"
        assert "delivered_at" not in document
        snapshot = await runtime.inbox.snapshot("monitor")
        assert snapshot["pending_count"] == 0 and snapshot["in_flight_count"] == 0
        await runtime.outbox.sync()
        assert len(runtime.outbox.entries) == 1


@pytest.mark.asyncio
async def test_cancelled_round_preserves_its_record_and_pending_input(tmp_path, monkeypatch):
    async with running_manager(tmp_path) as (runtime, context, wrapper):
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
        assert path.read_text() == SKIP
        assert earlier.read_text() == SKIP
        assert (await runtime.inbox.snapshot("monitor"))["pending_count"] == 1


@pytest.mark.asyncio
async def test_same_second_records_get_unique_paths(tmp_path):
    async with running_manager(tmp_path) as (runtime, context, _):
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


@pytest.mark.asyncio
async def test_same_second_round_skips_registered_path_after_record_is_moved(tmp_path, monkeypatch):
    async with running_manager(tmp_path) as (runtime, context, wrapper):
        await create_agent(runtime)
        now = runtime.now()
        monkeypatch.setattr(runtime, "now", lambda: now)
        agent = await runtime.definition("monitor")

        first_step = ProactiveAgentStep(app_context=context)
        first_paths = first_step._paths(agent, now)
        install_round(runtime, wrapper, monkeypatch, first_paths)
        (first,) = (await first_step(agent_name="monitor")).answer["notifications"]
        first_path = first_paths["record_path"]
        moved = first_path.rename(tmp_path / first_path.name)
        assert not (await runtime.outbox.list(state="pending"))["items"]

        await runtime.append_events(
            "monitor",
            [Event(id="event-2", event_ts=now, source="experiment", description="Another result")],
        )
        second_step = ProactiveAgentStep(app_context=context)
        second_paths = second_step._paths(agent, now)
        second_text = REMIND.format(priority="P1", body="\n").replace("Verified result", "Another result")
        install_round(runtime, wrapper, monkeypatch, second_paths, text=second_text)
        (second,) = (await second_step(agent_name="monitor")).answer["notifications"]
        assert second_paths["record_path"] == first_path.with_stem(f"{first_path.stem}_1")
        assert (await runtime.inbox.snapshot("monitor"))["pending_count"] == 0
        assert not first_path.exists()

        moved.rename(first_path)
        mailbox = await runtime.outbox.list(state="pending")
        assert {item["id"]: item["message"] for item in mailbox["items"]} == {
            first["id"]: "Verified result",
            second["id"]: "Another result",
        }
