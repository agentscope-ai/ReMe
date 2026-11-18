"""Single-turn file decisions and delivery in automatic proactive rounds."""

# pylint: disable=missing-function-docstring,protected-access

import asyncio
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, Mock

import frontmatter
import pytest

from reme.enumeration import ChunkEnum
from reme.schema import StreamChunk
from reme.steps.proactive.agent import ProactiveAgentStep
from .test_agent_runtime import create_agent, running_runtime


def install_round(
    runtime,
    wrapper,
    monkeypatch,
    paths,
    *,
    record="write",
    reminder="write",
    hook=None,
    failed=False,
):
    calls = []

    async def stream(query, **options):
        calls.append((query, options))
        assert "output_schema" not in options
        runtime.files.write(
            paths["record_path"],
            frontmatter.dumps(frontmatter.Post("New progress" if record == "write" else "", action=record)),
        )
        if record == "write":
            metadata = {"action": reminder}
            if reminder == "write":
                agent = await runtime.definition(options["activity_task_id"])
                metadata["priority"] = "P0" if agent.agent_type == "cron" else "P1"
            runtime.files.write(
                paths["reminder_path"],
                frontmatter.dumps(
                    frontmatter.Post(
                        "New result" if reminder == "write" else "No reminder needed",
                        **metadata,
                    ),
                ),
            )
            runtime.files.write(
                paths["today_summary_path"],
                f"## Today's records\n- [Progress]({paths['record_path'].name})\n"
                "\n## Today's reminders\n" + ("- New result\n" if reminder == "write" else ""),
            )
        if hook is not None:
            hook()
        result = {
            "session_id": options["resume"],
            "status": "failed" if failed else "completed",
            "error": "Model failed" if failed else None,
            "result": "Finished",
            # Deliberately conflicting response JSON must never become business output.
            "structured_output": {"notifications": [{"message": "Ignore this"}]},
            "usage": None,
            "raw": {},
        }
        yield StreamChunk(chunk_type=ChunkEnum.REPLY_END, status=result["status"], chunk=result)

    monkeypatch.setattr(wrapper, "reply_stream", stream)
    return calls


@pytest.mark.parametrize("agent_type", ["proactive", "cron"])
@pytest.mark.parametrize("language", ["en", "zh"])
@pytest.mark.asyncio
async def test_one_turn_saves_artifacts_and_publishes_through_runtime(tmp_path, monkeypatch, agent_type, language):
    async with running_runtime(tmp_path, language=language) as (
        runtime,
        context,
        wrapper,
    ):
        await create_agent(runtime)
        if agent_type == "cron":
            await runtime.update("monitor", {"agent_type": "cron", "cron": "0 * * * *"})
        now = runtime.now()
        monkeypatch.setattr(runtime, "now", lambda: now)
        step = ProactiveAgentStep(app_context=context)
        step.logger = Mock()
        paths = step._paths(await runtime.definition("monitor"), now)
        calls = install_round(runtime, wrapper, monkeypatch, paths)

        response = await step(agent_name="monitor")

        assert response.success and response.answer["processed"] == 1
        assert len(calls) == len(wrapper.created) == 1
        assert calls[0][1]["resume"] == "session-1"
        assert calls[0][1]["activity_source"] == "proactive"
        assert calls[0][1]["activity_task_id"] == "monitor"
        assert (await runtime.status("monitor"))["pending_events"] == 0
        await runtime.outbox.sync()
        assert len(runtime.outbox.committed) == 1
        item = (await runtime.notifications.read())["items"][0]
        assert item["message"] == "New result"
        assert item["priority"] == ("P0" if agent_type == "cron" else "P1")
        assert item["reminder_path"] == paths["reminder_path"].relative_to(tmp_path).as_posix()
        assert item["paths"] == [
            paths[key].relative_to(tmp_path).as_posix() for key in ("reminder_path", "record_path")
        ]
        document = frontmatter.loads(paths["reminder_path"].read_text())
        assert document.content == "New result" and document["delivered_at"]
        assert document["action"] == "write" and "message" not in document.metadata
        assert paths["record_path"].name in runtime.files.read(paths["today_summary_path"])
        assert wrapper.created[0]["sandbox"] == "workspace-write"
        logs = "\n".join(call.args[0] for call in step.logger.info.call_args_list)
        assert "result=validated notifications=1" in logs
        assert "elapsed=" in logs and str(paths["record_path"]) in logs


@pytest.mark.parametrize("existing_summary", [False, True])
@pytest.mark.asyncio
async def test_no_new_information_preserves_empty_decision_without_reminder(tmp_path, monkeypatch, existing_summary):
    async with running_runtime(tmp_path) as (runtime, context, wrapper):
        await create_agent(runtime)
        now = runtime.now()
        monkeypatch.setattr(runtime, "now", lambda: now)
        step = ProactiveAgentStep(app_context=context)
        paths = step._paths(await runtime.definition("monitor"), now)
        if existing_summary:
            runtime.files.write(paths["today_summary_path"], "# Previous progress\n")
        before = step._read_optional(paths["today_summary_path"])
        calls = install_round(runtime, wrapper, monkeypatch, paths, record="skip")

        response = await step(agent_name="monitor")

        assert response.success and response.answer["notifications"] == []
        assert len(calls) == 1
        document = frontmatter.loads(paths["record_path"].read_text())
        assert document["action"] == "skip" and not document.content
        assert not paths["reminder_path"].exists()
        assert step._read_optional(paths["today_summary_path"]) == before
        assert (await runtime.status("monitor"))["pending_events"] == 0
        await runtime.outbox.sync()
        assert not runtime.outbox.records and not runtime.outbox.issues


@pytest.mark.parametrize("existing_summary", [False, True])
@pytest.mark.asyncio
async def test_skipped_reminder_preserves_new_record_and_history(tmp_path, monkeypatch, existing_summary):
    async with running_runtime(tmp_path) as (runtime, context, wrapper):
        await create_agent(runtime)
        now = runtime.now()
        monkeypatch.setattr(runtime, "now", lambda: now)
        step = ProactiveAgentStep(app_context=context)
        paths = step._paths(await runtime.definition("monitor"), now)
        history = paths["record_path"].with_name("earlier.md")
        runtime.files.write(history, "# Earlier progress\n")
        if existing_summary:
            runtime.files.write(paths["today_summary_path"], "# Previous summary\n")
        summary_before = step._read_optional(paths["today_summary_path"])
        calls = install_round(runtime, wrapper, monkeypatch, paths, reminder="skip")

        response = await step(agent_name="monitor")

        assert response.success and response.answer["notifications"] == []
        assert len(calls) == 1
        record = frontmatter.loads(paths["record_path"].read_text())
        reminder = frontmatter.loads(paths["reminder_path"].read_text())
        assert record["action"] == "write" and record.content == "New progress"
        assert reminder["action"] == "skip" and reminder.content == "No reminder needed"
        summary = runtime.files.read(paths["today_summary_path"])
        assert summary != summary_before and paths["record_path"].name in summary
        assert "New result" not in summary
        assert history.read_text() == "# Earlier progress\n"
        assert (await runtime.status("monitor"))["pending_events"] == 0
        await runtime.outbox.sync()
        assert not runtime.outbox.records and not runtime.outbox.issues


@pytest.mark.parametrize("existing_summary", [False, True])
@pytest.mark.parametrize("defect", ["record_skip", "report_skip", "P1", "P2", "missing_priority"])
@pytest.mark.asyncio
async def test_cron_requires_a_record_and_p0_report_before_committing(tmp_path, monkeypatch, existing_summary, defect):
    async with running_runtime(tmp_path) as (runtime, context, wrapper):
        await create_agent(runtime)
        await runtime.update("monitor", {"agent_type": "cron", "cron": "0 * * * *"})
        now = runtime.now()
        monkeypatch.setattr(runtime, "now", lambda: now)
        step = ProactiveAgentStep(app_context=context)
        paths = step._paths(await runtime.definition("monitor"), now)
        if existing_summary:
            runtime.files.write(paths["today_summary_path"], "# Previous confirmed summary\n")
        before = step._read_optional(paths["today_summary_path"])

        def corrupt_priority():
            if defect not in ("P1", "P2", "missing_priority"):
                return
            document = frontmatter.loads(runtime.files.read(paths["reminder_path"]))
            if defect == "missing_priority":
                del document["priority"]
            else:
                document["priority"] = defect
            runtime.files.write(paths["reminder_path"], frontmatter.dumps(document))

        install_round(
            runtime,
            wrapper,
            monkeypatch,
            paths,
            record="skip" if defect == "record_skip" else "write",
            reminder="skip" if defect == "report_skip" else "write",
            hook=corrupt_priority,
        )
        with pytest.raises(ValueError, match="a cron task must write"):
            await step(agent_name="monitor")
        assert step._read_optional(paths["today_summary_path"]) == before
        assert not paths["record_path"].exists() and not paths["reminder_path"].exists()
        assert (await runtime.status("monitor"))["pending_events"] == 1
        await runtime.outbox.sync()
        assert not runtime.outbox.committed


@pytest.mark.parametrize("record", ["write", "skip"])
@pytest.mark.parametrize("existing_summary", [False, True])
@pytest.mark.asyncio
async def test_summary_updates_require_new_information_not_a_reminder(tmp_path, monkeypatch, record, existing_summary):
    async with running_runtime(tmp_path) as (runtime, context, wrapper):
        await create_agent(runtime)
        now = runtime.now()
        monkeypatch.setattr(runtime, "now", lambda: now)
        step = ProactiveAgentStep(app_context=context)
        paths = step._paths(await runtime.definition("monitor"), now)
        if existing_summary:
            runtime.files.write(paths["today_summary_path"], "# Earlier progress\n")
        install_round(
            runtime,
            wrapper,
            monkeypatch,
            paths,
            record=record,
            reminder="skip",
            hook=lambda: runtime.files.write(paths["today_summary_path"], "# Unexpected update\n"),
        )

        if record == "skip":
            with pytest.raises(ValueError, match="a skipped record must not change"):
                await step(agent_name="monitor")
            assert (await runtime.status("monitor"))["pending_events"] == 1
        else:
            response = await step(agent_name="monitor")
            assert response.success and response.answer["notifications"] == []
            assert runtime.files.read(paths["today_summary_path"]) == "# Unexpected update\n"
            assert (await runtime.status("monitor"))["pending_events"] == 0
        await runtime.outbox.sync()
        assert not runtime.outbox.committed


@pytest.mark.parametrize(
    "before,after",
    [
        (None, None),
        (None, ""),
        (None, " \n"),
        ("# Previous summary\n", ""),
        ("# Previous summary\n", None),
        ("# Previous summary\n", "# Previous summary\n"),
    ],
)
@pytest.mark.asyncio
async def test_new_record_requires_an_updated_nonempty_summary_without_reminder(tmp_path, monkeypatch, before, after):
    async with running_runtime(tmp_path) as (runtime, context, wrapper):
        await create_agent(runtime)
        now = runtime.now()
        monkeypatch.setattr(runtime, "now", lambda: now)
        step = ProactiveAgentStep(app_context=context)
        paths = step._paths(await runtime.definition("monitor"), now)
        if before is not None:
            runtime.files.write(paths["today_summary_path"], before)

        def corrupt_summary():
            if after is None:
                paths["today_summary_path"].unlink()
            else:
                runtime.files.write(paths["today_summary_path"], after)

        install_round(runtime, wrapper, monkeypatch, paths, reminder="skip", hook=corrupt_summary)
        message = (
            "requires a nonempty daily summary" if not after or not after.strip() else "must update its daily summary"
        )
        with pytest.raises(ValueError, match=message):
            await step(agent_name="monitor")
        assert step._read_optional(paths["today_summary_path"]) == before
        assert not paths["record_path"].exists() and not paths["reminder_path"].exists()
        assert (await runtime.status("monitor"))["pending_events"] == 1
        await runtime.outbox.sync()
        assert not runtime.outbox.committed


@pytest.mark.parametrize("key", ["record_path", "reminder_path"])
@pytest.mark.parametrize(
    "defect",
    [
        "missing",
        "blank",
        "missing_action",
        "invalid_action",
        "boolean_action",
        "empty_body",
    ],
)
@pytest.mark.asyncio
async def test_missing_or_invalid_required_documents_fail_instead_of_skipping(tmp_path, monkeypatch, key, defect):
    async with running_runtime(tmp_path) as (runtime, context, wrapper):
        await create_agent(runtime)
        now = runtime.now()
        monkeypatch.setattr(runtime, "now", lambda: now)
        step = ProactiveAgentStep(app_context=context)
        step.logger = Mock()
        paths = step._paths(await runtime.definition("monitor"), now)
        path = paths[key]

        def corrupt():
            if defect == "missing":
                path.unlink()
                return
            document = frontmatter.loads(runtime.files.read(path))
            if defect == "missing_action":
                del document["action"]
            elif defect in ("invalid_action", "boolean_action"):
                document["action"] = True if defect == "boolean_action" else "done"
            elif defect == "empty_body":
                document.content = " \n"
            runtime.files.write(path, "" if defect == "blank" else frontmatter.dumps(document))

        install_round(runtime, wrapper, monkeypatch, paths, hook=corrupt)
        with pytest.raises((FileNotFoundError, ValueError)):
            await step(agent_name="monitor")
        assert (await runtime.status("monitor"))["pending_events"] == 1
        await runtime.outbox.sync()
        assert not runtime.outbox.committed
        assert any("result=failed" in call.args[0] for call in step.logger.error.call_args_list)


@pytest.mark.parametrize("defect", ["record_body", "unexpected_reminder", "empty_reason"])
@pytest.mark.asyncio
async def test_skipped_decisions_require_consistent_artifacts(tmp_path, monkeypatch, defect):
    async with running_runtime(tmp_path) as (runtime, context, wrapper):
        await create_agent(runtime)
        now = runtime.now()
        monkeypatch.setattr(runtime, "now", lambda: now)
        step = ProactiveAgentStep(app_context=context)
        paths = step._paths(await runtime.definition("monitor"), now)

        def corrupt():
            if defect == "record_body":
                runtime.files.write(paths["record_path"], "---\naction: skip\n---\nUnexpected body\n")
            elif defect == "unexpected_reminder":
                runtime.files.write(
                    paths["reminder_path"],
                    "---\naction: skip\n---\nUnexpected decision\n",
                )
            else:
                runtime.files.write(paths["reminder_path"], "---\naction: skip\n---\n")

        install_round(
            runtime,
            wrapper,
            monkeypatch,
            paths,
            record="write" if defect == "empty_reason" else "skip",
            reminder="skip",
            hook=corrupt,
        )
        with pytest.raises(ValueError):
            await step(agent_name="monitor")
        assert (await runtime.status("monitor"))["pending_events"] == 1
        await runtime.outbox.sync()
        assert not runtime.outbox.committed


@pytest.mark.parametrize(
    "failure",
    [
        "missing_summary",
        "unchanged_summary",
        "empty_summary",
        "missing_priority",
        "invalid_priority",
    ],
)
@pytest.mark.asyncio
async def test_incomplete_reminder_keeps_events_pending_without_publishing(tmp_path, monkeypatch, failure):
    async with running_runtime(tmp_path) as (runtime, context, wrapper):
        await create_agent(runtime)
        now = runtime.now()
        monkeypatch.setattr(runtime, "now", lambda: now)
        step = ProactiveAgentStep(app_context=context)
        paths = step._paths(await runtime.definition("monitor"), now)
        if failure == "unchanged_summary":
            runtime.files.write(paths["today_summary_path"], "# Previous summary\n")

        def corrupt():
            if failure == "missing_summary":
                paths["today_summary_path"].unlink()
            elif failure in {"unchanged_summary", "empty_summary"}:
                runtime.files.write(
                    paths["today_summary_path"],
                    "# Previous summary\n" if failure == "unchanged_summary" else "",
                )
            else:
                document = frontmatter.loads(runtime.files.read(paths["reminder_path"]))
                if failure == "missing_priority":
                    del document["priority"]
                else:
                    document["priority"] = "P9"
                runtime.files.write(paths["reminder_path"], frontmatter.dumps(document))

        install_round(runtime, wrapper, monkeypatch, paths, hook=corrupt)
        with pytest.raises(ValueError):
            await step(agent_name="monitor")
        assert (await runtime.status("monitor"))["pending_events"] == 1
        await runtime.outbox.sync()
        assert not runtime.notifications.pending


@pytest.mark.asyncio
async def test_failed_turn_accepts_complete_valid_files(tmp_path, monkeypatch):
    async with running_runtime(tmp_path) as (runtime, context, wrapper):
        await create_agent(runtime)
        now = runtime.now()
        monkeypatch.setattr(runtime, "now", lambda: now)
        step = ProactiveAgentStep(app_context=context)
        paths = step._paths(await runtime.definition("monitor"), now)
        calls = install_round(runtime, wrapper, monkeypatch, paths, failed=True)

        response = await step(agent_name="monitor")
        assert response.success
        assert len(calls) == 1 and paths["today_summary_path"].exists()
        assert (await runtime.status("monitor"))["pending_events"] == 0
        await runtime.outbox.sync()
        assert len(runtime.outbox.committed) == 1


@pytest.mark.asyncio
async def test_timeout_accepts_complete_valid_files(tmp_path, monkeypatch):
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
        document = frontmatter.loads(runtime.files.read(paths["reminder_path"]))
        assert document["action"] == "write" and document["id"]
        snapshot = await runtime.inbox.snapshot("monitor")
        assert snapshot["pending_count"] == 0 and snapshot["in_flight_count"] == 0
        await runtime.outbox.sync()
        assert len(runtime.outbox.committed) == 1


@pytest.mark.parametrize("existing_summary", [False, True])
@pytest.mark.parametrize("failure", ["validation", "timeout", "exception", "cancel"])
@pytest.mark.asyncio
async def test_invalid_or_cancelled_round_restores_history_before_retry(
    tmp_path,
    monkeypatch,
    existing_summary,
    failure,
):
    async with running_runtime(tmp_path) as (runtime, context, wrapper):
        await create_agent(runtime)
        now = runtime.now()
        monkeypatch.setattr(runtime, "now", lambda: now)
        step = ProactiveAgentStep(app_context=context, timeout_seconds=0.1 if failure == "timeout" else 1200)
        paths = step._paths(await runtime.definition("monitor"), now)
        if existing_summary:
            runtime.files.write(paths["today_summary_path"], "# Previous confirmed summary\n")
        before = step._read_optional(paths["today_summary_path"])
        ready = asyncio.Event()
        install_round(
            runtime,
            wrapper,
            monkeypatch,
            paths,
            hook=lambda: paths["record_path"].unlink() if failure != "cancel" else None,
        )
        original = wrapper.reply_stream

        async def stream(query, **kwargs):
            async for chunk in original(query, **kwargs):
                yield chunk
            ready.set()
            if failure == "exception":
                raise OSError("Interrupted stream")
            if failure in ("timeout", "cancel"):
                await asyncio.Event().wait()

        monkeypatch.setattr(wrapper, "reply_stream", stream)
        task = asyncio.create_task(step(agent_name="monitor"))
        if failure == "cancel":
            await ready.wait()
            task.cancel()
        expected = {
            "validation": FileNotFoundError,
            "timeout": TimeoutError,
            "exception": OSError,
            "cancel": asyncio.CancelledError,
        }[failure]
        with pytest.raises(expected):
            await task
        assert step._read_optional(paths["today_summary_path"]) == before
        assert not paths["record_path"].exists() and not paths["reminder_path"].exists()
        assert (await runtime.status("monitor"))["pending_events"] == 1
        install_round(runtime, wrapper, monkeypatch, paths)
        step.timeout_seconds = 1200
        assert (await step(agent_name="monitor")).success
        await runtime.outbox.sync()
        assert (await runtime.status("monitor"))["pending_events"] == 0
        assert len(runtime.outbox.committed) == 1


@pytest.mark.parametrize("failure", ["save", "ack", "cancel_after_ack"])
@pytest.mark.asyncio
async def test_round_restores_only_before_durable_input_confirmation(tmp_path, monkeypatch, failure):
    async with running_runtime(tmp_path) as (runtime, context, wrapper):
        await create_agent(runtime)
        now = runtime.now()
        monkeypatch.setattr(runtime, "now", lambda: now)
        step = ProactiveAgentStep(app_context=context)
        paths = step._paths(await runtime.definition("monitor"), now)
        runtime.files.write(paths["today_summary_path"], "# Previous confirmed summary\n")
        before = runtime.files.read(paths["today_summary_path"])
        install_round(runtime, wrapper, monkeypatch, paths)
        if failure == "cancel_after_ack":
            ack = runtime.inbox.ack

            async def cancel_after_ack(*args, **kwargs):
                await ack(*args, **kwargs)
                raise asyncio.CancelledError

            monkeypatch.setattr(runtime.inbox, "ack", cancel_after_ack)
        else:
            target, method = (runtime.outbox, "save") if failure == "save" else (runtime.inbox, "ack")
            monkeypatch.setattr(target, method, AsyncMock(side_effect=OSError("Commit failed")))
        with pytest.raises(asyncio.CancelledError if failure == "cancel_after_ack" else OSError):
            await step(agent_name="monitor")
        await runtime.outbox.sync()
        if failure == "cancel_after_ack":
            assert runtime.files.read(paths["today_summary_path"]) != before
            assert paths["record_path"].exists() and paths["reminder_path"].exists()
            assert (await runtime.status("monitor"))["pending_events"] == 0
            assert len(runtime.outbox.committed) == 1
        else:
            assert runtime.files.read(paths["today_summary_path"]) == before
            assert not paths["record_path"].exists() and not paths["reminder_path"].exists()
            assert (await runtime.status("monitor"))["pending_events"] == 1
            assert not runtime.outbox.committed


@pytest.mark.parametrize(
    "timestamp,yesterday",
    [
        ("2026-01-01T00:00:00+08:00", "2025-12-31"),
        ("2024-03-01T00:00:00+08:00", "2024-02-29"),
    ],
)
@pytest.mark.asyncio
async def test_round_paths_keep_both_calendar_summaries_and_paired_filenames(tmp_path, timestamp, yesterday):
    async with running_runtime(tmp_path) as (runtime, context, _):
        await create_agent(runtime)
        agent = await runtime.definition("monitor")
        step = ProactiveAgentStep(app_context=context)
        now = datetime.fromisoformat(timestamp)
        paths = step._paths(agent, now)
        assert paths["today_summary_path"] == tmp_path / "daily" / f"{now:%Y-%m-%d}" / "monitor.md"
        assert paths["yesterday_summary_path"] == tmp_path / "daily" / yesterday / "monitor.md"
        assert paths["record_path"] == tmp_path / "daily" / f"{now:%Y-%m-%d}" / f"monitor_{now:%H%M%S}.md"
        assert paths["reminder_path"] == (
            tmp_path / "reme_agents/monitor/reminder" / f"{now:%Y-%m-%d}" / f"{now:%H%M%S}.md"
        )
        assert not paths["today_summary_path"].exists() and not paths["yesterday_summary_path"].exists()
        for delta in (-5, -2, 1):
            path = tmp_path / "daily" / f"{now + timedelta(days=delta):%Y-%m-%d}" / "monitor.md"
            runtime.files.write(path, "# Summary\n")
        runtime.files.write(tmp_path / "daily/not-a-date/monitor.md", "# Invalid date\n")
        paths = step._paths(agent, now)
        assert paths["yesterday_summary_path"] == tmp_path / "daily" / yesterday / "monitor.md"
        assert not paths["yesterday_summary_path"].exists()
        runtime.files.write(paths["yesterday_summary_path"], "# Yesterday\n")
        runtime.files.write(paths["today_summary_path"], "# Today\n")
        runtime.files.write(paths["record_path"], "# Existing details\n")
        paths = step._paths(agent, now)
        assert paths["today_summary_path"].read_text() == "# Today\n"
        assert paths["yesterday_summary_path"].read_text() == "# Yesterday\n"
        assert paths["record_path"].name == f"monitor_{now:%H%M%S}_1.md"
        assert paths["reminder_path"].name == f"{now:%H%M%S}_1.md"
        runtime.files.write(paths["reminder_path"], "---\npriority: P1\n---\nExisting reminder\n")
        paths = step._paths(agent, now)
        assert paths["record_path"].name == f"monitor_{now:%H%M%S}_2.md"
        assert paths["reminder_path"].name == f"{now:%H%M%S}_2.md"
