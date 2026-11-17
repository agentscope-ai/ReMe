"""File decisions, stage boundaries and delivery in automatic proactive rounds."""

# pylint: disable=missing-function-docstring,protected-access

import asyncio
from datetime import datetime, timedelta
from unittest.mock import Mock

import frontmatter
import pytest

from reme.enumeration import ChunkEnum
from reme.schema import StreamChunk
from reme.steps.proactive.agent import ProactiveAgentStep
from .test_agent_runtime import create_agent, running_runtime


def document_action(path):
    return frontmatter.loads(path.read_text())["action"]


def install_round(runtime, wrapper, monkeypatch, paths, *, record="write", reminder="write", hook=None, failed=None):
    calls = []

    async def stream(query, **options):
        stage = ("record", "update", "reminder", "summary")[len(calls)]
        calls.append((stage, query, options))
        if stage == "record":
            runtime.files.write(
                paths["detail_path"],
                frontmatter.dumps(
                    frontmatter.Post("New progress" if record == "write" else "No new information", action=record),
                ),
            )
        elif stage == "reminder":
            metadata = {"action": reminder}
            if reminder == "write":
                metadata["priority"] = "P1"
            runtime.files.write(
                paths["reminder_path"],
                frontmatter.dumps(frontmatter.Post("New result" if reminder == "write" else "Record only", **metadata)),
            )
        elif stage == "summary":
            runtime.files.write(paths["summary_path"], f"# Progress\n[Record]({paths['detail_path'].name})\n")
        if hook is not None:
            hook(stage)
        result = {
            "session_id": options["resume"],
            "status": "failed" if stage == failed else "completed",
            "error": "Model failed" if stage == failed else None,
            "result": f"Finished {stage}",
            # Deliberately conflicting response JSON must never become business output.
            "structured_output": {"notifications": [{"message": "Ignore this"}]},
            "usage": None,
            "raw": {},
        }
        yield StreamChunk(chunk_type=ChunkEnum.REPLY_END, status=result["status"], chunk=result)

    monkeypatch.setattr(wrapper, "reply_stream", stream)
    return calls


@pytest.mark.parametrize("reminder", ["write", "skip"])
@pytest.mark.asyncio
async def test_four_stages_share_session_and_summarize_record_without_history(tmp_path, monkeypatch, reminder):
    async with running_runtime(tmp_path) as (runtime, context, wrapper):
        await create_agent(runtime)
        now = runtime.now()
        monkeypatch.setattr(runtime, "now", lambda: now)
        step = ProactiveAgentStep(app_context=context)
        step.logger = Mock()
        paths = step._paths(await runtime.definition("monitor"), now)
        intermediate = {}

        def inspect(stage):
            if stage == "summary":
                intermediate.update(frontmatter.loads(runtime.files.read(paths["reminder_path"])).metadata)

        calls = install_round(runtime, wrapper, monkeypatch, paths, reminder=reminder, hook=inspect)
        response = await step(agent_name="monitor")
        assert response.success
        assert [stage for stage, _, _ in calls] == ["record", "update", "reminder", "summary"]
        assert len({options["resume"] for _, _, options in calls}) == 1
        assert len(wrapper.created) == 1
        assert intermediate["action"] == reminder and "id" not in intermediate
        if reminder == "write":
            assert intermediate["paths"] == [
                paths[key].relative_to(tmp_path).as_posix() for key in ("reminder_path", "detail_path")
            ]
        assert paths["detail_path"].name in runtime.files.read(paths["summary_path"])
        assert len(response.answer["notifications"]) == (reminder == "write")
        assert (await runtime.status("monitor"))["pending_events"] == 0
        await runtime.outbox.sync()
        assert len(runtime.outbox.committed) == (reminder == "write")
        logs = "\n".join(call.args[0] for call in step.logger.info.call_args_list)
        assert "stage=update result=unchanged" in logs
        assert "stage=summary result=created" in logs
        assert "elapsed=" in logs and str(paths["detail_path"]) in logs


@pytest.mark.parametrize("existing_summary", [False, True])
@pytest.mark.asyncio
async def test_no_new_information_writes_both_skip_files_with_one_model_call(tmp_path, monkeypatch, existing_summary):
    async with running_runtime(tmp_path) as (runtime, context, wrapper):
        await create_agent(runtime)
        now = runtime.now()
        monkeypatch.setattr(runtime, "now", lambda: now)
        step = ProactiveAgentStep(app_context=context)
        paths = step._paths(await runtime.definition("monitor"), now)
        if existing_summary:
            runtime.files.write(paths["summary_path"], "# Previous progress\n")
        before = step._read_optional(paths["summary_path"])
        calls = install_round(runtime, wrapper, monkeypatch, paths, record="skip")
        response = await step(agent_name="monitor")
        assert response.success and response.answer["notifications"] == []
        assert len(calls) == 1
        for key in ("detail_path", "reminder_path"):
            document = frontmatter.loads(runtime.files.read(paths[key]))
            assert document.metadata == {"action": "skip"}
            assert document.content == "No new information"
        assert step._read_optional(paths["summary_path"]) == before
        await runtime.outbox.sync()
        assert not runtime.outbox.records and not runtime.outbox.issues


@pytest.mark.parametrize("stage", ["record", "reminder"])
@pytest.mark.parametrize(
    "defect",
    ["missing", "blank", "missing_action", "invalid_action", "boolean_action", "empty_body"],
)
@pytest.mark.asyncio
async def test_missing_or_invalid_required_documents_fail_instead_of_skipping(tmp_path, monkeypatch, stage, defect):
    async with running_runtime(tmp_path) as (runtime, context, wrapper):
        await create_agent(runtime)
        now = runtime.now()
        monkeypatch.setattr(runtime, "now", lambda: now)
        step = ProactiveAgentStep(app_context=context)
        step.logger = Mock()
        paths = step._paths(await runtime.definition("monitor"), now)
        path = paths["detail_path" if stage == "record" else "reminder_path"]

        def corrupt(current):
            if current != stage:
                return
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

        calls = install_round(runtime, wrapper, monkeypatch, paths, hook=corrupt)
        with pytest.raises((FileNotFoundError, ValueError)):
            await step(agent_name="monitor")
        assert calls[-1][0] == stage
        assert (await runtime.status("monitor"))["pending_events"] == 1
        await runtime.outbox.sync()
        assert not runtime.outbox.committed
        assert any(f"stage={stage} result=failed" in call.args[0] for call in step.logger.error.call_args_list)


@pytest.mark.parametrize("stage", ["record", "update", "reminder", "summary"])
@pytest.mark.asyncio
async def test_failed_turn_stops_at_its_stage_and_keeps_inputs(tmp_path, monkeypatch, stage):
    async with running_runtime(tmp_path) as (runtime, context, wrapper):
        await create_agent(runtime)
        now = runtime.now()
        monkeypatch.setattr(runtime, "now", lambda: now)
        step = ProactiveAgentStep(app_context=context)
        paths = step._paths(await runtime.definition("monitor"), now)
        calls = install_round(runtime, wrapper, monkeypatch, paths, failed=stage)
        with pytest.raises(RuntimeError, match="Model failed"):
            await step(agent_name="monitor")
        assert calls[-1][0] == stage
        assert (await runtime.status("monitor"))["pending_events"] == 1
        await runtime.outbox.sync()
        assert not runtime.outbox.committed


@pytest.mark.parametrize("reminder", ["write", "skip"])
@pytest.mark.asyncio
async def test_rewriting_identical_summary_does_not_count_as_updated_progress(tmp_path, monkeypatch, reminder):
    async with running_runtime(tmp_path) as (runtime, context, wrapper):
        await create_agent(runtime)
        now = runtime.now()
        monkeypatch.setattr(runtime, "now", lambda: now)
        step = ProactiveAgentStep(app_context=context)
        paths = step._paths(await runtime.definition("monitor"), now)
        runtime.files.write(paths["summary_path"], "# Unchanged\n")

        def unchanged(stage):
            if stage == "summary":
                runtime.files.write(paths["summary_path"], "# Unchanged\n")

        install_round(runtime, wrapper, monkeypatch, paths, reminder=reminder, hook=unchanged)
        with pytest.raises(ValueError, match="new progress must update"):
            await step(agent_name="monitor")
        assert (await runtime.status("monitor"))["pending_events"] == 1
        await runtime.outbox.sync()
        assert not runtime.outbox.committed


@pytest.mark.parametrize("key", ["detail_path", "reminder_path"])
@pytest.mark.asyncio
async def test_summary_cannot_rewrite_validated_documents(tmp_path, monkeypatch, key):
    async with running_runtime(tmp_path) as (runtime, context, wrapper):
        await create_agent(runtime)
        now = runtime.now()
        monkeypatch.setattr(runtime, "now", lambda: now)
        step = ProactiveAgentStep(app_context=context)
        paths = step._paths(await runtime.definition("monitor"), now)

        def rewrite(stage):
            if stage == "summary":
                runtime.files.write(paths[key], "---\naction: skip\n---\nChanged my mind\n")

        install_round(runtime, wrapper, monkeypatch, paths, hook=rewrite)
        with pytest.raises(ValueError, match="must not change"):
            await step(agent_name="monitor")
        assert (await runtime.status("monitor"))["pending_events"] == 1


@pytest.mark.asyncio
async def test_timeout_during_later_stage_never_publishes_prepared_reminder(tmp_path, monkeypatch):
    async with running_runtime(tmp_path) as (runtime, context, wrapper):
        await create_agent(runtime)
        now = runtime.now()
        monkeypatch.setattr(runtime, "now", lambda: now)
        step = ProactiveAgentStep(app_context=context, timeout_seconds=0.5)
        paths = step._paths(await runtime.definition("monitor"), now)
        calls = install_round(runtime, wrapper, monkeypatch, paths)
        original = wrapper.reply_stream

        async def pause(query, **kwargs):
            if len(calls) == 3:
                await asyncio.Event().wait()
            async for chunk in original(query, **kwargs):
                yield chunk

        monkeypatch.setattr(wrapper, "reply_stream", pause)
        with pytest.raises(TimeoutError):
            await step(agent_name="monitor")
        document = frontmatter.loads(runtime.files.read(paths["reminder_path"]))
        assert document["action"] == "write" and "paths" in document and "id" not in document
        snapshot = await runtime.inbox.snapshot("monitor")
        assert snapshot["pending_count"] == 1 and snapshot["in_flight_count"] == 0
        await runtime.outbox.sync()
        assert not runtime.outbox.committed


@pytest.mark.asyncio
async def test_proactive_agent_step_saves_artifacts_and_publishes_through_runtime(tmp_path, monkeypatch):
    async with running_runtime(tmp_path) as (runtime, context, wrapper):
        await create_agent(runtime)
        now = runtime.now()
        monkeypatch.setattr(runtime, "now", lambda: now)
        paths = ProactiveAgentStep(app_context=context)._paths(await runtime.definition("monitor"), now)

        stages = []

        async def model_reply(_query, **kwargs):
            assert "output_schema" not in kwargs
            stage = ("record", "update", "reminder", "summary")[len(stages)]
            stages.append(stage)
            if stage == "record":
                runtime.files.write(paths["detail_path"], "---\naction: write\n---\n# Experiment results\n")
            elif stage == "reminder":
                runtime.files.write(paths["reminder_path"], "---\naction: write\npriority: P0\n---\nFinished\n")
            elif stage == "summary":
                runtime.files.write(paths["summary_path"], "# Experiment finished\n")
            return {
                "status": "completed",
                "session_id": kwargs["resume"],
                "result": f"Completed {stage}",
            }

        async def model_stream(query, **kwargs):
            result = await model_reply(query, **kwargs)
            yield StreamChunk(
                chunk_type=ChunkEnum.REPLY_END,
                event="end",
                session_id=kwargs["resume"],
                turn_id="automatic-turn",
                status="completed",
                chunk={
                    "structured_output": None,
                    "usage": None,
                    "error": None,
                    "raw": {},
                    **result,
                },
            )

        monkeypatch.setattr(wrapper, "reply_stream", model_stream)
        response = await ProactiveAgentStep(app_context=context)(agent_name="monitor")
        assert response.success and response.answer["processed"] == 1
        assert (await runtime.status("monitor"))["pending_events"] == 0
        await runtime.outbox.sync()
        item = (await runtime.notifications.read())["items"][0]
        assert stages == ["record", "update", "reminder", "summary"]
        assert document_action(paths["detail_path"]) == "write"
        assert item["message"] == "Finished" and item["priority"] == "P0"
        assert item["reminder_path"] == paths["reminder_path"].relative_to(tmp_path).as_posix()
        document = frontmatter.loads(paths["reminder_path"].read_text())
        assert document.content == "Finished" and document["delivered_at"]
        assert document["action"] == "write"
        assert "message" not in document.metadata
        assert not list(paths["reminder_dir"].glob("*.jsonl"))
        assert item["paths"] == [
            paths[key].relative_to(tmp_path).as_posix() for key in ("reminder_path", "detail_path")
        ]
        assert paths["detail_path"].name == f"monitor_{now:%H%M%S}.md"
        assert paths["reminder_path"].name == f"{now:%H%M%S}.md"
        assert wrapper.created[0]["sandbox"] == "workspace-write"


@pytest.mark.parametrize(
    "failure",
    [
        "missing_summary",
        "unchanged_summary",
        "empty_summary",
        "missing_priority",
        "invalid_priority",
        "empty_reminder",
        "missing_detail",
        "failed_reply",
    ],
)
@pytest.mark.asyncio
async def test_incomplete_round_keeps_events_pending_without_publishing(tmp_path, monkeypatch, failure):
    async with running_runtime(tmp_path) as (runtime, context, wrapper):
        await create_agent(runtime)
        now = runtime.now()
        monkeypatch.setattr(runtime, "now", lambda: now)
        step = ProactiveAgentStep(app_context=context)
        paths = step._paths(await runtime.definition("monitor"), now)
        if failure == "unchanged_summary":
            runtime.files.write(paths["summary_path"], "# Previous summary\n")

        stages = []

        async def model_reply(_query, **kwargs):
            stage = ("record", "update", "reminder", "summary")[len(stages)]
            stages.append(stage)
            if stage == "record" and failure != "missing_detail":
                runtime.files.write(paths["detail_path"], "---\naction: write\n---\n# Details\n")
            if stage == "reminder":
                priority = "P9" if failure == "invalid_priority" else "P0"
                header = "action: write\n" + ("" if failure == "missing_priority" else f"priority: {priority}\n")
                body = "" if failure == "empty_reminder" else "Finished\n"
                runtime.files.write(paths["reminder_path"], "---\n" + header + "---\n" + body)
            if stage == "summary" and failure not in {"missing_summary", "unchanged_summary"}:
                runtime.files.write(paths["summary_path"], "" if failure == "empty_summary" else "# Summary\n")
            result = {
                "status": "failed" if failure == "failed_reply" else "completed",
                "session_id": kwargs["resume"],
                "error": "Interrupted" if failure == "failed_reply" else None,
                "result": "Done",
                "structured_output": None,
                "usage": None,
                "raw": {},
            }
            yield StreamChunk(chunk_type=ChunkEnum.REPLY_END, chunk=result, status=result["status"])

        monkeypatch.setattr(wrapper, "reply_stream", model_reply)
        with pytest.raises((ValueError, FileNotFoundError, RuntimeError)):
            await step(agent_name="monitor")
        assert (await runtime.status("monitor"))["pending_events"] == 1
        await runtime.outbox.sync()
        assert not runtime.notifications.pending


@pytest.mark.parametrize("existing_summary", [False, True])
@pytest.mark.asyncio
async def test_round_without_reminder_uses_saved_files_instead_of_response_json(
    tmp_path,
    monkeypatch,
    existing_summary,
):
    async with running_runtime(tmp_path) as (runtime, context, wrapper):
        await create_agent(runtime)
        now = runtime.now()
        monkeypatch.setattr(runtime, "now", lambda: now)
        paths = ProactiveAgentStep(app_context=context)._paths(await runtime.definition("monitor"), now)
        summary_mtime = None
        if existing_summary:
            runtime.files.write(paths["summary_path"], "# No material change\n")
            summary_mtime = paths["summary_path"].stat().st_mtime_ns

        async def model_reply(_query, **kwargs):
            runtime.files.write(paths["detail_path"], "---\naction: skip\n---\nNo material change.\n")
            return {
                "status": "completed",
                "session_id": kwargs["resume"],
                "result": "No reminder needed.",
                "structured_output": {"notifications": [{"priority": "P0", "message": "Ignore response JSON"}]},
            }

        async def model_stream(query, **kwargs):
            result = await model_reply(query, **kwargs)
            yield StreamChunk(
                chunk_type=ChunkEnum.REPLY_END,
                event="end",
                session_id=kwargs["resume"],
                turn_id="automatic-turn",
                status="completed",
                chunk={
                    "structured_output": None,
                    "usage": None,
                    "error": None,
                    "raw": {},
                    **result,
                },
            )

        monkeypatch.setattr(wrapper, "reply_stream", model_stream)
        response = await ProactiveAgentStep(app_context=context)(agent_name="monitor")
        assert response.success and response.answer["notifications"] == []
        assert (await runtime.status("monitor"))["pending_events"] == 0
        if existing_summary:
            assert paths["summary_path"].stat().st_mtime_ns == summary_mtime
        else:
            assert not paths["summary_path"].exists()
        assert document_action(paths["detail_path"]) == "skip"
        assert document_action(paths["reminder_path"]) == "skip"
        await runtime.outbox.sync()
        assert not runtime.notifications.pending


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
        assert paths["summary_path"] == tmp_path / "daily" / f"{now:%Y-%m-%d}" / "monitor.md"
        assert paths["yesterday_summary_path"] == tmp_path / "daily" / yesterday / "monitor.md"
        assert paths["detail_path"] == tmp_path / "daily" / f"{now:%Y-%m-%d}" / f"monitor_{now:%H%M%S}.md"
        assert paths["reminder_path"] == (
            tmp_path / "reme_agents/monitor/reminder" / f"{now:%Y-%m-%d}" / f"{now:%H%M%S}.md"
        )
        assert not paths["summary_path"].exists() and not paths["yesterday_summary_path"].exists()
        for delta in (-5, -2, 1):
            path = tmp_path / "daily" / f"{now + timedelta(days=delta):%Y-%m-%d}" / "monitor.md"
            runtime.files.write(path, "# Summary\n")
        runtime.files.write(tmp_path / "daily/not-a-date/monitor.md", "# Invalid date\n")
        paths = step._paths(agent, now)
        assert paths["yesterday_summary_path"] == tmp_path / "daily" / yesterday / "monitor.md"
        assert not paths["yesterday_summary_path"].exists()
        runtime.files.write(paths["yesterday_summary_path"], "# Yesterday\n")
        runtime.files.write(paths["summary_path"], "# Today\n")
        runtime.files.write(paths["detail_path"], "# Existing details\n")
        paths = step._paths(agent, now)
        assert paths["summary_path"].read_text() == "# Today\n"
        assert paths["yesterday_summary_path"].read_text() == "# Yesterday\n"
        assert paths["detail_path"].name == f"monitor_{now:%H%M%S}_1.md"
        assert paths["reminder_path"].name == f"{now:%H%M%S}_1.md"
        runtime.files.write(paths["reminder_path"], "---\npriority: P1\n---\nExisting reminder\n")
        paths = step._paths(agent, now)
        assert paths["detail_path"].name == f"monitor_{now:%H%M%S}_2.md"
        assert paths["reminder_path"].name == f"{now:%H%M%S}_2.md"
