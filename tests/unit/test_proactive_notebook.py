"""The shared proactive runner uses scoped, replay-safe notebook tools."""

from unittest.mock import AsyncMock

import pytest

from reme.components.job import BaseJob
from reme.config import resolve_app_config
from reme.enumeration import ChunkEnum
from reme.schema import Event, StreamChunk
from reme.utils.proactive_record import split_record
from .test_proactive_runtime import running_manager


@pytest.mark.parametrize(
    "notify,needs_repair,body",
    [
        (False, False, None),
        (False, False, "- Release notes: follow up at the scheduled review."),
        (True, False, ""),
        (True, True, "- Release notes: verify the download steps."),
    ],
)
@pytest.mark.asyncio
async def test_notebook_uses_common_runner_and_tools(tmp_path, monkeypatch, notify, needs_repair, body):
    """Verify full history, exact duplicate handling, and the standard Remy handoff."""
    config = resolve_app_config(config="proactive", log_config=False)
    preset = next(
        item
        for item in config["components"]["proactive_manager"]["default"]["runtimes"]
        if item["agent_name"] == "proactive_notebook"
    )
    runtime_name = "release_notebook" if notify else "proactive_notebook"
    preset = {**preset, "agent_name": runtime_name}
    language = "zh" if notify else "en"
    async with running_manager(tmp_path, runtimes=[preset], language=language) as (manager, context, wrapper):
        names = ["run_proactive_agent", "todo", "idea"]
        jobs = {name: BaseJob(name=name, app_context=context, **config["jobs"][name]) for name in names}
        context.jobs.update(jobs)
        async with jobs["run_proactive_agent"], jobs["todo"], jobs["idea"]:
            todo = (
                await jobs["todo"](
                    action="create",
                    project="ReMe",
                    text="Verify release",
                    progress_text="Package built",
                )
            ).answer
            assert (await jobs["todo"](action="update", id=todo["id"], progress_text="Published to staging")).success
            idea = (await jobs["idea"](action="create", project="ReMe", text="Record checksums")).answer
            dialogue = "resource/codex_dialogue/example/dialogue.md"
            image = "resource/codex_dialogue/example/images/result.png"
            manager.files.write(manager.files.path(dialogue), "# Codex dialogue\n\nThe smoke test passed.\n")
            now = manager.now()
            monkeypatch.setattr(manager, "now", lambda: now)
            relative_record = f"daily/{now:%Y-%m-%d}/{runtime_name}_{now:%H%M%S}.md"
            record_path = tmp_path / relative_record
            event = Event(
                id="dialogue-1",
                source="codex_dialogue",
                event_ts=now,
                description="New evidence",
                paths=[dialogue, image],
            )
            await manager.append_events(runtime_name, [event])

            turns = []

            async def reply(query, **options):
                repairing = bool(turns)
                assert relative_record in query
                if repairing:
                    assert "reminder_message must not be blank" in query
                    assert options == turns[0]
                else:
                    assert "sandbox" not in options
                    assert "output_schema" not in options
                    assert f"`{event.id}`" in query
                    assert "`todo get`" in query
                    assert ("同一件事换个说法" if notify else "worded differently") in query
                    assert "The smoke test passed." not in query
                    assert f"- {dialogue}" in query
                    assert f"- {image}" in query
                    assert ("大白话" if notify else "plain language") in query
                turns.append(options)
                assert options["config"]["skills.include_instructions"] is True
                assert options["skills"] == ["manage-todos-ideas"]
                assert options["job_tools"] == ["todo", "idea"]
                bound = options["injected_job_kwargs"]

                existing = await jobs["todo"](action="get", id=todo["id"], **bound)
                assert existing.success
                history = existing.answer["progress"]
                if len(turns) == 1:
                    assert [note["text"] for note in history] == ["Package built", "Published to staging"]
                assert (await jobs["idea"](action="list", project="ReMe", **bound)).success
                assert not (
                    await jobs["todo"](
                        action="update",
                        id=todo["id"],
                        status="closed",
                        progress_text="Tests passed",
                        **bound,
                    )
                ).success
                assert not (await jobs["idea"](action="archive", id=idea["id"], **bound)).success
                assert not (
                    await jobs["todo"](
                        action="update",
                        id=todo["id"],
                        progress_text="Unknown",
                        event_id="other",
                        **bound,
                    )
                ).success

                for text in (history[0]["text"], "Tests passed", "Tests passed", "Windows test started"):
                    progress = await jobs["todo"](
                        action="update",
                        id=todo["id"],
                        status="in_progress",
                        progress_text=text,
                        **bound,
                    )
                    assert progress.success
                updated = await jobs["idea"](action="update", id=idea["id"], text="Include checksums", **bound)
                assert not updated.success
                if len(turns) == 1:
                    corrected = await jobs["todo"](
                        action="update",
                        id=todo["id"],
                        progress_match=history[0],
                        progress_text="Package built and signed",
                        **bound,
                    )
                    assert corrected.success
                    assert corrected.answer["progress"][0] == {**history[0], "text": "Package built and signed"}
                    stale = await jobs["todo"](action="update", id=todo["id"], progress_match=history[0], **bound)
                    assert not stale.success
                    withdrawn = await jobs["todo"](action="update", id=todo["id"], progress_match=history[1], **bound)
                    assert withdrawn.success
                result = "Finished"
                if notify or body:
                    priority = "P1" if notify else "skip"
                    message = "Review the release notes." if notify and (repairing or not needs_repair) else ""
                    manager.files.write(
                        record_path,
                        f'---\nreminder_priority: {priority}\nreminder_message: "{message}"\n---\n\n{body or ""}\n',
                    )
                else:
                    result = "NO_KEY_INFORMATION"
                yield StreamChunk(
                    chunk_type=ChunkEnum.REPLY_END,
                    status="completed",
                    chunk={
                        "status": "completed",
                        "result": result,
                        "session_id": options["resume"],
                        "usage": None,
                        "error": None,
                        "raw": {},
                        "structured_output": None,
                    },
                )

            monkeypatch.setattr(wrapper, "reply_stream", reply)

            async def dispatch(name, **kwargs):
                return await jobs[name](**kwargs)

            run_job = AsyncMock(side_effect=dispatch)
            await manager.tick(run_job)
            await manager.runtimes[runtime_name].worker
            run_job.assert_awaited_once_with("run_proactive_agent", agent_name=runtime_name)
            assert len(turns) == (2 if needs_repair else 1)
            assert (await manager.status(runtime_name))["pending_events"] == 0
            after = (await jobs["todo"](action="get", id=todo["id"])).answer
            assert after["status"] == "in_progress" and [row["text"] for row in after["progress"]] == [
                "Package built and signed",
                "Tests passed",
                "Windows test started",
            ]
            assert all(set(row) == {"time", "text"} for row in after["progress"])
            updated_idea = (await jobs["idea"](action="get", id=idea["id"])).answer
            assert updated_idea == idea
            records = list((tmp_path / "daily").glob("*/*.md"))
            assert len(records) == int(notify or bool(body))
            reminders = await manager.list_inputs("Remy")
            assert reminders["total"] == int(notify)
            if records:
                assert records == [record_path]
                metadata, saved_body = split_record(record_path.read_text())
                assert metadata == {
                    "reminder_priority": "P1" if notify else "skip",
                    "reminder_message": "Review the release notes." if notify else "",
                }
                assert saved_body.strip() == (body or "")
            if notify:
                assert reminders["items"][0]["read_at"] is None


@pytest.mark.asyncio
async def test_restored_notebook_keeps_its_configured_step(tmp_path):
    """Restore content and the execution step together under the same runtime name."""
    async with running_manager(tmp_path) as (manager, _, _):
        name = "proactive_notebook"
        await manager.create(
            {
                "agent_name": name,
                "task_description": "Keep this",
                "step_backend": "proactive_notebook_step",
            },
        )
        await manager.delete(name)
        await manager.create({"agent_name": name, "task_description": "Restored"})
        restored = await manager.definition(name)
        assert restored.step_backend == "proactive_notebook_step"
        assert manager.registry.task_path(restored).read_text() == "Keep this"
        with pytest.raises(ValueError, match="Notebook runs only on events"):
            await manager.update(name, {"cron": "0 8 * * *"})
