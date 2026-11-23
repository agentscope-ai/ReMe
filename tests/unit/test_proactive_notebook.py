"""The shared proactive runner uses scoped, replay-safe notebook tools."""

from unittest.mock import AsyncMock

import pytest

from reme.components.job import BaseJob
from reme.config import resolve_app_config
from reme.enumeration import ChunkEnum
from reme.schema import Event, StreamChunk
from .test_proactive_runtime import running_manager


@pytest.mark.parametrize("notify,needs_repair", [(False, False), (True, False), (True, True)])
@pytest.mark.asyncio
async def test_notebook_uses_common_runner_and_tools(tmp_path, monkeypatch, notify, needs_repair):
    """Verify tool scope, write replay, and the standard Remy handoff."""
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
            todo = (await jobs["todo"](action="create", project="ReMe", text="Verify release")).answer
            idea = (await jobs["idea"](action="create", project="ReMe", text="Record checksums")).answer
            dialogue = "resource/codex_dialogue/example/dialogue.md"
            image = "resource/codex_dialogue/example/images/result.png"
            manager.files.write(manager.files.path(dialogue), "# Codex dialogue\n\nThe smoke test passed.\n")
            event = Event(
                id="dialogue-1",
                source="codex_dialogue",
                event_ts=manager.now(),
                description="New evidence",
                paths=[dialogue, image],
            )
            await manager.append_events(runtime_name, [event])

            turns = []

            async def reply(query, **options):
                if turns:
                    assert "重新提交随记的结构化结果" in query
                    assert options == turns[0]
                else:
                    assert event.id in query
                    assert "> # Codex dialogue\n> \n> The smoke test passed." in query
                    assert f"`{dialogue}`" in query
                    assert f"- {dialogue}" not in query
                    assert f"- {image}" in query
                    assert ("大白话" if notify else "plain language") in query
                turns.append(options)
                assert options["sandbox"] == "read-only"
                assert options["skills"] == ["manage-todos-ideas"]
                assert options["job_tools"] == ["todo", "idea"]
                bound = options["injected_job_kwargs"]

                assert (await jobs["todo"](action="get", id=todo["id"], **bound)).success
                assert (await jobs["idea"](action="list", project="ReMe", **bound)).success
                assert not (await jobs["todo"](action="update", id=todo["id"], status="done", **bound)).success
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

                for text in ("Tests passed", "Replay of tests"):
                    progress = await jobs["todo"](
                        action="update",
                        id=todo["id"],
                        progress_text=text,
                        event_id=event.id,
                        **bound,
                    )
                    updated = await jobs["idea"](action="update", id=idea["id"], text=text, event_id=event.id, **bound)
                    assert progress.success and updated.success
                yield StreamChunk(
                    chunk_type=ChunkEnum.REPLY_END,
                    status="completed",
                    chunk={
                        "status": "completed",
                        "result": "Finished",
                        "session_id": options["resume"],
                        "usage": None,
                        "error": None,
                        "raw": {},
                        "structured_output": {
                            "reminder_priority": "P1" if notify else "skip",
                            "reminder_message": "Review the release notes." if notify else "Notebook updated.",
                            "body": (
                                "- Release notes: verify the download steps."
                                if notify and (not needs_repair or len(turns) > 1)
                                else ""
                            ),
                        },
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
            assert after["status"] == "open" and [row["text"] for row in after["progress"]] == ["Tests passed"]
            assert (await jobs["idea"](action="get", id=idea["id"])).answer["text"] == "Tests passed"
            records = list((tmp_path / "daily").glob("*/*.md"))
            assert len(records) == int(notify)
            if records:
                assert "- Release notes: verify the download steps." in records[0].read_text()
            assert (await manager.list_inputs("Remy"))["total"] == int(notify)


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
