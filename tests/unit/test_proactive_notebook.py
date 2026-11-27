"""Notebook domain tools enforce automatic-write boundaries and replay safety."""

# pylint: disable=missing-function-docstring,protected-access,redefined-outer-name

import pytest

from reme.schema import StreamChunk
from .proactive_helpers import running_system, queue_event
from .test_proactive_run import prepare_round, REMIND, SKIP


@pytest.mark.parametrize("notify", [False, True])
@pytest.mark.asyncio
async def test_notebook_reads_full_history_updates_progress_and_hands_off_decisions(tmp_path, monkeypatch, notify):
    async with running_system(
        tmp_path,
        runtimes=[{"agent_type": "event", "agent_name": "notes", "run_job": "run_notebook_agent"}],
    ) as (
        manager,
        context,
        wrapper,
    ):
        job = context.jobs["manage_notebook"]
        todo = (await job(kind="todo", action="create", text="Verify release", progress_text="Package built")).answer
        await job(kind="todo", action="update", id=todo["id"], progress_text="Published to staging")
        idea = (await job(kind="idea", action="create", text="Record checksums")).answer
        runtime, record = await prepare_round(manager, context, monkeypatch, name="notes")
        await queue_event(manager, name="notes")

        async def generate(query, **options):
            assert "sandbox" not in options
            assert options["job_tools"] == ["manage_notebook"]
            assert [spec["name"].rsplit("-", 1)[0] for spec in options["skills"]] == [
                "proactive-record",
                "manage-notebook",
            ]
            assert all(spec["content"] for spec in options["skills"]) and runtime.task_path.read_text() in query
            bound = options["injected_job_kwargs"]
            get = (await job(kind="todo", action="get", id=todo["id"], **bound)).answer
            assert [row["text"] for row in get["progress"]] == ["Package built", "Published to staging"]
            for values in [
                {"kind": "todo", "action": "create", "text": "New"},
                {"kind": "todo", "action": "update", "id": todo["id"], "status": "closed"},
                {"kind": "todo", "action": "update", "id": todo["id"], "text": "Rewrite"},
                {"kind": "idea", "action": "archive", "id": idea["id"]},
                {"kind": "idea", "action": "update", "id": idea["id"], "text": "Rewrite"},
            ]:
                assert not (await job(**values, **bound)).success
            for text in ["Package built", "Tests passed", "Tests passed"]:
                assert (
                    await job(
                        kind="todo",
                        action="update",
                        id=todo["id"],
                        progress_text=text,
                        status="in_progress",
                        **bound,
                    )
                ).success
            assert (
                await job(
                    kind="todo",
                    action="update",
                    id=todo["id"],
                    progress_match=get["progress"][0],
                    progress_text="Built and signed",
                    **bound,
                )
            ).success
            stale = await job(kind="todo", action="update", id=todo["id"], progress_match=get["progress"][0], **bound)
            assert not stale.success
            assert (
                await job(kind="todo", action="update", id=todo["id"], progress_match=get["progress"][1], **bound)
            ).success
            runtime.files.write(
                record,
                REMIND.format(priority="P1", body="\nNeeds user review\n") if notify else SKIP + "Progress recorded\n",
            )
            yield StreamChunk(
                chunk_type="reply_end",
                status="completed",
                chunk={
                    "status": "completed",
                    "result": "Finished",
                    "session_id": options["resume"],
                    "usage": None,
                    "raw": {},
                    "error": None,
                    "structured_output": None,
                },
            )

        monkeypatch.setattr(wrapper, "reply_stream", generate)
        response = await context.jobs["run_notebook_agent"](agent_name="notes")
        assert response.success, response.answer
        after = (await job(kind="todo", action="get", id=todo["id"])).answer
        assert after["status"] == "in_progress" and [row["text"] for row in after["progress"]] == [
            "Built and signed",
            "Tests passed",
        ]
        assert (await job(kind="idea", action="get", id=idea["id"])).answer == idea
        assert len(await context.components["agent_runtime"]["Remy"].inbox.pending_ids()) == int(notify)


@pytest.mark.asyncio
async def test_restore_preserves_notebook_role_and_event_only_contract(tmp_path):
    async with running_system(
        tmp_path,
        runtimes=[
            {"agent_type": "event", "agent_name": "notes", "run_job": "run_notebook_agent", "task_description": "Keep"},
        ],
    ) as (manager, _, _):
        await manager.delete("notes")
        await manager.create({"agent_type": "event", "agent_name": "notes"})
        runtime = await manager.resolve("notes")
        assert runtime.run_job == "run_notebook_agent" and runtime.task_path.read_text() == "Keep"
        with pytest.raises(ValueError):
            await manager.update("notes", {"schedule": "0 8 * * *"})
