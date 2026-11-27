"""One generic Job commits validated rounds using each Runtime binding."""

# pylint: disable=missing-function-docstring,protected-access,redefined-outer-name

import asyncio
from datetime import timedelta
from unittest.mock import AsyncMock

import pytest

from reme.schema import AgentActivity, Event, StreamChunk
from reme.steps.proactive.agent import ProactiveAgentStep
from .proactive_helpers import running_system, queue_event, runtime_config

SKIP = '---\nreminder_priority: skip\nreminder_message: ""\n---\n'
REMIND = "---\nreminder_priority: {priority}\nreminder_message: Verified result\n---{body}"


async def prepare_round(manager, _context, monkeypatch, name="monitor"):
    runtime = await manager.resolve(name)
    now = runtime.now()
    monkeypatch.setattr(runtime, "now", lambda: now)
    record = runtime.daily_directory / now.date().isoformat() / f"{name}_{now:%H%M%S}.md"
    return runtime, record


def install_round(
    wrapper,
    runtime,
    record,
    monkeypatch,
    *,
    text=None,
    result="Finished",
    status="completed",
):
    calls = []

    async def generate(query, **options):
        calls.append((query, options))
        for spec in options["skills"]:
            if spec.get("content"):
                manifest = options["_skills"].sources[spec["name"]] / "SKILL.md"
                assert spec["content"] in manifest.read_text(encoding="utf-8")
        if text is not None:
            runtime.files.write(record, text)
        yield StreamChunk(
            chunk_type="reply_end",
            status=status,
            chunk={
                "session_id": options["resume"],
                "status": status,
                "error": "Model failed" if status != "completed" else None,
                "result": result,
                "structured_output": None,
                "usage": None,
                "raw": {},
            },
        )

    monkeypatch.setattr(wrapper, "reply_stream", generate)
    return calls


@pytest.mark.parametrize("preset", ["proactive_memory", "proactive_note", "proactive_fin", "proactive_news"])
@pytest.mark.parametrize("agent_type", ["event", "time"])
@pytest.mark.parametrize("language", ["en", "zh"])
@pytest.mark.asyncio
async def test_generic_job_uses_runtime_skills_and_tools(tmp_path, monkeypatch, preset, agent_type, language):
    definition = runtime_config(preset)
    definition.update(agent_type=agent_type, schedule=None)
    if agent_type == "time":
        definition["schedule"] = "0 8 * * *"
    async with running_system(
        tmp_path,
        language=language,
        runtimes=[definition],
    ) as (
        manager,
        context,
        wrapper,
    ):
        runtime = await manager.resolve("monitor")
        if agent_type == "time":
            trigger = runtime.next_trigger(await runtime.definition())
            monkeypatch.setattr(runtime, "now", lambda: trigger)
            await runtime.inbox.append(Event(id="time-1", event_ts=trigger, source="time", description="Scheduled"))
        else:
            await queue_event(manager)
        runtime, record = await prepare_round(manager, context, monkeypatch)
        task = "Follow this project's {requirements}.\nReport completed experiments and pending decisions."
        runtime.task_path.write_text(task)
        calls = install_round(
            wrapper,
            runtime,
            record,
            monkeypatch,
            text=REMIND.format(priority="P1", body="\nEvidence\n"),
        )
        response = await context.jobs["run_proactive_agent"](agent_name="monitor")
        assert response.success, response.answer
        assert response.answer["processed"] == 1 and record.is_file()
        assert not await runtime.inbox.pending_ids()
        chat = context.components["agent_runtime"]["Remy"]
        assert len(await chat.inbox.pending_ids()) == 1
        delivery = (await runtime.outbox.entries())[0]
        assert (
            delivery.status == "delivered"
            and delivery.notification.record_path == record.relative_to(tmp_path).as_posix()
        )
        options = calls[0][1]
        assert calls[0][0].count(task) == 1 and runtime.task_path.read_text() == task
        assert options["approval_mode"] == "deny_all" and "output_schema" not in options
        expected_skills = ["proactive-record"]
        expected_tools = []
        if preset == "proactive_note":
            expected_skills.append("manage-notebook")
            expected_tools.append("manage_notebook")
        assert [spec["name"].rsplit("-", 1)[0] for spec in options["skills"]] == expected_skills
        assert options["job_tools"] == expected_tools
        assert len(calls) == len(wrapper.created) == 1
        assert "system_prompt" not in options
        if preset == "proactive_note":
            assert "sandbox" not in options and options["injected_job_kwargs"]["_automatic_notebook"]
        assert all(not source.exists() for source in options["_skills"].sources.values())


@pytest.mark.parametrize("skills", [[], ["manage-notebook"]])
@pytest.mark.parametrize("agent_type", ["event", "time"])
@pytest.mark.parametrize("language", ["en", "zh"])
@pytest.mark.asyncio
async def test_tasks_without_record_skill_complete_without_a_record(
    tmp_path,
    monkeypatch,
    skills,
    agent_type,
    language,
):
    definition = {"agent_name": "monitor", "agent_type": agent_type, "skills": skills}
    if agent_type == "time":
        definition["schedule"] = "0 8 * * *"
    async with running_system(tmp_path, language=language, runtimes=[definition]) as (manager, context, wrapper):
        runtime, record = await prepare_round(manager, context, monkeypatch)
        if agent_type == "time":
            await runtime.inbox.append(
                Event(id="time-1", event_ts=runtime.now(), source="time", description="Scheduled"),
            )
        else:
            await queue_event(manager)
        calls = install_round(wrapper, runtime, record, monkeypatch)
        response = await context.jobs["run_proactive_agent"](agent_name="monitor")
        assert response.success, response.answer
        assert response.answer == {"agent_name": "monitor", "processed": 1, "notifications": []}
        assert not await runtime.inbox.pending_ids() and not await runtime.outbox.entries()
        assert not await context.components["agent_runtime"]["Remy"].inbox.pending_ids()
        assert not record.exists()
        receipt = (await runtime.inbox.receipts())[-1]
        assert receipt.metadata["session_id"] and receipt.metadata["reminder_path"] is None
        assert receipt.metadata["notification"] is None
        query, options = calls[0]
        assert runtime.task_path.read_text() in query
        assert "Record path for this run" not in query and "本轮记录路径" not in query
        assert [spec["name"].rsplit("-", 1)[0] for spec in options["skills"]] == skills


@pytest.mark.parametrize("agent_type", ["event", "time"])
@pytest.mark.parametrize("language", ["en", "zh"])
@pytest.mark.asyncio
async def test_job_step_prompt_overrides_render_inputs_without_changing_runtime_binding(
    tmp_path,
    monkeypatch,
    agent_type,
    language,
):
    definition = {"agent_name": "monitor", "agent_type": agent_type}
    if agent_type == "time":
        definition["schedule"] = "0 8 * * *"
    async with running_system(tmp_path, language=language, runtimes=[definition]) as (manager, context, wrapper):
        runtime, record = await prepare_round(manager, context, monkeypatch)
        binding = runtime.session_options()
        task = "Read the {source} and report its result."
        runtime.files.write(runtime.task_path, task)
        attachment = "resource/input {batch}.md"
        runtime.files.write(runtime.files.path(attachment), "Evidence\n")
        event = Event(
            id="input-1",
            event_ts=runtime.now() - timedelta(hours=1),
            source="time" if agent_type == "time" else "test",
            description="Result {verified}",
            paths=[] if agent_type == "time" else [attachment],
        )
        await runtime.inbox.append(event)
        suffix = "_zh" if language == "zh" else ""
        context.jobs["run_proactive_agent"].step_specs[0][1]["prompt_dict"] = {
            f"query{suffix}": "Job task: {task_description}\nInputs: {event_batch}\n{record_instruction}",
            f"record_instruction{suffix}": "Record after completing the task: {record_path}",
            f"event_input{suffix}": "Observation: {description}\nAttachments: {paths}",
            f"time_input{suffix}": "Scheduled: {event_ts}",
        }
        calls = install_round(wrapper, runtime, record, monkeypatch, text=SKIP)
        response = await context.jobs["run_proactive_agent"](agent_name="monitor")
        assert response.success, response.answer
        query = calls[0][0]
        assert task in query and runtime.to_workspace_relative(record) in query
        if agent_type == "time":
            assert f"Scheduled: {event.event_ts.isoformat()}" in query
        else:
            assert "Observation: Result {verified}" in query and attachment in query
        assert not await runtime.inbox.pending_ids()
        assert runtime.session_options() == binding


@pytest.mark.parametrize("status", ["completed", "failed"])
@pytest.mark.parametrize("text", ["---\nbroken: [yaml\n---\nBusiness output\n", REMIND.format(priority="P1", body="")])
@pytest.mark.asyncio
async def test_tasks_without_record_skill_do_not_validate_deliver_or_clean_up_business_files(
    tmp_path,
    monkeypatch,
    status,
    text,
):
    async with running_system(
        tmp_path,
        runtimes=[{"agent_name": "monitor", "agent_type": "event", "skills": []}],
    ) as (manager, context, wrapper):
        runtime, record = await prepare_round(manager, context, monkeypatch)
        await queue_event(manager)
        install_round(wrapper, runtime, record, monkeypatch, text=text, status=status)
        response = await context.jobs["run_proactive_agent"](agent_name="monitor")
        completed = status == "completed"
        assert response.success is completed
        assert record.read_text() == text
        assert await runtime.inbox.pending_ids() == (set() if completed else {"event-1"})
        assert not (await runtime.inbox.snapshot())["in_flight_count"]
        assert not await runtime.outbox.entries()
        assert not await context.components["agent_runtime"]["Remy"].inbox.pending_ids()


@pytest.mark.parametrize("agent_type", ["event", "time"])
@pytest.mark.parametrize("result", ["Finished", "NO_KEY_INFORMATION"])
@pytest.mark.asyncio
async def test_default_record_skill_requires_a_record_except_for_an_explicit_event_skip(
    tmp_path,
    monkeypatch,
    agent_type,
    result,
):
    definition = {"agent_name": "monitor", "agent_type": agent_type}
    if agent_type == "time":
        definition["schedule"] = "0 8 * * *"
    async with running_system(tmp_path, runtimes=[definition]) as (manager, context, wrapper):
        runtime, record = await prepare_round(manager, context, monkeypatch)
        if agent_type == "time":
            await runtime.inbox.append(
                Event(id="time-1", event_ts=runtime.now(), source="time", description="Scheduled"),
            )
        else:
            await queue_event(manager)
        install_round(wrapper, runtime, record, monkeypatch, result=result)
        response = await context.jobs["run_proactive_agent"](agent_name="monitor")
        skipped = agent_type == "event" and result == "NO_KEY_INFORMATION"
        assert response.success is skipped
        assert bool(await runtime.inbox.pending_ids()) is not skipped
        assert not (await runtime.inbox.snapshot())["in_flight_count"]
        assert not record.exists() and not await runtime.outbox.entries()


@pytest.mark.parametrize("agent_type", ["event", "time"])
@pytest.mark.parametrize("language", ["en", "zh"])
@pytest.mark.asyncio
async def test_record_skill_refreshes_context_each_round_without_changing_cached_templates(
    tmp_path,
    monkeypatch,
    agent_type,
    language,
):
    definition = runtime_config("proactive_note")
    definition.update(agent_type=agent_type, schedule="0 8 * * *" if agent_type == "time" else None)
    definition.update(recent_history_hours=36, max_local_search_calls=7)
    async with running_system(tmp_path, language=language, runtimes=[definition]) as (manager, context, wrapper):
        runtime = await manager.resolve("monitor")
        templates = runtime.get_skills()
        first_time = runtime.now().replace(hour=9, minute=0, second=0, microsecond=0)
        previous = None
        for index in range(2):
            now = first_time + timedelta(days=index)
            monkeypatch.setattr(runtime, "now", lambda current=now: current)
            if agent_type == "time":
                await runtime.inbox.append(
                    Event(
                        id=f"time-{index}",
                        event_ts=now - timedelta(hours=1),
                        source="time",
                        description="Scheduled",
                    ),
                )
            else:
                await queue_event(manager, identifier=f"event-{index}")
            _, record = await prepare_round(manager, context, monkeypatch)
            calls = install_round(wrapper, runtime, record, monkeypatch, text=SKIP)
            response = await context.jobs["run_proactive_agent"](agent_name="monitor")
            assert response.success, response.answer
            skills = calls[0][1]["skills"]
            content = skills[0]["content"]
            directory = runtime.to_workspace_relative(runtime.daily_directory)
            relative = runtime.to_workspace_relative(record)
            expected = [now.isoformat(), "Asia/Shanghai", f"`{directory}`", f"`{relative}`"]
            if language == "zh":
                expected.extend(["最近 36 小时", "用了 7 次本地搜索工具"])
            else:
                expected.extend(["last 36 hours", "local search tool 7 times"])
            assert all(value in content for value in expected)
            if previous is not None:
                assert all(value not in content for value in previous)
            previous = now.isoformat(), relative
            assert skills[1] == templates[1]
            assert runtime.get_skills() == templates


@pytest.mark.parametrize("disabled", [False, True])
@pytest.mark.parametrize("phase", ["completed", "failed", "interrupted"])
@pytest.mark.asyncio
async def test_skipped_execution_preserves_previous_activity(tmp_path, disabled, phase):
    async with running_system(tmp_path, runtimes=[{"agent_type": "event", "agent_name": "monitor"}]) as (
        manager,
        context,
        wrapper,
    ):
        runtime = await manager.resolve("monitor")
        runtime.activity = AgentActivity(
            session_id="previous-session",
            turn_id="previous-turn",
            source="task",
            task_id="monitor",
            phase=phase,
            started_at=runtime.now(),
        )
        previous = runtime.activity.model_dump()
        if disabled:
            await manager.update("monitor", {"enabled": False})
        response = await context.jobs["run_proactive_agent"](agent_name="monitor")
        assert response.success and response.answer == {
            "agent_name": "monitor",
            **({"skipped": "disabled"} if disabled else {"processed": 0}),
        }
        assert runtime.activity.model_dump() == previous and not wrapper.turns
        assert "reme.agent_run" not in response.metadata


@pytest.mark.parametrize(
    "text,result",
    [(None, "NO_KEY_INFORMATION"), (SKIP, "Finished"), (SKIP + "Normal progress", "Finished")],
)
@pytest.mark.asyncio
async def test_event_skip_commits_without_creating_a_notification(tmp_path, monkeypatch, text, result):
    async with running_system(tmp_path, runtimes=[{"agent_type": "event", "agent_name": "monitor"}]) as (
        manager,
        context,
        wrapper,
    ):
        runtime, record = await prepare_round(manager, context, monkeypatch)
        await queue_event(manager)
        install_round(wrapper, runtime, record, monkeypatch, text=text, result=result)
        response = await context.jobs["run_proactive_agent"](agent_name="monitor")
        assert response.success, response.answer
        assert not await runtime.inbox.pending_ids() and not await runtime.outbox.entries()
        if text is not None:
            assert record.read_text() == text


@pytest.mark.parametrize("draft", [None, SKIP, REMIND.format(priority="P1", body="\nEvidence\n")])
@pytest.mark.asyncio
async def test_model_failure_discards_draft_without_consuming_inputs(tmp_path, monkeypatch, draft):
    async with running_system(tmp_path, runtimes=[{"agent_type": "event", "agent_name": "monitor"}]) as (
        manager,
        context,
        wrapper,
    ):
        runtime, record = await prepare_round(manager, context, monkeypatch)
        await queue_event(manager)
        install_round(wrapper, runtime, record, monkeypatch, text=draft, status="failed")
        response = await context.jobs["run_proactive_agent"](agent_name="monitor")
        assert not response.success
        assert (
            await runtime.inbox.pending_ids() == {"event-1"} and not (await runtime.inbox.snapshot())["in_flight_count"]
        )
        assert not await runtime.outbox.entries()
        assert not record.exists()


@pytest.mark.parametrize(
    "invalid",
    [
        "---\nreminder_priority: unknown\nreminder_message: Result\n---\nEvidence\n",
        '---\nreminder_priority: P1\nreminder_message: ""\n---\nEvidence\n',
        "---\nreminder_priority: [P1\nreminder_message: Broken YAML\n---\nEvidence\n",
    ],
)
@pytest.mark.asyncio
async def test_invalid_record_discards_draft_and_retries_inputs_in_a_new_round(
    tmp_path,
    monkeypatch,
    invalid,
):
    async with running_system(tmp_path, runtimes=[{"agent_type": "event", "agent_name": "monitor"}]) as (
        manager,
        context,
        wrapper,
    ):
        runtime, record = await prepare_round(manager, context, monkeypatch)
        await queue_event(manager)
        calls = install_round(wrapper, runtime, record, monkeypatch, text=invalid)
        response = await context.jobs["run_proactive_agent"](agent_name="monitor")
        assert not response.success
        assert len(calls) == len(wrapper.created) == 1 and not record.exists()
        assert await runtime.inbox.pending_ids() == {"event-1"}
        assert not (await runtime.inbox.snapshot())["in_flight_count"] and not await runtime.outbox.entries()
        assert response.metadata["reme.agent_run"]["model_status"] == "completed"
        assert response.metadata["reme.agent_run"]["result"] == "failed"

        valid = REMIND.format(priority="P1", body="\nEvidence\n")
        calls = install_round(wrapper, runtime, record, monkeypatch, text=valid)
        response = await context.jobs["run_proactive_agent"](agent_name="monitor")
        assert response.success, response.answer
        assert len(calls) == 1 and len(wrapper.created) == 2
        assert record.read_text() == valid
        assert not await runtime.inbox.pending_ids() and len(await runtime.outbox.entries()) == 1


@pytest.mark.parametrize("receipt_saved", [False, True])
@pytest.mark.parametrize("text", [SKIP, REMIND.format(priority="P1", body="\nEvidence\n")])
@pytest.mark.asyncio
async def test_ack_failure_discards_only_records_without_a_durable_receipt(
    tmp_path,
    monkeypatch,
    receipt_saved,
    text,
):
    async with running_system(tmp_path, runtimes=[{"agent_type": "event", "agent_name": "monitor"}]) as (
        manager,
        context,
        wrapper,
    ):
        runtime, record = await prepare_round(manager, context, monkeypatch)
        await queue_event(manager)
        install_round(wrapper, runtime, record, monkeypatch, text=text)
        replace = runtime.inbox._replace

        def fail(path, raw):
            if path.name == ("inbox_unread.jsonl" if receipt_saved else "inbox_read.jsonl"):
                raise OSError("Acknowledgement write failed")
            replace(path, raw)

        monkeypatch.setattr(runtime.inbox, "_replace", fail)
        response = await context.jobs["run_proactive_agent"](agent_name="monitor")
        assert not response.success and runtime.inbox.failed
        assert record.exists() is receipt_saved
        assert not await runtime.outbox.entries()

    async with running_system(tmp_path) as (manager, context, wrapper):
        runtime = await manager.resolve("monitor")
        assert await runtime.inbox.pending_ids() == (set() if receipt_saved else {"event-1"})
        await runtime.recover_handoffs()
        reminders = await context.components["agent_runtime"]["Remy"].inbox.pending_ids()
        assert bool(reminders) is (receipt_saved and text != SKIP)
        assert not wrapper.turns
        if receipt_saved:
            assert record.read_text() == text


@pytest.mark.parametrize("stage", ["prepare_reference", "deliver"])
@pytest.mark.asyncio
async def test_delivery_failure_preserves_committed_record_for_retry(tmp_path, monkeypatch, stage):
    async with running_system(tmp_path, runtimes=[{"agent_type": "event", "agent_name": "monitor"}]) as (
        manager,
        context,
        wrapper,
    ):
        runtime, record = await prepare_round(manager, context, monkeypatch)
        await queue_event(manager)
        text = REMIND.format(priority="P1", body="\nEvidence\n")
        install_round(wrapper, runtime, record, monkeypatch, text=text)
        with monkeypatch.context() as fault:
            fault.setattr(runtime.outbox, stage, AsyncMock(side_effect=OSError("Delivery unavailable")))
            response = await context.jobs["run_proactive_agent"](agent_name="monitor")
        assert response.success and runtime.delivery_error
        assert record.read_text() == text and not await runtime.inbox.pending_ids()
        assert not await context.components["agent_runtime"]["Remy"].inbox.pending_ids()
        await runtime.recover_handoffs()
        assert len(await context.components["agent_runtime"]["Remy"].inbox.pending_ids()) == 1
        assert record.read_text() == text and runtime.delivery_error is None


@pytest.mark.parametrize("committed", [False, True])
@pytest.mark.asyncio
async def test_cancellation_discards_only_uncommitted_records(tmp_path, monkeypatch, committed):
    async with running_system(tmp_path, runtimes=[{"agent_type": "event", "agent_name": "monitor"}]) as (
        manager,
        context,
        wrapper,
    ):
        runtime, record = await prepare_round(manager, context, monkeypatch)
        await queue_event(manager)
        text = REMIND.format(priority="P1", body="\nEvidence\n")
        if committed:
            install_round(wrapper, runtime, record, monkeypatch, text=text)
            monkeypatch.setattr(
                runtime.outbox,
                "prepare_reference",
                AsyncMock(side_effect=asyncio.CancelledError),
            )
            with pytest.raises(asyncio.CancelledError):
                await ProactiveAgentStep(app_context=context)(agent_name="monitor")
        else:
            written = asyncio.Event()

            async def stream(*_, **__):
                runtime.files.write(record, text)
                written.set()
                await asyncio.Event().wait()
                yield

            monkeypatch.setattr(wrapper, "reply_stream", stream)
            task = asyncio.create_task(ProactiveAgentStep(app_context=context)(agent_name="monitor"))
            await asyncio.wait_for(written.wait(), 5)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        assert record.exists() is committed
        assert await runtime.inbox.pending_ids() == (set() if committed else {"event-1"})
        assert not (await runtime.inbox.snapshot())["in_flight_count"]
        assert not await runtime.outbox.entries()


@pytest.mark.asyncio
async def test_failed_round_preserves_an_existing_record_at_the_original_path(tmp_path, monkeypatch):
    async with running_system(tmp_path, runtimes=[{"agent_type": "event", "agent_name": "monitor"}]) as (
        manager,
        context,
        wrapper,
    ):
        runtime, existing = await prepare_round(manager, context, monkeypatch)
        runtime.files.write(existing, "Existing user record\n")
        record = existing.with_stem(existing.stem + "_1")
        await queue_event(manager)
        install_round(wrapper, runtime, record, monkeypatch, text=SKIP, status="failed")
        response = await context.jobs["run_proactive_agent"](agent_name="monitor")
        assert not response.success
        assert existing.read_text() == "Existing user record\n" and not record.exists()
        assert await runtime.inbox.pending_ids() == {"event-1"}


@pytest.mark.parametrize("phase", ["session", "inference"])
@pytest.mark.asyncio
async def test_timeout_releases_claim_and_scoped_skills(tmp_path, monkeypatch, phase):
    async with running_system(
        tmp_path,
        runtimes=[runtime_config("proactive_note")],
    ) as (manager, context, wrapper):
        runtime, record = await prepare_round(manager, context, monkeypatch)
        await queue_event(manager)
        sources = []
        step = ProactiveAgentStep(app_context=context, timeout_seconds=10)

        async def stall(**options):
            sources.extend(options["_skills"].sources.values())
            assert len(sources) == 2 and all((source / "SKILL.md").is_file() for source in sources)
            step._timeout.reschedule(asyncio.get_running_loop().time())
            await asyncio.Event().wait()

        async def stream(*_, **options):
            sources.extend(options["_skills"].sources.values())
            assert len(sources) == 2 and all((source / "SKILL.md").is_file() for source in sources)
            runtime.files.write(record, REMIND.format(priority="P1", body="\nEvidence\n"))
            step._timeout.reschedule(asyncio.get_running_loop().time())
            await asyncio.Event().wait()
            yield

        if phase == "session":
            monkeypatch.setattr(wrapper, "create_session", stall)
        else:
            monkeypatch.setattr(wrapper, "reply_stream", stream)
        with pytest.raises(TimeoutError):
            await step(agent_name="monitor")
        assert (
            await runtime.inbox.pending_ids() == {"event-1"} and not (await runtime.inbox.snapshot())["in_flight_count"]
        )
        assert sources and all(not source.exists() for source in sources)
        assert not record.exists()


@pytest.mark.asyncio
async def test_cancelled_round_does_not_wait_for_optional_native_statistics(tmp_path, monkeypatch):
    async with running_system(tmp_path, runtimes=[{"agent_type": "event", "agent_name": "monitor"}]) as (
        manager,
        context,
        wrapper,
    ):
        runtime, _ = await prepare_round(manager, context, monkeypatch)
        await queue_event(manager)
        statistics_read = asyncio.Event()
        inference_started = asyncio.Event()

        async def stream(*_, **__):
            inference_started.set()
            step._timeout.reschedule(asyncio.get_running_loop().time())
            await asyncio.Event().wait()
            yield

        async def statistics(*_):
            statistics_read.set()
            await asyncio.Event().wait()

        monkeypatch.setattr(wrapper, "reply_stream", stream)
        monkeypatch.setattr(wrapper, "get_tool_calls", statistics)
        step = ProactiveAgentStep(app_context=context, timeout_seconds=10)
        with pytest.raises(TimeoutError):
            await asyncio.wait_for(step(agent_name="monitor"), 5)
        assert inference_started.is_set()
        assert not statistics_read.is_set()
        assert await runtime.inbox.pending_ids() == {"event-1"}
        assert not (await runtime.inbox.snapshot())["in_flight_count"]


@pytest.mark.parametrize("will_retry", [False, True])
@pytest.mark.asyncio
async def test_stream_errors_only_allow_commit_when_explicitly_retryable(tmp_path, monkeypatch, will_retry):
    async with running_system(tmp_path, runtimes=[{"agent_type": "event", "agent_name": "monitor"}]) as (
        manager,
        context,
        wrapper,
    ):
        runtime, record = await prepare_round(manager, context, monkeypatch)
        await queue_event(manager)
        install_round(wrapper, runtime, record, monkeypatch, text=SKIP)
        original = wrapper.reply_stream

        async def stream(*args, **kwargs):
            yield StreamChunk(chunk_type="error", chunk="Provider error", metadata={"will_retry": will_retry})
            async for chunk in original(*args, **kwargs):
                yield chunk

        monkeypatch.setattr(wrapper, "reply_stream", stream)
        response = await context.jobs["run_proactive_agent"](agent_name="monitor")
        assert response.success is will_retry
        assert record.exists() is will_retry
        assert await runtime.inbox.pending_ids() == (set() if will_retry else {"event-1"})
        assert not await runtime.outbox.entries()


@pytest.mark.asyncio
async def test_record_change_after_validation_preserves_inputs(tmp_path, monkeypatch):
    async with running_system(tmp_path, runtimes=[{"agent_type": "event", "agent_name": "monitor"}]) as (
        manager,
        context,
        wrapper,
    ):
        runtime, record = await prepare_round(manager, context, monkeypatch)
        await queue_event(manager)
        install_round(wrapper, runtime, record, monkeypatch, text=REMIND.format(priority="P1", body="\nEvidence\n"))
        commit = runtime.commit_round

        async def edit(*args):
            record.write_text(record.read_text() + "External change\n")
            return await commit(*args)

        monkeypatch.setattr(runtime, "commit_round", edit)
        response = await context.jobs["run_proactive_agent"](agent_name="monitor")
        assert not response.success and "External change" in record.read_text()
        assert await runtime.inbox.pending_ids() == {"event-1"}
