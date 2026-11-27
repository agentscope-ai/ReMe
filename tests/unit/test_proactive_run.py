"""One generic Job commits validated rounds using each Runtime binding."""

# pylint: disable=missing-function-docstring,protected-access,redefined-outer-name

import asyncio

import pytest

from reme.schema import Event, StreamChunk
from reme.steps.proactive.run import ProactiveAgentStep
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
                assert (options["_skills"].sources[spec["name"]] / "SKILL.md").is_file()
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
        assert all("$" + skill["name"] in calls[0][0] for skill in options["skills"])
        assert str(tmp_path) in options["system_prompt"] and runtime.now().isoformat() in options["system_prompt"]
        assert context.app_config.timezone in options["system_prompt"]
        if preset == "proactive_note":
            assert "sandbox" not in options and options["injected_job_kwargs"]["_automatic_notebook"]
        assert all(not source.exists() for source in options["_skills"].sources.values())


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
        response = await context.jobs[runtime.run_job](agent_name="monitor")
        assert response.success, response.answer
        assert not await runtime.inbox.pending_ids() and not await runtime.outbox.entries()
        if text is not None:
            assert record.read_text() == text


@pytest.mark.parametrize("draft", [None, SKIP, REMIND.format(priority="P1", body="\nEvidence\n")])
@pytest.mark.asyncio
async def test_model_failure_never_consumes_inputs_even_with_a_valid_draft(tmp_path, monkeypatch, draft):
    async with running_system(tmp_path, runtimes=[{"agent_type": "event", "agent_name": "monitor"}]) as (
        manager,
        context,
        wrapper,
    ):
        runtime, record = await prepare_round(manager, context, monkeypatch)
        await queue_event(manager)
        install_round(wrapper, runtime, record, monkeypatch, text=draft, status="failed")
        response = await context.jobs[runtime.run_job](agent_name="monitor")
        assert not response.success
        assert (
            await runtime.inbox.pending_ids() == {"event-1"} and not (await runtime.inbox.snapshot())["in_flight_count"]
        )
        assert not await runtime.outbox.entries()
        if draft is not None:
            assert record.read_text() == draft


@pytest.mark.parametrize(
    "invalid",
    [
        "---\nreminder_priority: unknown\nreminder_message: Result\n---\nEvidence\n",
        '---\nreminder_priority: P1\nreminder_message: ""\n---\nEvidence\n',
        "---\nreminder_priority: [P1\nreminder_message: Broken YAML\n---\nEvidence\n",
    ],
)
@pytest.mark.asyncio
async def test_invalid_record_keeps_inputs_and_draft_for_a_new_round_without_repair_query(
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
        response = await context.jobs[runtime.run_job](agent_name="monitor")
        assert not response.success
        assert len(calls) == len(wrapper.created) == 1 and record.read_text() == invalid
        assert await runtime.inbox.pending_ids() == {"event-1"}
        assert not (await runtime.inbox.snapshot())["in_flight_count"] and not await runtime.outbox.entries()
        assert response.metadata["reme.agent_run"]["model_status"] == "completed"
        assert response.metadata["reme.agent_run"]["result"] == "failed"

        retry_record = record.with_stem(record.stem + "_1")
        valid = REMIND.format(priority="P1", body="\nEvidence\n")
        calls = install_round(wrapper, runtime, retry_record, monkeypatch, text=valid)
        response = await context.jobs[runtime.run_job](agent_name="monitor")
        assert response.success, response.answer
        assert len(calls) == 1 and len(wrapper.created) == 2
        assert record.read_text() == invalid and retry_record.read_text() == valid
        assert not await runtime.inbox.pending_ids() and len(await runtime.outbox.entries()) == 1


@pytest.mark.parametrize("phase", ["session", "inference"])
@pytest.mark.asyncio
async def test_timeout_releases_claim_and_scoped_skills(tmp_path, monkeypatch, phase):
    async with running_system(
        tmp_path,
        runtimes=[runtime_config("proactive_note")],
    ) as (manager, context, wrapper):
        runtime, _ = await prepare_round(manager, context, monkeypatch)
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
        response = await context.jobs[runtime.run_job](agent_name="monitor")
        assert response.success is will_retry
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
        response = await context.jobs[runtime.run_job](agent_name="monitor")
        assert not response.success and "External change" in record.read_text()
        assert await runtime.inbox.pending_ids() == {"event-1"}
