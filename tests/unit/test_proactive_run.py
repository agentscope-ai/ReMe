"""Formal domain Jobs commit only validated, successful rounds."""

# pylint: disable=missing-function-docstring,protected-access,redefined-outer-name

import asyncio

import pytest

from reme.schema import StreamChunk
from reme.steps.proactive.task.base import BaseProactiveAgentStep
from reme.steps.proactive.task.initialize import initialize_task
from .proactive_helpers import running_system, queue_event

SKIP = '---\nreminder_priority: skip\nreminder_message: ""\n---\n'
REMIND = "---\nreminder_priority: {priority}\nreminder_message: Verified result\n---{body}"


async def prepare_round(manager, context, monkeypatch, name="monitor"):
    runtime = await manager.resolve(name)
    await initialize_task(runtime, context)
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
    repair=None,
):
    calls = []

    async def generate(query, **options):
        calls.append((query, options))
        assert any(spec.get("content") for spec in options["skills"])
        for spec in options["skills"]:
            if spec.get("content"):
                assert (options["_skills"].sources[spec["name"]] / "SKILL.md").is_file()
        current = text if len(calls) == 1 else repair
        if current is not None:
            runtime.files.write(record, current)
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


@pytest.mark.parametrize(
    "job,_backend",
    [
        ("run_memory_agent", "memory_agent_step"),
        ("run_notebook_agent", "notebook_agent_step"),
        ("run_finance_agent", "finance_agent_step"),
        ("run_news_agent", "news_agent_step"),
    ],
)
@pytest.mark.parametrize("language", ["en", "zh"])
@pytest.mark.asyncio
async def test_domain_jobs_share_executor_and_select_their_own_skills(tmp_path, monkeypatch, job, _backend, language):
    async with running_system(tmp_path, language=language, runtimes=[{"agent_name": "monitor", "run_job": job}]) as (
        manager,
        context,
        wrapper,
    ):
        runtime, record = await prepare_round(manager, context, monkeypatch)
        await queue_event(manager)
        calls = install_round(
            wrapper,
            runtime,
            record,
            monkeypatch,
            text=REMIND.format(priority="P1", body="\nEvidence\n"),
        )
        response = await context.jobs[job](agent_name="monitor")
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
        assert options["approval_mode"] == "deny_all" and "output_schema" not in options
        assert len(options["skills"]) == (3 if job == "run_notebook_agent" else 1)
        assert options["job_tools"] == (["manage_notebook"] if job == "run_notebook_agent" else [])
        if job == "run_notebook_agent":
            assert options["sandbox"] == "read-only" and options["injected_job_kwargs"]["_automatic_notebook"]
        assert not list(wrapper.project_skills_root.glob("proactive-record-*"))


@pytest.mark.parametrize(
    "text,result",
    [(None, "NO_KEY_INFORMATION"), (SKIP, "Finished"), (SKIP + "Normal progress", "Finished")],
)
@pytest.mark.asyncio
async def test_event_skip_commits_without_creating_a_notification(tmp_path, monkeypatch, text, result):
    async with running_system(tmp_path, runtimes=[{"agent_name": "monitor"}]) as (manager, context, wrapper):
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
    async with running_system(tmp_path, runtimes=[{"agent_name": "monitor"}]) as (manager, context, wrapper):
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
    ],
)
@pytest.mark.asyncio
async def test_one_repair_uses_same_session_and_preserves_body(tmp_path, monkeypatch, invalid):
    async with running_system(tmp_path, runtimes=[{"agent_name": "monitor"}]) as (manager, context, wrapper):
        runtime, record = await prepare_round(manager, context, monkeypatch)
        await queue_event(manager)
        repaired = REMIND.format(priority="P1", body="\nEvidence\n")
        calls = install_round(wrapper, runtime, record, monkeypatch, text=invalid, repair=repaired)
        response = await context.jobs[runtime.run_job](agent_name="monitor")
        assert response.success, response.answer
        assert len(calls) == 2 and calls[0][1] == calls[1][1] and len(wrapper.created) == 1
        assert record.read_text() == repaired


@pytest.mark.parametrize("phase", ["session", "inference"])
@pytest.mark.asyncio
async def test_timeout_releases_claim_and_scoped_skills(tmp_path, monkeypatch, phase):
    async with running_system(tmp_path, runtimes=[{"agent_name": "monitor"}]) as (manager, context, wrapper):
        runtime, _ = await prepare_round(manager, context, monkeypatch)
        await queue_event(manager)

        async def stall(**_):
            await asyncio.Event().wait()

        async def stream(*_, **__):
            await asyncio.Event().wait()
            yield

        if phase == "session":
            monkeypatch.setattr(wrapper, "create_session", stall)
        else:
            monkeypatch.setattr(wrapper, "reply_stream", stream)
        step = BaseProactiveAgentStep(app_context=context, timeout_seconds=0.03)
        with pytest.raises(TimeoutError):
            await step(agent_name="monitor")
        assert (
            await runtime.inbox.pending_ids() == {"event-1"} and not (await runtime.inbox.snapshot())["in_flight_count"]
        )
        assert not list(wrapper.project_skills_root.glob("proactive-record-*"))


@pytest.mark.asyncio
async def test_cancelled_round_does_not_wait_for_optional_native_statistics(tmp_path, monkeypatch):
    async with running_system(tmp_path, runtimes=[{"agent_name": "monitor"}]) as (manager, context, wrapper):
        runtime, _ = await prepare_round(manager, context, monkeypatch)
        await queue_event(manager)
        statistics_read = asyncio.Event()

        async def stream(*_, **__):
            await asyncio.Event().wait()
            yield

        async def statistics(*_):
            statistics_read.set()
            await asyncio.Event().wait()

        monkeypatch.setattr(wrapper, "reply_stream", stream)
        monkeypatch.setattr(wrapper, "get_tool_calls", statistics)
        step = BaseProactiveAgentStep(app_context=context, timeout_seconds=0.03)
        with pytest.raises(TimeoutError):
            await asyncio.wait_for(step(agent_name="monitor"), 0.5)
        assert not statistics_read.is_set()
        assert await runtime.inbox.pending_ids() == {"event-1"}
        assert not (await runtime.inbox.snapshot())["in_flight_count"]
        assert not list(wrapper.project_skills_root.glob("proactive-record-*"))


@pytest.mark.parametrize("will_retry", [False, True])
@pytest.mark.asyncio
async def test_stream_errors_only_allow_commit_when_explicitly_retryable(tmp_path, monkeypatch, will_retry):
    async with running_system(tmp_path, runtimes=[{"agent_name": "monitor"}]) as (manager, context, wrapper):
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
    async with running_system(tmp_path, runtimes=[{"agent_name": "monitor"}]) as (manager, context, wrapper):
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
