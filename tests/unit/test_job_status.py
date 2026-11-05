"""Execution status across job entry points, nested steps, retries, and shutdown."""

# pylint: disable=protected-access

import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx
import pytest
import yaml
from pydantic import ValidationError

from reme.application import Application
from reme.components import ApplicationContext
from reme.components.agent_wrapper import BaseAgentWrapper
from reme.components.job import BackgroundJob, BaseJob, CronJob, StreamJob
from reme.components.runtime_context import RuntimeContext
from reme.components.service import HttpService
from reme.enumeration import ChunkEnum
from reme.schema import ApplicationConfig, JobConfig, Response, StreamChunk
from reme.steps.base_step import BaseStep
from reme.utils.evaluation_interface import check_job_count


class _CallbackStep(BaseStep):
    """Run a test callback through the actual Step invocation boundary."""

    async def execute(self):
        return await self.kwargs["callback"](self)


@pytest.fixture(name="context")
def _context(tmp_path):
    """Use real application state without starting services or workspace components."""
    owner = ApplicationContext(workspace_dir=str(tmp_path))
    owner.registry.register(_CallbackStep, "callback_step")
    return owner


def _job(context, cls=BaseJob, name="test", **kwargs):
    """Register a job and let normal startup resolve any configured steps."""
    job = cls(name=name, app_context=context, **kwargs)
    context.jobs[name] = job
    return job


@pytest.mark.parametrize("cls,default", [(BaseJob, False), (StreamJob, False), (BackgroundJob, True), (CronJob, True)])
@pytest.mark.parametrize("option", [None, True, False])
def test_tracking_defaults_and_explicit_overrides(context, cls, default, option):
    """Typed defaults select by backend, and explicit switches never leak to job inputs."""
    config = JobConfig(track_execution=option)
    kwargs = {"cron": "* * * * *"} if cls is CronJob else {}
    job = _job(context, cls, **kwargs, **config.model_dump(exclude={"backend"}))
    assert job.track_execution is (default if option is None else option)
    assert "track_execution" not in job.kwargs


@pytest.mark.asyncio
async def test_foreground_tracking_is_opt_in_and_application_scoped(context):
    """Untracked calls leave no history, while tracked calls remain application-local."""
    job = _job(context)
    await job()
    assert context.get_job_status() == {}
    job.track_execution = True
    app = object.__new__(Application)
    app.context = context
    assert (await app.run_job("test")).success
    record = context.get_job_status()["test"][0]
    assert record["status"] == "succeeded"
    assert record["finished_at"] >= record["started_at"]
    assert not ApplicationContext().get_job_status()
    assert check_job_count("test", context) == 2


@pytest.mark.asyncio
async def test_nested_steps_report_live_path_and_restore_parent(context):
    """A watcher exposes its handler while active and then returns to its waiting step."""
    observations = []

    async def child(step):
        observations.append(context.get_job_status()["test"][0])
        assert step.context.data == {"request": "value"}

    async def parent(step):
        before = context.get_job_status()["test"][0]
        await step.dispatch_steps([{"backend": "callback_step", "callback": child}])
        after = context.get_job_status()["test"][0]
        assert before["step_path"] == after["step_path"] == ["callback_step"]
        assert before["step_started_at"] == after["step_started_at"]

    job = _job(context, track_execution=True, steps=[{"backend": "callback_step", "callback": parent}])
    await job.start()
    await job(request="value")
    assert observations[0]["step_path"] == ["callback_step", "callback_step"]
    assert observations[0]["status"] == "running"
    record = context.get_job_status()["test"][0]
    assert record["step_path"] == []
    assert record["step_started_at"] is None
    assert record["failed_step"] is None
    # Snapshots remain detached after the underlying execution has ended.
    assert observations[0]["finished_at"] is None


@pytest.mark.asyncio
@pytest.mark.parametrize("raise_error", [True, False])
async def test_failure_retains_innermost_step_for_exceptions_and_failed_responses(context, raise_error):
    """Both error channels identify the child, without replacing it with the parent path."""

    async def child(step):
        if raise_error:
            raise ValueError("failed child")
        step.context.response.success = False
        step.context.response.answer = "failed child"

    async def parent(step):
        await step.dispatch_steps([{"backend": "callback_step", "callback": child}])

    job = _job(context, track_execution=True, steps=[{"backend": "callback_step", "callback": parent}])
    await job.start()
    response = await job()
    assert not response.success
    record = context.get_job_status()["test"][0]
    assert record["status"] == "failed"
    assert record["failed_step"] == ["callback_step", "callback_step"]
    assert "failed child" in record["error"]
    assert record["step_path"] == []


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", ["recovered", "parent_failed", "sibling_failed", "wrapped_child_failed"])
async def test_recovered_step_errors_do_not_hide_the_actual_terminal_failure(context, outcome):
    """Caught child errors cannot leave a stale failure path on success or a later failure."""

    async def fail(_step):
        raise ValueError("child failed")

    context.registry.register(_CallbackStep, "first_step")
    context.registry.register(_CallbackStep, "second_step")

    async def parent(step):
        try:
            await step.dispatch_steps([{"backend": "first_step", "callback": fail}])
        except ValueError as exc:
            if outcome == "wrapped_child_failed":
                raise RuntimeError("parent wrapped the child error") from exc
        if outcome == "parent_failed":
            raise RuntimeError("parent failed independently")
        if outcome == "sibling_failed":
            await step.dispatch_steps([{"backend": "second_step", "callback": fail}])

    job = _job(context, track_execution=True, steps=[{"backend": "callback_step", "callback": parent}])
    await job.start()
    response = await job()
    record = context.get_job_status()["test"][0]
    expected = {
        "recovered": None,
        "parent_failed": ["callback_step"],
        "sibling_failed": ["callback_step", "second_step"],
        "wrapped_child_failed": ["callback_step", "first_step"],
    }
    assert response.success is (outcome == "recovered")
    assert record["failed_step"] == expected[outcome]


@pytest.mark.asyncio
async def test_stream_failure_records_status_and_preserves_error_done_protocol(context):
    """ERROR plus DONE remains the client contract even when the exception is handled."""

    async def fail(_step):
        raise RuntimeError("stream failed")

    job = _job(context, StreamJob, track_execution=True, steps=[{"backend": "callback_step", "callback": fail}])
    await job.start()
    app = object.__new__(Application)
    app.context = context
    chunks = [chunk async for chunk in app.run_stream_job("test")]
    assert [chunk.chunk_type for chunk in chunks] == [ChunkEnum.ERROR, ChunkEnum.DONE]
    assert chunks[-1].done
    record = context.get_job_status()["test"][0]
    assert record["status"] == "failed"
    assert record["error"] == "RuntimeError: stream failed"
    assert record["failed_step"] == ["callback_step"]


@pytest.mark.asyncio
@pytest.mark.parametrize("will_retry", [True, False])
async def test_chat_error_chunks_match_execution_status_without_changing_stream(context, will_retry):
    """A terminal ERROR is a failed run even when ChatStep returns normally."""

    class ErrorAgent(BaseAgentWrapper):
        """Model terminal and recoverable provider notifications without an external service."""

        async def reply(self, inputs, **kwargs):
            raise AssertionError("stream only")

        async def reply_stream(self, inputs, **kwargs):
            yield StreamChunk(chunk_type=ChunkEnum.ERROR, chunk="provider failed", metadata={"will_retry": will_retry})
            yield StreamChunk(chunk_type=ChunkEnum.CONTENT, chunk="partial or recovered")
            yield StreamChunk(chunk_type=ChunkEnum.REPLY_END)

    context.components["agent_wrapper"] = {"default": ErrorAgent()}
    job = _job(context, StreamJob, track_execution=True, steps=[{"backend": "chat_step"}])
    await job.start()
    queue = asyncio.Queue()
    await job(query="hello", stream_queue=queue)
    chunks = [queue.get_nowait() for _ in range(queue.qsize())]
    assert [chunk.chunk_type for chunk in chunks] == [
        ChunkEnum.ERROR,
        ChunkEnum.CONTENT,
        ChunkEnum.REPLY_END,
        ChunkEnum.DONE,
    ]
    record = context.get_job_status()["test"][0]
    assert record["status"] == ("succeeded" if will_retry else "failed")
    assert record["error"] == (None if will_retry else "provider failed")
    assert record["failed_step"] == (None if will_retry else ["chat_step"])


@pytest.mark.asyncio
async def test_late_child_step_exit_cannot_reopen_a_completed_record(context):
    """Cleanup from a child task cannot restore a step path after the Job has ended."""
    entered, release = asyncio.Event(), asyncio.Event()
    tasks = []

    async def child(_step):
        entered.set()
        await release.wait()
        raise RuntimeError("child finished after its job")

    async def parent(step):
        tasks.append(
            asyncio.create_task(
                _CallbackStep(app_context=context, backend="child_step", callback=child)(step.context),
            ),
        )
        await entered.wait()

    job = _job(context, track_execution=True, steps=[{"backend": "callback_step", "callback": parent}])
    await job.start()
    try:
        assert (await job()).success
        completed = context.get_job_status()["test"][0]
        release.set()
        await asyncio.gather(*tasks, return_exceptions=True)
        assert context.get_job_status()["test"][0] == completed
    finally:
        release.set()
        await asyncio.gather(*tasks, return_exceptions=True)


@pytest.mark.asyncio
async def test_same_name_concurrent_runs_and_cancellation_are_isolated(context):
    """Cancelling one invocation leaves its sibling's live step and result intact."""

    async def block(step):
        step.context["entered"].set()
        await step.context["release"].wait()

    job = _job(context, track_execution=True, steps=[{"backend": "callback_step", "callback": block}])
    await job.start()
    entered = [asyncio.Event(), asyncio.Event()]
    release = asyncio.Event()
    tasks = [asyncio.create_task(job(entered=event, release=release)) for event in entered]
    try:
        await asyncio.wait_for(asyncio.gather(*(event.wait() for event in entered)), 2)
        records = context.get_job_status()["test"]
        assert len({record["id"] for record in records}) == 2
        assert all(record["step_path"] == ["callback_step"] for record in records)
        tasks[0].cancel()
        with pytest.raises(asyncio.CancelledError):
            await tasks[0]
        records = context.get_job_status()["test"]
        assert [record["status"] for record in records] == ["running", "cancelled"]
        release.set()
        await tasks[1]
        assert [record["status"] for record in context.get_job_status()["test"]] == ["succeeded", "cancelled"]
    finally:
        release.set()
        await asyncio.gather(*tasks, return_exceptions=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("child_tracked", [True, False])
async def test_nested_job_has_its_own_tracking_switch(context, child_tracked):
    """An untracked child cannot overwrite its tracked caller's current step."""

    async def child(step):
        assert (step.context.job_execution is not None) is child_tracked
        assert context.get_job_status("parent")["parent"][0]["step_path"] == ["callback_step"]

    child_job = _job(
        context,
        name="child",
        track_execution=child_tracked,
        steps=[{"backend": "callback_step", "callback": child}],
    )
    await child_job.start()

    async def parent(step):
        await step.run_job("child")

    parent_job = _job(
        context,
        name="parent",
        track_execution=True,
        steps=[{"backend": "callback_step", "callback": parent}],
    )
    await parent_job.start()
    assert (await parent_job()).success
    assert bool(context.get_job_status("child")["child"]) is child_tracked


@pytest.mark.asyncio
async def test_background_restarts_have_separate_records_and_graceful_stop(context):
    """Supervisor attempts are recorded once each, and cooperative shutdown is stopped."""
    attempts = 0
    entered = asyncio.Event()

    async def body(step):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise RuntimeError("restart me")
        entered.set()
        await step.context.stop_event.wait()

    job = _job(context, BackgroundJob, backoff_base=0, steps=[{"backend": "callback_step", "callback": body}])
    await job.start()
    try:
        await asyncio.wait_for(entered.wait(), 2)
        assert [record["status"] for record in context.get_job_status()["test"]] == ["running", "failed"]
    finally:
        await job.close()
    assert [record["status"] for record in context.get_job_status()["test"]] == ["stopped", "failed"]
    assert check_job_count("test", context) == 2


@pytest.mark.asyncio
async def test_custom_background_body_is_tracked_by_supervisor(context):
    """Overriding __call__ still gets lifecycle tracking and a step-aware runtime context."""

    async def body(step):
        assert step.context.job_execution is not None
        assert context.get_job_status()["test"][0]["step_path"] == ["custom_step"]
        step.context.response.success = False
        step.context.response.answer = "custom failure"

    class CustomJob(BackgroundJob):
        """Simulate an extension providing its own job body."""

        async def __call__(self, **kwargs):
            runtime = RuntimeContext()
            await _CallbackStep(app_context=context, backend="custom_step", callback=body)(runtime)
            return runtime.response

    job = _job(context, CustomJob)
    await job.start()
    try:
        await asyncio.wait_for(job._task, 2)
        record = context.get_job_status()["test"][0]
        assert record["status"] == "failed"
        assert record["error"] == "custom failure"
        assert record["failed_step"] == ["custom_step"]
        assert check_job_count("test", context) == 1
    finally:
        await job.close()


@pytest.mark.asyncio
async def test_custom_background_super_call_preserves_runtime_overrides(context):
    """The supervisor's wrapper must not replace arguments supplied by an overridden body."""

    async def body(step):
        assert step.context["value"] == "from-body"

    class CustomJob(BackgroundJob):
        """An extension can still specialize inputs before invoking the configured steps."""

        async def __call__(self, **kwargs):
            return await super().__call__(value="from-body")

    job = _job(context, CustomJob, value="from-config", steps=[{"backend": "callback_step", "callback": body}])
    await job.start()
    try:
        await asyncio.wait_for(job._task, 2)
        records = context.get_job_status()["test"]
        assert len(records) == 1
        assert records[0]["status"] == "succeeded"
        assert check_job_count("test", context) == 1
    finally:
        await job.close()


@pytest.mark.asyncio
async def test_cron_records_triggers_not_scheduler_and_continues_after_failure(context):
    """Waiting creates no run, while each trigger gets a terminal result and fresh steps."""
    triggers = 0
    waits = 0

    async def body(_step):
        nonlocal triggers
        triggers += 1
        if triggers == 1:
            raise ValueError("first trigger failed")

    job = _job(context, CronJob, cron="* * * * *", steps=[{"backend": "callback_step", "callback": body}])

    async def wait(_delay):
        nonlocal waits
        waits += 1
        if waits == 1:
            assert context.get_job_status() == {"test": []}
        elif waits == 3:
            job._stop_event.set()

    job._wait_or_stop = wait
    job._next_fire_delay = lambda: 0
    await job.start()
    try:
        await asyncio.wait_for(job._task, 2)
        records = context.get_job_status()["test"]
        assert [record["status"] for record in records] == ["succeeded", "failed"]
        assert records[1]["failed_step"] == ["callback_step"]
        assert check_job_count("test", context) == 2
    finally:
        await job.close()


@pytest.mark.asyncio
async def test_worker_future_cancellation_does_not_claim_thread_body_has_stopped(context):
    """A timed-out executor Future cannot publish completion before its worker actually exits."""
    entered, release = threading.Event(), threading.Event()

    class ThreadJob(BackgroundJob):
        """Keep the worker alive after the outer Future has been cancelled."""

        async def __call__(self, **kwargs):
            entered.set()
            if not release.wait(timeout=3):
                raise TimeoutError("worker was not released")
            return Response()

    context.thread_pool = ThreadPoolExecutor(max_workers=1)
    job = _job(context, ThreadJob, use_thread_pool=True, close_timeout=0.01)
    try:
        await job.start()
        assert await asyncio.to_thread(entered.wait, 2)
        await job.close()
        assert context.get_job_status()["test"][0]["status"] == "running"
    finally:
        release.set()
        context.thread_pool.shutdown(wait=True)
    assert context.get_job_status()["test"][0]["status"] == "stopped"


@pytest.mark.asyncio
@pytest.mark.parametrize("limit", [0, 2])
async def test_retention_preserves_running_records_and_snapshots_are_detached(context, limit):
    """Only completed records count towards the cap, and clients cannot mutate stored paths."""
    context.app_config.job_history_limit = limit
    active = context.start_job_execution("test")
    context.enter_job_step(active, "waiting")
    job = _job(context, track_execution=True)
    for _ in range(4):
        await job()
    records = context.get_job_status(limit=20)["test"]
    assert len(records) == limit + 1
    assert records[-1]["status"] == "running"
    records[-1]["step_path"].append("tampered")
    assert context.get_job_status(status="running")["test"][0]["step_path"] == ["waiting"]
    context.finish_job_execution("test", active, "cancelled")
    assert len(context.get_job_status()["test"]) == limit
    assert check_job_count("test", context) == 5


@pytest.mark.parametrize("tracked", [True, False])
def test_concurrent_starts_count_every_call_independently_of_retained_history(context, tracked):
    """Worker threads share one atomic count even with optional and bounded execution records."""
    context.app_config.job_history_limit = 2
    job = _job(context, track_execution=tracked)
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(lambda _: asyncio.run(job()), range(32)))
    assert check_job_count("test", context) == 32
    assert len(context.get_job_status("test")["test"]) == (2 if tracked else 0)


@pytest.mark.asyncio
async def test_default_query_job_works_through_http_and_step_internal_calls(context):
    """The default config/registration produce a structured foreground query without self-history."""
    config_path = Path(__file__).parents[2] / "reme" / "config" / "default.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))["jobs"]["job_status"]
    query = _job(context, name="job_status", **config)
    await query.start()
    work = _job(context, name="work", track_execution=True)
    await work()
    app = object.__new__(Application)
    app.context = context
    service = HttpService(web_enabled=False, mcp_enabled=False)
    service.build_service(app)
    service.add_jobs(app)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=service.service), base_url="http://test") as client:
        response = await client.post("/job_status", json={"name": "work", "status": "succeeded", "limit": 1})
    assert response.status_code == 200
    assert response.json()["answer"]["work"][0]["status"] == "succeeded"
    caller = _CallbackStep(app_context=context, callback=None)
    assert (await caller.run_job("job_status")).success
    assert "job_status" not in context.get_job_status()


@pytest.mark.parametrize("kwargs", [{"limit": 0}, {"limit": True}, {"limit": "1"}, {"status": "unknown"}])
def test_query_rejects_invalid_filters(context, kwargs):
    """Direct callers receive the same validation as the service's documented schema."""
    with pytest.raises(ValueError):
        context.get_job_status(**kwargs)
    with pytest.raises(KeyError, match="not found"):
        context.get_job_status(name="missing")


def test_retention_limit_is_validated_by_application_config():
    """A negative retention limit cannot silently slice histories incorrectly."""
    with pytest.raises(ValidationError):
        ApplicationConfig(job_history_limit=-1)
