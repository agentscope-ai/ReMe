"""Automatic run statistics and failure backoff without external model calls."""

# pylint: disable=missing-function-docstring,protected-access

from datetime import timedelta
from unittest.mock import AsyncMock

import pytest

from reme.enumeration import ChunkEnum
from reme.schema import Response, StreamChunk
from reme.steps.proactive.agent import ProactiveAgentStep
from reme.utils.counter import global_counter_get
from .test_agent_runtime import create_agent, running_runtime
from .test_proactive_run import install_round


@pytest.mark.asyncio
async def test_statistics_separate_agent_types_and_count_snapshots_once(tmp_path, monkeypatch):
    async with running_runtime(tmp_path) as (runtime, context, wrapper):
        for mode in ("proactive", "cron"):
            await runtime.create({"agent_name": mode, "agent_type": "proactive", "task_description": "Check results"})
            await runtime.append(
                mode,
                {"id": mode, "event_ts": runtime.now(), "source": "experiment", "description": "Done"},
            )
            if mode == "cron":
                await runtime.update(mode, {"agent_type": "cron", "cron": "0 * * * *"})
            now = runtime.now()
            monkeypatch.setattr(runtime, "now", lambda current=now: current)
            step = ProactiveAgentStep(app_context=context)
            paths = step._paths(await runtime.definition(mode), now)
            install_round(runtime, wrapper, monkeypatch, paths)
            original = wrapper.reply_stream

            async def stream(query, _original=original, **options):
                for event in ("start", "delta", "end"):
                    yield StreamChunk(
                        chunk_type=ChunkEnum.TOOL_CALL,
                        event=event,
                        turn_id="turn-1",
                        tool_call_id="lookup-1",
                        tool_call_name="webSearch",
                    )
                yield StreamChunk(chunk_type=ChunkEnum.TOOL_RESULT, tool_call_id="lookup-1", event="end")
                yield StreamChunk(chunk_type=ChunkEnum.TOOL_CALL, tool_call_id="shell-1", tool_call_name="exec_command")
                yield StreamChunk(
                    chunk_type=ChunkEnum.TOOL_RESULT,
                    tool_call_id="lookup-2",
                    tool_call_name="webSearch",
                    event="end",
                )
                for tokens in (10, 15):
                    yield StreamChunk(chunk_type=ChunkEnum.USAGE, chunk={"input_tokens": tokens, "output_tokens": 7})
                async for chunk in _original(query, **options):
                    yield chunk

            monkeypatch.setattr(wrapper, "reply_stream", stream)
            response = await step(agent_name=mode)
            stats = response.metadata["reme.agent_run"]
            assert stats["agent_type"] == mode and stats["result"] == "completed"
            assert stats["model_status"] == "completed" and stats["session_id"]
            assert stats["tool_calls"] == 3 and stats["lookup_calls"] == 2
            assert stats["usage"] == {"input_tokens": 15, "output_tokens": 7, "total_tokens": 22}
            assert stats["elapsed_seconds"] >= 0
        for mode in ("proactive", "cron"):
            assert global_counter_get(context.metadata, ["agent_runs", mode, "runs"]) == 1
            assert global_counter_get(context.metadata, ["agent_runs", mode, "tool_calls"]) == 3
            assert global_counter_get(context.metadata, ["agent_runs", mode, "lookup_calls"]) == 2
            assert global_counter_get(context.metadata, ["agent_runs", mode, "total_tokens"]) == 22


@pytest.mark.parametrize("failure", ["model", "commit"])
@pytest.mark.asyncio
async def test_run_result_is_distinct_from_model_status(tmp_path, monkeypatch, failure):
    async with running_runtime(tmp_path) as (runtime, context, wrapper):
        await create_agent(runtime)
        now = runtime.now()
        monkeypatch.setattr(runtime, "now", lambda: now)
        step = ProactiveAgentStep(app_context=context)
        paths = step._paths(await runtime.definition("monitor"), now)
        install_round(runtime, wrapper, monkeypatch, paths, failed=failure == "model")
        if failure == "commit":
            monkeypatch.setattr(runtime.outbox, "save", AsyncMock(side_effect=OSError("Commit failed")))
            with pytest.raises(OSError, match="Commit failed"):
                await step(agent_name="monitor")
        else:
            assert (await step(agent_name="monitor")).success
        stats = step.context.response.metadata["reme.agent_run"]
        assert stats["result"] == ("failed" if failure == "commit" else "completed")
        assert stats["model_status"] == ("completed" if failure == "commit" else "failed")
        assert stats["usage"] is None
        assert global_counter_get(context.metadata, ["agent_runs", "proactive", stats["result"]]) == 1
        assert global_counter_get(context.metadata, ["agent_runs", "proactive", "usage_runs"]) == 0


@pytest.mark.parametrize(
    "retry_seconds,delays",
    [(5, [5, 10, 20, 40, 80, 160, 300, 300]), (200, [200, 300]), (500, [500, 500]), (0, [0, 0])],
)
@pytest.mark.asyncio
async def test_failure_backoff_is_capped_and_success_resets_it(tmp_path, monkeypatch, retry_seconds, delays):
    async with running_runtime(tmp_path) as (runtime, _, _):
        await create_agent(runtime)
        runtime.retry_seconds = retry_seconds
        now = runtime.now()
        monkeypatch.setattr(runtime, "now", lambda: now)
        job = AsyncMock(return_value=Response(success=False, answer="Provider unavailable"))
        execution = runtime.executions["monitor"]
        for delay in delays:
            await runtime._work("monitor", job)
            assert execution.retry_at == now + timedelta(seconds=delay)
        job.return_value = Response(success=True)
        await runtime._work("monitor", job)
        assert execution.retry_delay == 0 and execution.retry_at is None and execution.error is None
        job.return_value = Response(success=False, answer="Failed again")
        await runtime._work("monitor", job)
        assert execution.retry_at == now + timedelta(seconds=retry_seconds)
        assert (await runtime.status("monitor"))["pending_events"] == 1


@pytest.mark.asyncio
async def test_failure_before_model_start_is_counted_without_hiding_the_error(tmp_path):
    async with running_runtime(tmp_path) as (runtime, context, _):
        await create_agent(runtime)
        agent = await runtime.definition("monitor")
        runtime.files.write(runtime.agents.task_path(agent), "\n")
        step = ProactiveAgentStep(app_context=context)
        with pytest.raises(ValueError, match="TASK.md must not be blank"):
            await step(agent_name="monitor")
        stats = step.context.response.metadata["reme.agent_run"]
        assert stats["result"] == "failed" and stats["model_status"] == "not_started"
        assert stats["session_id"] is None and stats["usage"] is None
        assert stats["tool_calls"] == stats["lookup_calls"] == 0
        assert global_counter_get(context.metadata, ["agent_runs", "proactive", "failed"]) == 1
