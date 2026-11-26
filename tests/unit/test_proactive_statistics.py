"""Round telemetry counts unique native calls and preserves execution outcomes."""

# pylint: disable=missing-function-docstring,protected-access,redefined-outer-name

from datetime import timedelta

import pytest

from reme.schema import StreamChunk
from reme.utils.counter import global_counter_get
from .proactive_helpers import running_system, queue_event
from .test_proactive_run import prepare_round, install_round, SKIP


@pytest.mark.parametrize("status", ["completed", "failed"])
@pytest.mark.asyncio
async def test_round_statistics_and_counters_reflect_model_and_commit_outcomes(tmp_path, monkeypatch, status):
    async with running_system(tmp_path, runtimes=[{"agent_name": "monitor"}]) as (manager, context, wrapper):
        runtime, record = await prepare_round(manager, context, monkeypatch)
        await queue_event(manager)
        install_round(wrapper, runtime, record, monkeypatch, text=SKIP, status=status)
        original = wrapper.reply_stream

        async def stream(query, **options):
            for event in ("start", "delta", "end"):
                yield StreamChunk(
                    chunk_type="tool_call",
                    tool_call_id="lookup",
                    tool_call_name="webSearch",
                    event=event,
                )
            for tokens in (10, 15):
                yield StreamChunk(chunk_type="usage", chunk={"input_tokens": tokens, "output_tokens": 7})
            async for chunk in original(query, **options):
                yield chunk

        monkeypatch.setattr(wrapper, "reply_stream", stream)
        response = await context.jobs[runtime.run_job](agent_name="monitor")
        stats = response.metadata["reme.agent_run"]
        assert (
            response.success == (status == "completed")
            and stats["result"] == status
            and stats["model_status"] == status
        )
        assert stats["tool_calls"] == stats["web_lookup_calls"] == 1 and stats["usage"] == {
            "input_tokens": 15,
            "output_tokens": 7,
            "total_tokens": 22,
        }
        assert global_counter_get(context.metadata, ["agent_runs", "event", "runs"]) == 1
        assert global_counter_get(context.metadata, ["agent_runs", "event", status]) == 1
        assert global_counter_get(context.metadata, ["agent_runs", "event", "total_tokens"]) == 22


@pytest.mark.asyncio
async def test_exponential_backoff_is_capped_and_success_resets(tmp_path, monkeypatch):
    async with running_system(tmp_path, runtimes=[{"agent_name": "monitor"}]) as (manager, _, _):
        runtime = await manager.resolve("monitor")
        now = runtime.now()
        monkeypatch.setattr(runtime, "now", lambda: now)
        for seconds in [5, 10, 20, 40, 80, 160, 300, 300]:
            runtime.failed_round(OSError("Unavailable"))
            assert runtime.retry_at == now + timedelta(seconds=seconds)
        runtime.completed_round()
        assert runtime.retry_delay == 0 and runtime.retry_at is None and runtime.error is None
