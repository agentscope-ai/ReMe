"""Automatic run statistics and failure backoff without external model calls."""

# pylint: disable=missing-function-docstring,protected-access

from datetime import timedelta
from unittest.mock import AsyncMock
from unittest.mock import Mock

import pytest

from reme.enumeration import ChunkEnum
from reme.schema import Event, Response, StreamChunk
from reme.steps.proactive.agent import ProactiveAgentStep
from reme.utils.counter import global_counter_get
from .test_proactive_runtime import create_agent, running_runtime
from .test_proactive_run import create_round_task, install_round, queue_cron_trigger


@pytest.mark.asyncio
async def test_statistics_separate_agent_types_and_count_snapshots_once(tmp_path, monkeypatch):
    async with running_runtime(tmp_path) as (runtime, context, wrapper):
        for mode in ("proactive", "cron"):
            config = {"agent_name": mode, "task_description": "Check results"}
            if mode == "cron":
                await runtime.create({**config, "cron": "0 * * * *"})
                await queue_cron_trigger(runtime, mode, monkeypatch)
            else:
                await runtime.create(config)
                await runtime.append_events(
                    mode,
                    [Event(id=mode, event_ts=runtime.now(), source="experiment", description="Done")],
                )
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
            assert stats["tool_calls"] == 3 and stats["web_lookup_calls"] == 2
            assert stats["usage"] == {"input_tokens": 15, "output_tokens": 7, "total_tokens": 22}
            assert stats["elapsed_seconds"] >= 0
        for mode in ("proactive", "cron"):
            assert global_counter_get(context.metadata, ["agent_runs", mode, "runs"]) == 1
            assert global_counter_get(context.metadata, ["agent_runs", mode, "tool_calls"]) == 3
            assert global_counter_get(context.metadata, ["agent_runs", mode, "web_lookup_calls"]) == 2
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
            monkeypatch.setattr(runtime.inbox, "ack", AsyncMock(side_effect=OSError("Commit failed")))
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
        assert stats["tool_calls"] == stats["web_lookup_calls"] == 0
        assert global_counter_get(context.metadata, ["agent_runs", "proactive", "failed"]) == 1


@pytest.mark.parametrize(
    "command,expected",
    [
        ('cd daily; echo "history"; rg -n "status" . | head -20', True),
        ("rg one daily; rg two daily", True),
        ("cat TASK.md\nrg status daily", True),
        ("/usr/bin/grep -n status daily/*.md", True),
        ("rg --files daily | head -10", False),
        ("rg --files daily | grep work", True),
        ('echo "rg status daily"', False),
        ("python -c 'print(\"rg status daily\")'", False),
        ("cat TASK.md; ls daily", False),
        ('cat "unterminated', False),
    ],
)
def test_explicit_shell_searches_include_compound_commands(command, expected):
    call = {"name": "exec_command", "arguments": {"cmd": command}, "actions": [{"type": "unknown"}]}
    assert (ProactiveAgentStep._lookup_type(call) == "local_search_calls") is expected


@pytest.mark.parametrize("agent_type", ["proactive", "cron"])
@pytest.mark.asyncio
async def test_local_search_budget_warns_once_without_interrupting(tmp_path, monkeypatch, agent_type):
    async with running_runtime(tmp_path) as (runtime, context, wrapper):
        await create_round_task(runtime, monkeypatch, agent_type)
        now = runtime.now()
        monkeypatch.setattr(runtime, "now", lambda: now)
        step = ProactiveAgentStep(app_context=context, max_local_search_calls=3)
        step.logger = Mock()
        paths = step._paths(await runtime.definition("monitor"), now)
        install_round(runtime, wrapper, monkeypatch, paths)
        original = wrapper.reply_stream

        async def stream(query, **options):
            for kind, count in (("webSearch", 4), ("commandExecution", 5)):
                for index in range(count):
                    # Two search actions in one shell invocation are still one call.
                    payload = {}
                    if kind == "commandExecution":
                        payload = (
                            {"commandActions": [{"type": "search"}, {"type": "search"}]}
                            if index % 2
                            else {
                                "commandActions": [{"type": "unknown"}],
                                "command": "cd daily; rg one . | head; rg two .",
                            }
                        )
                    for event in ("start", "end"):
                        yield StreamChunk(
                            chunk_type=ChunkEnum.TOOL_CALL if event == "start" else ChunkEnum.TOOL_RESULT,
                            event=event,
                            tool_call_id=f"{kind}-{index}",
                            tool_call_name=kind,
                            chunk=payload,
                        )
            for kind in ("imageView", "clock", "fileChange", "commandExecution"):
                yield StreamChunk(
                    chunk_type=ChunkEnum.TOOL_CALL,
                    tool_call_id=f"ordinary-{kind}",
                    tool_call_name=kind,
                    chunk={"commandActions": [{"type": "read", "path": str(paths["task_path"])}]},
                )
            async for chunk in original(query, **options):
                yield chunk

        monkeypatch.setattr(wrapper, "reply_stream", stream)
        response = await step(agent_name="monitor")
        assert response.success and paths["record_path"].exists()
        stats = response.metadata["reme.agent_run"]
        assert stats["tool_calls"] == 13
        assert stats["web_lookup_calls"] == 4 and stats["local_search_calls"] == 5
        assert stats["max_local_search_calls"] == 3
        warnings = [call.args[0] for call in step.logger.warning.call_args_list]
        assert len(warnings) == 1 and "local_search_calls" in warnings[0]


@pytest.mark.asyncio
async def test_saved_calls_fill_stream_gaps_without_double_counting(tmp_path, monkeypatch):
    async with running_runtime(tmp_path) as (runtime, context, wrapper):
        await create_agent(runtime)
        now = runtime.now()
        monkeypatch.setattr(runtime, "now", lambda: now)
        step = ProactiveAgentStep(app_context=context)
        paths = step._paths(await runtime.definition("monitor"), now)
        install_round(runtime, wrapper, monkeypatch, paths)
        original = wrapper.reply_stream

        async def stream(query, **options):
            yield StreamChunk(
                chunk_type=ChunkEnum.TOOL_CALL,
                tool_call_id="one",
                tool_call_name="webSearch",
                turn_id="turn-1",
            )
            async for chunk in original(query, **options):
                yield chunk

        monkeypatch.setattr(wrapper, "reply_stream", stream)
        saved = AsyncMock(
            return_value=[
                {"id": "one", "name": "web_search", "arguments": {}},
                {"id": "image", "name": "view_image", "arguments": {"path": "image.png"}},
                {"id": "clock", "name": "curr_time", "arguments": {}},
                {"id": "rejected-patch", "name": "apply_patch", "arguments": "invalid patch"},
                {"id": "history", "name": "exec_command", "arguments": {}, "actions": [{"type": "search"}]},
            ],
        )
        monkeypatch.setattr(wrapper, "get_tool_calls", saved)
        response = await step(agent_name="monitor")
        stats = response.metadata["reme.agent_run"]
        assert response.success and stats["tool_calls"] == 5
        assert stats["web_lookup_calls"] == stats["local_search_calls"] == 1
        assert stats["tool_call_source"] == "native"
        saved.assert_awaited_once_with("session-1", "turn-1")


@pytest.mark.parametrize("fragments", [False, True])
@pytest.mark.parametrize("tool_name", ["Bash", "commandExecution"])
@pytest.mark.asyncio
async def test_search_arguments_survive_stream_fragments_and_tool_results(tmp_path, monkeypatch, fragments, tool_name):
    async with running_runtime(tmp_path) as (runtime, context, wrapper):
        await create_agent(runtime)
        now = runtime.now()
        monkeypatch.setattr(runtime, "now", lambda: now)
        step = ProactiveAgentStep(app_context=context)
        paths = step._paths(await runtime.definition("monitor"), now)
        install_round(runtime, wrapper, monkeypatch, paths)
        original = wrapper.reply_stream

        async def stream(query, **options):
            yield StreamChunk(
                chunk_type=ChunkEnum.TOOL_CALL,
                event="start",
                tool_call_id="search-1",
                tool_call_name=tool_name,
                chunk=None if fragments else {"command": "rg error daily"},
            )
            if fragments:
                for fragment in ('{"command":', '"rg error daily"}'):
                    yield StreamChunk(
                        chunk_type=ChunkEnum.TOOL_CALL,
                        event="delta",
                        tool_call_id="search-1",
                        chunk=fragment,
                    )
            yield StreamChunk(
                chunk_type=ChunkEnum.TOOL_RESULT,
                event="delta",
                tool_call_id="search-1",
                chunk="ordinary output",
            )
            yield StreamChunk(
                chunk_type=ChunkEnum.TOOL_RESULT,
                event="end",
                tool_call_id="search-1",
                tool_call_name=tool_name,
                chunk={"output": "one match"},
            )
            async for chunk in original(query, **options):
                yield chunk

        monkeypatch.setattr(wrapper, "reply_stream", stream)
        stats = (await step(agent_name="monitor")).metadata["reme.agent_run"]
        assert stats["tool_calls"] == stats["local_search_calls"] == 1
        assert stats["web_lookup_calls"] == 0
