"""Real workspace persistence through AgentScope's shared streaming/collection path."""

# pylint: disable=protected-access

import asyncio
from types import SimpleNamespace
from uuid import uuid4

from agentscope.event import ModelCallEndEvent, ReplyEndEvent
from agentscope.message import AssistantMsg, Usage
from agentscope.permission import PermissionMode
import pytest

from reme.components.agent_wrapper.as_agent_wrapper import AsAgentWrapper
from reme.components.application_context import ApplicationContext
from reme.utils import global_counter_get_all


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("api", "outcome"),
    [(api, outcome) for api in ("reply", "stream") for outcome in ("complete", "error", "cancel")]
    + [("stream", "close")],
)
async def test_reply_persists_only_bound_workspace_on_all_exits(tmp_path, monkeypatch, api, outcome):
    """Public return modes save resumable beta state on completion, error, cancellation and close."""
    config = {"workspace_dir": {"alpha": str(tmp_path / "alpha"), "beta": str(tmp_path / "beta")}}
    context = ApplicationContext(**config)
    wrapper = AsAgentWrapper(name="research", app_context=context, as_llm="")
    session_id = str(uuid4())
    entered, closed = asyncio.Event(), asyncio.Event()
    saved = []
    dump_state = wrapper._dump_state

    async def persist(state, kwargs):
        saved.append(kwargs["workspace"])
        await dump_state(state, kwargs)

    async def build_agent(inputs, **kwargs):
        state = await wrapper._load_state(kwargs, PermissionMode.DEFAULT)
        state.summary = f"beta-{outcome}"
        message = AssistantMsg(
            name="agent",
            content="beta answer",
            finished_reason="completed",
            usage=Usage(input_tokens=3, output_tokens=2),
        )
        state.context.append(message)

        async def native_stream(_inputs, *, yield_final_msg):
            assert yield_final_msg
            try:
                yield ModelCallEndEvent(reply_id=message.id, input_tokens=3, output_tokens=2)
                entered.set()
                if outcome == "error":
                    raise RuntimeError("native stream failed")
                if outcome in {"cancel", "close"}:
                    await asyncio.Event().wait()
                yield ReplyEndEvent(session_id=session_id, reply_id=message.id, finished_reason="completed")
                yield message
            finally:
                closed.set()

        return SimpleNamespace(state=state, reply_stream=native_stream), inputs

    monkeypatch.setattr(wrapper, "_build_agent", build_agent)
    monkeypatch.setattr(wrapper, "_dump_state", persist)

    async def consume():
        kwargs = {"session_id": session_id, "workspace": "beta"}
        if api == "reply":
            return await wrapper.reply("hello", **kwargs)
        chunks = [chunk async for chunk in wrapper.reply_stream("hello", **kwargs)]
        return chunks[-1].chunk

    task = None
    try:
        if outcome == "close":
            stream = wrapper.reply_stream("hello", session_id=session_id, workspace="beta")
            await anext(stream)
            await stream.aclose()
        elif outcome == "cancel":
            task = asyncio.create_task(consume())
            await asyncio.wait_for(entered.wait(), 2)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(task, 2)
        elif outcome == "error":
            with pytest.raises(RuntimeError, match="native stream failed"):
                await consume()
        else:
            result = await consume()
            assert result["result"] == "beta answer"
            assert result["status"] == "completed"
            assert result["usage"]["total_tokens"] == 5

        assert closed.is_set()
        assert saved == ["beta"]
        beta_file = tmp_path / "beta" / "mem_session" / "agentscope" / f"{session_id}.jsonl"
        assert beta_file.is_file()
        assert not (tmp_path / "alpha" / "mem_session" / "agentscope" / f"{session_id}.jsonl").exists()
        # Recreate the application context to verify the durable result, not a live object cache.
        restarted = AsAgentWrapper(app_context=ApplicationContext(**config), as_llm="")
        state = await restarted._load_state({"resume": session_id, "workspace": "beta"}, PermissionMode.DEFAULT)
        assert state.summary == f"beta-{outcome}"
        with pytest.raises(FileNotFoundError):
            await restarted._load_state({"resume": session_id, "workspace": "alpha"}, PermissionMode.DEFAULT)
        counters = global_counter_get_all(context.metadata, ["__token_counter", "research"])
        assert counters["children"]["total_tokens"]["value"] == 5
    finally:
        if task is not None:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        await wrapper.close()
