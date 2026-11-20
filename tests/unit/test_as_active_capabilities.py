"""Production AgentScope forks and append queues with the real SDK and offline models."""

# pylint: disable=missing-function-docstring,protected-access

import asyncio
from types import SimpleNamespace

import pytest
from agentscope.message import AssistantMsg, TextBlock, ToolCallBlock
from agentscope.model import ChatResponse, ChatUsage
from agentscope.permission import PermissionBehavior, PermissionDecision
from agentscope.state import AgentState
from agentscope.tool import FunctionTool, ToolChunk, Toolkit

from reme.components.agent_wrapper import AsAgentWrapper
from reme.core.context import ApplicationContext
from reme.components.agent_wrapper.as_state_io import AsStateHandler
from .test_as_agent_sessions import OfflineModel


def make_wrapper(root, model=None, context=None, **kwargs):
    wrapper = AsAgentWrapper(
        name="probe",
        as_llm="",
        session_retention_days=0,
        app_context=context or ApplicationContext(workspace_dir=str(root)),
        **kwargs,
    )
    wrapper.as_llm = SimpleNamespace(model=model or OfflineModel())
    return wrapper


@pytest.mark.asyncio
async def test_shared_read_only_hint_excludes_write_and_exec_even_with_a_custom_toolkit(tmp_path):
    source = make_wrapper(tmp_path)
    custom = Toolkit(tools=source._builtin_tools(["write"]))
    agent, _ = await source._build_agent(
        None,
        state=AgentState(),
        toolkit=custom,
        use_builtin_tools=True,
        sandbox="read-only",
    )
    names = {schema["function"]["name"] for schema in await agent.toolkit.get_tool_schemas()}
    assert {tool.name for tool in source._builtin_tools(["read", "glob", "grep"])}.issubset(names)
    assert not {tool.name for tool in source._builtin_tools(["write", "edit", "bash"])} & names


@pytest.mark.asyncio
async def test_creation_rejects_an_existing_target_and_fork_requires_a_source(tmp_path):
    source = make_wrapper(tmp_path)
    session_id = await source.create_session()
    path = await source.get_session_path(session_id)
    original = path.read_bytes()
    with pytest.raises(FileExistsError):
        await source.create_session(session_id=session_id)
    with pytest.raises(ValueError, match="requires resume"):
        await source.create_session(fork_session=True)
    assert path.read_bytes() == original and not source.as_llm.model.calls


@pytest.mark.asyncio
async def test_live_fork_is_detached_and_does_not_wait_for_or_mutate_the_source(tmp_path):
    model = OfflineModel()
    model.paused = True
    source = make_wrapper(tmp_path, model)
    source_id = await source.create_session()
    running = asyncio.create_task(source.reply("Source question", resume=source_id))
    try:
        await asyncio.wait_for(model.entered.wait(), 2)
        child_id = await asyncio.wait_for(source.create_session(resume=source_id, fork_session=True), 1)
        child = make_wrapper(tmp_path, context=source.app_context)
        result = await child.reply("Child question", resume=child_id)
        assert result["result"] == "final answer" and not running.done()
        assert child_id != source_id
        assert "Child question" not in (await source.read_session(source_id)).model_dump_json()
        detached = (await child.read_session(child_id)).model_dump_json()
        assert "Source question" in detached and "draft" in detached
        model.release.set()
        assert (await running)["result"] == "final answer"
        assert (await child.read_session(child_id)).model_dump_json() == detached
    finally:
        model.release.set()
        await asyncio.gather(running, return_exceptions=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("role", ["user", "assistant"])
@pytest.mark.parametrize("outcome", ["timeout", "cancelled"])
async def test_active_append_keeps_one_receipt_across_timeout_cancellation_and_retry(tmp_path, role, outcome):
    model = OfflineModel()
    model.paused = True
    source = make_wrapper(tmp_path, model)
    session_id = await source.create_session()
    running = asyncio.create_task(source.reply("Source question", resume=session_id))
    observers = []
    try:
        await asyncio.wait_for(model.entered.wait(), 2)
        first = asyncio.create_task(
            source.append_session(
                session_id,
                "Observation",
                message_id="observation",
                role=role,
                delivery="immediate",
                timeout=0.02 if outcome == "timeout" else 30,
            ),
        )
        observers.append(first)
        queue = source._sessions._runtimes[session_id].appends
        async with asyncio.timeout(2):
            while "reme-proactive-observation" not in queue.pending:
                await asyncio.sleep(0)
        if outcome == "cancelled":
            first.cancel()
        with pytest.raises(TimeoutError if outcome == "timeout" else asyncio.CancelledError):
            await first
        assert "Observation" not in (await source._sessions.read(session_id)).state.model_dump_json()
        with pytest.raises(ValueError, match="different content"):
            await source.append_session(
                session_id,
                "Different",
                message_id="observation",
                role=role,
                delivery="immediate",
            )
        retry = asyncio.create_task(
            source.append_session(
                session_id,
                "Observation",
                message_id="observation",
                role=role,
                delivery="immediate",
                expected_revision="stale",
            ),
        )
        observers.append(retry)
        await asyncio.sleep(0)
        assert not retry.done()
        model.release.set()
        result, receipt = await asyncio.gather(running, retry)
        assert result["result"] == "final answer" and len(model.calls) == 1
        persisted = await source.read_session(session_id)
        assert next(message for message in persisted.messages if message.id == receipt.id) == receipt
        assert sum(message.id == receipt.id for message in persisted.messages) == 1
        assert (
            await source.append_session(
                session_id,
                "Observation",
                message_id="observation",
                role=role,
                expected_revision="stale",
            )
            == receipt
        )
        assert not source._sessions._runtimes
    finally:
        model.release.set()
        await asyncio.gather(running, *observers, return_exceptions=True)


@pytest.mark.asyncio
async def test_retry_waits_for_commit_and_finalization_does_not_lose_new_arrivals(tmp_path, monkeypatch):
    model = OfflineModel()
    model.paused = True
    source = make_wrapper(tmp_path, model)
    session_id = await source.create_session()
    running = asyncio.create_task(source.reply("Source question", resume=session_id))
    writing, commit = asyncio.Event(), asyncio.Event()
    tasks = []
    try:
        await asyncio.wait_for(model.entered.wait(), 2)
        save = source._sessions.save

        async def paused_save(session):
            if any(message.id == "reme-proactive-first" for message in session.messages) and not writing.is_set():
                writing.set()
                await commit.wait()
            return await save(session)

        monkeypatch.setattr(source._sessions, "save", paused_save)
        tasks.append(
            asyncio.create_task(source.append_session(session_id, "First", message_id="first", delivery="immediate")),
        )
        await asyncio.sleep(0)
        model.release.set()
        await asyncio.wait_for(writing.wait(), 2)
        tasks.append(
            asyncio.create_task(source.append_session(session_id, "First", message_id="first", delivery="immediate")),
        )
        tasks.append(
            asyncio.create_task(
                source.append_session(session_id, "Later", message_id="later", role="user", delivery="immediate"),
            ),
        )
        await asyncio.sleep(0)
        assert not any(task.done() for task in tasks)
        commit.set()
        await asyncio.wait_for(asyncio.gather(running, *tasks), 2)
        persisted = await source.read_session(session_id)
        assert sum(message.id == "reme-proactive-first" for message in persisted.messages) == 1
        assert sum(message.id == "reme-proactive-later" for message in persisted.messages) == 1
        assert len(model.calls) == 1 and not source._sessions._runtimes
    finally:
        model.release.set()
        commit.set()
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(running, *tasks, return_exceptions=True)


class CompressionModel(OfflineModel):
    """Pause native compression without changing its context replacement behavior."""

    context_size = 1000

    async def count_tokens(self, *_args, **_kwargs):
        return 900

    async def generate_structured_output(self, **_kwargs):
        self.entered.set()
        await self.release.wait()
        return SimpleNamespace(
            content={"summary": "Earlier context"},
            finished_reason="completed",
            usage=ChatUsage(time=0, input_tokens=1, output_tokens=1),
        )


@pytest.mark.asyncio
async def test_active_observation_survives_sdk_compression_and_enters_next_model_input(tmp_path):
    model = CompressionModel()
    source = make_wrapper(
        tmp_path,
        model,
        context_config={
            "summary_template": "{summary}",
            "summary_schema": {"type": "object", "properties": {"summary": {"type": "string"}}},
        },
    )
    session_id = await source.create_session()
    running = asyncio.create_task(source.reply("Question", resume=session_id))
    pending = None
    try:
        await asyncio.wait_for(model.entered.wait(), 2)
        pending = asyncio.create_task(
            source.append_session(
                session_id,
                "KEEP_DURING_COMPRESSION",
                message_id="observation",
                role="user",
                delivery="immediate",
            ),
        )
        await asyncio.sleep(0)
        assert not pending.done()
        model.release.set()
        _, receipt = await asyncio.gather(running, pending)
        persisted = await source.read_session(session_id)
        assert any(message.id == receipt.id for message in persisted.messages)
        assert any("KEEP_DURING_COMPRESSION" in message.model_dump_json() for message in model.calls[0]["messages"])
        assert len(model.calls) == 1
    finally:
        model.release.set()
        await asyncio.gather(*(task for task in (running, pending) if task is not None), return_exceptions=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("state", ["allowed", "asking", "submitted"])
async def test_fork_closes_unfinished_tools_only_in_the_child_without_model_calls(tmp_path, state):
    source = make_wrapper(tmp_path)
    session_id = await source.create_session()
    handler = AsStateHandler(await source.get_session_path(session_id))
    native = await handler.load()
    native.context.append(
        AssistantMsg(
            name="probe",
            id=native.reply_id,
            content=[ToolCallBlock(id="tool", name="not_executed", input="{}", state=state)],
        ),
    )
    await handler.dump(native)
    original = handler.path.read_bytes()
    child_id = await source.create_session(resume=session_id, fork_session=True)
    child = await AsStateHandler(await source.get_session_path(child_id)).load()
    assert handler.path.read_bytes() == original and not source.as_llm.model.calls
    assert child.context[-1].get_content_blocks("tool_call")[0].state == "finished"
    assert child.context[-1].get_content_blocks("tool_result")[0].state == "interrupted"
    assert (await source.reply("Continue independently", resume=child_id))["result"] == "final answer"


@pytest.mark.asyncio
async def test_fork_neither_cancels_nor_repeats_a_tool_actually_running_in_source(tmp_path):
    entered, release = asyncio.Event(), asyncio.Event()
    calls = []

    async def paused_tool() -> ToolChunk:
        """Pause this one native execution until the test releases it."""
        calls.append("started")
        entered.set()
        await release.wait()
        calls.append("finished")
        return ToolChunk(content=[TextBlock(text="Tool completed")], state="success")

    class ToolModel(OfflineModel):
        """Request one real asynchronous tool call, then complete normally."""

        async def __call__(self, **kwargs):
            if not self.calls:
                self.calls.append(kwargs)
                return ChatResponse(
                    content=[ToolCallBlock(id="tool", name="paused_tool", input="{}")],
                    is_last=True,
                    usage=ChatUsage(time=0, input_tokens=1, output_tokens=1),
                )
            return await super().__call__(**kwargs)

    tool = FunctionTool(
        paused_tool,
        permission=PermissionDecision(behavior=PermissionBehavior.ALLOW, message="Allow the isolated test tool"),
    )
    source = make_wrapper(tmp_path, ToolModel(), toolkit=Toolkit(tools=[tool]))
    session_id = await source.create_session()
    running = asyncio.create_task(source.reply("Use the tool", resume=session_id))
    try:
        await asyncio.wait_for(entered.wait(), 2)
        child_id = await asyncio.wait_for(source.create_session(resume=session_id, fork_session=True), 1)
        child = make_wrapper(tmp_path, context=source.app_context)
        assert (await child.reply("Continue independently", resume=child_id))["result"] == "final answer"
        assert calls == ["started"] and not running.done()
        release.set()
        assert (await running)["result"] == "final answer"
        assert calls == ["started", "finished"]
    finally:
        release.set()
        await asyncio.gather(running, return_exceptions=True)
