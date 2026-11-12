"""AgentScope session contracts exercised with the real SDK and an offline model."""

# pylint: disable=protected-access,redefined-outer-name

import asyncio
import threading
from types import SimpleNamespace
from uuid import uuid4

import pytest
from agentscope.agent import Agent
from agentscope.formatter import OpenAIChatFormatter
from agentscope.message import AssistantMsg, Base64Source, DataBlock, HintBlock, TextBlock, ToolCallBlock, UserMsg
from agentscope.model import ChatResponse, ChatUsage

from reme.components.agent_wrapper import AsAgentWrapper
from reme.components.application_context import ApplicationContext
from reme.enumeration import ChunkEnum
from reme.utils import AsStateHandler


class OfflineModel:
    """Replace only the provider boundary; AgentScope handles its real events and state."""

    model = "offline-session-test"
    context_size = 100000
    formatter = OpenAIChatFormatter()

    def __init__(self):
        self.calls = []
        self.entered = asyncio.Event()
        self.release = asyncio.Event()
        self.paused = False

    async def count_tokens(self, *_args, **_kwargs):
        """Keep normal replies below the automatic compression threshold."""
        return 10

    async def __call__(self, **kwargs):
        """Return native responses, optionally pausing after a real text delta."""
        self.calls.append(kwargs)
        response = ChatResponse(
            content=[TextBlock(text="final answer")],
            is_last=True,
            usage=ChatUsage(time=0, input_tokens=10, output_tokens=5),
        )
        if not self.paused:
            return response

        async def stream():
            yield ChatResponse(content=[TextBlock(text="draft")], is_last=False)
            self.entered.set()
            await self.release.wait()
            yield response

        return stream()


@pytest.fixture
def wrapper(tmp_path):
    """Bind a real wrapper to an isolated workspace and offline model."""
    agent = AsAgentWrapper(
        name="agent",
        as_llm="",
        session_retention_days=0,
        app_context=ApplicationContext(workspace_dir=str(tmp_path)),
    )
    agent.as_llm = SimpleNamespace(model=OfflineModel())
    return agent


@pytest.mark.asyncio
@pytest.mark.parametrize("role", ["user", "assistant"])
async def test_empty_session_and_proactive_append_survive_restart_without_a_model(wrapper, role):
    """Creation and retry-safe observations work without a bound provider."""
    session_id = await wrapper.create_session()
    assert await wrapper.session_exists(session_id.upper())
    assert await wrapper.get_session_path(session_id) == wrapper.session_path / f"{session_id}.jsonl"
    message = await wrapper.append_session(
        session_id,
        "before the first turn",
        message_id="one",
        delivery="immediate",
        role=role,
    )
    assert message.source == "proactive" and message.delivery == "immediate"
    assert not wrapper.as_llm.model.calls
    restarted = AsAgentWrapper(
        name="agent",
        as_llm="",
        app_context=ApplicationContext(workspace_dir=str(wrapper.workspace_path)),
    )
    snapshot = await restarted.read_session(session_id)
    assert snapshot.messages == [message] and snapshot.last_turn_id is None
    assert await restarted.append_session(session_id, message.content, message_id="one", role=role) == message
    native = await AsStateHandler(await restarted.get_session_path(session_id)).load()
    assert len(native.context) == 1 and native.context[0].role == role
    with pytest.raises(ValueError, match="different content"):
        await restarted.append_session(session_id, "changed", message_id="one", role=role)
    with pytest.raises(ValueError, match="role"):
        await restarted.append_session(
            session_id,
            message.content,
            message_id="one",
            role="user" if role == "assistant" else "assistant",
        )
    await restarted.release_session(session_id)
    assert await restarted.session_exists(session_id)


@pytest.mark.asyncio
async def test_live_history_and_status_do_not_wait_for_generation(wrapper):
    """Reads stay available while appends serialize behind the live turn."""
    session_id = await wrapper.create_session()
    wrapper.as_llm.model.paused = True
    reply = asyncio.create_task(wrapper.reply("question", resume=session_id))
    await asyncio.wait_for(wrapper.as_llm.model.entered.wait(), 2)
    try:
        snapshot = await asyncio.wait_for(wrapper.read_session(session_id), 0.5)
        assert snapshot.state.status == "active"
        assert [message.content for message in snapshot.messages] == ["question", "draft"]
        assert not snapshot.messages[-1].completed
        snapshot.messages[-1].content = "mutated"
        assert (await wrapper.read_session(session_id)).messages[-1].content == "draft"
        with pytest.raises(TimeoutError):
            await wrapper.append_session(session_id, "later", message_id="timeout", timeout=0.01)
        pending = asyncio.create_task(wrapper.append_session(session_id, "later", message_id="later"))
        await asyncio.sleep(0)
        assert not pending.done()
        wrapper.as_llm.model.release.set()
        result = await reply
        message = await pending
    finally:
        wrapper.as_llm.model.release.set()
        await reply
    snapshot = await wrapper.read_session(session_id)
    assert result["status"] == "completed" and message.turn_id == snapshot.last_turn_id
    assert snapshot.state.status == "idle"
    assert [message.content for message in snapshot.messages] == ["question", "final answer", "later"]
    assert snapshot.messages[1].completed
    assert not wrapper._sessions._runtimes


@pytest.mark.asyncio
async def test_turn_guard_and_accepted_retry_have_distinct_semantics(wrapper):
    """Stale new deliveries fail while retries recover previously accepted messages."""
    session_id = await wrapper.create_session()
    await wrapper.reply("first", resume=session_id)
    turn = (await wrapper.read_session(session_id)).last_turn_id
    accepted = await wrapper.append_session(session_id, "once", message_id="guarded", expected_turn_id=turn)
    await wrapper.reply("second", resume=session_id)
    with pytest.raises(RuntimeError, match="changed"):
        await wrapper.append_session(session_id, "stale", message_id="new", expected_turn_id=turn)
    assert await wrapper.append_session(session_id, "once", message_id="guarded", expected_turn_id=turn) == accepted
    native = await AsStateHandler(await wrapper.get_session_path(session_id)).load()
    assert sum(message.id == accepted.id for message in native.context) == 1


@pytest.mark.asyncio
async def test_compaction_retains_history_and_forks_copy_complete_state(wrapper, monkeypatch):
    """Compression changes model context while forks retain independent complete state."""
    session_id = await wrapper.create_session()
    await wrapper.reply("keep this history", resume=session_id)
    await wrapper.append_session(session_id, "also keep", message_id="source")
    before = await wrapper.read_session(session_id)

    async def compress(agent, _config):
        agent.state.summary = "compressed"
        agent.state.context.clear()
        agent.state.middle_context = {"saved": True}
        agent.state.tool_context.activated_groups = ["saved-group"]

    monkeypatch.setattr(Agent, "compress_context", compress)
    await wrapper.compact_session(session_id)
    assert (await wrapper.read_session(session_id)).messages == before.messages
    options = wrapper._sessions.options({"resume": session_id, "fork_session": True})
    async with wrapper._sessions.open(options, "bypass") as fork:
        assert fork.state.middle_context == {"saved": True}
        assert fork.state.tool_context.activated_groups == ["saved-group"]
        expected = before.model_copy(deep=True)
        for message in expected.messages:
            for chunk in message.chunks:
                chunk.session_id = options["session_id"]
        assert fork.messages == expected.messages
        fork.state.middle_context["fork_only"] = True
        await wrapper._sessions.save(fork)
    source = await AsStateHandler(await wrapper.get_session_path(session_id)).load()
    assert source.middle_context == {"saved": True} and source.summary == "compressed"
    forked = await wrapper.read_session(options["session_id"])
    assert forked.messages[-1].origin_session_id == session_id


@pytest.mark.asyncio
async def test_explicit_session_id_keeps_fresh_execution_semantics(wrapper):
    """Stable resource IDs retain their existing fresh-context behavior."""
    session_id = str(uuid4())
    await wrapper.reply("first resource version", session_id=session_id)
    await wrapper.reply("second resource version", session_id=session_id)
    inputs = wrapper.as_llm.model.calls[-1]["messages"]
    assert not any(message.get_text_content() == "first resource version" for message in inputs)
    assert [message.content for message in (await wrapper.read_session(session_id)).messages] == [
        "first resource version",
        "final answer",
        "second resource version",
        "final answer",
    ]


@pytest.mark.asyncio
async def test_parked_tool_calls_are_not_idle_after_restart(wrapper):
    """Persistent approval waits prevent unrelated observations from hiding pending tools."""
    session_id = await wrapper.create_session()
    handler = AsStateHandler(await wrapper.get_session_path(session_id))
    state = await handler.load()
    state.context.append(
        AssistantMsg(
            name="agent",
            id=state.reply_id,
            content=[ToolCallBlock(id="tool", name="example", input="{}", state="asking")],
        ),
    )
    await handler.dump(state)
    status = await wrapper.get_session_status(session_id)
    assert status.status == "active" and status.active_flags
    with pytest.raises(RuntimeError, match="idle"):
        await wrapper.append_session(session_id, "not yet", message_id="parked")


@pytest.mark.asyncio
async def test_cancelled_reply_saves_partial_history_and_releases_the_session(wrapper):
    """SDK cancellation fallbacks preserve caller cancellation and settled display history."""
    session_id = await wrapper.create_session()
    wrapper.as_llm.model.paused = True
    task = asyncio.create_task(wrapper.reply("interrupt", resume=session_id))
    await asyncio.wait_for(wrapper.as_llm.model.entered.wait(), 2)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    snapshot = await wrapper.read_session(session_id)
    assert snapshot.state.status == "idle" and snapshot.messages[-1].completed
    assert snapshot.messages[-1].chunks[-1].status == "interrupted"
    assert not wrapper._sessions._runtimes


@pytest.mark.asyncio
async def test_cancelled_append_finishes_one_atomic_commit_before_unlocking(wrapper, monkeypatch):
    """Repeated cancellation cannot separate a native observation from its durable ID."""
    session_id = await wrapper.create_session()
    entered, release = threading.Event(), threading.Event()
    original = AsStateHandler._write

    def blocked_write(handler, content):
        entered.set()
        assert release.wait(5)
        original(handler, content)

    monkeypatch.setattr(AsStateHandler, "_write", blocked_write)
    pending = asyncio.create_task(wrapper.append_session(session_id, "committed", message_id="cancel"))
    assert await asyncio.to_thread(entered.wait, 2)
    pending.cancel()
    await asyncio.sleep(0)
    pending.cancel()
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await pending
    monkeypatch.setattr(AsStateHandler, "_write", original)
    await wrapper.append_session(session_id, "committed", message_id="cancel")
    snapshot = await wrapper.read_session(session_id)
    state = await AsStateHandler(await wrapper.get_session_path(session_id)).load()
    assert len(snapshot.messages) == len(state.context) == 1
    assert not wrapper._sessions._runtimes


@pytest.mark.asyncio
async def test_failed_append_does_not_publish_history_or_lose_the_previous_state(wrapper, monkeypatch):
    """Failed persistence leaves the old session available for a clean retry."""
    session_id = await wrapper.create_session()
    before = (await wrapper.get_session_path(session_id)).read_bytes()
    original = AsStateHandler._write

    def fail_write(_handler, _content):
        raise OSError("disk failure")

    monkeypatch.setattr(AsStateHandler, "_write", fail_write)
    with pytest.raises(OSError, match="disk failure"):
        await wrapper.append_session(session_id, "retry", message_id="retry")
    assert (await wrapper.get_session_path(session_id)).read_bytes() == before
    monkeypatch.setattr(AsStateHandler, "_write", original)
    await wrapper.append_session(session_id, "retry", message_id="retry")
    assert len((await wrapper.read_session(session_id)).messages) == 1


@pytest.mark.asyncio
async def test_session_paths_reject_traversal_and_symlink_escapes(wrapper, tmp_path):
    """Native state reads cannot escape the configured session directory."""
    for value in ("../outside", "", str(uuid4()) + "\n"):
        with pytest.raises(ValueError):
            await wrapper.session_exists(value)
    session_id = str(uuid4())
    outside = tmp_path.parent / f"{session_id}.jsonl"
    wrapper.session_path.mkdir(parents=True)
    (wrapper.session_path / f"{session_id}.jsonl").symlink_to(outside)
    with pytest.raises(ValueError, match="escapes"):
        await wrapper.read_session(session_id)


@pytest.mark.asyncio
async def test_explicit_external_session_directory_remains_supported(tmp_path):
    """Keep the existing mem_session_dir configuration independent of workspace_dir."""
    root = tmp_path / "sessions"
    agent = AsAgentWrapper(
        as_llm="",
        app_context=ApplicationContext(workspace_dir=str(tmp_path / "workspace"), mem_session_dir=str(root)),
    )
    session_id = await agent.create_session()
    assert (await agent.get_session_path(session_id)).parent == root / "agentscope"
    await agent.append_session(session_id, "external storage", message_id="external", role="user")
    assert (await agent.read_session(session_id)).messages[0].content == "external storage"


@pytest.mark.asyncio
async def test_wrappers_share_one_session_owner_and_rebind_tools_per_invocation(wrapper):
    """Shared storage reuses ownership while each call receives its own instructions."""
    other = AsAgentWrapper(name="other", as_llm="", app_context=wrapper.app_context)
    other.as_llm = wrapper.as_llm
    assert other._sessions is wrapper._sessions
    session_id = await wrapper.create_session()
    await wrapper.reply("memory task", resume=session_id, system_prompt="memory instructions")
    await other.reply("dream task", resume=session_id, system_prompt="dream instructions")
    calls = wrapper.as_llm.model.calls
    assert calls[0]["messages"][0].get_text_content() == "memory instructions"
    assert calls[1]["messages"][0].get_text_content() == "dream instructions"


@pytest.mark.asyncio
async def test_closed_stream_settles_history_and_releases_its_lease(wrapper):
    """An early consumer disconnect settles the live turn without leaving a lease."""
    session_id = await wrapper.create_session()
    wrapper.as_llm.model.paused = True
    stream = wrapper.reply_stream("close early", resume=session_id)
    while (await anext(stream)).chunk_type != ChunkEnum.CONTENT:
        pass
    await stream.aclose()
    snapshot = await wrapper.read_session(session_id)
    assert snapshot.state.status == "idle"
    assert snapshot.messages[-1].chunks[-1].status == "interrupted"
    assert not wrapper._sessions._runtimes


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "kwargs",
    [
        {"content": ""},
        {"message_id": "../escape"},
        {"timeout": 0},
        {"timeout": float("nan")},
        {"delivery": "unknown"},
        {"role": "system"},
        {"role": None},
    ],
)
async def test_invalid_append_never_modifies_the_session(wrapper, kwargs):
    """Input validation precedes all session writes."""
    session_id = await wrapper.create_session()
    path = await wrapper.get_session_path(session_id)
    before = path.read_bytes()
    options = {"content": "message", "message_id": "valid", **kwargs}
    with pytest.raises(ValueError):
        await wrapper.append_session(session_id, **options)
    assert path.read_bytes() == before


@pytest.mark.asyncio
async def test_resume_and_fork_validate_before_touching_source_history(wrapper):
    """Missing sessions and conflicting identifiers never overwrite another conversation."""
    missing = str(uuid4())
    assert not await wrapper.session_exists(missing)
    with pytest.raises(FileNotFoundError):
        await wrapper.reply("missing", resume=missing)
    with pytest.raises(ValueError, match="cannot be used"):
        await wrapper.reply("conflict", resume=missing, session_id=str(uuid4()))
    with pytest.raises(ValueError, match="different"):
        await wrapper.reply("same fork", resume=missing, session_id=missing, fork_session=True)
    assert not wrapper._sessions._runtimes


@pytest.mark.asyncio
async def test_old_state_history_preserves_media_and_handles_runtime_hints(wrapper):
    """Legacy snapshots retain attached data while runtime hints remain diagnostic."""
    session_id = await wrapper.create_session()
    handler = AsStateHandler(await wrapper.get_session_path(session_id))
    state = await handler.load()
    state.context = [
        UserMsg(name="user", content=[DataBlock(source=Base64Source(media_type="image/png", data="aW1hZ2U="))]),
        AssistantMsg(name="agent", content=[HintBlock(hint="internal", source="runtime"), TextBlock(text="answer")]),
    ]
    await handler.dump(state)
    snapshot = await wrapper.read_session(session_id)
    assert snapshot.messages[0].chunks[0].media_type == "image/png"
    assert snapshot.messages[0].chunks[0].chunk == "aW1hZ2U="
    assert snapshot.messages[1].chunks[0].metadata["visibility"] == "diagnostic"
    assert snapshot.messages[1].content == "answer"


@pytest.mark.asyncio
async def test_invalid_parked_reply_does_not_replace_existing_history(wrapper):
    """A missing confirmation leaves the prior waiting turn and its anchor intact."""
    session_id = await wrapper.create_session()
    handler = AsStateHandler(await wrapper.get_session_path(session_id))
    state = await handler.load()
    state.context.append(
        AssistantMsg(
            name="agent",
            id=state.reply_id,
            content=[ToolCallBlock(id="tool", name="example", input="{}", state="asking")],
        ),
    )
    await handler.dump(state)
    before = await wrapper.read_session(session_id)
    with pytest.raises(ValueError):
        await wrapper.reply("missing confirmation", resume=session_id)
    after = await wrapper.read_session(session_id)
    assert after.messages == before.messages and after.last_turn_id == before.last_turn_id
    assert after.state.status == "active"


@pytest.mark.asyncio
async def test_appended_first_message_keeps_its_original_turn_anchor(wrapper):
    """Pre-turn observations remain anchored before the conversation's first generated reply."""
    session_id = await wrapper.create_session()
    for role in ("user", "assistant"):
        await wrapper.append_session(session_id, f"initial {role}", role=role, message_id=role)
    await wrapper.reply("start", resume=session_id)
    snapshot = await wrapper.read_session(session_id)
    assert all(message.turn_id is None for message in snapshot.messages[:2])
    assert all(message.turn_id == snapshot.last_turn_id for message in snapshot.messages[2:])
