"""Claude session persistence, no-inference injection and concurrency boundaries."""

import asyncio
import json
import os
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest
from claude_agent_sdk import AssistantMessage, ResultMessage, StreamEvent, TextBlock
from claude_agent_sdk import project_key_for_directory

from reme.components.agent_wrapper.cc_agent_wrapper import CcAgentWrapper
from reme.components.agent_wrapper.cc_session_store import CcFileSessionStore
from reme.components.agent_wrapper.cc_sessions import native_id
from reme.components.proactive import ChatAgentRuntime
from reme.core.context import ApplicationContext

# pylint: disable=protected-access,redefined-outer-name


def write_entries(wrapper, options, frames):
    """Simulate only the CLI's native persistence boundary, within tmp_path."""
    session_id = options.session_id if options.fork_session else options.resume or options.session_id
    data = wrapper._sessions.location(options)
    path = wrapper._sessions.native_path(session_id, data)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as output:
        for frame in frames:
            entry = {"timestamp": datetime.now(timezone.utc).isoformat(), **frame, "sessionId": session_id}
            if frame.get("isSynthetic") and frame.get("type") == "user":
                entry["isMeta"] = True
            output.write(json.dumps(entry) + "\n")


@pytest.fixture
def wrapper(tmp_path, monkeypatch):
    """Keep all native writes isolated, replacing only the SDK execution boundary."""
    instance = CcAgentWrapper(
        app_context=ApplicationContext(workspace_dir=str(tmp_path)),
        api_key="test-key",
        model="test-model",
    )

    async def inject(options, frames):
        if options.session_store is not None:
            project = project_key_for_directory(options.cwd)
            target = options.session_id if options.fork_session else options.resume or options.session_id
            if options.fork_session:
                frames = (
                    await options.session_store.load({"project_key": project, "session_id": options.resume}) or []
                ) + frames
            await options.session_store.append({"project_key": project, "session_id": target}, frames)
            return
        if options.fork_session:
            data = instance._sessions.location(options)
            source = instance._sessions.native_path(options.resume, data)
            frames = [json.loads(line) for line in source.read_text().split("\n") if line.strip()] + frames
        write_entries(instance, options, frames)

    monkeypatch.setattr(instance, "_inject", AsyncMock(side_effect=inject))
    return instance


@pytest.mark.asyncio
async def test_create_empty_native_session_is_hidden_resumable_and_idle(wrapper):
    """Empty creation persists only the hidden seed, without model execution."""
    session_id = await wrapper.create_session()
    assert await wrapper.session_exists(session_id)
    assert (await wrapper.get_session_path(session_id)).is_file()
    snapshot = await wrapper.read_session(session_id)
    assert snapshot.messages == [] and snapshot.last_turn_id is None
    assert snapshot.state.status == "idle"
    options, frames = wrapper._inject.call_args.args
    assert options.system_prompt == {"type": "preset", "preset": "claude_code", "snapshot": False}
    assert options.resume is None and options.session_id == session_id
    assert frames[0]["uuid"] == native_id(session_id, "seed")
    assert frames[0]["shouldQuery"] is False
    await wrapper.release_session(session_id)
    assert await wrapper.session_exists(session_id)


@pytest.mark.parametrize("language", ["en", "zh"])
@pytest.mark.asyncio
async def test_chat_role_reaches_creation_resume_and_read_only_fork(wrapper, monkeypatch, language):
    """Native sessions receive the same preset and the current appended role on every path."""
    wrapper.app_context.app_config.language = language
    chat = ChatAgentRuntime(name="Remy", app_context=wrapper.app_context, agent_wrapper=wrapper)
    observed = []

    async def query(prompt, options):
        observed.append(options)
        write_entries(wrapper, options, [frame async for frame in prompt])
        yield ResultMessage(
            subtype="success",
            duration_ms=1,
            duration_api_ms=1,
            is_error=False,
            num_turns=1,
            session_id=options.session_id or options.resume,
            result="Done",
        )

    monkeypatch.setattr(CcAgentWrapper, "_stream_messages", staticmethod(query))
    async with chat:
        binding = chat.session_options()
        session_id = await chat.ensure_session(binding)
        assert wrapper._inject.call_args.args[0].system_prompt == binding["system_prompt"]
        await wrapper.reply("Continue", resume=session_id, **binding)
        await wrapper.reply(
            "Assess notifications",
            resume=session_id,
            fork_session=True,
            sandbox="read-only",
            system_prompt=chat.session_options()["system_prompt"],
        )
        assert len(observed) == 2
        assert all(options.system_prompt == binding["system_prompt"] for options in observed)
        assert observed[0].resume == observed[1].resume == session_id
        assert not observed[0].fork_session and observed[1].fork_session

        chat.prompt.load_prompt_dict({"system_prompt": "Updated role.", "system_prompt_zh": "更新后的角色。"})
        updated = chat.session_options()
        assert await chat.ensure_session(updated) != session_id
        options = wrapper._inject.call_args.args[0]
        assert options.resume == session_id and options.fork_session
        assert options.system_prompt == updated["system_prompt"]
        assert options.system_prompt["append"] != binding["system_prompt"]["append"]
        assert options.system_prompt["snapshot"] is False


@pytest.mark.asyncio
async def test_revision_guard_and_turn_anchor_remain_distinct_after_user_injection(wrapper):
    """Native user injections change display content without becoming model-turn anchors."""
    session_id = await wrapper.create_session()
    before = await wrapper.read_session(session_id)
    await wrapper.append_session(session_id, "Stop reminders", message_id="stop", role="user")
    after = await wrapper.read_session(session_id)
    assert after.last_turn_id is None and after.revision != before.revision
    with pytest.raises(RuntimeError, match="Session changed"):
        await wrapper.append_session(session_id, "Old reminder", message_id="old", expected_revision=before.revision)
    await wrapper.append_session(session_id, "Current reminder", message_id="current", expected_revision=after.revision)
    await wrapper.append_session(session_id, "Stop reminders", message_id="stop", role="user", expected_revision="old")
    assert len((await wrapper.read_session(session_id)).messages) == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("external", [False, True])
@pytest.mark.parametrize("fork", [False, True])
async def test_continue_resolves_latest_main_session_and_holds_its_lease(wrapper, monkeypatch, external, fork):
    """Resolve continuation before execution, including native forks and external stores."""
    data = wrapper._session_defaults()
    root = wrapper.workspace_path / "external" if external else wrapper.session_path / "claude_config" / "projects"
    store = CcFileSessionStore(root)
    if external:
        wrapper.kwargs["session_store"] = store
    project = project_key_for_directory(data.cwd)
    source_id = str(uuid4())
    for index, (identifier, sidechain) in enumerate([(source_id, False), (str(uuid4()), True), ("invalid", False)]):
        key = {"project_key": project, "session_id": identifier}
        await store.append(
            key,
            [
                {
                    "type": "user",
                    "uuid": str(uuid4()),
                    "isSidechain": sidechain,
                    "message": {"role": "user", "content": "earlier"},
                },
            ],
        )
        os.utime(root / project / f"{identifier}.jsonl", (index + 1, index + 1))
    started, release = asyncio.Event(), asyncio.Event()

    async def query(prompt, options):
        assert options.resume == source_id and not options.continue_conversation
        session_id = options.session_id if fork else options.resume
        assert (session_id != source_id) == fork
        frames = [frame async for frame in prompt]
        await store.append({"project_key": project, "session_id": session_id}, frames)
        started.set()
        await release.wait()
        yield ResultMessage(
            subtype="success",
            duration_ms=1,
            duration_api_ms=1,
            is_error=False,
            num_turns=1,
            session_id=session_id,
            result="continued",
        )

    monkeypatch.setattr(CcAgentWrapper, "_stream_messages", staticmethod(query))
    pending = asyncio.create_task(wrapper.reply("continue", continue_conversation=True, fork_session=fork))
    try:
        await asyncio.wait_for(started.wait(), 2)
        assert (await wrapper.get_session_status(source_id)).status == ("idle" if fork else "active")
        if fork:
            await wrapper.append_session(source_id, "independent", message_id="independent")
        else:
            with pytest.raises(TimeoutError):
                await wrapper.append_session(source_id, "queued", message_id="queued", timeout=0.01)
        release.set()
        result = await pending
    finally:
        release.set()
        await asyncio.gather(pending, return_exceptions=True)
    assert result["result"] == "continued"
    snapshot = await wrapper.read_session(result["session_id"])
    assert snapshot.messages[-1].content == "continue"
    assert snapshot.state.status == "idle"


@pytest.mark.asyncio
async def test_continue_without_history_starts_a_new_session(wrapper, monkeypatch):
    """Native --continue starts fresh when no resumable transcript exists."""

    async def query(prompt, options):
        assert options.resume is None and not options.continue_conversation
        frames = [frame async for frame in prompt]
        write_entries(wrapper, options, frames)
        yield ResultMessage(
            subtype="success",
            duration_ms=1,
            duration_api_ms=1,
            is_error=False,
            num_turns=1,
            session_id=options.session_id,
            result="new session",
        )

    monkeypatch.setattr(CcAgentWrapper, "_stream_messages", staticmethod(query))
    result = await wrapper.reply("start", continue_conversation=True)
    assert (await wrapper.read_session(result["session_id"])).messages[0].content == "start"


@pytest.mark.asyncio
async def test_fork_cannot_acquire_the_same_session_twice_with_different_uuid_case(wrapper):
    """UUID casing cannot bypass the fork collision check and deadlock the shared lease."""
    session_id = await wrapper.create_session()
    with pytest.raises(ValueError, match="different session_id"):
        async with asyncio.timeout(0.2):
            await wrapper.reply("same session", resume=session_id.upper(), session_id=session_id, fork_session=True)
    assert (await wrapper.get_session_status(session_id)).status == "idle"


@pytest.mark.asyncio
async def test_resume_with_explicit_session_id_requires_a_fork_before_any_writes(wrapper, monkeypatch):
    """An invalid SDK request cannot create metadata under a different target ID."""

    def query(*_args, **_kwargs):
        raise AssertionError("Invalid resume must not reach the SDK")

    monkeypatch.setattr(CcAgentWrapper, "_stream_messages", staticmethod(query))
    session_id = await wrapper.create_session()
    target = str(uuid4())
    with pytest.raises(ValueError, match="session_id cannot be used with resume"):
        await wrapper.reply("wrong target", resume=session_id, session_id=target)
    assert not wrapper._sessions.path(target).exists()
    assert wrapper._inject.await_count == 1


@pytest.mark.asyncio
async def test_file_store_rejects_final_jsonl_symlink_before_read_or_write(tmp_path):
    """Containment must check the final filename after adding its extension."""
    root = tmp_path / "store"
    project = root / "project"
    project.mkdir(parents=True)
    outside = tmp_path / "outside.jsonl"
    outside.write_text('{"uuid":"original"}\n')
    (project / "session.jsonl").symlink_to(outside)
    store = CcFileSessionStore(root)
    key = {"project_key": "project", "session_id": "session"}
    for operation in (lambda: store.load(key), lambda: store.append(key, [{"uuid": "extra"}])):
        with pytest.raises(ValueError, match="escapes root"):
            await operation()
    assert outside.read_text() == '{"uuid":"original"}\n'


@pytest.mark.asyncio
@pytest.mark.parametrize("role", ["user", "assistant"])
async def test_append_role_is_durable_and_idempotent_across_wrapper_restart(wrapper, role):
    """Both roles preserve provenance and retry identity after reloading metadata."""
    session_id = await wrapper.create_session()
    message = await wrapper.append_session(
        session_id,
        "exact content",
        message_id="one",
        role=role,
        delivery="immediate",
    )
    assert message.role == role and message.source == "proactive"
    assert message.origin_session_id == session_id and message.delivery == "immediate"
    _, frames = wrapper._inject.call_args.args
    assert frames[0]["type"] == frames[0]["message"]["role"] == role
    assert len(frames) == (2 if role == "assistant" else 1)
    assert all(frame["shouldQuery"] is False for frame in frames)
    reloaded = CcAgentWrapper(app_context=wrapper.app_context)
    assert (await reloaded.read_session(session_id)).messages == [message]
    calls = wrapper._inject.await_count
    assert await reloaded.append_session(session_id, "exact content", message_id="one", role=role) == message
    assert wrapper._inject.await_count == calls
    for changed in ({"role": "assistant" if role == "user" else "user"}, {"content": "changed"}):
        with pytest.raises(ValueError, match="different content or role"):
            await reloaded.append_session(
                session_id,
                changed.get("content", "exact content"),
                message_id="one",
                role=changed.get("role", role),
            )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "kwargs",
    [
        {"role": "system"},
        {"role": []},
        {"message_id": "../escape"},
        {"content": " "},
        {"timeout": float("nan")},
        {"timeout": True},
        {"timeout": 0},
        {"delivery": "unknown"},
        {"delivery": []},
    ],
)
async def test_invalid_append_has_no_native_side_effects(wrapper, kwargs):
    """Invalid input cannot inject or create provenance records."""
    session_id = await wrapper.create_session()
    arguments = {"content": "text", "message_id": "one", **kwargs}
    with pytest.raises(ValueError):
        await wrapper.append_session(session_id, **arguments)
    assert wrapper._inject.await_count == 1
    assert (await wrapper.read_session(session_id)).messages == []


@pytest.mark.asyncio
async def test_missing_sessions_and_escaping_paths_fail_explicitly(wrapper, tmp_path):
    """Missing and escaping sessions fail before any native execution."""
    session_id = str(uuid4())
    assert not await wrapper.session_exists(session_id)
    assert not (await wrapper.get_session_path(session_id)).exists()
    for operation in (wrapper.read_session, wrapper.get_session_status):
        with pytest.raises(FileNotFoundError):
            await operation(session_id)
    for operation in (wrapper.session_exists, wrapper.get_session_path, wrapper.read_session):
        with pytest.raises(ValueError):
            await operation("../escape")
    session_id = await wrapper.create_session()
    path = await wrapper.get_session_path(session_id)
    path.unlink()
    outside = tmp_path.parent / f"outside-{uuid4().hex}.jsonl"
    path.symlink_to(outside)
    with pytest.raises(ValueError, match="escapes"):
        await wrapper.read_session(session_id)


@pytest.mark.asyncio
async def test_explicit_external_session_directory_remains_supported(tmp_path, monkeypatch):
    """Configured storage may live outside the workspace while lookups stay contained."""
    instance = CcAgentWrapper(
        app_context=ApplicationContext(
            workspace_dir=str(tmp_path / "workspace"),
            mem_session_dir=str(tmp_path / "sessions"),
        ),
    )

    async def inject(options, frames):
        write_entries(instance, options, frames)

    monkeypatch.setattr(instance, "_inject", AsyncMock(side_effect=inject))
    session_id = await instance.create_session()
    message = await instance.append_session(session_id, "external storage", message_id="external", role="user")
    assert await instance.session_exists(session_id)
    assert (await instance.get_session_path(session_id)).is_relative_to(tmp_path / "sessions")
    assert (await instance.read_session(session_id)).messages == [message]


@pytest.mark.asyncio
async def test_native_commit_recovers_after_metadata_publication_fails(wrapper, monkeypatch):
    """Recover an accepted UUID even if its publication or original turn guard failed."""
    session_id = await wrapper.create_session()
    original = wrapper._sessions.save
    fail = True

    async def save(identifier, data):
        nonlocal fail
        if fail and any(record.accepted for record in data.appends):
            fail = False
            raise OSError("publication failed")
        await original(identifier, data)

    monkeypatch.setattr(wrapper._sessions, "save", save)
    with pytest.raises(OSError, match="publication failed"):
        await wrapper.append_session(session_id, "once", message_id="recover")
    assert (await wrapper.read_session(session_id)).messages[0].content == "once"
    calls = wrapper._inject.await_count
    message = await wrapper.append_session(session_id, "once", message_id="recover", expected_turn_id="stale")
    assert message.content == "once" and wrapper._inject.await_count == calls


@pytest.mark.asyncio
async def test_cancelled_append_finishes_native_commit_before_releasing_lease(wrapper, monkeypatch):
    """A cancelled caller cannot let another writer race its accepted injection."""
    session_id = await wrapper.create_session()
    written, release = asyncio.Event(), asyncio.Event()

    async def inject(options, frames):
        write_entries(wrapper, options, frames)
        written.set()
        await release.wait()

    monkeypatch.setattr(wrapper, "_inject", AsyncMock(side_effect=inject))
    pending = asyncio.create_task(wrapper.append_session(session_id, "once", message_id="cancel"))
    await written.wait()
    pending.cancel()
    await asyncio.sleep(0)
    assert (await wrapper.get_session_status(session_id)).status == "active"
    assert not pending.done()
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await pending
    assert (await wrapper.get_session_status(session_id)).status == "idle"
    await wrapper.append_session(session_id, "once", message_id="cancel")
    assert wrapper._inject.await_count == 1


@pytest.mark.asyncio
async def test_reads_are_nonblocking_and_append_waits_for_the_generation_lease(wrapper, monkeypatch):
    """Reads expose partial content while append queues behind a live generation."""
    session_id = await wrapper.create_session()
    started, release = asyncio.Event(), asyncio.Event()

    async def query(prompt, options):
        inputs = [frame async for frame in prompt]
        write_entries(wrapper, options, inputs)
        for event in (
            {"type": "message_start", "message": {"id": "model-message"}},
            {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
            {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "partial"}},
        ):
            yield StreamEvent(uuid=str(uuid4()), session_id=session_id, event=event)
        started.set()
        await release.wait()
        write_entries(
            wrapper,
            options,
            [
                {
                    "type": "assistant",
                    "uuid": str(uuid4()),
                    "message": {
                        "id": "model-message",
                        "role": "assistant",
                        "model": "test",
                        "content": [{"type": "text", "text": "final"}],
                    },
                },
            ],
        )
        yield AssistantMessage(content=[TextBlock(text="final")], model="test", message_id="model-message")
        yield ResultMessage(
            subtype="success",
            duration_ms=1,
            duration_api_ms=1,
            is_error=False,
            num_turns=1,
            session_id=session_id,
            result="final",
        )

    monkeypatch.setattr(CcAgentWrapper, "_stream_messages", staticmethod(query))
    task = asyncio.create_task(wrapper.reply("question", resume=session_id))
    await started.wait()
    async with asyncio.timeout(0.2):
        assert (await wrapper.get_session_status(session_id)).status == "active"
        snapshot = await wrapper.read_session(session_id)
    assert [message.content for message in snapshot.messages] == ["question", "partial"]
    assert not snapshot.messages[-1].completed
    with pytest.raises(TimeoutError):
        await wrapper.append_session(session_id, "later", message_id="timeout", timeout=0.01)
    pending = asyncio.create_task(wrapper.append_session(session_id, "later", message_id="later", role="user"))
    await asyncio.sleep(0)
    assert not pending.done()
    release.set()
    result = await task
    assert result["result"] == "final"
    await pending
    snapshot = await wrapper.read_session(session_id)
    assert [message.content for message in snapshot.messages] == ["question", "final", "later"]
    assert snapshot.last_turn_id == snapshot.messages[0].id
    assert snapshot.state.status == "idle"
    with pytest.raises(RuntimeError, match="changed"):
        await wrapper.append_session(session_id, "stale", message_id="guard", expected_turn_id="old")


@pytest.mark.asyncio
async def test_approval_wait_is_persisted_and_not_appendable(wrapper):
    """Parked native work remains active between wrapper invocations."""
    session_id = await wrapper.create_session()
    data = await wrapper._sessions.load(session_id, wrapper._session_defaults())
    data.waiting = True
    await wrapper._sessions.save(session_id, data)
    assert (await wrapper.get_session_status(session_id)).active_flags == ["waitingOnApprovalOrInput"]
    with pytest.raises(RuntimeError, match="idle"):
        await wrapper.append_session(session_id, "blocked", message_id="waiting")


@pytest.mark.asyncio
async def test_status_and_exists_do_not_read_the_native_transcript(wrapper, monkeypatch):
    """Polling activity must not parse the full, potentially large transcript."""
    session_id = await wrapper.create_session()
    monkeypatch.setattr(wrapper._sessions, "entries", AsyncMock(side_effect=AssertionError("history read")))
    assert await wrapper.session_exists(session_id)
    assert (await wrapper.get_session_status(session_id)).status == "idle"


@pytest.mark.asyncio
async def test_metadata_cannot_redirect_native_reads_outside_the_configured_location(wrapper, tmp_path):
    """Workspace metadata cannot authorize reading another Claude config directory."""
    session_id = await wrapper.create_session()
    data = await wrapper._sessions.load(session_id, wrapper._session_defaults())
    data.config_dir = str(tmp_path.parent / "other-claude-config")
    await wrapper._sessions.save(session_id, data)
    with pytest.raises(ValueError, match="configured location"):
        await wrapper.read_session(session_id)


@pytest.mark.asyncio
async def test_explicit_new_session_and_fork_cannot_replace_existing_native_history(wrapper, monkeypatch):
    """Reject colliding new/fork IDs without launching the native SDK."""
    session_id = await wrapper.create_session()
    query = AsyncMock()
    monkeypatch.setattr(CcAgentWrapper, "_stream_messages", staticmethod(query))
    with pytest.raises(FileExistsError):
        await wrapper.reply("replace", session_id=session_id)
    with pytest.raises(ValueError, match="different"):
        await wrapper.reply("replace", session_id=session_id, resume=session_id, fork_session=True)
    query.assert_not_called()
    assert (await wrapper.read_session(session_id)).messages == []


@pytest.mark.asyncio
async def test_native_history_cannot_silently_override_a_committed_append(wrapper):
    """Treat a changed native payload as corruption, not a successful retry."""
    session_id = await wrapper.create_session()
    await wrapper.append_session(session_id, "original", message_id="one")
    path = await wrapper.get_session_path(session_id)
    entries = [json.loads(line) for line in path.read_text().split("\n") if line]
    identifier = native_id(session_id, "append:one")
    for entry in entries:
        if entry.get("uuid") == identifier:
            entry["message"]["content"] = [{"type": "text", "text": "changed"}]
    path.write_text("".join(json.dumps(entry) + "\n" for entry in entries), encoding="utf-8")
    with pytest.raises(ValueError, match="differs"):
        await wrapper.read_session(session_id)
    with pytest.raises(ValueError, match="differs"):
        await wrapper.append_session(session_id, "original", message_id="one")


@pytest.mark.asyncio
async def test_external_store_preserves_unicode_line_separators(tmp_path):
    """JSONL delimiters must not split valid Unicode text inside JSON strings."""
    store = CcFileSessionStore(tmp_path)
    key = {"project_key": "project", "session_id": str(uuid4())}
    entries = [{"type": "user", "uuid": str(uuid4()), "message": {"content": "one\u2028two\u2029three"}}]
    await store.append(key, entries)
    assert await store.load(key) == entries


@pytest.mark.asyncio
async def test_history_retains_pre_compaction_messages_and_correlates_tools(wrapper):
    """Group native model blocks and attach tool results without creating user bubbles."""
    session_id = await wrapper.create_session()
    data = await wrapper._sessions.load(session_id, wrapper._session_defaults())
    prompt_id = str(uuid4())
    common = {"role": "assistant", "model": "test", "id": "model-one"}
    write_entries(
        wrapper,
        wrapper._injection_options(session_id, data),
        [
            {"type": "user", "uuid": prompt_id, "message": {"role": "user", "content": "old question"}},
            {
                "type": "assistant",
                "uuid": str(uuid4()),
                "message": {**common, "content": [{"type": "thinking", "thinking": "reason", "signature": "sig"}]},
            },
            {
                "type": "assistant",
                "uuid": str(uuid4()),
                "message": {
                    **common,
                    "content": [{"type": "tool_use", "id": "call-one", "name": "Read", "input": {"path": "a"}}],
                },
            },
            {
                "type": "user",
                "uuid": str(uuid4()),
                "message": {
                    "role": "user",
                    "content": [{"type": "tool_result", "tool_use_id": "call-one", "content": "tool output"}],
                },
            },
            {
                "type": "assistant",
                "uuid": str(uuid4()),
                "message": {**common, "id": "model-two", "content": [{"type": "text", "text": "old answer"}]},
            },
            {
                "type": "user",
                "uuid": str(uuid4()),
                "isCompactSummary": True,
                "message": {"role": "user", "content": "context summary"},
            },
        ],
    )
    snapshot = await wrapper.read_session(session_id)
    assert [message.content for message in snapshot.messages] == ["old question", "", "old answer"]
    assert snapshot.last_turn_id == prompt_id
    tool_message = snapshot.messages[1]
    assert [chunk.chunk_type.value for chunk in tool_message.chunks] == ["think", "tool_call", "tool_result"]
    assert tool_message.chunks[1].tool_call_id == tool_message.chunks[2].tool_call_id == "call-one"
    assert tool_message.chunks[2].status == "completed"
    assert all(chunk.turn_id == prompt_id for chunk in tool_message.chunks)


@pytest.mark.asyncio
async def test_closing_reply_stream_closes_sdk_and_releases_session_lease(wrapper, monkeypatch):
    """Early consumer exit cannot leave a native stream or session lease running."""
    session_id = await wrapper.create_session()
    closed = []

    async def query(**_kwargs):
        try:
            yield StreamEvent(
                uuid=str(uuid4()),
                session_id=session_id,
                event={"type": "message_start", "message": {"id": "one"}},
            )
            await asyncio.Event().wait()
        finally:
            closed.append(True)

    monkeypatch.setattr(CcAgentWrapper, "_stream_messages", staticmethod(query))
    stream = wrapper.reply_stream("question", resume=session_id)
    await anext(stream)
    await stream.aclose()
    assert closed == [True]
    assert (await wrapper.get_session_status(session_id)).status == "idle"
    assert not wrapper._sessions.runtimes


@pytest.mark.asyncio
@pytest.mark.parametrize("turns,tokens", [(0, 0), (1, 1)])
async def test_injection_requires_zero_inference_and_closes_the_sdk(monkeypatch, turns, tokens):
    """Reject a model-bearing receipt and always close the SDK client."""
    closed = []
    sent = []

    class Client:
        """Minimal public SDK lifecycle and event stream."""

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            closed.append(True)

        async def query(self, inputs):
            """Collect input frames without native or network side effects."""
            sent.extend([frame async for frame in inputs])

        async def receive_messages(self):
            """Return the specific receipt being verified."""
            yield ResultMessage(
                subtype="success",
                duration_ms=1,
                duration_api_ms=0,
                is_error=False,
                num_turns=turns,
                session_id=str(uuid4()),
                usage={"input_tokens": tokens},
            )

    monkeypatch.setattr("claude_agent_sdk.ClaudeSDKClient", lambda **_kwargs: Client())
    if turns:
        with pytest.raises(RuntimeError, match="unexpectedly invoked"):
            await CcAgentWrapper._inject(SimpleNamespace(), [{"type": "user"}])
    else:
        await CcAgentWrapper._inject(SimpleNamespace(), [{"type": "user"}])
    assert closed == [True] and sent == [{"type": "user"}]


@pytest.mark.asyncio
async def test_resume_restores_saved_cwd_and_rejects_explicit_project_changes(wrapper, tmp_path, monkeypatch):
    """The SDK and history reader must use the same native project key."""
    original = tmp_path / "original"
    original.mkdir()
    session_id = await wrapper.create_session(cwd=str(original))
    other = tmp_path / "other"
    other.mkdir()
    wrapper.kwargs["cwd"] = str(other)

    async def query(prompt, options):
        assert options.cwd == str(original)
        write_entries(wrapper, options, [frame async for frame in prompt])
        yield ResultMessage(
            subtype="success",
            duration_ms=1,
            duration_api_ms=1,
            is_error=False,
            num_turns=1,
            session_id=session_id,
            result="Resumed",
        )

    monkeypatch.setattr(CcAgentWrapper, "_stream_messages", staticmethod(query))
    assert (await wrapper.reply("Continue", resume=session_id))["status"] == "completed"
    assert (await wrapper.read_session(session_id)).messages[0].content == "Continue"
    with pytest.raises(ValueError, match="original cwd"):
        await wrapper.reply("Continue", resume=session_id, cwd=str(other))


@pytest.mark.asyncio
@pytest.mark.parametrize("accepted", [False, True])
async def test_failed_query_updates_anchor_only_after_native_input_acceptance(wrapper, monkeypatch, accepted):
    """Rejected setup preserves both the last conversation anchor and pending approval state."""
    session_id = await wrapper.create_session()
    data = await wrapper._sessions.load(session_id, wrapper._session_defaults())
    previous = data.last_turn_id = str(uuid4())
    data.waiting = True
    await wrapper._sessions.save(session_id, data)

    async def query(prompt, options):
        if accepted:
            write_entries(wrapper, options, [frame async for frame in prompt])
        raise RuntimeError("Rejected query")
        yield  # pylint: disable=unreachable

    monkeypatch.setattr(CcAgentWrapper, "_stream_messages", staticmethod(query))
    with pytest.raises(RuntimeError, match="Rejected query"):
        await wrapper.reply("Continue", resume=session_id)
    saved = await wrapper._sessions.load(session_id, wrapper._session_defaults())
    assert (saved.last_turn_id != previous) is accepted
    assert saved.waiting is not accepted
