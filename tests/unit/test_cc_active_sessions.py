"""Production Claude client ownership, active snapshots and durable append receipts."""

# pylint: disable=missing-function-docstring,protected-access,redefined-outer-name

import asyncio
from types import SimpleNamespace
from uuid import uuid4

import pytest
from claude_agent_sdk import AssistantMessage, ResultMessage, TextBlock

from . import test_cc_agent_sessions

wrapper = test_cc_agent_sessions.wrapper
write_entries = test_cc_agent_sessions.write_entries


class NativeClient:
    """Replace native transport only; keep wrapper leases, queues, parsing and disk checks."""

    def __init__(self, wrapper, options, state):
        self.wrapper, self.options, self.state = wrapper, options, state
        self.session_id = options.session_id or options.resume
        self.messages = asyncio.Queue()
        self.inputs = []
        self.worker = None
        self.generated = False
        self._query = SimpleNamespace(wait_for_result_and_end_input=self.end_input)

    async def __aenter__(self):
        self.state.clients.append(self)
        return self

    async def __aexit__(self, *_args):
        if self.worker is not None:
            self.worker.cancel()
            await asyncio.gather(self.worker, return_exceptions=True)
        self.state.closed.append(self.session_id)

    async def query(self, stream):
        frames = [frame async for frame in stream]
        if all(frame.get("shouldQuery") is False for frame in frames):
            self.inputs.extend(frames)
            self.state.dispatches.append(frames)
            self.state.submitted.set()
            if self.generated:
                await self.flush()
            if self.state.lose_receipt:
                self.state.lose_receipt = False
                raise OSError("Lost native write receipt")
            return
        write_entries(self.wrapper, self.options, frames)
        self.state.entered.set()
        self.worker = asyncio.create_task(self.generate())

    async def generate(self):
        if self.session_id == self.state.pause_id:
            await self.state.release.wait()
        identifier = str(uuid4())
        write_entries(
            self.wrapper,
            self.options,
            [
                {
                    "type": "assistant",
                    "uuid": str(uuid4()),
                    "message": {
                        "id": identifier,
                        "role": "assistant",
                        "model": "offline",
                        "content": [{"type": "text", "text": "Native final reply"}],
                    },
                },
            ],
        )
        await self.messages.put(
            AssistantMessage(content=[TextBlock(text="Native final reply")], model="offline", message_id=identifier),
        )
        await self.messages.put(
            ResultMessage(
                subtype="success",
                duration_ms=1,
                duration_api_ms=1,
                is_error=False,
                num_turns=1,
                session_id=self.session_id,
                result="Native final reply",
                usage={"input_tokens": 11, "output_tokens": 3},
            ),
        )
        self.generated = True
        await self.flush()

    async def flush(self):
        if self.inputs:
            write_entries(self.wrapper, self.options, self.inputs)
            self.inputs.clear()
            await self.messages.put(
                ResultMessage(
                    subtype="success",
                    duration_ms=1,
                    duration_api_ms=0,
                    is_error=False,
                    num_turns=self.state.append_turns,
                    session_id=self.session_id,
                    result="",
                    usage={"input_tokens": 0, "output_tokens": 0},
                ),
            )

    async def receive_messages(self):
        while True:
            message = await self.messages.get()
            if message is None:
                return
            yield message

    async def end_input(self):
        if self.worker is not None:
            await self.worker
        await self.messages.put(None)


def install_client(monkeypatch, wrapper, source_id, *, lose_receipt=False):
    state = SimpleNamespace(
        pause_id=source_id,
        entered=asyncio.Event(),
        release=asyncio.Event(),
        submitted=asyncio.Event(),
        clients=[],
        closed=[],
        dispatches=[],
        lose_receipt=lose_receipt,
        append_turns=0,
    )
    monkeypatch.setattr("claude_agent_sdk.ClaudeSDKClient", lambda options: NativeClient(wrapper, options, state))
    return state


@pytest.mark.asyncio
async def test_creation_rejects_conflicting_source_options_before_writing(wrapper, tmp_path):
    source = await wrapper.create_session()
    path = await wrapper.get_session_path(source)
    original = path.read_bytes()
    wrapper._inject.reset_mock()
    with pytest.raises(ValueError, match="cannot be used"):
        await wrapper.create_session(resume=source, session_id=str(uuid4()))
    with pytest.raises(ValueError, match="original cwd"):
        await wrapper.create_session(resume=source, fork_session=True, cwd=str(tmp_path / "other"))
    assert path.read_bytes() == original
    wrapper._inject.assert_not_awaited()
    assert not wrapper._sessions.runtimes


@pytest.mark.asyncio
@pytest.mark.parametrize("cancelled", [False, True])
async def test_interrupted_append_admission_releases_runtime_and_retries_saved_intent(wrapper, monkeypatch, cancelled):
    session_id = await wrapper.create_session()
    save = wrapper._sessions.save

    async def interrupted_save(identifier, data):
        await save(identifier, data)
        raise asyncio.CancelledError() if cancelled else OSError("Lost intent receipt")

    monkeypatch.setattr(wrapper._sessions, "save", interrupted_save)
    with pytest.raises(asyncio.CancelledError if cancelled else OSError):
        await wrapper.append_session(session_id, "Observation", message_id="observation")
    assert not wrapper._sessions.runtimes
    assert not (await wrapper.read_session(session_id)).messages
    monkeypatch.setattr(wrapper._sessions, "save", save)
    receipt = await wrapper.append_session(
        session_id,
        "Observation",
        message_id="observation",
        expected_revision="stale",
    )
    assert (await wrapper.read_session(session_id)).messages == [receipt]
    assert not wrapper._sessions.runtimes


@pytest.mark.asyncio
@pytest.mark.parametrize("role", ["user", "assistant"])
@pytest.mark.parametrize("outcome", ["timeout", "cancelled", "lost_receipt"])
async def test_live_client_accepts_once_and_keeps_the_model_result_separate(wrapper, monkeypatch, role, outcome):
    session_id = await wrapper.create_session()
    state = install_client(monkeypatch, wrapper, session_id, lose_receipt=outcome == "lost_receipt")
    running = asyncio.create_task(wrapper.reply("Question", resume=session_id))
    observers = []
    try:
        await asyncio.wait_for(state.entered.wait(), 2)
        first = asyncio.create_task(
            wrapper.append_session(
                session_id,
                "Observation",
                message_id="observation",
                role=role,
                delivery="immediate",
                timeout=0.05 if outcome == "timeout" else 30,
            ),
        )
        observers.append(first)
        await asyncio.wait_for(state.submitted.wait(), 2)
        assert not running.done() and len(state.clients) == 1
        if outcome == "cancelled":
            first.cancel()
        error = {"timeout": TimeoutError, "cancelled": asyncio.CancelledError, "lost_receipt": OSError}[outcome]
        with pytest.raises(error):
            await first
        with pytest.raises(ValueError, match="different content"):
            await wrapper.append_session(
                session_id,
                "Different",
                message_id="observation",
                role=role,
                delivery="immediate",
            )
        retry = asyncio.create_task(
            wrapper.append_session(
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
        assert not retry.done() and len(state.dispatches) == 1
        state.release.set()
        result, receipt = await asyncio.gather(running, retry)
        assert result["result"] == "Native final reply" and result["status"] == "completed"
        assert result["usage"] == {"input_tokens": 11, "output_tokens": 3, "total_tokens": 14}
        assert sum(message.id == receipt.id for message in (await wrapper.read_session(session_id)).messages) == 1
        assert (
            await wrapper.append_session(
                session_id,
                "Observation",
                message_id="observation",
                role=role,
                expected_turn_id="stale",
            )
            == receipt
        )
        assert state.closed == [session_id] and not wrapper._sessions.runtimes
    finally:
        state.release.set()
        await asyncio.gather(running, *observers, return_exceptions=True)


@pytest.mark.asyncio
async def test_generation_cancellation_releases_append_waiters_and_retry_recovers_intent(wrapper, monkeypatch):
    session_id = await wrapper.create_session()
    state = install_client(monkeypatch, wrapper, session_id)
    running = asyncio.create_task(wrapper.reply("Question", resume=session_id))
    pending = None
    try:
        await asyncio.wait_for(state.entered.wait(), 2)
        pending = asyncio.create_task(
            wrapper.append_session(session_id, "Observation", message_id="observation", delivery="immediate"),
        )
        await asyncio.wait_for(state.submitted.wait(), 2)
        running.cancel()
        with pytest.raises(asyncio.CancelledError):
            await running
        with pytest.raises(RuntimeError, match="ended before"):
            await pending
        assert not wrapper._sessions.runtimes
        receipt = await wrapper.append_session(
            session_id,
            "Observation",
            message_id="observation",
            expected_revision="stale",
        )
        assert receipt.content == "Observation"
        assert sum(message.id == receipt.id for message in (await wrapper.read_session(session_id)).messages) == 1
    finally:
        state.release.set()
        await asyncio.gather(*(task for task in (running, pending) if task is not None), return_exceptions=True)


@pytest.mark.asyncio
async def test_append_result_cannot_replace_the_source_result_or_invoke_a_model(wrapper, monkeypatch):
    session_id = await wrapper.create_session()
    state = install_client(monkeypatch, wrapper, session_id)
    state.append_turns = 1
    running = asyncio.create_task(wrapper.reply("Question", resume=session_id))
    pending = None
    try:
        await asyncio.wait_for(state.entered.wait(), 2)
        pending = asyncio.create_task(
            wrapper.append_session(session_id, "Observation", message_id="observation", delivery="immediate"),
        )
        await asyncio.wait_for(state.submitted.wait(), 2)
        state.release.set()
        with pytest.raises(RuntimeError, match="unexpectedly invoked a model"):
            await running
        receipt = await pending
        assert receipt.content == "Observation"
        assert not wrapper._sessions.runtimes
    finally:
        state.release.set()
        await asyncio.gather(*(task for task in (running, pending) if task is not None), return_exceptions=True)


@pytest.mark.asyncio
async def test_fork_creation_and_query_do_not_acquire_or_change_the_running_source(wrapper, monkeypatch):
    source = await wrapper.create_session()
    state = install_client(monkeypatch, wrapper, source)
    running = asyncio.create_task(wrapper.reply("Source question", resume=source))
    try:
        await asyncio.wait_for(state.entered.wait(), 2)
        child = await asyncio.wait_for(wrapper.create_session(resume=source, fork_session=True), 1)
        assert child != source and not running.done()
        assert "Source question" in (await wrapper.read_session(child)).model_dump_json()
        assert (await wrapper.reply("Child question", resume=child))["result"] == "Native final reply"
        assert not running.done()
        assert "Child question" not in (await wrapper.read_session(source)).model_dump_json()
        state.release.set()
        assert (await running)["result"] == "Native final reply"
    finally:
        state.release.set()
        await asyncio.gather(running, return_exceptions=True)


@pytest.mark.asyncio
async def test_concurrent_append_batches_wait_for_previous_native_persistence(wrapper, monkeypatch):
    session_id = await wrapper.create_session()
    state = install_client(monkeypatch, wrapper, session_id)
    running = asyncio.create_task(wrapper.reply("Question", resume=session_id))
    tasks = []
    try:
        await asyncio.wait_for(state.entered.wait(), 2)
        tasks.append(
            asyncio.create_task(
                wrapper.append_session(
                    session_id,
                    "Assistant observation",
                    message_id="assistant",
                    delivery="immediate",
                ),
            ),
        )
        await asyncio.wait_for(state.submitted.wait(), 2)
        tasks.append(
            asyncio.create_task(
                wrapper.append_session(
                    session_id,
                    "User observation",
                    message_id="user",
                    role="user",
                    delivery="immediate",
                ),
            ),
        )
        queue = wrapper._sessions.runtimes[session_id].appends
        async with asyncio.timeout(2):
            while len(queue.pending) != 2:
                await asyncio.sleep(0)
        async with queue.lock:
            assert len(state.dispatches) == 1
            assert not queue.pending["reme-proactive-user"].submitted
        state.release.set()
        result, assistant, user = await asyncio.wait_for(asyncio.gather(running, *tasks), 2)
        assert result["result"] == "Native final reply" and len(state.dispatches) == 2
        snapshot = await wrapper.read_session(session_id)
        assert [message.content for message in snapshot.messages if message.source == "proactive"] == [
            assistant.content,
            user.content,
        ]
    finally:
        state.release.set()
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(running, *tasks, return_exceptions=True)
