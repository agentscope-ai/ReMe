"""Completed automatic-memory requests can be replayed after a client timeout."""

# pylint: disable=protected-access,missing-function-docstring

import asyncio
from types import SimpleNamespace

import pytest
from agentscope.message import Msg

from reme.components.runtime_context import RuntimeContext
from reme.steps.evolve.auto_memory import AutoMemoryStep


def _step(tmp_path, request_id="batch-1", **kwargs):
    step = AutoMemoryStep()
    step.file_store = SimpleNamespace(workspace_path=tmp_path)
    step.context = RuntimeContext(
        request_id=request_id,
        session_id="session-1",
        messages=[{"id": "m1", "content": "fact"}],
        date="2026-10-21",
        **kwargs,
    )
    return step


@pytest.mark.asyncio
@pytest.mark.parametrize("first_is_msg", [False, True])
@pytest.mark.parametrize("replay_is_msg", [False, True])
async def test_message_objects_and_dicts_share_a_stable_request_fingerprint(tmp_path, first_is_msg, replay_is_msg):
    message = Msg.model_validate(
        {
            "id": "stable-message",
            "name": "user",
            "role": "user",
            "content": [{"type": "text", "text": "fact"}],
            "created_at": "2026-10-21T12:00:00+08:00",
        },
    )
    original = message.model_dump(mode="json")
    first = _step(tmp_path)
    first.context["messages"] = [message if first_is_msg else original]

    async def execute():
        first.context.response.answer = "recorded"

    first._execute = execute
    await first.execute()
    assert first.context.response.success
    assert message.model_dump(mode="json") == original
    replay = _step(tmp_path)
    replay.context["messages"] = [Msg.model_validate(original) if replay_is_msg else original]

    async def unexpected():
        pytest.fail("replaying a supported message form must not invoke the agent")

    replay._execute = unexpected
    await replay.execute()
    assert replay.context.response == first.context.response


@pytest.mark.asyncio
async def test_completed_request_survives_restart_without_repeating_agent(tmp_path):
    first = _step(tmp_path)

    async def execute():
        first.context.response.answer = "recorded"
        first.context.response.metadata = {"path": "daily/note.md"}
        first.context["changes"] = [{"change": "modified", "path": "daily/note.md"}]

    first._execute = execute
    await first.execute()
    replay = _step(tmp_path)

    async def unexpected():
        pytest.fail("completed request must not invoke the memory agent")

    replay._execute = unexpected
    await replay.execute()
    assert replay.context.response == first.context.response
    assert replay.context["changes"] == first.context["changes"]


@pytest.mark.asyncio
async def test_concurrent_duplicate_waits_for_the_original_request(tmp_path):
    first, duplicate = _step(tmp_path), _step(tmp_path)
    started, release = asyncio.Event(), asyncio.Event()
    calls = []

    async def execute():
        calls.append("agent")
        started.set()
        await release.wait()

    first._execute = duplicate._execute = execute
    original = asyncio.create_task(first.execute())
    await started.wait()
    retry = asyncio.create_task(duplicate.execute())
    await asyncio.sleep(0)
    assert calls == ["agent"]
    release.set()
    await asyncio.wait_for(asyncio.gather(original, retry), 2)
    assert calls == ["agent"]


@pytest.mark.asyncio
async def test_failed_request_can_retry_but_reusing_a_completed_id_for_other_input_fails(tmp_path):
    step = _step(tmp_path)

    async def fail():
        step.context.response.success = False

    step._execute = fail
    await step.execute()
    assert not list(tmp_path.rglob("*.json"))

    retry = _step(tmp_path)

    async def succeed():
        return None

    retry._execute = succeed
    await retry.execute()
    assert retry.context.response.success
    conflicting = _step(tmp_path)
    conflicting.context["messages"] = [{"id": "m2", "content": "different fact"}]
    await conflicting.execute()
    assert not conflicting.context.response.success
    assert "different input" in conflicting.context.response.answer


@pytest.mark.asyncio
async def test_receipts_obey_workspace_permissions(tmp_path):
    step = _step(tmp_path, _allowed_paths=["daily"])
    with pytest.raises(PermissionError):
        await step.execute()
    assert not list(tmp_path.rglob("*.json"))


@pytest.mark.asyncio
@pytest.mark.parametrize("request_id", ["", "x" * 129, 123])
async def test_rejects_invalid_request_id_before_running_agent(tmp_path, request_id):
    step = _step(tmp_path, request_id)
    await step.execute()
    assert not step.context.response.success
    assert "request_id" in step.context.response.answer
    assert not list(tmp_path.iterdir())
