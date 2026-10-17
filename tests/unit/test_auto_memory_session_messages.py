"""Session transcript saves stay consistent under concurrent writers."""

# pylint: disable=protected-access,missing-function-docstring

import asyncio
from types import SimpleNamespace

from agentscope.message import Msg, TextBlock
import pytest

import reme.steps.evolve.auto_memory as auto_memory
from reme.steps.evolve.auto_memory import AutoMemoryStep

_SESSION = "race-session"


def _msg(message_id: str, timestamp: str) -> Msg:
    return Msg.model_validate(
        {
            "id": message_id,
            "name": "Alice",
            "role": "user",
            "created_at": timestamp,
            "content": [TextBlock(text=f"text of {message_id}")],
        },
    )


@pytest.fixture(name="step")
def memory_step(tmp_path):
    step = AutoMemoryStep()
    step.file_store = SimpleNamespace(workspace_path=tmp_path)
    return step


class _ReadGate:
    """Pause chosen transcript reads right after they captured the file content."""

    def __init__(self, gated_reads: int = 1):
        self.pending = gated_reads
        self.entered = asyncio.Event()
        self.release = asyncio.Event()

    def patch(self, monkeypatch):
        real_open = auto_memory.aiofiles.open

        def patched_open(path, mode="r", **kwargs):
            context_manager = real_open(path, mode, **kwargs)
            if mode != "r" or self.pending <= 0:
                return context_manager
            return _GatedRead(context_manager, self)

        monkeypatch.setattr(auto_memory.aiofiles, "open", patched_open)


class _GatedRead:
    def __init__(self, context_manager, gate):
        self._context_manager = context_manager
        self._gate = gate

    async def __aenter__(self):
        handle = await self._context_manager.__aenter__()
        gate = self._gate
        gate.pending -= 1

        async def read():
            content = await handle.read()
            gate.entered.set()
            await gate.release.wait()
            return content

        return SimpleNamespace(read=read)

    async def __aexit__(self, *exc_info):
        return await self._context_manager.__aexit__(*exc_info)


def _assert_transcript(path, expected_ids):
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    messages = [Msg.model_validate_json(line) for line in lines]
    assert [message.id for message in messages] == expected_ids
    assert len({message.id for message in messages}) == len(messages)
    assert [message.created_at for message in messages] == sorted(message.created_at for message in messages)


async def _race(step, gate, gated_messages, other_messages):
    """Run one gated save against one inline save; release the gate mid-flight."""
    task_gated = asyncio.create_task(step._save_session_messages(_SESSION, gated_messages))
    await asyncio.wait_for(gate.entered.wait(), timeout=5)
    task_other = asyncio.create_task(step._save_session_messages(_SESSION, other_messages))
    await asyncio.sleep(0.1)
    gate.release.set()
    await asyncio.wait_for(asyncio.gather(task_gated, task_other), timeout=5)


@pytest.mark.asyncio
async def test_concurrent_appends_merge_disjoint_messages_in_order(step, tmp_path, monkeypatch):
    await step._save_session_messages(_SESSION, [_msg("m0", "2026-10-01T10:00:00")])

    gate = _ReadGate(gated_reads=1)
    gate.patch(monkeypatch)
    # The gated save sees only m0; the inline save appends m2 while it is paused.
    await _race(step, gate, [_msg("m1", "2026-10-01T10:01:00")], [_msg("m2", "2026-10-01T10:02:00")])

    _assert_transcript(step._session_path(_SESSION), ["m0", "m1", "m2"])


@pytest.mark.asyncio
async def test_concurrent_append_survives_rewrite_race(step, tmp_path, monkeypatch):
    await step._save_session_messages(_SESSION, [_msg("m0", "2026-10-01T10:00:00")])
    await step._save_session_messages(_SESSION, [_msg("m1", "2026-10-01T10:01:00")])

    gate = _ReadGate(gated_reads=1)
    gate.patch(monkeypatch)
    # The gated save inserts m_mid before m1, forcing the full-rewrite branch;
    # the inline save appends m2 against the transcript the gated save already read.
    await _race(
        step,
        gate,
        [_msg("m_mid", "2026-10-01T10:00:30")],
        [_msg("m2", "2026-10-01T10:02:00")],
    )

    _assert_transcript(step._session_path(_SESSION), ["m0", "m_mid", "m1", "m2"])
