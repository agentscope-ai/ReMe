"""Session transcript saves stay consistent under concurrent writers."""

# pylint: disable=protected-access,missing-function-docstring

import asyncio
from types import SimpleNamespace

from agentscope.message import Msg, TextBlock
import pytest

from reme.steps.evolve import auto_memory
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


class _LoggedLock:
    """Record lock contention; writer B signals once it is queued on the lock."""

    def __init__(self, lock, tag, interleaving):
        self._lock = lock
        self._tag = tag
        self._interleaving = interleaving

    async def __aenter__(self):
        self._interleaving.log.append((self._tag, "lock-wait"))
        if self._tag == "B":
            self._interleaving.b_lock_waiting.set()
        await self._lock.acquire()
        self._interleaving.log.append((self._tag, "lock-acquired"))
        return self

    async def __aexit__(self, *exc_info):
        self._lock.release()
        return None


class _LoggedRead:
    """Record read progress; writer A pauses after reading until B awaits the lock."""

    def __init__(self, context_manager, tag, interleaving):
        self._context_manager = context_manager
        self._tag = tag
        self._interleaving = interleaving

    async def __aenter__(self):
        handle = await self._context_manager.__aenter__()
        interleaving, tag = self._interleaving, self._tag
        interleaving.log.append((tag, "read-start"))

        async def read():
            content = await handle.read()
            interleaving.log.append((tag, "read-done"))
            if tag == "A":
                await interleaving.b_lock_waiting.wait()
            return content

        return SimpleNamespace(read=read)

    async def __aexit__(self, *exc_info):
        return await self._context_manager.__aexit__(*exc_info)


class _LoggedWrite:
    """Record write completion once the file handle is closed and flushed."""

    def __init__(self, context_manager, tag, interleaving):
        self._context_manager = context_manager
        self._tag = tag
        self._interleaving = interleaving

    async def __aenter__(self):
        self._interleaving.log.append((self._tag, "write-start"))
        return await self._context_manager.__aenter__()

    async def __aexit__(self, *exc_info):
        result = await self._context_manager.__aexit__(*exc_info)
        self._interleaving.log.append((self._tag, "write-done"))
        return result


class _Interleaving:
    """Deterministic two-writer interleaving driven by events, not sleeps.

    Writer A snapshots the transcript, then pauses until writer B is queued on
    the transcript lock. Without mutual exclusion B would read and write against
    A's stale snapshot in between; with the lock B cannot start reading until A
    finished writing.
    """

    def __init__(self):
        self.log: list[tuple[str, str]] = []
        self.b_lock_waiting = asyncio.Event()
        self.task_a: asyncio.Task | None = None
        self.task_b: asyncio.Task | None = None

    def patch(self, monkeypatch):
        real_get_path_lock = auto_memory.get_path_lock
        real_open = auto_memory.aiofiles.open

        async def get_path_lock(target):
            return _LoggedLock(await real_get_path_lock(target), self._tag(), self)

        def open_(path, mode="r", **kwargs):
            context_manager = real_open(path, mode, **kwargs)
            tag = self._tag()
            if mode == "r":
                return _LoggedRead(context_manager, tag, self)
            if mode in ("a", "w"):
                return _LoggedWrite(context_manager, tag, self)
            return context_manager

        monkeypatch.setattr(auto_memory, "get_path_lock", get_path_lock)
        monkeypatch.setattr(auto_memory.aiofiles, "open", open_)

    def _tag(self):
        task = asyncio.current_task()
        if task is self.task_a:
            return "A"
        if task is self.task_b:
            return "B"
        return None

    async def run(self, step, a_messages, b_messages):
        self.task_a = asyncio.create_task(step._save_session_messages(_SESSION, a_messages))
        await asyncio.sleep(0)  # Let writer A acquire the transcript lock before B starts.
        self.task_b = asyncio.create_task(step._save_session_messages(_SESSION, b_messages))
        await asyncio.wait_for(asyncio.gather(self.task_a, self.task_b), timeout=5)


def _assert_transcript(path, expected_ids):
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    messages = [Msg.model_validate_json(line) for line in lines]
    assert [message.id for message in messages] == expected_ids
    assert len({message.id for message in messages}) == len(messages)
    assert [message.created_at for message in messages] == sorted(message.created_at for message in messages)


def _assert_serialized(interleaving):
    log = interleaving.log
    # Writer B queued on the transcript lock while writer A held it: A's
    # post-read pause only ends once B attempted the lock, so B's lock-wait
    # must precede A's write.
    assert log.index(("B", "lock-wait")) < log.index(("A", "write-start"))
    # Mutual exclusion: B acquires the lock and re-reads only after A's write
    # completed, so B always merges against A's fresh transcript state.
    assert log.index(("A", "write-done")) < log.index(("B", "lock-acquired"))
    assert log.index(("B", "lock-acquired")) < log.index(("B", "read-start"))


@pytest.mark.asyncio
async def test_concurrent_appends_merge_disjoint_messages_in_order(step, monkeypatch):
    await step._save_session_messages(_SESSION, [_msg("m0", "2026-10-01T10:00:00")])

    interleaving = _Interleaving()
    interleaving.patch(monkeypatch)
    # Writer A snapshots [m0]; writer B appends m2 while A is paused.
    await interleaving.run(step, [_msg("m1", "2026-10-01T10:01:00")], [_msg("m2", "2026-10-01T10:02:00")])

    _assert_transcript(step._session_path(_SESSION), ["m0", "m1", "m2"])
    _assert_serialized(interleaving)


@pytest.mark.asyncio
async def test_concurrent_append_survives_rewrite_race(step, monkeypatch):
    await step._save_session_messages(_SESSION, [_msg("m0", "2026-10-01T10:00:00")])
    await step._save_session_messages(_SESSION, [_msg("m1", "2026-10-01T10:01:00")])

    interleaving = _Interleaving()
    interleaving.patch(monkeypatch)
    # Writer A inserts m_mid before m1, forcing the full-rewrite branch against
    # its stale snapshot; writer B appends m2 while A is paused.
    await interleaving.run(
        step,
        [_msg("m_mid", "2026-10-01T10:00:30")],
        [_msg("m2", "2026-10-01T10:02:00")],
    )

    _assert_transcript(step._session_path(_SESSION), ["m0", "m_mid", "m1", "m2"])
    _assert_serialized(interleaving)
