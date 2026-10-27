"""Local executions must not leave same-group background processes running."""

import asyncio
import os

import psutil
import pytest

from reme.utils.local_backend import LocalProcessBackend


@pytest.mark.asyncio
@pytest.mark.skipif(os.name != "posix", reason="requires POSIX shell process groups")
@pytest.mark.parametrize("exit_code", [0, 7])
async def test_completed_command_stops_redirected_background_process(tmp_path, exit_code):
    """Completion preserves the result and kills children even after all pipes close."""
    (tmp_path / "note.txt").write_text("note\n", encoding="utf-8")
    pid_file = tmp_path / "child.pid"
    command = (
        "tail -f note.txt < /dev/null > /dev/null 2>&1 &\n"
        "echo $! > child.pid\n"
        "printf 'output\\n'\n"
        "printf 'diagnostic\\n' >&2\n"
        f"exit {exit_code}"
    )
    try:
        result = await asyncio.wait_for(
            LocalProcessBackend().exec_shell(["/bin/sh", "-c", command], cwd=str(tmp_path), timeout=100),
            timeout=3,
        )
        assert result.exit_code == exit_code
        assert result.stdout == b"output\n"
        assert result.stderr == b"diagnostic\n"
        pid = int(pid_file.read_text(encoding="ascii"))

        async def wait_stopped():
            while True:
                try:
                    if psutil.Process(pid).status() == psutil.STATUS_ZOMBIE:
                        return
                except psutil.NoSuchProcess:
                    return
                await asyncio.sleep(0.01)

        # Signal delivery and orphan reaping can lag the executor's return.
        await asyncio.wait_for(wait_stopped(), timeout=1)
    finally:
        if pid_file.exists():
            try:
                psutil.Process(int(pid_file.read_text(encoding="ascii"))).kill()
            except psutil.NoSuchProcess:
                pass


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel", [False, True])
async def test_inherited_pipe_cleanup_is_bounded(monkeypatch, cancel):
    """A surviving child with an output pipe must not hold the timeout response open."""
    from unittest.mock import Mock

    backend = LocalProcessBackend()
    started = asyncio.Event()
    cancelled = asyncio.Event()

    class StuckPipe:
        """Model an inherited pipe held open after termination."""

        async def read(self, _size):
            """Wait forever until the caller cancels pipe draining."""
            started.set()
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.set()

    class StuckProcess:
        """Model a child that keeps its output pipe open after termination."""

        pid = 12345
        _transport = Mock()
        stdout = StuckPipe()
        stderr = StuckPipe()
        stdin = None

        async def wait(self):
            """A stale process transport must also have a bounded wait."""
            await asyncio.Event().wait()

    async def start(*_args, **_kwargs):
        return StuckProcess()

    monkeypatch.setattr(backend, "_start_process", start)
    monkeypatch.setattr(backend, "_kill_process", lambda _process: None)
    monkeypatch.setattr("reme.utils.local_backend._CLEANUP_TIMEOUT", 0.01)

    task = asyncio.create_task(backend.exec_shell(["true"], timeout=None if cancel else 0.01))
    await asyncio.wait_for(started.wait(), timeout=1)
    if cancel:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, timeout=1)
    else:
        result = await asyncio.wait_for(task, timeout=1)
        assert result.exit_code == -1 and result.stderr == b"timed out"
    assert cancelled.is_set()
    StuckProcess._transport.close.assert_called_once()  # pylint: disable=protected-access
