"""Cancellation during startup or shutdown must not abandon sandbox resources."""

# pylint: disable=protected-access

import asyncio
import errno
import os
import sys
from pathlib import Path
from unittest.mock import AsyncMock

import anyio
import pytest

from reme.utils.file_sandbox.path_bubble_backend import PathBubbleBackend
from reme.utils.file_sandbox.path_seatbelt_backend import LaunchctlError, PathSeatbeltBackend
from reme.utils.local_backend import LocalProcessBackend


@pytest.mark.asyncio
@pytest.mark.parametrize("scope_name", ["fail_after", "move_on_after"])
async def test_anyio_deadline_during_successful_seatbelt_bootout(tmp_path, monkeypatch, scope_name):
    """A deadline during normal bootout must retain AnyIO's cancellation identity."""
    backend = PathSeatbeltBackend(cwd=tmp_path, read_only_paths=[], writable_paths=[])
    monkeypatch.setattr("reme.utils.file_sandbox.path_seatbelt_backend.sys.platform", "darwin")
    monkeypatch.setattr("reme.utils.file_sandbox.path_seatbelt_backend.shutil.which", lambda _: "sandbox-exec")
    monkeypatch.setattr(backend, "_has_gui_launchd", AsyncMock(return_value=True))
    monkeypatch.setattr(backend, "_wait_status", AsyncMock(return_value=0))
    booted_out = False
    control = None
    scope = None

    async def launchctl(*args):
        nonlocal booted_out, control
        if args[0] == "bootstrap":
            control = Path(args[-1]).parent
        elif args[0] == "kickstart":
            return "12345"
        elif args[0] == "bootout":
            assert control.exists()
            # Expire only once the completed command enters normal cleanup.
            scope.deadline = anyio.current_time()
            await asyncio.sleep(0.02)
            assert control.exists()
            booted_out = True
        else:
            pytest.fail(f"Unexpected launchctl call: {args[0]}")
        return ""

    monkeypatch.setattr(backend, "_launchctl", launchctl)

    async def run_scope():
        nonlocal scope
        with getattr(anyio, scope_name)(None) as active_scope:
            scope = active_scope
            await backend.exec_shell(["true"])

    if scope_name == "fail_after":
        with pytest.raises(TimeoutError):
            await run_scope()
    else:
        await run_scope()
    assert scope.cancelled_caught
    assert booted_out
    assert not control.exists()


@pytest.mark.asyncio
async def test_repeated_cancellation_finishes_fifo_resource_release(tmp_path, monkeypatch):
    """Even setup failure must drain reader finalizers before closing control resources."""
    backend = PathSeatbeltBackend(cwd=tmp_path, read_only_paths=[], writable_paths=[])
    monkeypatch.setattr("reme.utils.file_sandbox.path_seatbelt_backend.sys.platform", "darwin")
    monkeypatch.setattr("reme.utils.file_sandbox.path_seatbelt_backend.shutil.which", lambda _: "sandbox-exec")
    monkeypatch.setattr(backend, "_has_gui_launchd", AsyncMock(return_value=True))
    started = asyncio.Event()
    stopping = asyncio.Event()
    reply = asyncio.Event()
    fds = set()
    stopped = set()
    control = None

    async def read_fifo(fd, *_args):
        fds.add(fd)
        if len(fds) == 2:
            started.set()
        try:
            await asyncio.Event().wait()
        finally:
            stopping.set()
            await reply.wait()
            stopped.add(fd)

    async def bootstrap(_domain, plist, _service):
        nonlocal control
        control = plist.parent
        await started.wait()
        raise RuntimeError("bootstrap failed")

    monkeypatch.setattr(backend, "_read_fifo", read_fifo)
    monkeypatch.setattr(backend, "_bootstrap", bootstrap)
    task = asyncio.create_task(backend.exec_shell(["true"]))
    try:
        await asyncio.wait_for(stopping.wait(), timeout=1)
        for _ in range(2):
            task.cancel()
            await asyncio.sleep(0)
            await asyncio.sleep(0)
            assert not task.done()
            assert control.exists()
        reply.set()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, timeout=1)
        assert stopped == fds
        assert not control.exists()
        for fd in fds:
            with pytest.raises(OSError) as exc:
                os.fstat(fd)
            assert exc.value.errno == errno.EBADF
    finally:
        reply.set()
        await asyncio.gather(task, return_exceptions=True)


@pytest.mark.asyncio
async def test_repeated_cancellation_waits_for_operation_and_backend_cleanup():
    """Repeated cancellation cannot interrupt the operation's asynchronous finalizer."""
    entered = asyncio.Event()
    stopping = asyncio.Event()
    stopped = asyncio.Event()
    reply = asyncio.Event()
    released = []

    async def operation():
        try:
            entered.set()
            await asyncio.Event().wait()
        finally:
            stopping.set()
            await reply.wait()
            stopped.set()

    async def cleanup(completed):
        assert stopped.is_set()
        released.append(completed)

    task = asyncio.create_task(LocalProcessBackend._execute(operation(), timeout=None, cleanup=cleanup))
    try:
        await asyncio.wait_for(entered.wait(), timeout=1)
        task.cancel()
        await asyncio.wait_for(stopping.wait(), timeout=1)
        task.cancel()
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        assert not task.done()
        reply.set()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, timeout=1)
        assert released == [False]
    finally:
        reply.set()
        await asyncio.gather(task, return_exceptions=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", ["cancel", "success", "timeout"])
async def test_execution_cleanup_unregisters_despite_repeated_cancellation(tmp_path, monkeypatch, outcome):
    """Keep launchd control files until bootout finishes even after repeated cancellation."""
    backend = PathSeatbeltBackend(cwd=tmp_path, read_only_paths=[], writable_paths=[])
    monkeypatch.setattr("reme.utils.file_sandbox.path_seatbelt_backend.sys.platform", "darwin")
    monkeypatch.setattr("reme.utils.file_sandbox.path_seatbelt_backend.shutil.which", lambda _: "sandbox-exec")
    monkeypatch.setattr(backend, "_has_gui_launchd", AsyncMock(return_value=True))
    monkeypatch.setattr(backend, "_kill_process", lambda _process: None)
    running = asyncio.Event()
    bootout_entered = asyncio.Event()
    bootout_reply = asyncio.Event()
    services = set()
    control = None

    async def launchctl(*args):
        nonlocal control
        if args[0] == "bootstrap":
            control = Path(args[-1]).parent
            services.add("sandbox")
        elif args[0] == "kickstart":
            return "12345"
        elif args[0] == "bootout":
            assert control.exists()
            bootout_entered.set()
            await bootout_reply.wait()
            assert control.exists()
            services.clear()
        else:
            pytest.fail(f"Unexpected launchctl call: {args[0]}")
        return ""

    async def wait_status(*_args):
        running.set()
        if outcome == "success":
            return 0
        await asyncio.Event().wait()
        return 0

    monkeypatch.setattr(backend, "_launchctl", launchctl)
    monkeypatch.setattr(backend, "_wait_status", wait_status)
    task = asyncio.create_task(backend.exec_shell(["true"], timeout=0.05 if outcome == "timeout" else None))
    try:
        await asyncio.wait_for(running.wait(), timeout=1)
        if outcome == "cancel":
            task.cancel()
        await asyncio.wait_for(bootout_entered.wait(), timeout=1)
        for _ in range(1 if outcome == "cancel" else 2):
            task.cancel()
            await asyncio.sleep(0)
            await asyncio.sleep(0)
            assert not task.done()
            assert services
            assert control.exists()
        bootout_reply.set()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, timeout=1)
        assert not services
        assert not control.exists()
    finally:
        bootout_reply.set()
        await asyncio.gather(task, return_exceptions=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("late_registration", [False, True])
@pytest.mark.parametrize("cancel_twice", [False, True])
async def test_cancelled_bootstrap_waits_and_unregisters(tmp_path, monkeypatch, late_registration, cancel_twice):
    """Unregister after a pending bootstrap settles, including a late successful reply."""
    backend = PathSeatbeltBackend(cwd=tmp_path, read_only_paths=[], writable_paths=[])
    monkeypatch.setattr("reme.utils.file_sandbox.path_seatbelt_backend.sys.platform", "darwin")
    monkeypatch.setattr("reme.utils.file_sandbox.path_seatbelt_backend.shutil.which", lambda _: "sandbox-exec")
    monkeypatch.setattr(backend, "_has_gui_launchd", AsyncMock(return_value=True))
    entered = asyncio.Event()
    reply = asyncio.Event()
    bootout_entered = asyncio.Event()
    bootout_reply = asyncio.Event()
    services = set()
    calls = []
    control = None

    async def launchctl(*args):
        nonlocal control
        calls.append(args[0])
        if args[0] == "bootstrap":
            control = Path(args[-1]).parent
            if not late_registration:
                services.add("sandbox")
            entered.set()
            await reply.wait()
            services.add("sandbox")
        elif args[0] == "bootout":
            assert reply.is_set(), "bootout must not race an unfinished registration"
            assert control.exists(), "keep control files until the service is removed"
            bootout_entered.set()
            await bootout_reply.wait()
            services.clear()
        else:
            pytest.fail(f"Unexpected launchctl call: {args[0]}")
        return ""

    monkeypatch.setattr(backend, "_launchctl", launchctl)
    task = asyncio.create_task(backend.exec_shell(["true"]))
    await asyncio.wait_for(entered.wait(), timeout=1)
    task.cancel()
    await asyncio.sleep(0)
    assert not task.done()
    reply.set()
    if cancel_twice:
        await asyncio.wait_for(bootout_entered.wait(), timeout=1)
        task.cancel()
        await asyncio.sleep(0)
        assert not task.done()
    bootout_reply.set()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, timeout=1)
    assert calls == ["bootstrap", "bootout"]
    assert not services
    assert not control.exists()


@pytest.mark.asyncio
@pytest.mark.parametrize("accepted", [False, True])
async def test_failed_bootstrap_attempts_unregister(tmp_path, monkeypatch, accepted):
    """A failed reply may still hide an accepted registration; absence alone is benign."""
    backend = PathSeatbeltBackend(cwd=tmp_path, read_only_paths=[], writable_paths=[])
    plist = tmp_path / "job.plist"
    calls = []
    services = set()

    async def launchctl(*args):
        calls.append(args[0])
        if args[0] == "bootstrap":
            if accepted:
                services.add("sandbox")
            raise LaunchctlError("registration reply failed", 5)
        if not services:
            raise LaunchctlError("service not found", errno.ESRCH)
        services.clear()
        return ""

    monkeypatch.setattr(backend, "_launchctl", launchctl)
    with pytest.raises(LaunchctlError, match="registration reply failed"):
        await backend._bootstrap("gui/501", plist, "gui/501/sandbox")
    assert calls == ["bootstrap", "bootout"]
    assert not services


@pytest.mark.asyncio
async def test_bootstrap_rollback_reports_unregistration_failure(tmp_path, monkeypatch):
    """Permission and other cleanup failures must not be mistaken for a missing service."""
    backend = PathSeatbeltBackend(cwd=tmp_path, read_only_paths=[], writable_paths=[])
    launchctl = AsyncMock(side_effect=[LaunchctlError("bootstrap failed", 5), LaunchctlError("bootout denied", 1)])
    monkeypatch.setattr(backend, "_launchctl", launchctl)
    with pytest.raises(LaunchctlError, match="bootout denied"):
        await backend._bootstrap("gui/501", tmp_path / "job.plist", "gui/501/sandbox")


@pytest.mark.asyncio
@pytest.mark.parametrize("memory_limit", [0, 1])
@pytest.mark.parametrize("spawn_fails", [False, True])
async def test_cancelled_bubble_start_reaps_before_removing_cgroup(tmp_path, monkeypatch, memory_limit, spawn_fails):
    """A delayed process handle is reclaimed before its cgroup is removed."""
    backend = PathBubbleBackend(cwd=tmp_path, read_only_paths=[], writable_paths=[], memory_limit_mib=memory_limit)
    monkeypatch.setattr("reme.utils.file_sandbox.path_bubble_backend.sys.platform", "linux")
    monkeypatch.setattr("reme.utils.file_sandbox.path_bubble_backend.shutil.which", lambda _: "bwrap")
    group = tmp_path / "cgroup"

    def create_group():
        group.mkdir()
        return group

    monkeypatch.setattr(backend, "_create_cgroup", create_group)
    entered = asyncio.Event()
    reply = asyncio.Event()
    processes = []
    create_process = asyncio.create_subprocess_exec

    async def delayed_spawn(*_args, **kwargs):
        if not spawn_fails:
            processes.append(await create_process(sys.executable, "-c", "import time;time.sleep(60)", **kwargs))
        entered.set()
        await reply.wait()
        if spawn_fails:
            raise FileNotFoundError("spawn failed after cancellation")
        return processes[0]

    monkeypatch.setattr("reme.utils.local_backend.asyncio.create_subprocess_exec", delayed_spawn)
    task = asyncio.create_task(backend._start_process(["true"], str(tmp_path)))
    try:
        await asyncio.wait_for(entered.wait(), timeout=2)
        task.cancel()
        await asyncio.sleep(0)
        task.cancel()
        await asyncio.sleep(0)
        assert not task.done()
        assert group.exists() is bool(memory_limit)
        reply.set()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, timeout=2)
        assert all(process.returncode is not None for process in processes)
        assert not group.exists()
        assert not backend._process_cgroups
    finally:
        reply.set()
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        for process in processes:
            if process.returncode is None:
                process.kill()
                await process.communicate()
