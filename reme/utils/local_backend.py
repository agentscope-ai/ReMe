"""Shared subprocess execution with bounded timeout and cancellation cleanup."""

import asyncio
import os
import signal
import sys
from collections.abc import Awaitable, Callable

import psutil
from agentscope.tool import ExecResult, LocalBackend

from .async_utils import finish_cleanup

_CLEANUP_TIMEOUT = 0.25
OUTPUT_LIMIT_BYTES = 1024 * 1024


class OutputLimitExceeded(RuntimeError):
    """Identify the output stream that exceeded its byte budget."""


def capture_output(buffer: bytearray, chunk: bytes, stream: str, exceeded: asyncio.Future) -> None:
    """Retain at most one stream's budget, signalling overflow without blocking draining."""
    remaining = OUTPUT_LIMIT_BYTES - len(buffer)
    buffer.extend(chunk[:remaining])
    if len(chunk) > remaining and not exceeded.done():
        exceeded.set_result(stream)


def output_limit_result(stdout: bytes, stderr: bytes, stream: str) -> ExecResult:
    """Return bounded partial output and an explicit failure diagnostic."""
    message = f"\nOutput limit exceeded: {stream} exceeds {OUTPUT_LIMIT_BYTES} bytes; command terminated.\n".encode()
    diagnostic = (stderr[: max(0, OUTPUT_LIMIT_BYTES - len(message))] + message)[:OUTPUT_LIMIT_BYTES]
    return ExecResult(exit_code=1, stdout=stdout[:OUTPUT_LIMIT_BYTES], stderr=diagnostic)


class ProcessTimeout(TimeoutError):
    """Carry the reaped process output for callers with their own timeout protocol."""

    def __init__(self, result: ExecResult):
        super().__init__("timed out")
        self.result = result


class LocalProcessBackend(LocalBackend):
    """Keep local file helpers; terminate commands exceeding 1 MiB on either output stream."""

    async def exec_shell(
        self,
        command: list[str],
        *,
        cwd: str | None = None,
        timeout: float | None = None,
        raise_errors: bool = False,
    ) -> ExecResult:
        """Run an argv in its own session and collect output within the deadline."""
        try:
            process = await self._start_process(command, cwd)
        except OSError as exc:
            if raise_errors:
                raise
            return ExecResult(exit_code=127, stdout=b"", stderr=str(exc).encode("utf-8"))
        return await self._communicate(process, timeout=timeout, raise_errors=raise_errors)

    async def _start_process(self, command: list[str], cwd: str | None, *, stdin=None):
        """Start a host process; sandboxes override platform-specific setup."""
        return await self._spawn_process(command, cwd=cwd, stdin=stdin)

    async def _spawn_process(self, command: list[str], **kwargs):
        """Capture output in a separate process group for every execution backend."""
        spawning = asyncio.create_task(
            asyncio.create_subprocess_exec(
                *command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                start_new_session=os.name == "posix",
                **kwargs,
            ),
        )
        try:
            return await asyncio.shield(spawning)
        except asyncio.CancelledError:

            async def rollback():
                outcome = (await asyncio.gather(spawning, return_exceptions=True))[0]
                if isinstance(outcome, BaseException):
                    return
                exceeded = asyncio.get_running_loop().create_future()
                communication = asyncio.create_task(
                    self._collect_output(outcome, None, bytearray(), bytearray(), exceeded),
                )
                await self._stop_process(outcome, communication)

            await finish_cleanup(rollback())
            raise

    async def _release_process(self, process: asyncio.subprocess.Process) -> None:
        """Release any platform resources owned by a completed execution."""

    @staticmethod
    async def _watch_output(operation: Awaitable, exceeded: asyncio.Future):
        """Stop waiting as soon as either captured stream exceeds its budget."""
        task = asyncio.ensure_future(operation)
        try:
            await asyncio.wait((task, exceeded), return_when=asyncio.FIRST_COMPLETED)
            if exceeded.done():
                raise OutputLimitExceeded(exceeded.result())
            return await task
        finally:
            task.cancel()
            await finish_cleanup(asyncio.gather(task, return_exceptions=True))

    @staticmethod
    async def _collect_output(process, data, stdout, stderr, exceeded):
        """Drain both pipes concurrently while retaining bounded byte buffers."""

        async def read(reader, buffer, name):
            while chunk := await reader.read(65536):
                capture_output(buffer, chunk, name, exceeded)

        async def write():
            if process.stdin is not None:
                try:
                    if data is not None:
                        process.stdin.write(data)
                        await process.stdin.drain()
                except (BrokenPipeError, ConnectionResetError):
                    pass
                finally:
                    process.stdin.close()

        tasks = [
            asyncio.create_task(read(process.stdout, stdout, "stdout")),
            asyncio.create_task(read(process.stderr, stderr, "stderr")),
            asyncio.create_task(write()),
        ]
        try:
            await asyncio.gather(*tasks)
            await process.wait()
            return bytes(stdout), bytes(stderr)
        finally:
            for task in tasks:
                task.cancel()
            await finish_cleanup(asyncio.gather(*tasks, return_exceptions=True))

    @staticmethod
    async def _execute(operation: Awaitable, *, timeout: float | None, cleanup: Callable[[bool], Awaitable[None]]):
        """Apply one deadline and always clean up success, failure, or cancellation."""
        task = asyncio.ensure_future(operation)
        completed = False
        try:
            result = await asyncio.wait_for(asyncio.shield(task), timeout=timeout)
            completed = True
            return result
        finally:

            async def finalize():
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
                await cleanup(completed)

            await finish_cleanup(finalize())

    async def _communicate(
        self,
        process: asyncio.subprocess.Process,
        *,
        data: bytes | None = None,
        timeout: float | None = None,
        raise_errors: bool = False,
    ) -> ExecResult:
        """Collect pipes and release the process, preserving each caller's timeout protocol."""
        stdout_buffer, stderr_buffer = bytearray(), bytearray()
        exceeded = asyncio.get_running_loop().create_future()
        communication = asyncio.create_task(
            self._collect_output(process, data, stdout_buffer, stderr_buffer, exceeded),
        )

        async def cleanup(_completed):
            try:
                # Closed pipes do not imply that redirected background children exited.
                await self._stop_process(process, communication)
            finally:
                await self._release_process(process)

        try:
            stdout, stderr = await self._execute(
                self._watch_output(asyncio.shield(communication), exceeded),
                timeout=timeout,
                cleanup=cleanup,
            )
        except OutputLimitExceeded as exc:
            return output_limit_result(bytes(stdout_buffer), bytes(stderr_buffer), str(exc))
        except TimeoutError:
            if raise_errors:
                stdout, stderr = communication.result() if not communication.cancelled() else (b"", b"")
                raise ProcessTimeout(ExecResult(exit_code=process.returncode, stdout=stdout, stderr=stderr)) from None
            return ExecResult(exit_code=-1, stdout=b"", stderr=b"timed out")
        return ExecResult(exit_code=process.returncode, stdout=stdout, stderr=stderr)

    @staticmethod
    def _kill_process(process: asyncio.subprocess.Process) -> None:
        """Kill the group and known detached children on macOS and Windows."""
        if os.name != "posix" or sys.platform == "darwin":
            if os.name == "posix":
                try:
                    os.killpg(process.pid, signal.SIGSTOP)
                except ProcessLookupError:
                    pass
            try:
                parent = psutil.Process(process.pid)
                for child in reversed(parent.children(recursive=True)):
                    try:
                        child.kill()
                    except (psutil.Error, OSError):
                        pass
                if os.name != "posix":
                    parent.kill()
            except (psutil.Error, OSError):
                pass
        if os.name == "posix":
            # The launcher may have exited while a pipeline still holds stdout.
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass

    async def _stop_process(self, process: asyncio.subprocess.Process, communication: asyncio.Task) -> None:
        """Kill descendants and bound draining even if a child escaped the group."""
        self._kill_process(process)
        try:
            await asyncio.wait_for(asyncio.shield(communication), timeout=_CLEANUP_TIMEOUT)
        except TimeoutError:
            communication.cancel()
            await asyncio.gather(communication, return_exceptions=True)
            # asyncio has no public API to close a Process's captured pipes.
            # An escaped descendant must not keep these transports alive.
            process._transport.close()  # pylint: disable=protected-access
            try:
                await asyncio.wait_for(process.wait(), timeout=_CLEANUP_TIMEOUT)
            except TimeoutError:
                pass
