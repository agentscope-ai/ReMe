"""Utility functions for executing code and shell commands.

This module provides helper functions for running Python code and shell commands,
with support for async execution and output capture.
"""

import asyncio
import contextlib
import concurrent.futures
from io import StringIO


async def run_shell_command(cmd: str, timeout: float | None = 30) -> tuple[str, str, int]:
    """Execute a shell command asynchronously.

    Args:
        cmd: The shell command to execute.
        timeout: Maximum time to wait for command completion in seconds. None for no timeout.

    Returns:
        A tuple containing (stdout, stderr, return_code) as strings and integer.

    Raises:
        TimeoutError: If the command does not complete within the timeout.
    """
    process = await asyncio.create_subprocess_shell(
        cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )

    try:
        if timeout:
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=timeout)
        else:
            stdout, stderr = await process.communicate()
    except (asyncio.TimeoutError, TimeoutError) as e:
        # Kill the child process to avoid orphaned / zombie processes
        process.kill()
        await process.wait()
        raise TimeoutError(f"Shell command timed out after {timeout}s") from e

    return (
        stdout.decode("utf-8", errors="ignore"),
        stderr.decode("utf-8", errors="ignore"),
        process.returncode,
    )


def exec_code(
    code: str,
    timeout: float | None = 30,
    executor: concurrent.futures.ThreadPoolExecutor | None = None,
) -> str:
    """Execute Python code and capture the output.

    Args:
        code: The Python code string to execute.
        timeout: Maximum time to wait for execution in seconds. None for no timeout.
        executor: Optional thread pool executor to use. If None, a temporary
            single-thread executor is created (and shut down after the call).
            Pass a shared executor to amortize thread-creation overhead across
            multiple calls.

    Returns:
        The captured stdout output, or the error message if execution fails.

    Raises:
        TimeoutError: If execution exceeds the timeout.
    """

    def _run() -> str:
        redirected_output = StringIO()
        with contextlib.redirect_stdout(redirected_output):
            exec(code)
        return redirected_output.getvalue()

    def _submit(pool: concurrent.futures.ThreadPoolExecutor) -> str:
        future = pool.submit(_run)
        return future.result(timeout=timeout)

    try:
        if executor is not None:
            return _submit(executor)
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            return _submit(pool)

    except concurrent.futures.TimeoutError as e:
        raise TimeoutError(f"Code execution timed out after {timeout}s") from e

    except Exception as e:
        return str(e)

    except BaseException as e:
        return str(e)


async def async_exec_code(
    code: str,
    timeout: float | None = 30,
    executor: concurrent.futures.ThreadPoolExecutor | None = None,
) -> str:
    """Execute Python code asynchronously and capture the output.

    Runs the code in a thread executor to avoid blocking the event loop,
    with async-friendly timeout via ``asyncio.wait_for``.

    Args:
        code: The Python code string to execute.
        timeout: Maximum time to wait for execution in seconds. None for no timeout.
        executor: Optional thread pool executor. If None, the default event-loop
            executor is used.

    Returns:
        The captured stdout output, or the error message if execution fails.

    Raises:
        TimeoutError: If execution exceeds the timeout.
    """

    def _run() -> str:
        redirected_output = StringIO()
        with contextlib.redirect_stdout(redirected_output):
            exec(code)
        return redirected_output.getvalue()

    loop = asyncio.get_running_loop()

    try:
        coro = loop.run_in_executor(executor, _run)
        if timeout is not None:
            return await asyncio.wait_for(coro, timeout=timeout)
        return await coro

    except asyncio.TimeoutError as e:
        raise TimeoutError(f"Code execution timed out after {timeout}s") from e

    except Exception as e:
        return str(e)

    except BaseException as e:
        return str(e)
