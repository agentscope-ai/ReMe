"""Tests for reme.core.utils.execute_utils."""

import concurrent.futures

import pytest

from reme.core.utils import (
    async_exec_code,
    exec_code,
    run_shell_command,
)


# ---------------------------------------------------------------------------
# run_shell_command
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_shell_command_basic():
    """Basic shell command returns stdout and exit code 0."""
    stdout, _stderr, rc = await run_shell_command("echo hello")
    assert stdout.strip() == "hello"
    assert rc == 0


@pytest.mark.asyncio
async def test_run_shell_command_stderr():
    """Shell command captures stderr output."""
    _stdout, stderr, rc = await run_shell_command("echo error >&2")
    assert "error" in stderr
    assert rc == 0


@pytest.mark.asyncio
async def test_run_shell_command_nonzero_exit():
    """Shell command returns non-zero exit code."""
    _stdout, _stderr, rc = await run_shell_command("exit 42")
    assert rc == 42


@pytest.mark.asyncio
async def test_run_shell_command_timeout():
    """Shell command raises TimeoutError when exceeding timeout."""
    with pytest.raises(TimeoutError):
        await run_shell_command("sleep 10", timeout=0.5)


# ---------------------------------------------------------------------------
# exec_code (sync)
# ---------------------------------------------------------------------------


def test_exec_code_basic():
    """exec_code captures print output."""
    result = exec_code("print('hello')")
    assert result.strip() == "hello"


def test_exec_code_multiline():
    """exec_code handles multiline code."""
    code = "for i in range(3):\n    print(i)"
    result = exec_code(code)
    assert result.strip() == "0\n1\n2"


def test_exec_code_exception_returns_message():
    """exec_code returns exception message on error."""
    result = exec_code("raise ValueError('boom')")
    assert "boom" in result


def test_exec_code_no_output():
    """exec_code returns empty string when no output."""
    result = exec_code("x = 1 + 1")
    assert result == ""


def test_exec_code_timeout():
    """exec_code raises TimeoutError when exceeding timeout."""
    with pytest.raises(TimeoutError, match="timed out"):
        exec_code("import time; time.sleep(10)", timeout=0.5)


def test_exec_code_with_shared_executor():
    """exec_code works with a shared ThreadPoolExecutor."""
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        r1 = exec_code("print('a')", executor=pool)
        r2 = exec_code("print('b')", executor=pool)
    assert r1.strip() == "a"
    assert r2.strip() == "b"


def test_exec_code_no_timeout():
    """exec_code works with timeout=None."""
    result = exec_code("print('ok')", timeout=None)
    assert result.strip() == "ok"


# ---------------------------------------------------------------------------
# async_exec_code
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_async_exec_code_basic():
    """async_exec_code captures print output."""
    result = await async_exec_code("print('async hello')")
    assert result.strip() == "async hello"


@pytest.mark.asyncio
async def test_async_exec_code_exception():
    """async_exec_code returns exception message on error."""
    result = await async_exec_code("raise RuntimeError('async boom')")
    assert "async boom" in result


@pytest.mark.asyncio
async def test_async_exec_code_timeout():
    """async_exec_code raises TimeoutError when exceeding timeout."""
    with pytest.raises(TimeoutError, match="timed out"):
        await async_exec_code("import time; time.sleep(10)", timeout=0.5)


@pytest.mark.asyncio
async def test_async_exec_code_with_executor():
    """async_exec_code works with a shared ThreadPoolExecutor."""
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        result = await async_exec_code("print('pooled')", executor=pool)
    assert result.strip() == "pooled"


@pytest.mark.asyncio
async def test_async_exec_code_no_timeout():
    """async_exec_code works with timeout=None."""
    result = await async_exec_code("print('no limit')", timeout=None)
    assert result.strip() == "no limit"
