"""Common utilities: hashing, async stream task execution, HTTP helpers."""

import asyncio
import hashlib
import json
import socket
import subprocess
import sys
import time
from collections.abc import AsyncGenerator, Callable
from contextlib import asynccontextmanager
from typing import Any, Literal

from .logger_utils import get_logger
from ..constants import REME_DEFAULT_HOST, REME_DEFAULT_PORT
from ..enumeration import ChunkEnum
from ..schema import StreamChunk


def hash_text(text: str, encoding: str = "utf-8") -> str:
    """Return SHA-256 hex digest of text."""
    return hashlib.sha256(text.encode(encoding)).hexdigest()


def _format_chunk(
    chunk: StreamChunk,
    output_format: Literal["str", "bytes", "chunk"],
) -> str | bytes | StreamChunk:
    """Render a StreamChunk in the requested transport format."""
    if output_format == "chunk":
        return chunk
    data = "data:[DONE]\n\n" if chunk.done else f"data:{chunk.model_dump_json()}\n\n"
    return data.encode() if output_format == "bytes" else data


async def execute_stream_task(
    stream_queue: asyncio.Queue[StreamChunk],
    task: asyncio.Task[Any],
    task_name: str | None = None,
    output_format: Literal["str", "bytes", "chunk"] = "str",
) -> AsyncGenerator[str | bytes | StreamChunk, None]:
    """Yield chunks from stream_queue while monitoring task; cancels task on exit.

    output_format: "str"/"bytes" emit SSE frames, "chunk" emits raw StreamChunk.
    """
    logger = get_logger()
    consumer: asyncio.Task[StreamChunk] | None = None
    try:
        while True:
            consumer = get_chunk = asyncio.create_task(stream_queue.get())
            done, _pending = await asyncio.wait({get_chunk, task}, return_when=asyncio.FIRST_COMPLETED)

            # Producer still running — relay the next chunk and continue.
            if task not in done:
                chunk = get_chunk.result()
                yield _format_chunk(chunk, output_format)
                if chunk.done:
                    return
                continue

            # Producer finished. Capture any pending chunk, then stop the consumer wait
            # so we can inspect task state safely.
            pending_chunk: StreamChunk | None = None
            if get_chunk in done:
                pending_chunk = get_chunk.result()
            else:
                get_chunk.cancel()
                try:
                    await get_chunk
                except asyncio.CancelledError:
                    pass

            # Surface task failure first — an exception trumps trailing data.
            if task.cancelled():
                msg = f"Task cancelled: {task_name}" if task_name else "Task cancelled"
                raise asyncio.CancelledError(msg)
            exc = task.exception()
            if exc is not None:
                log_msg = f"Task error in {task_name}: {exc}" if task_name else f"Task error: {exc}"
                logger.error(log_msg, exc_info=exc)
                raise exc

            # Producer ended cleanly — flush pending + drain queue so no chunk is lost,
            # then emit the terminal sentinel.
            if pending_chunk is not None:
                yield _format_chunk(pending_chunk, output_format)
                if pending_chunk.done:
                    return
            while not stream_queue.empty():
                chunk = stream_queue.get_nowait()
                yield _format_chunk(chunk, output_format)
                if chunk.done:
                    return

            yield _format_chunk(StreamChunk(chunk_type=ChunkEnum.DONE, chunk="", done=True), output_format)
            return

    finally:
        # Cancel consumer wait if still pending (e.g. on consumer aclose).
        if consumer is not None and not consumer.done():
            consumer.cancel()
            try:
                await consumer
            except asyncio.CancelledError:
                pass
        # Cancel producer task if still running to avoid resource leaks.
        if not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass


def _pick_free_port(host: str = REME_DEFAULT_HOST) -> int:
    """Bind to port 0 and return the OS-assigned free port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((host, 0))
        return s.getsockname()[1]


async def _wait_reme_ready(host: str, port: int, timeout: float) -> None:
    """Poll find_reme until it reports 'reme' or timeout elapses."""
    from .service_utils import find_reme

    deadline = time.time() + timeout
    while time.time() < deadline:
        status = await find_reme(host, port)
        if status == "reme":
            return
        await asyncio.sleep(0.2)
    raise TimeoutError(f"ReMe service did not become ready at {host}:{port} within {timeout}s")


@asynccontextmanager
async def mock_reme_server(
    host: str = REME_DEFAULT_HOST,
    port: int | None = None,
    config: str | None = None,
    extra_args: list[str] | None = None,
    startup_timeout: float = 120.0,
    shutdown_timeout: float = 10.0,
    log_to_file: bool = False,
    enable_logo: bool = False,
):
    """Spawn `reme start` as a subprocess and yield (host, port) once ready.

    Auto-picks a free port when port is None. Subprocess is terminated on exit.
    """
    logger = get_logger()
    if port is None:
        port = _pick_free_port(host)

    cmd: list[str] = [
        sys.executable,
        "-m",
        "reme.reme",
        "start",
        f"service.host={host}",
        f"service.port={port}",
        f"log_to_file={'true' if log_to_file else 'false'}",
        f"enable_logo={'true' if enable_logo else 'false'}",
    ]
    if config:
        cmd.append(f"config={config}")
    if extra_args:
        cmd.extend(extra_args)

    logger.info(f"Launching mock reme server: {' '.join(cmd)}")
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    try:
        await _wait_reme_ready(host, port, startup_timeout)
        yield host, port
    except Exception:
        # Capture early-exit output for diagnostics.
        if proc.poll() is not None and proc.stdout is not None:
            tail = proc.stdout.read()
            logger.error(f"reme server exited early. output:\n{tail}")
        raise
    finally:
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=shutdown_timeout)
            except subprocess.TimeoutExpired:
                logger.warning("reme server did not terminate gracefully, killing")
                proc.kill()
                proc.wait(timeout=shutdown_timeout)
        if proc.stdout is not None:
            try:
                proc.stdout.close()
            except Exception:
                pass


async def call_action(
    action: str,
    host: str = REME_DEFAULT_HOST,
    port: int = REME_DEFAULT_PORT,
    timeout: float = 30.0,
    **kwargs,
) -> dict | str:
    """POST to /{action}; return parsed JSON (dict) for JSON endpoints, raw text for SSE."""
    from ..components.client.http_client import HttpClient

    pieces: list[str] = []
    async with HttpClient(host=host, port=port, timeout=timeout) as client:
        async for chunk in client.stream_chunks(action, **kwargs):
            payload = chunk.chunk
            pieces.append(payload if isinstance(payload, str) else json.dumps(payload, ensure_ascii=False))
    raw = "".join(pieces)
    try:
        return json.loads(raw)
    except (ValueError, json.JSONDecodeError):
        return raw


async def call_and_check(
    action: str,
    host: str = REME_DEFAULT_HOST,
    port: int = REME_DEFAULT_PORT,
    validator: Callable[[Any], bool] | None = None,
    expected: Any = None,
    timeout: float = 30.0,
    **kwargs,
) -> Any:
    """Call action and verify response. Raises AssertionError on mismatch.

    - validator(result) -> bool: custom predicate.
    - expected: deep-equality target (compared to result, or to result[key] when expected is dict).
    """
    result = await call_action(action, host=host, port=port, timeout=timeout, **kwargs)
    if validator is not None and not validator(result):
        raise AssertionError(f"validator rejected response for action={action!r}: {result!r}")
    if expected is not None:
        if isinstance(expected, dict) and isinstance(result, dict):
            for k, v in expected.items():
                if result.get(k) != v:
                    raise AssertionError(
                        f"action={action!r} expected {k}={v!r}, got {result.get(k)!r} (full: {result!r})",
                    )
        elif result != expected:
            raise AssertionError(f"action={action!r} expected {expected!r}, got {result!r}")
    return result
