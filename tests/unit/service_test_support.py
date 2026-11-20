"""Support helpers for testing ReMe services and jobs."""

import asyncio
import json
import socket
import subprocess
import sys
import time
from contextlib import asynccontextmanager

from reme.constants import REME_DEFAULT_HOST, REME_DEFAULT_PORT
from reme.reme import find_reme
from reme.utils.logger_utils import get_logger


def _pick_free_port(host: str = REME_DEFAULT_HOST) -> int:
    """Bind to port 0 and return the OS-assigned free port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((host, 0))
        return s.getsockname()[1]


async def _wait_reme_ready(host: str, port: int, timeout: float) -> None:
    """Poll find_reme until it reports 'reme' or timeout elapses."""
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
    from reme.components.client.http_client import HttpClient

    pieces: dict[str, str] = {}
    result = None
    async with HttpClient(host=host, port=port, timeout=timeout) as client:
        async for chunk in client.stream_chunks(action, **kwargs):
            payload = chunk.chunk
            if chunk.chunk_type == "error" and chunk.metadata.get("will_retry") is not True:
                raise RuntimeError(str(payload))
            if chunk.chunk_type == "reply_end" and isinstance(payload, dict):
                if payload["status"] != "completed":
                    raise RuntimeError(payload["error"] or f"Agent reply ended: {payload['status']}")
                result = payload["result"]
                continue
            if payload is None or chunk.metadata.get("visibility") == "diagnostic":
                continue
            key = f"{chunk.chunk_type}:{chunk.block_id}" if chunk.block_id else str(len(pieces))
            text = payload if isinstance(payload, str) else json.dumps(payload, ensure_ascii=False)
            pieces[key] = pieces.get(key, "") + text if chunk.event == "delta" else text
    raw = result if result is not None else "".join(pieces.values())
    try:
        return json.loads(raw)
    except ValueError:
        return raw
