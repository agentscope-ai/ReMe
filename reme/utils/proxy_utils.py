"""Optional SSH proxy helpers for outbound HTTP downloads."""

import asyncio
import os
import socket
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass

REME_PROXY_IP = "REME_PROXY_IP"
REME_PROXY_ACCOUNT = "REME_PROXY_ACCOUNT"
_PROXY_HOST = "127.0.0.1"


@dataclass(frozen=True)
class SshProxyConfig:
    """SSH destination used to create a local SOCKS proxy."""

    host: str
    account: str

    @property
    def destination(self) -> str:
        """Return the OpenSSH destination argument."""
        return f"{self.account}@{self.host}"


def get_ssh_proxy_config() -> SshProxyConfig | None:
    """Read the optional SSH proxy configuration from the environment.

    Both variables must be configured together. Returning ``None`` keeps callers
    on their existing direct network path.
    """
    host = os.getenv(REME_PROXY_IP, "").strip()
    account = os.getenv(REME_PROXY_ACCOUNT, "").strip()
    if not host and not account:
        return None
    if not host or not account:
        missing = REME_PROXY_IP if not host else REME_PROXY_ACCOUNT
        raise ValueError(f"{missing} is required when configuring the ReMe SSH proxy")
    return SshProxyConfig(host=host, account=account)


def _pick_free_port() -> int:
    """Return an available loopback TCP port for the SSH forwarding process."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((_PROXY_HOST, 0))
        return int(sock.getsockname()[1])


async def _wait_for_proxy(process: asyncio.subprocess.Process, port: int, timeout: float) -> None:
    """Wait until the SOCKS listener accepts connections or SSH exits."""
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while loop.time() < deadline:
        if process.returncode is not None:
            detail = ""
            if process.stderr is not None:
                detail = (await process.stderr.read()).decode(errors="replace").strip()
            suffix = f": {detail}" if detail else ""
            raise RuntimeError(f"SSH proxy exited with status {process.returncode}{suffix}")
        try:
            _reader, writer = await asyncio.open_connection(_PROXY_HOST, port)
        except OSError:
            await asyncio.sleep(0.05)
            continue
        writer.close()
        await writer.wait_closed()
        return
    raise TimeoutError(f"SSH proxy did not become ready on {_PROXY_HOST}:{port} within {timeout:g}s")


async def _stop_proxy(process: asyncio.subprocess.Process) -> None:
    """Terminate an SSH forwarding process without leaking it on cancellation."""
    if process.returncode is not None:
        return
    try:
        process.terminate()
    except ProcessLookupError:
        return
    try:
        await asyncio.wait_for(process.wait(), timeout=5.0)
    except TimeoutError:
        try:
            process.kill()
        except ProcessLookupError:
            return
        await process.wait()


@asynccontextmanager
async def ssh_socks_proxy(*, connect_timeout: float = 10.0) -> AsyncIterator[str | None]:
    """Yield a temporary local SOCKS5 URL backed by the configured SSH host.

    When ``REME_PROXY_IP`` and ``REME_PROXY_ACCOUNT`` are absent, yields ``None``
    and does not start a process. The caller owns no process state beyond this
    context manager.
    """
    config = get_ssh_proxy_config()
    if config is None:
        yield None
        return

    timeout = max(1.0, float(connect_timeout))
    port = _pick_free_port()
    process = await asyncio.create_subprocess_exec(
        "ssh",
        "-N",
        "-D",
        f"{_PROXY_HOST}:{port}",
        "-o",
        "BatchMode=yes",
        "-o",
        "ExitOnForwardFailure=yes",
        "-o",
        "StrictHostKeyChecking=accept-new",
        "-o",
        f"ConnectTimeout={max(1, int(timeout))}",
        "-o",
        "LogLevel=ERROR",
        "--",
        config.destination,
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        await _wait_for_proxy(process, port, timeout)
        yield f"socks5://{_PROXY_HOST}:{port}"
    finally:
        await _stop_proxy(process)
