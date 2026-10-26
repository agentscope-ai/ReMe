"""Long-lived SSH-backed HTTP outbound proxy."""

import asyncio
import contextlib
import importlib.util
import math
import shutil
import socket
import sys
import time
from asyncio.subprocess import Process

from .base import BaseOutboundProxy, OutboundProxyEndpoint
from ..component_registry import R

_LOOPBACK = "127.0.0.1"
_START_ATTEMPTS = 3
_PROCESS_CLOSE_TIMEOUT = 5.0
_PROCESS_OUTPUT_LIMIT = 4096
_PORT_CONFLICT_MARKERS = ("address already in use", "cannot listen to port")


class _PortUnavailableError(RuntimeError):
    """A selected listener port was claimed before its subprocess bound it."""


@R.register("ssh_http")
class SshHttpOutboundProxy(BaseOutboundProxy):
    """Expose a stable local HTTP proxy backed by an OpenSSH SOCKS tunnel."""

    def __init__(
        self,
        host: str = "",
        account: str = "",
        connect_timeout: float = 10.0,
        monitor_interval: float = 1.0,
        restart_initial_delay: float = 1.0,
        restart_max_delay: float = 30.0,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.host = host
        self.account = account
        self.connect_timeout = connect_timeout
        self.monitor_interval = monitor_interval
        self.restart_initial_delay = restart_initial_delay
        self.restart_max_delay = restart_max_delay

        self._ssh_executable: str | None = None
        self._socks_port: int | None = None
        self._http_port: int | None = None
        self._ssh_process: Process | None = None
        self._bridge_process: Process | None = None
        self._monitor_task: asyncio.Task[None] | None = None
        self._closing = False

    async def _start(self) -> None:
        self._validate_configuration()
        self._closing = False

        try:
            for attempt in range(_START_ATTEMPTS):
                self._socks_port, self._http_port = self._pick_distinct_ports()
                try:
                    await self._start_processes()
                    break
                except _PortUnavailableError:
                    await self._stop_processes()
                    if attempt + 1 == _START_ATTEMPTS:
                        raise
                    self.logger.warning("Outbound proxy listener port was claimed; selecting new ports.")
            else:  # pragma: no cover - loop either breaks or raises
                raise RuntimeError("Outbound proxy failed to allocate listener ports.")

            assert self._http_port is not None
            self._endpoint = OutboundProxyEndpoint(http_url=f"http://{_LOOPBACK}:{self._http_port}")
            self._monitor_task = asyncio.create_task(
                self._monitor_processes(),
                name=f"outbound-proxy-monitor:{self.name}",
            )
            self.logger.info(
                f"Outbound proxy ready http={_LOOPBACK}:{self._http_port} " f"socks={_LOOPBACK}:{self._socks_port}",
            )
        except BaseException:
            await self._stop_processes()
            self._endpoint = None
            self._socks_port = None
            self._http_port = None
            raise

    async def _close(self) -> None:
        self._closing = True
        cleanup = asyncio.create_task(self._cleanup())
        try:
            await asyncio.shield(cleanup)
        except asyncio.CancelledError:
            await cleanup
            raise

    async def _cleanup(self) -> None:
        try:
            if self._monitor_task is not None:
                self._monitor_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await self._monitor_task
                self._monitor_task = None
            await self._stop_processes()
        finally:
            self._endpoint = None
            self._socks_port = None
            self._http_port = None

    def _validate_configuration(self) -> None:
        if not isinstance(self.host, str) or not self.host.strip():
            raise ValueError("Outbound proxy configuration invalid: host is required.")
        if not isinstance(self.account, str) or not self.account.strip():
            raise ValueError("Outbound proxy configuration invalid: account is required.")

        numeric_fields = (
            "connect_timeout",
            "monitor_interval",
            "restart_initial_delay",
            "restart_max_delay",
        )
        for field in numeric_fields:
            value = getattr(self, field)
            if isinstance(value, bool):
                raise ValueError(f"Outbound proxy configuration invalid: {field} must be positive.")
            try:
                normalized = float(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Outbound proxy configuration invalid: {field} must be positive.") from exc
            if not math.isfinite(normalized) or normalized <= 0:
                raise ValueError(f"Outbound proxy configuration invalid: {field} must be positive.")
            setattr(self, field, normalized)

        self._ssh_executable = shutil.which("ssh")
        if self._ssh_executable is None:
            raise RuntimeError("Outbound proxy configuration invalid: ssh executable was not found.")
        if importlib.util.find_spec("pproxy") is None:
            raise RuntimeError(
                "SSH HTTP outbound proxy requires pproxy; install ReMe with the 'core' extra: reme-ai[core].",
            )

    @staticmethod
    def _pick_free_port() -> int:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
            listener.bind((_LOOPBACK, 0))
            return int(listener.getsockname()[1])

    @classmethod
    def _pick_distinct_ports(cls) -> tuple[int, int]:
        socks_port = cls._pick_free_port()
        http_port = cls._pick_free_port()
        while http_port == socks_port:
            http_port = cls._pick_free_port()
        return socks_port, http_port

    async def _start_processes(self) -> None:
        await self._start_ssh()
        try:
            await self._start_bridge()
        except BaseException:
            await self._stop_process(self._ssh_process)
            self._ssh_process = None
            raise

    async def _start_ssh(self) -> None:
        assert self._ssh_executable is not None
        assert self._socks_port is not None
        destination = f"{self.account}@{self.host}"
        command = (
            self._ssh_executable,
            "-N",
            "-D",
            f"{_LOOPBACK}:{self._socks_port}",
            "-o",
            "BatchMode=yes",
            "-o",
            "ExitOnForwardFailure=yes",
            "-o",
            "StrictHostKeyChecking=accept-new",
            "-o",
            f"ConnectTimeout={max(1, math.ceil(self.connect_timeout))}",
            "-o",
            "LogLevel=ERROR",
            "--",
            destination,
        )
        self._ssh_process = await asyncio.create_subprocess_exec(
            *command,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            await self._wait_for_listener(
                self._ssh_process,
                self._socks_port,
                self.connect_timeout,
                "SSH proxy",
            )
        except asyncio.CancelledError:
            await self._stop_process(self._ssh_process)
            raise
        except BaseException as exc:
            await self._raise_start_error(self._ssh_process, exc, "SSH proxy exited before readiness")

    async def _start_bridge(self) -> None:
        assert self._socks_port is not None
        assert self._http_port is not None
        command = (
            sys.executable,
            "-m",
            "pproxy",
            "-l",
            f"http://{_LOOPBACK}:{self._http_port}",
            "-r",
            f"socks5://{_LOOPBACK}:{self._socks_port}",
        )
        self._bridge_process = await asyncio.create_subprocess_exec(
            *command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        try:
            await self._wait_for_listener(
                self._bridge_process,
                self._http_port,
                self.connect_timeout,
                "HTTP bridge",
            )
        except asyncio.CancelledError:
            await self._stop_process(self._bridge_process)
            raise
        except BaseException as exc:
            await self._raise_start_error(self._bridge_process, exc, "HTTP bridge exited before readiness")

    async def _raise_start_error(self, process: Process, cause: BaseException, prefix: str) -> None:
        detail = await self._read_process_output(process)
        await self._stop_process(process)
        if self._is_port_conflict(detail):
            raise _PortUnavailableError(f"{prefix}: {detail}") from cause
        suffix = f": {detail}" if detail else ""
        raise RuntimeError(f"{prefix}{suffix}") from cause

    @staticmethod
    def _is_port_conflict(detail: str) -> bool:
        lowered = detail.lower()
        return any(marker in lowered for marker in _PORT_CONFLICT_MARKERS)

    @staticmethod
    async def _wait_for_listener(process: Process, port: int, timeout: float, label: str) -> None:
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout
        while loop.time() < deadline:
            if process.returncode is not None:
                raise RuntimeError(f"{label} process exited with status {process.returncode}.")
            try:
                _reader, writer = await asyncio.open_connection(_LOOPBACK, port)
                writer.close()
                await writer.wait_closed()
                return
            except OSError:
                await asyncio.sleep(min(0.05, max(0.0, deadline - loop.time())))
        raise TimeoutError(f"{label} listener did not become ready on {_LOOPBACK}:{port} within {timeout:g}s.")

    async def _monitor_processes(self) -> None:
        attempt = 0
        stable_since = time.monotonic()
        while True:
            await asyncio.sleep(self.monitor_interval)
            if self._closing:
                return

            restart_ssh = self._ssh_process is None or self._ssh_process.returncode is not None
            restart_bridge = self._bridge_process is None or self._bridge_process.returncode is not None
            if not restart_ssh and not restart_bridge:
                if time.monotonic() - stable_since >= self.monitor_interval:
                    attempt = 0
                continue

            delay = min(self.restart_initial_delay * (2**attempt), self.restart_max_delay)
            self.logger.warning(
                f"Outbound proxy reconnecting ssh={restart_ssh} bridge={restart_bridge} delay={delay:g}s",
            )
            await asyncio.sleep(delay)
            if self._closing:
                return

            try:
                if restart_ssh:
                    await self._restart_ssh()
                if restart_bridge:
                    await self._restart_bridge()
            except Exception as exc:
                attempt += 1
                stable_since = time.monotonic()
                self.logger.error(f"Outbound proxy reconnect failed error={type(exc).__name__}: {exc}")
            else:
                stable_since = time.monotonic()

    async def _restart_ssh(self) -> None:
        await self._stop_process(self._ssh_process)
        self._ssh_process = None
        try:
            await self._start_ssh()
        except BaseException:
            await self._stop_process(self._ssh_process)
            self._ssh_process = None
            raise

    async def _restart_bridge(self) -> None:
        await self._stop_process(self._bridge_process)
        self._bridge_process = None
        try:
            await self._start_bridge()
        except BaseException:
            await self._stop_process(self._bridge_process)
            self._bridge_process = None
            raise

    async def _stop_processes(self) -> None:
        await self._stop_process(self._bridge_process)
        self._bridge_process = None
        await self._stop_process(self._ssh_process)
        self._ssh_process = None

    @staticmethod
    async def _stop_process(process: Process | None) -> None:
        if process is None:
            return
        if process.returncode is not None:
            await process.wait()
            return
        with contextlib.suppress(ProcessLookupError):
            process.terminate()
        try:
            await asyncio.wait_for(process.wait(), timeout=_PROCESS_CLOSE_TIMEOUT)
        except asyncio.TimeoutError:
            with contextlib.suppress(ProcessLookupError):
                process.kill()
            await process.wait()

    @staticmethod
    async def _read_process_output(process: Process) -> str:
        stream = process.stderr if process.stderr is not None else process.stdout
        if stream is None or process.returncode is None:
            return ""
        data = await stream.read(_PROCESS_OUTPUT_LIMIT + 1)
        truncated = len(data) > _PROCESS_OUTPUT_LIMIT
        text = data[:_PROCESS_OUTPUT_LIMIT].decode(errors="replace").strip()
        return f"{text}…" if truncated else text
