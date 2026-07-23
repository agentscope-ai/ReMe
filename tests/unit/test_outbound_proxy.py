"""Tests for application-scoped outbound proxy components."""

# pylint: disable=missing-function-docstring,protected-access

import asyncio
import os
import sys
from collections.abc import Callable

import pytest

from reme.application import Application
from reme.components import R
from reme.components.outbound_proxy import FixedHttpOutboundProxy, SshHttpOutboundProxy
from reme.enumeration import ComponentEnum


class FakeProcess:
    """Minimal asyncio subprocess stand-in with observable shutdown."""

    def __init__(self, label: str, events: list[str]) -> None:
        self.label = label
        self.events = events
        self.returncode: int | None = None
        self.stdout = asyncio.StreamReader()
        self.stderr = asyncio.StreamReader()
        self._finished = asyncio.Event()

    async def wait(self) -> int:
        await self._finished.wait()
        assert self.returncode is not None
        return self.returncode

    def terminate(self) -> None:
        self.events.append(f"terminate:{self.label}")
        self.exit(-15)

    def kill(self) -> None:
        self.events.append(f"kill:{self.label}")
        self.exit(-9)

    def exit(self, returncode: int, output: str = "") -> None:
        if self.returncode is not None:
            return
        self.returncode = returncode
        encoded = output.encode()
        self.stdout.feed_data(encoded)
        self.stdout.feed_eof()
        self.stderr.feed_data(encoded)
        self.stderr.feed_eof()
        self._finished.set()


async def _ready_listener(*_args) -> None:
    return None


async def _wait_until(predicate: Callable[[], bool], timeout: float = 1.0) -> None:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while not predicate():
        if loop.time() >= deadline:
            raise AssertionError("condition did not become true")
        await asyncio.sleep(0.005)


def test_outbound_proxy_backends_are_registered() -> None:
    assert R.get(ComponentEnum.OUTBOUND_PROXY, "fixed_http") is FixedHttpOutboundProxy
    assert R.get(ComponentEnum.OUTBOUND_PROXY, "ssh_http") is SshHttpOutboundProxy


@pytest.mark.asyncio
async def test_application_builds_and_manages_fixed_http_proxy(tmp_path) -> None:
    app = Application(
        workspace_dir=str(tmp_path),
        enable_logo=False,
        log_to_console=False,
        log_to_file=False,
        service={"backend": "cli"},
        components={
            "outbound_proxy": {
                "default": {
                    "backend": "fixed_http",
                    "url": "http://127.0.0.1:18080",
                },
            },
        },
    )

    component = app.context.components[ComponentEnum.OUTBOUND_PROXY]["default"]
    assert isinstance(component, FixedHttpOutboundProxy)

    await app.start()
    assert component.http_url == "http://127.0.0.1:18080"
    await app.close()

    with pytest.raises(RuntimeError, match="start the component"):
        _ = component.endpoint


@pytest.mark.asyncio
async def test_fixed_http_publishes_endpoint_and_merges_environment(monkeypatch) -> None:
    monkeypatch.setenv("HTTP_PROXY", "http://ambient.example:8080")
    component = FixedHttpOutboundProxy(url="http://127.0.0.1:18080")
    base = {"CUSTOM": "value", "NO_PROXY": "example.com"}

    with pytest.raises(RuntimeError, match="start the component"):
        _ = component.http_url

    await component.start()
    merged = component.merge_environment(base)

    assert component.http_url == "http://127.0.0.1:18080"
    assert base == {"CUSTOM": "value", "NO_PROXY": "example.com"}
    assert os.environ["HTTP_PROXY"] == "http://ambient.example:8080"
    assert merged["CUSTOM"] == "value"
    for key in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
        assert merged[key] == component.http_url
    assert merged["NO_PROXY"] == "127.0.0.1,localhost,::1"
    assert merged["no_proxy"] == "127.0.0.1,localhost,::1"

    await component.close()
    with pytest.raises(RuntimeError, match="start the component"):
        _ = component.endpoint


@pytest.mark.parametrize(
    ("url", "message"),
    [
        ("", "must use http"),
        ("https://proxy.example:8080", "must use http"),
        ("http://proxy.example", "include host and port"),
        ("http://user@proxy.example:8080", "must not contain userinfo"),
        ("http://proxy.example:8080?mode=x", "must not contain query or fragment"),
        ("http://proxy.example:8080#fragment", "must not contain query or fragment"),
        ("http://proxy.example:not-a-port", "malformed"),
    ],
)
@pytest.mark.asyncio
async def test_fixed_http_rejects_invalid_urls(url: str, message: str) -> None:
    component = FixedHttpOutboundProxy(url=url)

    with pytest.raises(ValueError, match=message):
        await component.start()

    assert component.is_started is False
    with pytest.raises(RuntimeError, match="start the component"):
        _ = component.endpoint


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"host": "", "account": "agent"}, "host is required"),
        ({"host": "proxy.example", "account": ""}, "account is required"),
        ({"host": "proxy.example", "account": "agent", "connect_timeout": 0}, "connect_timeout"),
        ({"host": "proxy.example", "account": "agent", "monitor_interval": -1}, "monitor_interval"),
        ({"host": "proxy.example", "account": "agent", "restart_initial_delay": "bad"}, "restart_initial_delay"),
        ({"host": "proxy.example", "account": "agent", "restart_max_delay": float("inf")}, "restart_max_delay"),
    ],
)
@pytest.mark.asyncio
async def test_ssh_http_validates_configuration(kwargs: dict, message: str) -> None:
    component = SshHttpOutboundProxy(**kwargs)

    with pytest.raises(ValueError, match=message):
        await component.start()


@pytest.mark.asyncio
async def test_ssh_http_requires_ssh_executable(monkeypatch) -> None:
    monkeypatch.setattr("reme.components.outbound_proxy.ssh_http.shutil.which", lambda _name: None)
    component = SshHttpOutboundProxy(host="proxy.example", account="agent")

    with pytest.raises(RuntimeError, match="ssh executable was not found"):
        await component.start()


@pytest.mark.asyncio
async def test_ssh_http_starts_expected_commands_and_closes_bridge_first(monkeypatch) -> None:
    events: list[str] = []
    commands: list[tuple[str, ...]] = []
    processes: list[FakeProcess] = []

    async def fake_spawn(*command, **kwargs):
        assert kwargs
        commands.append(command)
        label = "ssh" if command[0] == "/usr/bin/ssh" else "bridge"
        process = FakeProcess(label, events)
        processes.append(process)
        return process

    monkeypatch.setattr("reme.components.outbound_proxy.ssh_http.shutil.which", lambda _name: "/usr/bin/ssh")
    monkeypatch.setattr("reme.components.outbound_proxy.ssh_http.asyncio.create_subprocess_exec", fake_spawn)
    component = SshHttpOutboundProxy(host="proxy.example", account="agent")
    monkeypatch.setattr(component, "_pick_distinct_ports", lambda: (43123, 43124))
    monkeypatch.setattr(component, "_wait_for_listener", _ready_listener)

    await component.start()

    assert component.http_url == "http://127.0.0.1:43124"
    assert commands[0] == (
        "/usr/bin/ssh",
        "-N",
        "-D",
        "127.0.0.1:43123",
        "-o",
        "BatchMode=yes",
        "-o",
        "ExitOnForwardFailure=yes",
        "-o",
        "StrictHostKeyChecking=accept-new",
        "-o",
        "ConnectTimeout=10",
        "-o",
        "LogLevel=ERROR",
        "--",
        "agent@proxy.example",
    )
    assert commands[1] == (
        sys.executable,
        "-m",
        "pproxy",
        "-l",
        "http://127.0.0.1:43124",
        "-r",
        "socks5://127.0.0.1:43123",
    )

    await component.close()

    assert events == ["terminate:bridge", "terminate:ssh"]
    assert all(process.returncode == -15 for process in processes)


@pytest.mark.asyncio
async def test_ssh_http_cleans_up_when_initial_readiness_fails(monkeypatch) -> None:
    events: list[str] = []
    process = FakeProcess("ssh", events)

    async def fake_spawn(*_command, **_kwargs):
        return process

    async def fail_readiness(*_args):
        raise TimeoutError("not ready")

    monkeypatch.setattr("reme.components.outbound_proxy.ssh_http.shutil.which", lambda _name: "/usr/bin/ssh")
    monkeypatch.setattr("reme.components.outbound_proxy.ssh_http.asyncio.create_subprocess_exec", fake_spawn)
    component = SshHttpOutboundProxy(host="proxy.example", account="agent")
    monkeypatch.setattr(component, "_pick_distinct_ports", lambda: (43123, 43124))
    monkeypatch.setattr(component, "_wait_for_listener", fail_readiness)

    with pytest.raises(RuntimeError, match="SSH proxy exited before readiness"):
        await component.start()

    assert events == ["terminate:ssh"]
    assert component.is_started is False
    with pytest.raises(RuntimeError, match="start the component"):
        _ = component.endpoint


@pytest.mark.asyncio
async def test_ssh_http_reselects_ports_only_before_endpoint_is_published(monkeypatch) -> None:
    events: list[str] = []
    commands: list[tuple[str, ...]] = []
    selected_ports = iter(((43123, 43124), (43125, 43126)))

    async def fake_spawn(*command, **_kwargs):
        commands.append(command)
        label = "ssh" if command[0] == "/usr/bin/ssh" else "bridge"
        process = FakeProcess(label, events)
        if label == "ssh" and len(commands) == 1:
            process.exit(255, "bind [127.0.0.1]:43123: Address already in use")
        return process

    async def fake_readiness(process, *_args):
        if process.returncode is not None:
            raise RuntimeError("process exited")

    monkeypatch.setattr("reme.components.outbound_proxy.ssh_http.shutil.which", lambda _name: "/usr/bin/ssh")
    monkeypatch.setattr("reme.components.outbound_proxy.ssh_http.asyncio.create_subprocess_exec", fake_spawn)
    component = SshHttpOutboundProxy(host="proxy.example", account="agent")
    monkeypatch.setattr(component, "_pick_distinct_ports", lambda: next(selected_ports))
    monkeypatch.setattr(component, "_wait_for_listener", fake_readiness)

    await component.start()

    assert component.http_url == "http://127.0.0.1:43126"
    assert [command[0] for command in commands] == ["/usr/bin/ssh", "/usr/bin/ssh", sys.executable]
    await component.close()


@pytest.mark.asyncio
async def test_ssh_http_monitor_restarts_on_original_ports(monkeypatch) -> None:
    events: list[str] = []
    commands: list[tuple[str, ...]] = []
    processes: list[FakeProcess] = []

    async def fake_spawn(*command, **_kwargs):
        commands.append(command)
        label = "ssh" if command[0] == "/usr/bin/ssh" else "bridge"
        process = FakeProcess(label, events)
        processes.append(process)
        return process

    monkeypatch.setattr("reme.components.outbound_proxy.ssh_http.shutil.which", lambda _name: "/usr/bin/ssh")
    monkeypatch.setattr("reme.components.outbound_proxy.ssh_http.asyncio.create_subprocess_exec", fake_spawn)
    component = SshHttpOutboundProxy(
        host="proxy.example",
        account="agent",
        monitor_interval=0.005,
        restart_initial_delay=0.005,
    )
    monkeypatch.setattr(component, "_pick_distinct_ports", lambda: (43123, 43124))
    monkeypatch.setattr(component, "_wait_for_listener", _ready_listener)
    await component.start()
    endpoint = component.endpoint

    processes[0].exit(7)
    await _wait_until(lambda: len(processes) == 3)

    assert component.endpoint is endpoint
    assert commands[2][0] == "/usr/bin/ssh"
    assert "127.0.0.1:43123" in commands[2]
    assert sum(command[0] == sys.executable for command in commands) == 1

    processes[1].exit(8)
    await _wait_until(lambda: len(processes) == 4)

    assert component.endpoint is endpoint
    assert commands[3] == (
        sys.executable,
        "-m",
        "pproxy",
        "-l",
        "http://127.0.0.1:43124",
        "-r",
        "socks5://127.0.0.1:43123",
    )

    await component.close()
