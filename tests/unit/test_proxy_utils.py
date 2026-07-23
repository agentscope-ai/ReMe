"""Tests for the optional SSH-backed download proxy."""

import asyncio

import pytest

from reme.utils import proxy_utils


def test_get_ssh_proxy_config_requires_both_values(monkeypatch):
    """A partial SSH destination should fail before starting a process."""
    monkeypatch.setenv(proxy_utils.REME_PROXY_IP, "proxy.example.com")
    monkeypatch.delenv(proxy_utils.REME_PROXY_ACCOUNT, raising=False)

    with pytest.raises(ValueError, match="REME_PROXY_ACCOUNT"):
        proxy_utils.get_ssh_proxy_config()


def test_get_ssh_proxy_config_reads_destination(monkeypatch):
    """The two environment variables form one OpenSSH destination."""
    monkeypatch.setenv(proxy_utils.REME_PROXY_IP, "proxy.example.com")
    monkeypatch.setenv(proxy_utils.REME_PROXY_ACCOUNT, "reme-user")

    config = proxy_utils.get_ssh_proxy_config()

    assert config is not None
    assert config.destination == "reme-user@proxy.example.com"


@pytest.mark.asyncio
async def test_ssh_socks_proxy_starts_and_stops_forwarder(monkeypatch):
    """The context starts a dynamic forward and terminates it on exit."""
    monkeypatch.setenv(proxy_utils.REME_PROXY_IP, "proxy.example.com")
    monkeypatch.setenv(proxy_utils.REME_PROXY_ACCOUNT, "reme-user")
    monkeypatch.setattr(proxy_utils, "_pick_free_port", lambda: 43123)
    commands: list[tuple[str, ...]] = []

    class FakeProcess:
        """Minimal asyncio subprocess stand-in for lifecycle assertions."""

        returncode = None
        stderr = None
        terminated = False

        def terminate(self):
            """Record graceful termination."""
            self.terminated = True

        async def wait(self):
            """Simulate a process that exits successfully."""
            self.returncode = 0
            return 0

    process = FakeProcess()

    async def fake_create_subprocess_exec(*args, **_kwargs):
        commands.append(args)
        return process

    async def fake_wait_for_proxy(actual_process, port, timeout):
        assert actual_process is process
        assert port == 43123
        assert timeout == 7.0

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_create_subprocess_exec)
    monkeypatch.setattr(proxy_utils, "_wait_for_proxy", fake_wait_for_proxy)

    async with proxy_utils.ssh_socks_proxy(connect_timeout=7.0) as proxy:
        assert proxy == "socks5://127.0.0.1:43123"

    assert process.terminated is True
    assert commands[0][-2:] == ("--", "reme-user@proxy.example.com")


@pytest.mark.asyncio
async def test_ssh_socks_proxy_is_noop_without_configuration(monkeypatch):
    """An absent proxy configuration preserves direct networking."""
    monkeypatch.delenv(proxy_utils.REME_PROXY_IP, raising=False)
    monkeypatch.delenv(proxy_utils.REME_PROXY_ACCOUNT, raising=False)

    async with proxy_utils.ssh_socks_proxy() as proxy:
        assert proxy is None
