"""Regression tests for BaseClient service address resolution."""

import pytest

from reme.components.client.http_client import HttpClient
from reme.components.client.mcp_client import MCPClient
from reme.constants import REME_DEFAULT_HOST, REME_DEFAULT_PORT, REME_SERVICE_INFO


@pytest.fixture(autouse=True)
def clear_service_info(monkeypatch):
    """Address resolution runs without service discovery unless opted in."""
    monkeypatch.delenv(REME_SERVICE_INFO, raising=False)


def test_explicit_port_only_keeps_port_and_defaults_host():
    """An explicit port is not discarded when the host is omitted."""
    assert HttpClient(port=8123).base_url == f"http://{REME_DEFAULT_HOST}:8123"

    client = MCPClient(port=8123)
    assert client.host == REME_DEFAULT_HOST
    assert client.port == 8123


def test_explicit_host_only_keeps_host_and_defaults_port():
    """An explicit host is not discarded when the port is omitted."""
    assert HttpClient(host="192.0.2.10").base_url == f"http://192.0.2.10:{REME_DEFAULT_PORT}"

    client = MCPClient(host="192.0.2.10")
    assert client.host == "192.0.2.10"
    assert client.port == REME_DEFAULT_PORT


def test_ipv6_host_builds_a_valid_http_base_url():
    """IPv6 literals are bracketed by URL construction rather than string interpolation."""
    assert HttpClient(host="::1", port=8123).base_url == "http://[::1]:8123"


def test_explicit_fields_win_over_service_discovery(monkeypatch):
    """Discovery fills only the missing field; explicit fields are kept."""
    monkeypatch.setenv(REME_SERVICE_INFO, '{"host": "10.0.0.1", "port": 9000}')

    assert HttpClient(port=8123).base_url == "http://10.0.0.1:8123"
    assert HttpClient(host="192.0.2.10").base_url == "http://192.0.2.10:9000"
    assert HttpClient(host="192.0.2.10", port=8123).base_url == "http://192.0.2.10:8123"

    client = MCPClient(port=8123)
    assert client.host == "10.0.0.1"
    assert client.port == 8123

    client = MCPClient(host="192.0.2.10")
    assert client.host == "192.0.2.10"
    assert client.port == 9000


def test_service_discovery_fills_both_fields_when_nothing_is_explicit(monkeypatch):
    """Without explicit arguments the discovered address is used as before."""
    monkeypatch.setenv(REME_SERVICE_INFO, '{"host": "10.0.0.1", "port": 9000}')

    assert HttpClient().base_url == "http://10.0.0.1:9000"

    client = MCPClient()
    assert client.host == "10.0.0.1"
    assert client.port == 9000


def test_ipv6_service_discovery_builds_a_valid_http_base_url(monkeypatch):
    """Discovered IPv6 service addresses use the same URL-safe construction."""
    monkeypatch.setenv(REME_SERVICE_INFO, '{"host": "::1", "port": 9000}')

    assert HttpClient().base_url == "http://[::1]:9000"


def test_invalid_service_info_keeps_explicit_fields_and_defaults_the_rest(monkeypatch):
    """An undecodable discovery payload still preserves explicit fields."""
    monkeypatch.setenv(REME_SERVICE_INFO, "not-json")

    assert HttpClient(port=8123).base_url == f"http://{REME_DEFAULT_HOST}:8123"
    assert HttpClient(host="192.0.2.10").base_url == f"http://192.0.2.10:{REME_DEFAULT_PORT}"
    assert HttpClient().base_url == f"http://{REME_DEFAULT_HOST}:{REME_DEFAULT_PORT}"


def test_incomplete_service_info_keeps_explicit_fields_and_defaults_the_rest(monkeypatch):
    """A discovery payload missing host/port keys preserves explicit fields."""
    monkeypatch.setenv(REME_SERVICE_INFO, '{"host": "10.0.0.1"}')

    assert HttpClient(port=8123).base_url == f"http://{REME_DEFAULT_HOST}:8123"
    assert HttpClient().base_url == f"http://{REME_DEFAULT_HOST}:{REME_DEFAULT_PORT}"


def test_mcp_stdio_transport_skips_network_address_resolution():
    """The stdio transport never resolves a network service address."""
    client = MCPClient(transport="stdio", command="python", args=["server.py"])

    assert not hasattr(client, "host")
    assert not hasattr(client, "port")
