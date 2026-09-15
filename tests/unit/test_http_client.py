"""Tests for HTTP client display formatting."""

# pylint: disable=protected-access

from reme.components.client.http_client import HttpClient
from reme.constants import REME_DEFAULT_HOST, REME_DEFAULT_PORT


def test_default_client_uses_loopback_address():
    """Clients connect to loopback when no service address is configured."""
    client = HttpClient()

    assert client.base_url == f"http://{REME_DEFAULT_HOST}:{REME_DEFAULT_PORT}"


def test_client_converts_wildcard_bind_address_to_loopback():
    """A service's wildcard bind address is not advertised as a destination."""
    client = HttpClient(host="0.0.0.0", port=8123)

    assert client.base_url == "http://127.0.0.1:8123"


def test_format_for_display_hides_metadata_by_default():
    """Response metadata is available structurally but not shown in normal CLI output."""
    client = HttpClient()
    text = '{"answer":"0.4.0.7","success":true,"metadata":{"version":"0.4.0.7"}}'

    assert client._format_for_display(text) == "0.4.0.7\n✅"


def test_format_for_display_shows_metadata_when_requested():
    """show_metadata=true opts into metadata display."""
    client = HttpClient(show_metadata=True)
    text = '{"answer":"0.4.0.7","success":true,"metadata":{"version":"0.4.0.7"}}'

    assert client._format_for_display(text) == '0.4.0.7\n✅ {"version": "0.4.0.7"}'
