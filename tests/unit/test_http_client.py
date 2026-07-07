"""Tests for HTTP client display formatting."""

# pylint: disable=protected-access

from reme.components.client.http_client import HttpClient


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
