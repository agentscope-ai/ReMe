"""Focused tests for TuShare source selection."""

# pylint: disable=protected-access

import sys
from types import SimpleNamespace

from reme.utils.tushare import create_tushare_api


def test_tushare_uses_configured_mirror_or_sdk_default(monkeypatch):
    """A mirror overrides the SDK endpoint while an empty setting leaves it unchanged."""
    api = SimpleNamespace(_DataApi__http_url="https://api.tushare.pro")
    monkeypatch.setitem(sys.modules, "tushare", SimpleNamespace(pro_api=lambda _token: api))

    monkeypatch.delenv("TUSHARE_MIRROR_URL", raising=False)
    assert create_tushare_api("token")._DataApi__http_url == "https://api.tushare.pro"
    assert api._DataApi__timeout == 600

    monkeypatch.setenv("TUSHARE_MIRROR_URL", "http://112.124.63.173:4000/tushare/")
    assert create_tushare_api("token")._DataApi__http_url == "http://112.124.63.173:4000/tushare"
