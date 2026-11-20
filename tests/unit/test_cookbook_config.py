"""Access restrictions for the cookbook application."""

from reme.config.config_parser import _load_config


def test_cookbook_dingtalk_sender_is_private(monkeypatch):
    """Sending workspace files to DingTalk is restricted to local jobs."""
    for name in ("DINGTALK_APP_KEY", "DINGTALK_APP_SECRET", "DINGTALK_ROBOT_CODE"):
        monkeypatch.setenv(name, "test-value")

    assert _load_config("cookbook")["jobs"]["dingtalk_send"]["enable_serve"] is False
