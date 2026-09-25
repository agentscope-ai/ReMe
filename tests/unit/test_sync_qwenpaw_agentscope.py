"""Tests for the QwenPaw AgentScope dependency sync."""

import pytest

from scripts.sync_qwenpaw_agentscope import sync_pin

QWENPAW = '[project]\ndependencies = ["agentscope[model-ollama]==2.0.8"]\n'
REME = '[project.optional-dependencies]\nas = ["agentscope[model-ollama]==2.0.7.post1"]\n'


def test_sync_pin_updates_only_reme_pin():
    """A stable QwenPaw pin replaces ReMe's exact pin."""
    version, updated = sync_pin(QWENPAW, REME)
    assert version == "2.0.8"
    assert updated == REME.replace("2.0.7.post1", "2.0.8")


@pytest.mark.parametrize(
    "upstream",
    [
        '[project]\ndependencies = ["agentscope[model-ollama]>=2.0.8"]\n',
        '[project]\ndependencies = ["agentscope[model-ollama]==2.0.9rc1"]\n',
        '[project]\ndependencies = ["other==1.0"]\n',
    ],
)
def test_sync_pin_rejects_missing_nonexact_or_prerelease_upstream_pin(upstream):
    """An absent or ambiguous QwenPaw pin never changes ReMe's manifest."""
    with pytest.raises(ValueError):
        sync_pin(upstream, REME)
