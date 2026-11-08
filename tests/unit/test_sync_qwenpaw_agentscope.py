"""Tests for the QwenPaw AgentScope dependency sync."""

import pytest

from scripts.sync_qwenpaw_agentscope import sync_pin

QWENPAW = '[project]\ndependencies = ["agentscope[model-ollama]==2.0.8"]\n'
REME = (
    "[project.optional-dependencies]\n"
    'as = ["agentscope[model-ollama]==2.0.7.post1"]\n'
    'decision = ["agentscope[classifier-jev]==2.0.7.post1"]\n'
)


def test_sync_pin_updates_both_reme_pins():
    """The model and decision extras must resolve to the same stable version."""
    version, updated = sync_pin(QWENPAW, REME)
    assert version == "2.0.8"
    assert updated == REME.replace("2.0.7.post1", "2.0.8")


def test_sync_pin_accepts_qwenpaw_classifier_extra():
    """Additional host extras do not change ReMe's independently selected extras."""
    upstream = '[project]\ndependencies = ["agentscope[model-ollama,classifier-jev]==2.0.11"]\n'
    version, updated = sync_pin(upstream, REME)
    assert version == "2.0.11"
    assert updated == REME.replace("2.0.7.post1", "2.0.11")


def test_sync_pin_preserves_already_aligned_manifest():
    """A repeated sync does not rewrite an aligned manifest."""
    current = REME.replace("2.0.7.post1", "2.0.8")
    assert sync_pin(QWENPAW, current) == ("2.0.8", current)


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
