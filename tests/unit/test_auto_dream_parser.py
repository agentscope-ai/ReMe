"""Regression tests for structured Dream replies with reasoning wrappers."""

import json

import pytest

from reme.steps.evolve.dream.utils import parse_structured_reply


@pytest.mark.parametrize("strict", [False, True])
@pytest.mark.parametrize(
    "reasoning",
    [
        "<think>Reasoning mentions X: Y before the answer.</think>",
        '<think type="analysis">Reasoning mentions X: Y before the answer.</think>',
        "< thought >Reasoning mentions X: Y before the answer.</ thought >",
        "Reasoning mentions X: Y before the answer.</think>",
        "<think>Reasoning mentions X: Y before the answer.",
        "<think>Earlier sketch { has X: Y but no closing brace</think>",
    ],
)
def test_parse_structured_reply_extracts_json_after_reasoning_tags(strict, reasoning):
    """Reasoning envelopes must not hide valid JSON from strict or legacy callers."""
    expected = {"units": [{"name": "recovered", "summary": "supported"}]}
    reply = f"{reasoning}\n{json.dumps(expected)}"

    assert parse_structured_reply(reply, strict=strict) == expected


def test_parse_structured_reply_balances_json_around_braces_in_strings():
    """Balanced-span extraction ignores JSON delimiters inside escaped string values."""
    expected = {"units": [{"summary": 'literal } brace, ] bracket, and \\"quoted\\" text'}]}
    reply = f"<think>Reason: not the answer</think>\nResult: {json.dumps(expected)} trailing prose"

    assert parse_structured_reply(reply, strict=True) == expected


def test_parse_structured_reply_does_not_unwrap_json_arrays():
    """A nested object does not turn an array reply into a supported object reply."""
    reply = '[{"units": []}]'

    assert parse_structured_reply(reply) == {}
    with pytest.raises(ValueError, match="expected a JSON/YAML object"):
        parse_structured_reply(reply, strict=True)
