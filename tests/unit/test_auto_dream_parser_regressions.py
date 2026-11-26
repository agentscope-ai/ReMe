"""Regression coverage for structured Dream replies containing JSON and reasoning text."""

import json

import pytest

from reme.steps.evolve.dream.utils import parse_structured_reply


def _receipt(summary: str) -> dict:
    """Build a receipt with all required extraction fields and a source path."""
    return {
        "units": [
            {
                "name": "parser-contract",
                "bucket": "wiki",
                "summary": summary,
                "paths": ["daily/2026-11-16/notes.md"],
            },
        ],
    }


@pytest.mark.parametrize("strict", [False, True])
def test_parse_structured_reply_uses_final_json_after_reasoning(strict):
    """A rejected draft inside a reasoning block must not become accepted memory."""
    draft = _receipt("A proposed memory that is explicitly rejected after checking the source.")
    expected = {"units": []}
    reply = (
        f"<think>First draft:\n{json.dumps(draft)}\n"
        "Reject this draft: the source supports no durable memory.</think>\n"
        f"{json.dumps(expected)}"
    )

    assert parse_structured_reply(reply, strict=strict) == expected


@pytest.mark.parametrize("strict", [False, True])
def test_parse_structured_reply_preserves_reasoning_tags_in_json_strings(strict):
    """Reasoning markers that are part of a JSON string are literal memory content."""
    expected = _receipt("Keep <think>reasoning</think> and <thought>analysis</thought> as literal examples.")

    assert parse_structured_reply(json.dumps(expected), strict=strict) == expected


@pytest.mark.parametrize("strict", [False, True])
def test_parse_structured_reply_preserves_yaml_root_with_json_mapping(strict):
    """A JSON flow mapping inside valid YAML must not replace the root receipt."""
    expected = _receipt("A YAML sequence may contain a JSON flow mapping.")
    reply = f"units:\n  - {json.dumps(expected['units'][0])}\n"

    assert parse_structured_reply(reply, strict=strict) == expected


@pytest.mark.parametrize("strict", [False, True])
def test_reasoning_envelope_preserves_literal_tags_in_answer(strict):
    """Recovery must not rewrite literal tags in the selected answer either."""
    expected = _receipt('Keep <think>literal</think> and "quoted" text.')
    reply = f'<think type="analysis">Discard this draft.</think>\n{json.dumps(expected)}'
    assert parse_structured_reply(reply, strict=strict) == expected


@pytest.mark.parametrize("strict", [False, True])
def test_final_answer_wins_over_larger_unwrapped_draft(strict):
    """Recovery chooses the final answer, not an earlier larger JSON example."""
    draft = _receipt("A long preliminary draft, not the final answer.")
    reply = f'Draft follows.\n{json.dumps(draft)}\nFinal follows.\n{{"units": []}}'
    assert parse_structured_reply(reply, strict=strict) == {"units": []}


@pytest.mark.parametrize(
    "reply",
    [
        '<think>{"units": [{"summary": "draft"}]}</think>',
        '<think>draft</think>\n[{"units": []}]',
        'Draft follows.\n{"units": []}\nFinal follows.\n[{"units": []}]',
    ],
)
def test_recovery_does_not_accept_reasoning_only_or_final_arrays(reply):
    """Absent or unsupported final answers cannot promote an earlier draft."""
    assert parse_structured_reply(reply) == {}
    with pytest.raises(ValueError):
        parse_structured_reply(reply, strict=True)


@pytest.mark.parametrize("strict", [False, True])
@pytest.mark.parametrize(
    "scalar",
    [
        "'Keep <think>literal</think> tags'",
        "|\n  Keep <think>literal</think> tags",
    ],
)
def test_fenced_yaml_preserves_literal_reasoning_tags(strict, scalar):
    """Reasoning markers in a fenced YAML document belong to its payload."""
    import yaml

    document = f"note: {scalar}\n"
    reply = f"<think>Ignore draft.</think>\n```yaml\n{document}```"
    assert parse_structured_reply(reply, strict=strict) == yaml.safe_load(document)


def test_fenced_reasoning_draft_is_not_an_answer():
    """A fence inside a reasoning envelope must not escape draft exclusion."""
    reply = '<think>```json\n{"units": [{"summary": "draft"}]}\n```</think>\n{"units": []}'
    assert parse_structured_reply(reply, strict=True) == {"units": []}


@pytest.mark.parametrize("strict", [False, True])
@pytest.mark.parametrize(
    "scalar",
    [
        "Keep <think>literal</think> tags",
        "'Keep <think>literal</think> tags'",
        "|\n  Keep <think>literal</think> tags",
    ],
)
def test_unfenced_yaml_after_reasoning_preserves_literal_tags(strict, scalar):
    """Envelope recovery must not reinterpret YAML scalar contents as markers."""
    import yaml

    document = f"note: {scalar}\n"
    reply = f"<think>Ignore draft.</think>\n{document}"
    assert parse_structured_reply(reply, strict=strict) == yaml.safe_load(document)


@pytest.mark.parametrize("strict", [False, True])
@pytest.mark.parametrize("prefix", ["", "<think>Discard this draft.</think>\n"])
def test_fenced_yaml_preserves_root_with_json_flow_mapping(strict, prefix):
    """A fenced YAML receipt retains its root around nested JSON mappings."""
    expected = _receipt("A flow mapping is nested, not the whole receipt.")
    document = f"units:\n  - {json.dumps(expected['units'][0])}\n"
    reply = f"{prefix}```yaml\n{document}```"
    assert parse_structured_reply(reply, strict=strict) == expected


@pytest.mark.parametrize("strict", [False, True])
def test_final_fenced_yaml_wins_over_earlier_json_draft(strict):
    """Final complete YAML is authoritative over a preceding fenced draft."""
    reply = '```json\n{"draft": true}\n```\n```yaml\nunits:\n  - {"name": "final"}\n```'
    assert parse_structured_reply(reply, strict=strict) == {"units": [{"name": "final"}]}


@pytest.mark.parametrize("strict", [False, True])
def test_final_fenced_array_does_not_promote_earlier_draft(strict):
    """An unsupported final list cannot fall back to an earlier object."""
    reply = '```json\n{"draft": true}\n```\n```json\n[{"units": []}]\n```'
    if strict:
        with pytest.raises(ValueError, match="expected a JSON/YAML object"):
            parse_structured_reply(reply, strict=True)
    else:
        assert parse_structured_reply(reply) == {}
