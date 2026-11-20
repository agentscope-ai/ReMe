"""Additional review cases for final-answer selection and orphan reasoning."""

import json

import pytest

from reme.steps.evolve.dream.utils import parse_structured_reply


def _draft() -> str:
    """Return a schema-valid draft that must not override the final reply."""
    return json.dumps(
        {
            "units": [
                {
                    "name": "discarded-draft",
                    "bucket": "wiki",
                    "summary": "A preliminary draft rejected after checking its source.",
                    "paths": ["daily/2026-11-18/notes.md"],
                },
            ],
        },
    )


@pytest.mark.parametrize("strict", [False, True])
def test_bare_final_answer_overrides_earlier_fenced_draft(strict):
    """A final object outside fences must outrank a preceding fenced example."""
    reply = f'Draft:\n```json\n{_draft()}\n```\nFinal:\n{{"units": []}}'

    assert parse_structured_reply(reply, strict=strict) == {"units": []}


@pytest.mark.parametrize("strict", [False, True])
def test_bare_final_array_does_not_promote_earlier_fenced_draft(strict):
    """Reject a final array rather than accepting an earlier fenced object."""
    reply = f'Draft:\n```json\n{_draft()}\n```\nFinal:\n[{{"units": []}}]'

    if strict:
        with pytest.raises(ValueError, match="expected a JSON/YAML object"):
            parse_structured_reply(reply, strict=True)
    else:
        assert parse_structured_reply(reply) == {}


@pytest.mark.parametrize("strict", [False, True])
def test_final_json_survives_unterminated_draft_string_after_orphan_marker(strict):
    """An incomplete reasoning sketch cannot hide a complete final object."""
    reply = '<think>Previous sketch {"summary": "unfinished text\nFinal:\n{"units": []}'

    assert parse_structured_reply(reply, strict=strict) == {"units": []}


@pytest.mark.parametrize("strict", [False, True])
@pytest.mark.parametrize(
    "draft, final, expected",
    [
        ('{"draft": true}', '```json\n{"units": []}\n```', {"units": []}),
        ('```yaml\nunits:\n  - {"name": "draft"}\n```', '{"units": []}', {"units": []}),
        ('{"draft": true}', '```yaml\nunits:\n  - {"name": "final"}\n```', {"units": [{"name": "final"}]}),
        ('```json\n{"draft": true}\n```', 'units:\n  - {"name": "final"}', {"units": [{"name": "final"}]}),
    ],
)
def test_final_region_preserves_mapping_roots(strict, draft, final, expected):
    """Fenced and bare final mappings use response order and preserve YAML roots."""
    reply = f"Draft follows.\n{draft}\n{final}"
    assert parse_structured_reply(reply, strict=strict) == expected


@pytest.mark.parametrize("strict", [False, True])
@pytest.mark.parametrize("final", ['[{"units": []}]', '```json\n[{"units": []}]\n```'])
def test_final_array_blocks_draft_fallback(strict, final):
    """A final array remains unsupported regardless of fence placement."""
    reply = f'Draft follows.\n{{"draft": true}}\nFinal follows.\n{final}'
    if strict:
        with pytest.raises(ValueError, match="expected a JSON/YAML object"):
            parse_structured_reply(reply, strict=True)
    else:
        assert parse_structured_reply(reply) == {}


@pytest.mark.parametrize("strict", [False, True])
def test_recovery_keeps_json_looking_strings_opaque(strict):
    """Quoted containers are payloads, not independent answer candidates."""
    expected = {"units": [{"summary": 'Keep {"draft": true} and [1, 2] literally.'}]}
    reply = f"Result follows.\n{json.dumps(expected)}\nTrailing prose."
    assert parse_structured_reply(reply, strict=strict) == expected


@pytest.mark.parametrize("strict", [False, True])
def test_orphan_draft_cannot_hide_final_array(strict):
    """Malformed reasoning must not expose an object nested in the final list."""
    reply = '<think>Previous sketch {"summary": "unfinished text\nFinal:\n[{"units": []}]'
    if strict:
        with pytest.raises(ValueError, match="expected a JSON/YAML object"):
            parse_structured_reply(reply, strict=True)
    else:
        assert parse_structured_reply(reply) == {}


@pytest.mark.parametrize("strict", [False, True])
def test_cleaned_yaml_block_scalar_keeps_literal_fence(strict):
    """A complete YAML answer keeps Markdown examples inside its block scalar."""
    reply = '<think>Ignore draft.</think>\nnote: |\n  ```json\n  {"literal": true}\n  ```\n'
    expected = {"note": '```json\n{"literal": true}\n```\n'}
    assert parse_structured_reply(reply, strict=strict) == expected


@pytest.mark.parametrize("strict", [False, True])
def test_recovered_json_string_keeps_literal_fence(strict):
    """Fences within a valid JSON string are not Markdown answer boundaries."""
    expected = {"note": '```json\n{"literal": true}\n```'}
    reply = f"Result follows.\n{json.dumps(expected)}"
    assert parse_structured_reply(reply, strict=strict) == expected


def test_cleaned_final_array_reports_object_requirement():
    """Array diagnostics must not reuse a preceding envelope's YAML error."""
    with pytest.raises(ValueError, match="expected a JSON/YAML object"):
        parse_structured_reply('<think>X: Y</think>\n[{"units": []}]', strict=True)
