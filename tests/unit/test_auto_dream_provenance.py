"""CI-scope coverage for the auto_dream provenance assertion logic.

``tests/integration/test_auto_dream.py`` needs a live LLM endpoint, so its
assertions never run in CI. The provenance check lives in
``tests/integration/_provenance.py`` (kept dependency-light for exactly this
reason) and is exercised here against the offline string cases for every link
form the integrate prompt permits — and the forms it must reject.
"""

import sys
from pathlib import Path

import pytest

INTEGRATION_DIR = Path(__file__).resolve().parents[1] / "integration"
sys.path.insert(0, str(INTEGRATION_DIR))

# pylint: disable=wrong-import-position
from _provenance import targets_with_provenance  # noqa: E402

SOURCE = "daily/2026-05-28/auth-refactor/notes.md"


@pytest.mark.parametrize(
    ("body", "expected"),
    [
        # Bare list line (the pre-#428 form the old substring asserted).
        (f"- [[{SOURCE}]]", True),
        # Contextual sentence, no alias — the prompt's required form.
        (f"The session record at [[{SOURCE}]] motivated this change.", True),
        # Contextual sentence with the prompt's canonical alias example.
        (f"See [[{SOURCE}|the session record]] for the full discussion.", True),
        # Anchored link and anchored alias: the anchor is a separate regex
        # group, so the target still compares equal.
        (f"Decided in [[{SOURCE}#Decision]].", True),
        (f"See [[{SOURCE}#Decision|the decision]] for details.", True),
        # Embed form.
        (f"![[{SOURCE}]]", True),
        # Unterminated alias span: not a wikilink, must not count.
        (f"See [[{SOURCE}|the session record", False),
        # No provenance at all.
        ("No links here.", False),
        # Link present but to a different source target.
        ("Points elsewhere: [[daily/2026-05-28/auth-refactor/other.md]].", False),
    ],
)
def test_targets_with_provenance(body: str, expected: bool):
    """Every prompt-permitted link form counts; malformed or foreign links do not."""
    target_texts = {"digest/session-review.md": body}
    assert bool(targets_with_provenance(target_texts, SOURCE)) is expected


def test_targets_with_provenance_lists_only_matching_targets():
    """Only targets that actually link to the source are listed."""
    target_texts = {
        "digest/with-link.md": f"Derived from [[{SOURCE}|the session record]].",
        "digest/no-link.md": "Independent summary.",
    }
    assert targets_with_provenance(target_texts, SOURCE) == ["digest/with-link.md"]
