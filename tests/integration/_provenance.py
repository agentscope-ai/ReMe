"""Provenance-link predicate shared by the auto_dream integration assertion and its CI unit coverage.

Kept dependency-light (stdlib + ``WikilinkHandler`` only) so ``tests/unit``
can import it without LLM credentials or the integration workspace fixture.
"""

from reme.utils.documents.wikilink_handler import WikilinkHandler


def targets_with_provenance(target_texts: dict[str, str], source_path: str) -> list[str]:
    """Return the targets whose text links to ``source_path`` at least once.

    Uses the shared wikilink grammar (``WikilinkHandler.iter_matches``) rather
    than substring matching, so every form the integrate prompt permits counts:
    bare links, contextual sentences, aliases, embeds, and anchored links (the
    anchor lives in its own regex group, so the target still compares equal).
    Unterminated ``[[...`` spans the parser rejects do not count, and links to
    any other target do not either.
    """
    return [
        rel
        for rel, text in target_texts.items()
        if any(match.target == source_path for match in WikilinkHandler.iter_matches(text))
    ]
