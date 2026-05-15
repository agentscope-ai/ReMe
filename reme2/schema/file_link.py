"""FileLink — typed wikilink between vault files.

One type, two states (the value of ``path`` distinguishes them):

    * **pre-resolution** — ``path`` holds the raw wikilink target as
      written, e.g. ``"Foo"`` or ``"topics/Bar"``.
    * **resolved**       — ``path`` holds the vault-relative path
      file_graph stores, e.g. ``"topics/Foo/Foo.md"``.

The extractor (``utils.link_parser.iter_links``) produces the
pre-resolution form from body text. The batch resolver
(``utils.link_parser.resolve_links``, or one-shot ``text_to_links``)
delegates to ``utils.path_resolver.resolve`` to rewrite ``path`` to
the resolved form, expanding short-path ambiguity into one
``FileLink`` per candidate.

file_graph trusts ``link.path`` directly for adjacency: it only ever
stores resolved links. The pre-resolution form is internal pipeline
plumbing.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class FileLink(BaseModel):
    """Typed wikilink — ``(path, anchor, predicate)``.

    Single type for both states (see module docstring): ``path`` is
    a raw wikilink target before resolution, a vault-relative resolved
    path after.

    Fields:
        path       wikilink target. Raw target text in pre-resolution
                   form (e.g. ``"Foo"``, ``"topics/Bar"``);
                   vault-relative resolved path in stored form
                   (e.g. ``"topics/Foo/Foo.md"``).
        anchor     heading or block anchor (text after ``#`` in the
                   wikilink). Pass-through across resolution.
        predicate  Dataview-style typed-link predicate; ``None`` for bare.
    """

    model_config = ConfigDict(extra="forbid")

    path: str = Field(
        ...,
        description=(
            "Wikilink target. Pre-resolution: the raw target as written "
            "(e.g. 'Foo'). Resolved: the vault-relative path file_graph "
            "stores. Short-path ambiguity is resolved BEFORE construction "
            "of the resolved form by emitting one FileLink per candidate."
        ),
    )
    anchor: str | None = Field(
        default=None,
        description="Heading or block anchor (text after '#'). None if absent.",
    )
    predicate: str | None = Field(
        default=None,
        description="Typed-link predicate (Dataview-style). None for bare [[X]].",
    )
