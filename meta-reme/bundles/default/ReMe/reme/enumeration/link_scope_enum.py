"""Link-scope enumeration: which edges to return from get_inlinks / get_outlinks."""

from enum import Enum


class LinkScopeEnum(str, Enum):
    """Which subset of edges a link query should return.

    A file-graph edge is **real** when both endpoints are indexed nodes,
    and **virtual** (dangling / pending) when one endpoint is a placeholder
    — a path referenced by a wikilink but never upserted, or upserted
    then deleted.
    """

    REAL = "real"

    VIRTUAL = "virtual"

    ALL = "all"
