"""MCP step shells — the @R.register classes the model invokes as MCP tools.

Three groups:

    Hot-write primitives that bypass services and land on MFS directly:
        sync           — idempotent event-folder upsert (event log)
        memory_toolkit — memory_create / update / property_update /
                         rename / delete / archive / get / list /
                         links / backlinks / resolve_wikilink /
                         count_tokens. Each step lives in
                         `reme2.memory.memory_toolkit` and exposes
                         BOTH an `execute()` (this MCP path) and a
                         same-named class method (the agent toolkit
                         path used by the Ingestor's ReActAgent).

    Service delegates (thin shells over the three memory services):
        memory_retriever — memory_search + memory_graph_search; delegate
                           to the Retriever component
                           (`reme2.memory.retriever`).
        memory_lint      — read-only projection of the Maintainer's lint
                           findings (`reme2.memory.maintainer`).

Topic creation is owned by the Ingestor (`reme2.memory.ingestor`) — its
LLM-driven R-M-W loop decides when a new topic is warranted, applies the
schema preset, and routes the write through `memory_create`. There is no
separate `topic_create` MCP tool.

The Ingestor's MCP face (`ingest`) is registered from
`reme2.memory.ingestor`. Importing this package triggers all the step
registrations hosted here and in `reme2.memory.memory_toolkit`.
"""

from ...memory import memory_toolkit  # noqa: F401  -- triggers @R.register for memory_*
from . import memory_lint  # noqa: F401  -- triggers @R.register for memory_lint
from . import memory_retriever  # noqa: F401  -- memory_search / memory_graph_search
from .sync import Sync

__all__ = [
    "Sync",
]
