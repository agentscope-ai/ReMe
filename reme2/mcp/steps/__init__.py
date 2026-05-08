"""MCP step shells — the @R.register classes the model invokes as MCP tools.

Two groups:

    Hot-write primitives that bypass services and land on MFS directly:
        sync          — idempotent event-folder upsert (event log)
        topic_create  — typed topic creation (hard schema gates)
        memory_io     — memory_create / update / property_update / rename /
                        delete / archive / get / list / links / backlinks /
                        resolve_wikilink / count_tokens

    Service delegate:
        memory_retriever — memory_search + memory_graph_search; thin
                           wrappers that delegate to the configured
                           Retriever component (`reme2.memory.retriever`).

The Ingestor's MCP face (`ingest`) is registered from
`reme2.memory.ingestor`, since it's a service step rather than an MFS
primitive. Importing this package triggers all the step registrations
hosted here.
"""

from . import memory_io   # noqa: F401  -- triggers @R.register for memory_*
from . import memory_retriever  # noqa: F401  -- memory_search / memory_graph_search
from .sync import Sync
from .topic_create import TopicCreate

__all__ = [
    "Sync",
    "TopicCreate",
]
