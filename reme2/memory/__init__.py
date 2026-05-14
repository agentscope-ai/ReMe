"""Memory subsystem — agent-facing services + tools on top of the core engine.

Three services (read / cold-write / treatment) plus the agent-callable
write/read tools that operate on the Memory File System. Every
``@R.register`` step a host agent might invoke lives here, regardless
of whether it's reached via MCP, HTTP, or direct Python.

    Services
        retriever.py    Read service — V + K + Graph BFS fusion + intent
                        routing. Registered as ``hybrid``; consumed by
                        the search shells below.
        ingestor.py     Cold-write service — LLM-driven R-M-W curator
                        (also registers an ``ingest`` MCP face).
        maintainer.py   Treatment service — Merge / Split / Decay / Lint,
                        woken by cron or thresholds.
        summarizer.py   Auxiliary used by the services.

    Tools (each ``@R.register`` exposes a step the agent invokes by name)
        memory_io.py        Pure engine API (no schema, no Step boilerplate).
        memory_toolkit.py   12 ``memory_*`` primitives — get / list / links /
                            backlinks / resolve_wikilink / count_tokens +
                            create / update / property_update / rename /
                            delete / archive.
        memory_search.py    ``memory_search`` + ``memory_graph_search`` —
                            shells over the Retriever service.
        memory_lint.py      ``memory_lint`` — narrowed projection of the
                            Maintainer's lint pass.
        sync.py             ``sync`` — hot-path event-folder upsert
                            (deterministic, no LLM).

Importing this package triggers every ``@R.register`` so configs that
name these tools resolve at boot.
"""

from . import retriever  # noqa: F401  -- @R.register("hybrid")
from . import ingestor  # noqa: F401  -- @R.register("ingestor")
from . import maintainer  # noqa: F401  -- @R.register("maintainer")
from . import memory_toolkit  # noqa: F401  -- @R.register("memory_*") x12
from . import memory_search  # noqa: F401  -- @R.register("memory_search", "memory_graph_search")
from . import memory_lint  # noqa: F401  -- @R.register("memory_lint")
from . import sync  # noqa: F401  -- @R.register("sync")
