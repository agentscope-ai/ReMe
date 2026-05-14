"""Memory subsystem — agent-facing services + tools on top of the core engine.

Three services (read / cold-write / treatment) plus the agent-callable
tools that operate on the Memory File System. Every ``@R.register``
step a host agent might invoke lives here, regardless of whether it's
reached via MCP, HTTP, or direct Python.

    Services
        retriever.py    Read service — V + K + Graph BFS fusion + intent
                        routing. Registered as ``hybrid``; consumed by
                        ``memory_search`` / ``memory_graph_search``.
        ingestor.py     Cold-write service — LLM-driven R-M-W curator
                        (also registers an ``ingest`` MCP face).
        maintainer.py   Treatment service — Merge / Split / Decay / Lint,
                        woken by cron or thresholds. Composes the
                        atomic ``check_*`` primitives below.
        summarizer.py   Auxiliary used by the services.

    Tool surfaces (each ``@R.register`` exposes a step the agent
    invokes by name).

    agent_toolkit.py    The 11 agent tools across three categories
                        (build via ``build_agent_toolkit``):
                            memory_*   get / create / update_body /
                                       update_meta / search
                            file_*     download / upload / delete /
                                       list / move
                            graph_*    traverse
                        Plus ``memory_graph_search`` (MCP-only,
                        not in the agent toolkit binding).

    lint_toolkit.py     Atomic vault-health checks (build via
                        ``build_lint_toolkit``):
                            check_dangling
                            check_orphans
                            check_collisions
                            check_schema
                        Read-only, separate category — for maintainer
                        / CLI / scheduled-job use.

    sync.py             ``sync`` — hot-path event-folder upsert
                        (deterministic, no LLM).

    memory_io.py        Pure engine API — no schema, no Step
                        boilerplate. The toolkits wrap it.

Importing this package triggers every ``@R.register`` so configs that
name these tools resolve at boot.
"""

from . import retriever  # noqa: F401  -- @R.register("hybrid")
from . import ingestor  # noqa: F401  -- @R.register("ingestor")
from . import maintainer  # noqa: F401  -- @R.register("maintainer")
from . import agent_toolkit  # noqa: F401  -- 11 agent tools + memory_graph_search
from . import lint_toolkit  # noqa: F401  -- 4 check_* atomic primitives
from . import sync  # noqa: F401  -- @R.register("sync")

from .agent_toolkit import AGENT_TOOL_NAMES, build_agent_toolkit
from .lint_toolkit import LINT_TOOL_NAMES, build_lint_toolkit


__all__ = [
    "AGENT_TOOL_NAMES",
    "LINT_TOOL_NAMES",
    "build_agent_toolkit",
    "build_lint_toolkit",
]
