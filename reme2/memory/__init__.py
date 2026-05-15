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

    Tool surfaces — split by category, each cohesive and self-
    contained. ``agent_toolkit`` is just a thin orchestrator that
    composes them into one Toolkit for hosts that want all 13 at once.

        memory_toolkit.py   memory_get / memory_create /
                            memory_update_body / memory_update_meta /
                            memory_search
                            + memory_graph_search (MCP-only)
                            + schema policy helpers (path template,
                              status state machine,
                              create_with_schema, update_status)

        file_toolkit.py     file_download / file_upload /
                            file_delete / file_list / file_move
                            + path resolution + session temp dir

        graph_toolkit.py    graph_traverse + adjacency BFS

        event_toolkit.py    event_open / event_complete
                            (uses memory schema gates so the event
                            index follows the memory schema)

        agent_toolkit.py    AGENT_TOOL_NAMES + build_agent_toolkit

        lint_toolkit.py     Atomic vault-health checks (build via
                            ``build_lint_toolkit``):
                                check_dangling
                                check_orphans
                                check_collisions
                                check_schema
                            Read-only, separate category — for
                            maintainer / CLI / scheduled-job use.

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

# Tool surfaces — each module's @R.register decorators fire on import.
from . import memory_toolkit  # noqa: F401  -- 5 memory_* tools + memory_graph_search
from . import file_toolkit  # noqa: F401  -- 5 file_* tools
from . import graph_toolkit  # noqa: F401  -- graph_traverse
from . import event_toolkit  # noqa: F401  -- event_open / event_complete
from . import agent_toolkit  # noqa: F401  -- composition: AGENT_TOOL_NAMES
from . import lint_toolkit  # noqa: F401  -- 4 check_* atomic primitives

from .agent_toolkit import AGENT_TOOL_NAMES, build_agent_toolkit
from .lint_toolkit import LINT_TOOL_NAMES, build_lint_toolkit


__all__ = [
    "AGENT_TOOL_NAMES",
    "LINT_TOOL_NAMES",
    "build_agent_toolkit",
    "build_lint_toolkit",
]
