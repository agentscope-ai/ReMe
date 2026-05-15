"""Agent toolkit — composition entry point for the agent's tool surface.

The actual tool implementations are split by category across four
modules, each cohesive and self-contained:

    memory_toolkit.py   memory_get / memory_create / memory_update_body /
                        memory_update_meta / memory_search
                        + memory_graph_search (MCP-only)
                        + schema policy helpers (path template, status
                          state machine, create_with_schema, update_status)

    file_toolkit.py     file_download / file_upload / file_delete /
                        file_list / file_move
                        + path resolution + session temp dir

    graph_toolkit.py    graph_traverse + adjacency BFS

    event_toolkit.py    event_open / event_complete
                        — uses memory_toolkit's schema gates so the
                        event index follows the memory schema

This module just composes them into one ``Toolkit`` for hosts that
want all 13 tools bound at once. Hosts that want a subset can import
the per-category builders and pass through a shared ``Toolkit``.

Lint tools (``check_dangling`` / ``check_orphans`` / ``check_collisions``
/ ``check_schema``) live in ``lint_toolkit.py`` — separate factory,
not bound here by default.
"""

from __future__ import annotations

from agentscope.tool import Toolkit

from ..component import R
from ..enumeration import ComponentEnum
from .event_toolkit import EVENT_TOOL_NAMES
from .file_toolkit import FILE_TOOL_NAMES
from .graph_toolkit import GRAPH_TOOL_NAMES
from .memory_toolkit import MEMORY_TOOL_NAMES


# Order: memory → file → graph → event. Within memory, search comes
# last (retrieval is a separate concern from CRUD); the rest follow
# their CRUD order. memory_graph_search stays out — it's MCP-only.
AGENT_TOOL_NAMES: tuple[str, ...] = (
    *MEMORY_TOOL_NAMES,
    *FILE_TOOL_NAMES,
    *GRAPH_TOOL_NAMES,
    *EVENT_TOOL_NAMES,
)


def build_agent_toolkit(
    app_context,
    audit: list[dict] | None = None,
    toolkit: Toolkit | None = None,
) -> Toolkit:
    """Bind every agent tool's method to an agentscope ``Toolkit``.

    For each name in ``AGENT_TOOL_NAMES``, instantiates the registered
    BaseStep against ``app_context``, attaches the shared ``audit``
    list, and registers the same-named class method as a tool function.
    agentscope introspects the method signature directly — no separate
    JSON schema layer.
    """
    toolkit = toolkit or Toolkit()
    for name in AGENT_TOOL_NAMES:
        step_cls = R.get(ComponentEnum.STEP, name)
        if step_cls is None:
            continue
        instance = step_cls(app_context=app_context)
        instance.audit = audit  # type: ignore[attr-defined]
        toolkit.register_tool_function(
            getattr(instance, name),
            namesake_strategy="override",
        )
    return toolkit
