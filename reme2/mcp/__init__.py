"""reme2.mcp — MCP transport layer.

Just the MCP server bootstrap (config loading, env-var overrides,
sidecar HTTP). All agent-facing tools live in ``reme2.memory`` and
register themselves there; this package only wires them into the MCP
transport via the ``mcp`` service component.

Dependency direction is strict: ``reme2.mcp → reme2.memory`` (and
through it ``reme2.component`` / ``reme2.utils``). Importing
``reme2.memory`` triggers all ``@R.register`` decorators for
``memory_*`` / ``sync`` / ``memory_search`` / ``memory_lint`` /
``ingest`` / ``maintainer`` / ``hybrid``.

Files:
    server.py — MCP server bootstrap (defaults to ../config/service.yaml).
    test/     — end-to-end profile smoke tests.
"""

__version__ = "0.1.0"
