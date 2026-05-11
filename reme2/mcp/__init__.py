"""reme2.mcp — MCP interface layer.

The Agent-facing surface: server entrypoint + step shells that wrap
the three services in `reme2.memory` (Retriever, Ingestor, Maintainer)
plus the hot-write primitive (`sync`) and the raw `memory_*` write/read
tools that bypass services and land directly on the Memory File System.

This package depends on `reme2.memory`, `reme2.utils`, `reme2.component`
— never the reverse. The Memory schema (the typed shape of every
frontmatter) lives under `reme2.memory.schema/` so the services that
own validation can use it without an import cycle through this
transport layer.

Sub-packages:
    steps/   - all @R.register MCP step shells (memory_toolkit,
               memory_retriever, memory_lint, sync).
    server.py - MCP server bootstrap (defaults to ../config/service.yaml).
"""

__version__ = "0.1.0"
