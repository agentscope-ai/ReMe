"""reme2.mcp — MCP interface layer.

The Agent-facing surface: server entrypoint + step shells that wrap
the three services in `reme2.memory` (Retriever, Ingestor, Maintainer)
plus hot-write primitives (sync, topic_create, memory_*) that bypass
services and land directly on the Memory File System.

This package depends on `reme2.memory`, `reme2.schema.vault`,
`reme2.utils`, `reme2.component` — never the reverse. Domain types
(Topic / Event), pure helpers (path builders, naming), and the write
primitives all live outside `mcp/` so memory-layer services can use
them without forming an import cycle through the transport layer.

Sub-packages:
    steps/   - all @R.register MCP step shells (memory_io, memory_retriever,
               sync, topic_create).
    server.py - MCP server bootstrap (defaults to ../config/full.yaml).
"""

__version__ = "0.1.0"
