"""Service interfaces and concrete implementations for FlowLLM.

This subpackage defines the standard service abstraction used to connect flows
to external runtimes and tools, and provides reference implementations:

- ``BaseService``: Defines the abstract lifecycle and request/response
  contract that all services follow.
- ``CmdService``: Executes local command-line subprocesses as a service.
- ``HttpService``: Exposes a simple HTTP/JSON service implementation.
- ``MCPService``: Integrates with Model Context Protocol (MCP) servers.

Typical usage:
    from flowllm.core.service import BaseService, HttpService

The module-level ``__all__`` re-exports the primary service classes for
convenience.
"""

from .base_service import BaseService
from .cmd_service import CmdService
from .http_service import HttpService
from .mcp_service import MCPService

__all__ = [
    "BaseService",
    "CmdService",
    "HttpService",
    "MCPService",
]
