"""service"""

from .base_service import BaseService
from .cmd_service import CmdService
from .http_service import HttpService
from .mcp_service import MCPService
from ..registry_factory import R

__all__ = [
    "BaseService",
    "CmdService",
    "HttpService",
    "MCPService",
]

R.services.register("cmd")(CmdService)
R.services.register("http")(HttpService)
R.services.register("mcp")(MCPService)
