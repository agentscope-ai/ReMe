"""Application-scoped outbound HTTP proxy components."""

from .base import BaseOutboundProxy, OutboundProxyEndpoint
from .fixed_http import FixedHttpOutboundProxy
from .ssh_http import SshHttpOutboundProxy

__all__ = [
    "BaseOutboundProxy",
    "FixedHttpOutboundProxy",
    "OutboundProxyEndpoint",
    "SshHttpOutboundProxy",
]
