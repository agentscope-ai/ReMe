"""Public contract shared by outbound proxy backends."""

from collections.abc import Mapping
from dataclasses import dataclass

from ..base_component import BaseComponent
from ...enumeration import ComponentEnum

_LOCAL_BYPASS = "127.0.0.1,localhost,::1"
_PROXY_VARIABLES = (
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
    "http_proxy",
    "https_proxy",
    "all_proxy",
)


@dataclass(frozen=True, slots=True)
class OutboundProxyEndpoint:
    """Stable public endpoint exposed for the lifetime of a started proxy."""

    http_url: str


class BaseOutboundProxy(BaseComponent):
    """Base component exposing one application-scoped HTTP proxy endpoint."""

    component_type = ComponentEnum.OUTBOUND_PROXY

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._endpoint: OutboundProxyEndpoint | None = None

    @property
    def endpoint(self) -> OutboundProxyEndpoint:
        """Return the ready endpoint, failing when the component is not started."""
        if self._endpoint is None:
            raise RuntimeError("Outbound proxy endpoint is unavailable; start the component first.")
        return self._endpoint

    @property
    def http_url(self) -> str:
        """Return the public HTTP proxy URL."""
        return self.endpoint.http_url

    def merge_environment(self, base: Mapping[str, str] | None = None) -> dict[str, str]:
        """Return a proxy-aware environment without mutating the input or process environment."""
        environment = dict(base or {})
        environment.update(dict.fromkeys(_PROXY_VARIABLES, self.http_url))
        environment["NO_PROXY"] = _LOCAL_BYPASS
        environment["no_proxy"] = _LOCAL_BYPASS
        return environment
