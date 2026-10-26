"""Outbound proxy backend wrapping an existing HTTP proxy."""

from urllib.parse import urlsplit

from .base import BaseOutboundProxy, OutboundProxyEndpoint
from ..component_registry import R


@R.register("fixed_http")
class FixedHttpOutboundProxy(BaseOutboundProxy):
    """Publish an externally managed HTTP proxy without owning its lifecycle."""

    def __init__(self, url: str = "", **kwargs) -> None:
        super().__init__(**kwargs)
        self.url = url

    async def _start(self) -> None:
        self._endpoint = OutboundProxyEndpoint(http_url=self._validate_url())

    async def _close(self) -> None:
        self._endpoint = None

    def _validate_url(self) -> str:
        try:
            parsed = urlsplit(self.url)
            port = parsed.port
        except ValueError as exc:
            raise ValueError("Outbound proxy configuration invalid: malformed fixed HTTP proxy URL.") from exc

        if parsed.scheme != "http":
            raise ValueError("Outbound proxy configuration invalid: fixed proxy URL must use http://.")
        if not parsed.hostname or port is None:
            raise ValueError("Outbound proxy configuration invalid: fixed proxy URL must include host and port.")
        if parsed.username is not None or parsed.password is not None:
            raise ValueError("Outbound proxy configuration invalid: fixed proxy URL must not contain userinfo.")
        if parsed.query or parsed.fragment:
            raise ValueError(
                "Outbound proxy configuration invalid: fixed proxy URL must not contain query or fragment.",
            )
        return self.url
