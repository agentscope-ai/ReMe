"""HTTP transport and authentication for native DashScope speech APIs."""

import os

import httpx
from pydantic import BaseModel, Field


class _SpeechTransportOptions(BaseModel):
    """Validate transport settings without provider credentials."""

    base_url: str = Field(default="https://dashscope.aliyuncs.com/api/v1", min_length=1)
    timeout: float = Field(default=60, gt=0, allow_inf_nan=False)


class DashScopeSpeechClient:
    """Own an asynchronous HTTP client; resolve credentials only on use."""

    def __init__(
        self,
        api_key: str = "",
        base_url: str = "https://dashscope.aliyuncs.com/api/v1",
        timeout: float = 60,
    ):
        self.api_key = api_key
        self.options = _SpeechTransportOptions(base_url=base_url.rstrip("/"), timeout=timeout)
        self.client: httpx.AsyncClient | None = None

    def start(self) -> None:
        """Create the client when its owning component starts."""
        self.client = httpx.AsyncClient(base_url=self.options.base_url + "/", timeout=self.options.timeout)

    async def close(self) -> None:
        """Release the client when its owning component closes."""
        if self.client is not None:
            await self.client.aclose()
            self.client = None

    def headers(self) -> dict[str, str]:
        """Resolve optional credentials only when making a request."""
        api_key = (self.api_key or os.getenv("DASHSCOPE_API_KEY", "")).strip()
        if not api_key:
            raise ValueError("DASHSCOPE_API_KEY is required")
        return {"Authorization": f"Bearer {api_key}"}

    @staticmethod
    async def check_response(response: httpx.Response) -> None:
        """Keep provider error details without exposing request headers or audio."""
        if response.is_error:
            await response.aread()
            try:
                body = response.json()
            except ValueError:
                body = None
            if isinstance(body, dict):
                detail = ": ".join(str(body[key])[:1000] for key in ("code", "message", "request_id") if body.get(key))
                if detail:
                    raise httpx.HTTPStatusError(
                        f"DashScope HTTP {response.status_code}: {detail}",
                        request=response.request,
                        response=response,
                    )
        response.raise_for_status()

    @staticmethod
    def check_error(body: dict) -> None:
        """Reject provider errors returned inside successful HTTP responses."""
        if body.get("code"):
            raise RuntimeError(
                f"DashScope {body['code']}: {body.get('message', '')} " f"(request_id={body.get('request_id', '')})",
            )
