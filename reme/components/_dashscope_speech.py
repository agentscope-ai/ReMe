"""Shared lifecycle and authentication for native DashScope speech APIs."""

import os

import httpx
from pydantic import BaseModel, Field

from .base_component import BaseComponent


class _SpeechTransportOptions(BaseModel):
    """Validate transport settings without provider credentials."""

    base_url: str = Field(default="https://dashscope.aliyuncs.com/api/v1", min_length=1)
    timeout: float = Field(default=60, gt=0, allow_inf_nan=False)


class DashScopeSpeechComponent(BaseComponent):
    """Own an asynchronous HTTP client; resolve credentials only on use."""

    def __init__(
        self,
        api_key: str = "",
        base_url: str = "https://dashscope.aliyuncs.com/api/v1",
        timeout=60,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.api_key = api_key
        self.options = _SpeechTransportOptions(base_url=base_url.rstrip("/"), timeout=timeout)
        self.client: httpx.AsyncClient | None = None

    async def _start(self) -> None:
        self.client = httpx.AsyncClient(base_url=self.options.base_url + "/", timeout=self.options.timeout)

    async def _close(self) -> None:
        if self.client is not None:
            await self.client.aclose()
            self.client = None

    def _headers(self) -> dict[str, str]:
        api_key = (self.api_key or os.getenv("DASHSCOPE_API_KEY", "")).strip()
        if not api_key:
            raise ValueError("DASHSCOPE_API_KEY is required")
        return {"Authorization": f"Bearer {api_key}"}

    @staticmethod
    async def _check_response(response: httpx.Response) -> None:
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
    def _check_error(body: dict) -> None:
        if body.get("code"):
            raise RuntimeError(
                f"DashScope {body['code']}: {body.get('message', '')} " f"(request_id={body.get('request_id', '')})",
            )
