"""Scoped TuShare client construction with optional explicit proxy routing."""

import json
from functools import partial
from typing import Any


class _ProxiedTushareApi:
    """TuShare DataApi-compatible adapter using one explicit HTTP proxy."""

    def __init__(self, api: Any, token: str, proxy_url: str) -> None:
        http_url = getattr(api, "_DataApi__http_url", None)
        if not isinstance(http_url, str) or not http_url:
            raise RuntimeError(
                "Unsupported tushare SDK: DataApi HTTP endpoint is unavailable",
            )
        self._http_url = http_url.rstrip("/")
        self._timeout = getattr(api, "_DataApi__timeout", 30)
        self._token = token
        self._proxy_url = proxy_url

    def query(self, api_name: str, fields: str = "", **kwargs):
        """Query one TuShare endpoint through the configured proxy."""
        import pandas as pd
        import requests

        params = dict(kwargs)
        params.setdefault("ts_type_name", self._http_url)
        request = {
            "api_name": api_name,
            "token": self._token,
            "params": params,
            "fields": fields,
        }
        with requests.Session() as session:
            session.trust_env = False
            response = session.post(
                f"{self._http_url}/{api_name}",
                json=request,
                timeout=self._timeout,
                proxies={"http": self._proxy_url, "https": self._proxy_url},
            )
        if not response:
            return pd.DataFrame()
        result = json.loads(response.text)
        if result["code"] != 0:
            raise RuntimeError(result["msg"])
        data = result["data"]
        return pd.DataFrame(data["items"], columns=data["fields"])

    def __getattr__(self, name: str):
        return partial(self.query, name)


def create_tushare_api(token: str, *, proxy_url: str | None = None):
    """Create a TuShare DataApi, optionally pinned to one explicit proxy."""
    try:
        import tushare as ts
    except ImportError as exc:  # pragma: no cover - optional core dependency.
        raise RuntimeError("tushare is required for market-data research") from exc

    api = ts.pro_api(token)
    if proxy_url is None:
        return api
    return _ProxiedTushareApi(api, token, proxy_url)
