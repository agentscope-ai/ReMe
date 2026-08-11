"""TuShare client construction with an optional mirror endpoint."""

import os


def create_tushare_api(token: str):
    """Create a TuShare DataApi, using TUSHARE_MIRROR_URL when configured."""
    try:
        import tushare as ts
    except ImportError as exc:  # pragma: no cover - optional core dependency.
        raise RuntimeError("tushare is required for market-data research") from exc

    api = ts.pro_api(token)
    api._DataApi__timeout = 600  # pylint: disable=protected-access
    if mirror_url := os.getenv("TUSHARE_MIRROR_URL", "").strip():
        api._DataApi__http_url = mirror_url.rstrip("/")  # pylint: disable=protected-access
    return api
