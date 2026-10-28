"""Preserve the Meta-ReMe distinction between candidate and infrastructure errors."""

INFRA_TYPES = {
    "APIConnectionError",
    "APITimeoutError",
    "ConnectError",
    "ConnectionError",
    "ConnectTimeout",
    "NetworkError",
    "PoolTimeout",
    "RateLimitError",
    "ReadError",
    "ReadTimeout",
    "RemoteProtocolError",
    "ServiceUnavailableError",
    "TimeoutError",
    "TimeoutException",
    "WriteError",
    "WriteTimeout",
    "SandboxCommandTimeoutError",
}


def failure_kind(exception, stage: str) -> str:
    """Only candidate execution errors count against candidate scores."""
    exception = exception if isinstance(exception, dict) else {}
    status = exception.get("status_code")
    if isinstance(status, int) and (status in {408, 429} or status >= 500):
        return "infra_error"
    name = str(exception.get("type", ""))
    module = str(exception.get("module", ""))
    if name in INFRA_TYPES or (
        module.startswith(("httpx", "httpcore", "aiohttp", "requests", "urllib3"))
        and any(word in name.lower() for word in ("connection", "network", "timeout", "protocol"))
    ):
        return "infra_error"
    return "candidate_failure" if stage in {"construction", "answer"} else "infra_error"


def construction_exception(build: dict):
    """Find the first failed construction result's structured exception."""
    for job in build.get("jobs", []):
        result = job.get("result", {})
        if not result.get("success"):
            return result.get("exception")
    return build.get("exception")
