"""Internal embedding results and normalized provider errors."""

import re
from dataclasses import dataclass

_PROVIDER_WIDE_STATUS_CODES = frozenset({401, 403, 404})


# AgentScope's DashScope wrapper serializes the response into a RuntimeError.
_STATUS_IN_MESSAGE = re.compile(r'"status_code"\s*:\s*(\d{3})')


_CODE_IN_MESSAGE = re.compile(r'"code"\s*:\s*"([^"]+)"')


_INPUT_DEPENDENT_ERROR_TYPES = frozenset(
    {
        "ContextLengthError",
        "ContextWindowExceededError",
        "InputLengthError",
        "TokenLimitError",
    },
)


_INPUT_DEPENDENT_MARKERS = (
    "context length",
    "context_length",
    "context window",
    "input length",
    "input is too long",
    "maximum context",
    "reduce the length",
    "token limit",
    "too many tokens",
    "too long",
)


@dataclass(frozen=True)
class _EmbeddingResult:
    """Keep recovered vectors together with the failure state that ends isolation."""

    embeddings: list[list[float] | None] | None = None
    terminal_failure: bool = False


@dataclass(frozen=True)
class _EmbeddingError:
    """Normalize provider error shapes once before applying the retry policy."""

    status: int | None
    code: str | None
    text: str
    error_type: str

    @classmethod
    def from_exception(cls, error: Exception) -> "_EmbeddingError":
        """Read attributes, structured bodies, and serialized DashScope responses."""
        message = str(error)
        body = getattr(error, "body", None)
        status = None
        for value in (
            getattr(error, "status_code", None),
            getattr(error, "code", None),
            getattr(getattr(error, "response", None), "status_code", None),
        ):
            # Integer codes are HTTP statuses; strings name business errors.
            if isinstance(value, int) and not isinstance(value, bool):
                status = value
                break
        if status is None and (match := _STATUS_IN_MESSAGE.search(message)):
            status = int(match.group(1))

        code = getattr(error, "code", None)
        if not isinstance(code, str) or not code:
            details = body.get("error", body) if isinstance(body, dict) else None
            code = details.get("code") if isinstance(details, dict) else None
        if not isinstance(code, str) or not code:
            match = _CODE_IN_MESSAGE.search(message)
            code = match.group(1) if match else None
        text = f"{message} {body}" if isinstance(body, dict) else message
        return cls(status, code, text.lower(), type(error).__name__)

    @property
    def is_insufficient_quota(self) -> bool:
        """Quota exhaustion takes precedence over a generic HTTP 429."""
        return self.code == "insufficient_quota"

    @property
    def is_rate_limited(self) -> bool:
        """Recognize rate limits while preserving the separate quota policy."""
        return not self.is_insufficient_quota and (self.code == "rate_limit_exceeded" or self.status == 429)

    @property
    def is_provider_wide(self) -> bool:
        """Credentials, missing routes, and server outages cannot be isolated per input."""
        return (
            self.status is not None and (self.status in _PROVIDER_WIDE_STATUS_CODES or 500 <= self.status < 600)
        ) or self.error_type in ("AuthenticationError", "PermissionDeniedError", "NotFoundError", "InternalServerError")

    @property
    def is_input_dependent(self) -> bool:
        """Only input-length rejections may split; a 413 always requires a text signal."""
        if self.status not in (None, 400, 413):
            return False
        if self.status != 413 and self.error_type in _INPUT_DEPENDENT_ERROR_TYPES:
            return True
        return any(marker in self.text for marker in _INPUT_DEPENDENT_MARKERS)
