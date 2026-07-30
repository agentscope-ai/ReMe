"""Backend-neutral token accounting contracts."""

from typing import Any

from pydantic import BaseModel, Field, model_validator


class TokenUsage(BaseModel):
    """Token usage for one completed agent invocation.

    ``input_tokens`` is the complete prompt size.  Cache-read and cache-write
    tokens are included in it, but are also exposed separately when the
    backend reports them.  ``reasoning_tokens`` is a subset of output tokens.
    A ``None`` cache field means that the backend did not report it; it must
    not be interpreted as zero.
    """

    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    cache_read_tokens: int | None = Field(default=None, ge=0)
    cache_write_tokens: int | None = Field(default=None, ge=0)
    reasoning_tokens: int | None = Field(default=None, ge=0)
    total_tokens: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def _set_total(self) -> "TokenUsage":
        self.total_tokens = self.input_tokens + self.output_tokens
        return self

    @classmethod
    def from_provider(
        cls,
        usage: Any,
        *,
        input_includes_cache: bool,
    ) -> "TokenUsage":
        """Normalize a provider usage object or mapping.

        Providers that report cache tokens separately from normal input (for
        example Claude) pass ``False``.  Providers whose input count already
        includes cached input (for example Codex/OpenAI) pass ``True``.
        """

        def get(*names: str) -> int | None:
            for name in names:
                value = usage.get(name) if isinstance(usage, dict) else getattr(usage, name, None)
                if value is not None:
                    return int(value)
            return None

        reported_input = get("input_tokens", "prompt_tokens") or 0
        cache_read = get(
            "cache_read_input_tokens",
            "cache_input_tokens",
            "cached_input_tokens",
        )
        cache_write = get("cache_creation_input_tokens", "cache_write_input_tokens")
        input_tokens = reported_input
        if not input_includes_cache:
            input_tokens += (cache_read or 0) + (cache_write or 0)
        return cls(
            input_tokens=input_tokens,
            output_tokens=get("output_tokens", "completion_tokens") or 0,
            cache_read_tokens=cache_read,
            cache_write_tokens=cache_write,
            reasoning_tokens=get("reasoning_output_tokens", "reasoning_tokens"),
        )

    @classmethod
    def combine(cls, usages: list["TokenUsage"]) -> "TokenUsage":
        """Combine completed model calls without turning unknown into zero."""
        optional = ("cache_read_tokens", "cache_write_tokens", "reasoning_tokens")
        values: dict[str, int | None] = {
            "input_tokens": sum(item.input_tokens for item in usages),
            "output_tokens": sum(item.output_tokens for item in usages),
        }
        for field in optional:
            reported = [getattr(item, field) for item in usages if getattr(item, field) is not None]
            values[field] = sum(reported) if reported else None
        return cls(**values)
