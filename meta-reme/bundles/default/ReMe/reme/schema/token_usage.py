"""Backend-neutral token accounting contracts."""

from typing import Any

from pydantic import BaseModel, Field, model_validator


class TokenUsage(BaseModel):
    """Portable token usage reported for one completed agent invocation.

    Only the provider's top-level input and output counters are retained.
    Provider-specific cache and reasoning breakdowns are intentionally
    excluded, so values may not share identical billing semantics across
    providers. ``total_tokens`` is always their derived sum.
    """

    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    total_tokens: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def _set_total(self) -> "TokenUsage":
        self.total_tokens = self.input_tokens + self.output_tokens
        return self

    @classmethod
    def from_provider(
        cls,
        usage: Any,
    ) -> "TokenUsage":
        """Keep only a provider's portable top-level input/output counters."""

        def get(*names: str) -> int | None:
            for name in names:
                value = usage.get(name) if isinstance(usage, dict) else getattr(usage, name, None)
                if value is not None:
                    return int(value)
            return None

        return cls(
            input_tokens=get("input_tokens", "prompt_tokens") or 0,
            output_tokens=get("output_tokens", "completion_tokens") or 0,
        )

    @classmethod
    def combine(cls, usages: list["TokenUsage"]) -> "TokenUsage":
        """Combine completed model calls into one full-invocation usage."""
        return cls(
            input_tokens=sum(item.input_tokens for item in usages),
            output_tokens=sum(item.output_tokens for item in usages),
        )
