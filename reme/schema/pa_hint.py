"""Bounded sampling state for optional, time-appropriate proactive hints."""

from typing import Literal

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field

HintPeriod = Literal["morning", "midday", "evening"]


class PaHintProbabilities(BaseModel):
    """Independent probabilities for the three daily hint opportunities."""

    model_config = ConfigDict(extra="forbid")

    morning: float = Field(default=1.0, ge=0, le=1, strict=True, allow_inf_nan=False)
    midday: float = Field(default=0.5, ge=0, le=1, strict=True, allow_inf_nan=False)
    evening: float = Field(default=0.5, ge=0, le=1, strict=True, allow_inf_nan=False)


class PaHintDraw(BaseModel):
    """One daily opportunity, including a probability miss or an ignored hint."""

    model_config = ConfigDict(extra="forbid")

    sampled_at: AwareDatetime
    round_id: str
    hint_index: int | None = Field(default=None, ge=1, le=6, strict=True)


class PaHintState(BaseModel):
    """Keep the latest draw and last selected variant for each of three periods."""

    model_config = ConfigDict(extra="forbid")

    draws: dict[HintPeriod, PaHintDraw] = Field(default_factory=dict)
    recent: dict[HintPeriod, int] = Field(default_factory=dict)
