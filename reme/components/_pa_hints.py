"""Sample optional proactive hints once per local day and eligible period."""

# pylint: disable=protected-access

from __future__ import annotations

from datetime import datetime
import random
from typing import TYPE_CHECKING

from ..schema.pa_hint import HintPeriod, PaHintDraw, PaHintState
from ..schema.pa_proactive import PaProactiveConfig
from ..utils.async_utils import complete_in_thread

if TYPE_CHECKING:
    from .pa_runtime import PaRuntime


class ProactiveHints:
    """Own bounded, durable draws; the agent still decides whether to speak."""

    def __init__(self, runtime: PaRuntime):
        self.runtime = runtime

    @staticmethod
    def _period(config: PaProactiveConfig, now: datetime) -> HintPeriod | None:
        if not config.is_active(now):
            return None
        minute = now.hour * 60 + now.minute
        if 6 <= now.hour < 12:
            return "morning"
        if 12 <= now.hour < 14:
            return "midday"
        end = int(config.end_time[:2]) * 60 + int(config.end_time[2:])
        # Closing hints stay in the evening, even when active hours end next morning.
        evening_start = max(18 * 60, end - 30) if config.start_time < config.end_time else 21 * 60
        return "evening" if minute >= evening_start else None

    async def select(self, config: PaProactiveConfig, now: datetime) -> str | None:
        """Persist misses too, so later wakeups and restarts cannot reroll a slot."""
        if not config.hints_enabled or self.runtime.session_id is None:
            return None
        period = self._period(config, now)
        if period is None:
            return None
        current = self.runtime.current_round
        if current is None:
            raise RuntimeError("Proactive hints require an active round")
        if current.session_id != self.runtime.session_id:
            return None
        hint = await complete_in_thread(self._draw, config, now, period, current.id)
        current.hint_period = period if hint is not None else None
        return hint

    def _draw(self, config: PaProactiveConfig, now: datetime, period: HintPeriod, round_id: str) -> str | None:
        path = self.runtime._metadata_file("hint_state.json")
        try:
            with path.open("rb") as handle:
                content = handle.read(4097)
        except FileNotFoundError:
            state = PaHintState()
        else:
            if len(content) > 4096:
                raise ValueError("hint_state.json exceeds 4096 bytes")
            state = PaHintState.model_validate_json(content)
        previous = state.draws.get(period)
        if previous is not None and previous.sampled_at.astimezone(now.tzinfo).date() >= now.date():
            return None
        hint_index = None
        if random.random() < getattr(config.hint_probabilities, period):
            hint_index = random.choice([index for index in range(1, 7) if index != state.recent.get(period)])
            state.recent[period] = hint_index
        state.draws[period] = PaHintDraw(sampled_at=now, round_id=round_id, hint_index=hint_index)
        self.runtime._write_json(path, state.model_dump(mode="json"))
        return f"hint_{period}_{hint_index}" if hint_index is not None else None
