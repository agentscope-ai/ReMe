"""User-owned proactive wakeup configuration."""

from datetime import datetime, timedelta
import math

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, StrictBool, field_validator

from .pa_hint import PaHintProbabilities


class PaProactiveConfig(BaseModel):
    """Active hours use the application timezone; intervals follow round completion."""

    model_config = ConfigDict(extra="forbid")

    start_time: str = Field(pattern=r"^(?:[01][0-9]|2[0-3])[0-5][0-9]$")
    end_time: str = Field(pattern=r"^(?:[01][0-9]|2[0-3])[0-5][0-9]$")
    interval_minutes: float = Field(gt=0, allow_inf_nan=False)
    enabled: StrictBool = True
    tasks_follow_active_hours: StrictBool = True
    hints_enabled: StrictBool = True
    hint_probabilities: PaHintProbabilities = Field(default_factory=PaHintProbabilities)

    @field_validator("interval_minutes", mode="before")
    @classmethod
    def validate_interval(cls, value):
        """Reject booleans, strings and delays that overflow timer arithmetic."""
        if type(value) not in (int, float) or not math.isfinite(value) or not math.isfinite(value * 60):
            raise ValueError("interval_minutes must be a positive finite number")
        return value

    def is_active(self, now: datetime) -> bool:
        """Start is inclusive, end is exclusive; equal endpoints mean all day."""
        current, start, end = now.hour * 100 + now.minute, int(self.start_time), int(self.end_time)
        if start == end:
            return True
        return start <= current < end if start < end else current >= start or current < end

    def next_start(self, now: datetime) -> datetime:
        """Return the next opening of the active window."""
        start = now.replace(hour=int(self.start_time[:2]), minute=int(self.start_time[2:]), second=0, microsecond=0)
        return start if start > now else start + timedelta(days=1)

    def active_window(self, now: datetime) -> tuple[datetime, datetime]:
        """Return the most recently opened window, including its exclusive deadline."""
        start = self.next_start(now) - timedelta(days=1)
        end = start.replace(hour=int(self.end_time[:2]), minute=int(self.end_time[2:]))
        if end <= start:
            end += timedelta(days=1)
        return start, end

    def next_wakeup(self, anchor: datetime) -> datetime:
        """Wait after an active round, then move an out-of-window deadline to the next opening."""
        if not self.is_active(anchor):
            return self.next_start(anchor)
        wakeup = anchor + timedelta(minutes=self.interval_minutes)
        return wakeup if self.is_active(wakeup) else self.next_start(wakeup)


class PaNotificationWindow(BaseModel):
    """A rebuildable display policy shared by all notification surfaces."""

    model_config = ConfigDict(extra="forbid")

    starts_at: AwareDatetime
    ends_at: AwareDatetime
    closing_message_id: str | None = None
    enabled: bool = True

    def allows(self, now: datetime, message_id: str | None = None) -> bool:
        """After a closing decision, only that message may still be delivered."""
        return (
            self.enabled
            and self.starts_at <= now < self.ends_at
            and (self.closing_message_id is None or self.closing_message_id == message_id)
        )
