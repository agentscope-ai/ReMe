"""auto_memory (beam) — AutoMemoryStep with timestamp interpolation and daily_write date default."""

from datetime import datetime, timedelta

from agentscope.message import Msg

from ...evolve.auto_memory import AutoMemoryStep, _normalize_msg_timestamp
from ....components import R


def _parse_iso_seconds(value: str) -> datetime | None:
    """Parse an ISO-8601 timestamp that is precise to at least seconds.

    Accepts formats like:
        2026-07-01T14:30:00
        2026-07-01T14:30:00Z
        2026-07-01T14:30:00+08:00
        2026-07-01T14:30:00.123456

    Rejects date-only (``2026-07-01``) or minute-only (``2026-07-01T14:30``).
    Returns ``None`` when the value does not satisfy the requirements.
    """
    text = str(value).strip()
    # Minimum valid: YYYY-MM-DDTHH:MM:SS = 19 chars
    if len(text) < 19:
        return None
    # Must contain 'T' separator and at least HH:MM:SS after it
    if "T" not in text:
        return None
    time_part = text.split("T", 1)[1]
    # time_part must start with HH:MM:SS (8 chars minimum)
    if len(time_part) < 8 or time_part[2] != ":" or time_part[5] != ":":
        return None
    try:
        return datetime.fromisoformat(text)
    except (ValueError, TypeError):
        return None


def _interpolate_timestamps(items: list[dict]) -> list[dict]:
    """Fill missing ``created_at`` fields via linear interpolation.

    Rules (backward-compatible — returns *items* unchanged when no message
    carries a ``created_at`` value):

    1. If **no** message has ``created_at`` → return as-is (system time used
       later by AgentScope's ``Msg`` constructor).
    2. Messages **before** the first timestamped message → inherit the first
       timestamp.
    3. Messages **after** the last timestamped message → inherit the last
       timestamp.
    4. Messages **between** two timestamped anchors → linearly interpolated.
    """
    # Pass 1: normalize aliases and collect anchors
    normalized: list[dict] = []
    anchors: list[tuple[int, datetime]] = []  # (index, parsed_dt)

    for i, raw_item in enumerate(items):
        if not isinstance(raw_item, dict):
            normalized.append(raw_item)
            continue
        item = _normalize_msg_timestamp(raw_item)
        normalized.append(item)
        ca = item.get("created_at")
        if ca:
            dt = _parse_iso_seconds(str(ca))
            if dt is not None:
                anchors.append((i, dt))

    # No anchors → fully backward-compatible, let Msg use system time
    if not anchors:
        return normalized

    # Pass 2: interpolate
    result: list[dict] = []
    for i, item in enumerate(normalized):
        if not isinstance(item, dict):
            result.append(item)
            continue
        # Already has a valid parsed anchor — keep it
        if any(idx == i for idx, _ in anchors):
            result.append(item)
            continue

        # Find the nearest preceding and following anchors
        prev_anchor: tuple[int, datetime] | None = None
        next_anchor: tuple[int, datetime] | None = None
        for idx, dt in anchors:
            if idx < i:
                prev_anchor = (idx, dt)
        for idx, dt in anchors:
            if idx > i:
                next_anchor = (idx, dt)
                break

        # Determine interpolated time
        if prev_anchor is None:
            # Before the first anchor
            interpolated_dt = anchors[0][1]
        elif next_anchor is None:
            # After the last anchor
            interpolated_dt = anchors[-1][1]
        else:
            # Between two anchors — linear
            prev_idx, prev_dt = prev_anchor
            next_idx, next_dt = next_anchor
            span = next_idx - prev_idx
            ratio = (i - prev_idx) / span
            delta_seconds = (next_dt - prev_dt).total_seconds()
            interpolated_dt = prev_dt + timedelta(seconds=delta_seconds * ratio)

        item = {**item, "created_at": interpolated_dt.isoformat()}
        result.append(item)

    return result


@R.register("beam_auto_memory_step")
class BeamAutoMemoryStep(AutoMemoryStep):
    """AutoMemoryStep variant that interpolates timestamps and pins daily_write to the resolved day."""

    def _build_messages(self, raw_messages: list) -> list[Msg]:
        # Interpolate timestamps: if any message carries created_at, fill in
        # the rest via linear interpolation so the whole session has coherent
        # time ordering (see _interpolate_timestamps docstring for rules).
        interpolated = _interpolate_timestamps(
            [item if not isinstance(item, dict) else dict(item) for item in raw_messages],
        )
        return [self._to_msg(item) for item in interpolated]

    def _reply_extra_kwargs(self, day: str) -> dict:
        return {"tool_defaults": {"daily_write": {"date": day}}}
