"""auto_memory (beam) — AutoMemoryStep with timestamp interpolation and daily_write date default."""

from datetime import datetime, timedelta

from agentscope.message import Msg

from reme.steps.evolve.auto_memory import AutoMemoryStep, _normalize_msg_timestamp
from reme.steps.file_io import validate_session_id

# Runtime-context key carrying the 0-based line offset of the current segment
# inside the full session file (segmented ingestion of long sessions).
_LINE_OFFSET_KEY = "beam_line_offset"


def _msg_word_count(msg: Msg) -> int:
    """Whitespace-separated word count of a message's text content."""
    return len((msg.get_text_content() or "").split())


def split_turn_segments(messages: list[Msg], max_words: int) -> list[tuple[int, list[Msg]]]:
    """Split *messages* into segments of complete turns bounded by *max_words*.

    A turn starts at every ``user`` message and spans all following non-user
    messages (assistant replies). Segments only break at turn boundaries, so a
    ``user + assistant`` exchange is never split across two segments. A single
    turn larger than *max_words* still becomes its own (oversized) segment.

    Returns ``[(offset, msgs), ...]`` where ``offset`` is the 0-based index of
    the segment's first message in *messages* — i.e. its line number in the
    session file minus 1. ``max_words <= 0`` disables splitting.
    """
    if not messages:
        return []
    if max_words <= 0:
        return [(0, list(messages))]

    # Group into turns: each user message opens a new turn.
    turns: list[list[Msg]] = []
    for msg in messages:
        if msg.role == "user" or not turns:
            turns.append([msg])
        else:
            turns[-1].append(msg)

    segments: list[tuple[int, list[Msg]]] = []
    current: list[Msg] = []
    current_words = 0
    offset = 0
    for turn in turns:
        turn_words = sum(_msg_word_count(m) for m in turn)
        if current and current_words + turn_words > max_words:
            segments.append((offset, current))
            offset += len(current)
            current = []
            current_words = 0
        current.extend(turn)
        current_words += turn_words
    if current:
        segments.append((offset, current))
    return segments


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


class BeamAutoMemoryStep(AutoMemoryStep):
    """AutoMemoryStep variant that interpolates timestamps for BEAM sessions.

    Date pinning for ``daily_write`` is handled by the base class through the
    agent wrapper's server-owned ``injected_job_kwargs``.

    Long sessions are ingested incrementally: the message list is split into
    segments of complete turns (``max_segment_words`` step config, default
    10000 words per segment) and each segment runs one agent pass — the first
    pass creates the daily note, later passes merge into it. Line numbers
    shown to the agent always refer to the ORIGINAL session file, not the
    segment.
    """

    async def execute(self):
        assert self.context is not None
        raw_messages = self.context.get("messages") or []
        session_id: str = self.context.get("session_id", "")
        max_words = int(self.kwargs.get("max_segment_words", 10000) or 0)

        messages = self._build_messages(raw_messages)
        segments = split_turn_segments(messages, max_words)

        if len(segments) <= 1:
            self.context[_LINE_OFFSET_KEY] = 0
            await super().execute()
            return

        # Persist the FULL session file up-front so every segment's [Ln] labels
        # (and the note's source markers) match the final on-disk layout. The
        # per-segment saves inside the base execute become no-op appends.
        if session_id and validate_session_id(session_id) is None:
            await self._save_session_messages(session_id, messages)

        base_hint = self.context.get("memory_hint", "") or ""
        answers: list[str] = []
        failed_segments = 0
        self.logger.info(
            f"[{self.name}] segmented ingestion session_id={session_id!r} "
            f"messages={len(messages)} segments={len(segments)} max_segment_words={max_words}",
        )
        for part, (offset, segment) in enumerate(segments, start=1):
            part_hint = (
                f"This is part {part}/{len(segments)} of one long session "
                f"(lines {offset + 1}-{offset + len(segment)} of the session file). "
                "Earlier parts were already recorded; extract ONLY from the turns shown below."
            )
            self.context["messages"] = segment
            self.context["memory_hint"] = f"{base_hint}\n{part_hint}".strip()
            self.context[_LINE_OFFSET_KEY] = offset
            # One bad segment (e.g. transiently corrupted note frontmatter) must
            # not lose the remaining segments of the session.
            try:
                await super().execute()
                ok = bool(self.context.response.success)
                summary = self.context.response.answer or ""
            except Exception as exc:  # pylint: disable=broad-exception-caught
                ok = False
                summary = f"Error: {exc}"
                self.logger.warning(
                    f"[{self.name}] segment {part}/{len(segments)} failed session_id={session_id!r}: {exc}",
                )
            if not ok:
                failed_segments += 1
            answers.append(f"[part {part}/{len(segments)}] {summary}")

        # Restore caller-visible context and aggregate the response.
        self.context["messages"] = raw_messages
        self.context["memory_hint"] = base_hint
        self.context.response.success = failed_segments < len(segments)
        self.context.response.answer = "\n".join(answers)
        self.context.response.metadata.update(
            {"n_messages": len(messages), "segments": len(segments), "failed_segments": failed_segments},
        )

    def _build_messages(self, raw_messages: list) -> list[Msg]:
        # Interpolate timestamps: if any message carries created_at, fill in
        # the rest via linear interpolation so the whole session has coherent
        # time ordering (see _interpolate_timestamps docstring for rules).
        interpolated = _interpolate_timestamps(
            [item if not isinstance(item, dict) else dict(item) for item in raw_messages],
        )
        return [self._to_msg(item) for item in interpolated]

    def _format_history(self, messages: list[Msg]) -> str:
        # Annotate every turn with its physical line number in the session
        # file so the agent can cite information sources as
        # [[<session file path>#L<start>-L<end>]]. A BEAM batch is already
        # in chronological order, matching the one-message-per-line layout
        # written by AutoMemoryStep._save_session_messages, so number each turn
        # by its position plus the segment's offset into the full session file
        # (no re-sorting). When the session is ingested in segments, the offset
        # keeps [Ln] labels aligned with the ORIGINAL file, not the segment.
        session_id = self.context.get("session_id", "") if self.context is not None else ""
        line_offset = int(self.context.get(_LINE_OFFSET_KEY, 0) or 0) if self.context is not None else 0
        source_path = self._session_source_path(session_id)

        turns: list[str] = []
        for line, msg in enumerate(messages, start=line_offset + 1):
            text = (msg.get_text_content() or "").strip()
            if not text:
                continue
            speaker = msg.name or msg.role or "?"
            turns.append(f"[L{line} | {speaker} @ {msg.created_at}]\n{text}")

        if not turns:
            return "(empty)"

        first_line = line_offset + 1
        last_line = line_offset + len(messages)
        preamble = (
            f"Source file: {source_path}\n"
            f"(Excerpt covering lines {first_line}-{last_line} of the source file. Each turn below is "
            "prefixed with [Ln] = its line number in the source file; copy these numbers verbatim "
            "into source markers.)"
        )
        return preamble + "\n\n" + "\n\n".join(turns)

    def _reply_extra_kwargs(self, day: str) -> dict:
        return {"tool_defaults": {"daily_write": {"date": day}}}
