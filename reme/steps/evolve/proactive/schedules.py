"""Long-lived reminder registry: due-clock maths, maintenance and candidates.

The registry is the ``schedules`` section of the ``daily/_proactive.yaml``
truth source. It exists because a schedule mentioned weeks ago still has to
fire on its due date - long after the originating daily note left the extract
scan window and long after the matching topic was trimmed by
``carry_forward_days``. Every change the pipeline observes is appended to the
entry's ``history``, so a reschedule or a cancellation stays auditable.

Everything here is deterministic: no LLM calls and no file writes. The extract
step owns the clock scan (``schedule_view`` / ``reminder_candidates_from_view``)
and the topics step owns persistence (``apply_schedule_updates``,
``upsert_schedule_from_reminder``, ``trim_schedules`` followed by
``save_state``).
"""

import calendar
import datetime as dt
import re

from ....schema import ProactiveSchedule, ProactiveStateFile, ProactiveTopic
from ....schema.proactive import (
    RECURRENCES,
    REMINDER_TRIGGERS,
    SCHEDULE_ACTIONS,
    clamp_confidence,
)
from ....utils import get_logger
from ..dream.utils import clean_paths
from .utils import norm_path, safe_date, topic_id

logger = get_logger(log_to_file=False)


# Triggers that deserve a reminder; ``later``/``unscheduled``/``closed`` stay
# in the prompt view but never become candidates.


DUE_TRIGGERS = REMINDER_TRIGGERS


_RECURRING = ("daily", "weekly", "monthly", "yearly")


_ISO_DUE_RE = re.compile(
    r"^(\d{4})-(\d{1,2})-(\d{1,2})(?:[T\s]+(\d{1,2}):(\d{2})(?::(\d{2}))?)?\s*(?:[+-]\d{2}:?\d{2}|Z)?$",
)


_CJK_DATE_RE = re.compile(r"(\d{4})\s*年\s*(\d{1,2})\s*月\s*(\d{1,2})\s*[日号]?")


_CJK_MONTH_DAY_RE = re.compile(r"(?<![\d年.])(\d{1,2})\s*月\s*(\d{1,2})\s*[日号]?")


_CJK_CLOCK_RE = re.compile(r"(\d{1,2})\s*[点时:：]\s*(\d{1,2})?\s*分?")


# ---------------------------------------------------------------------------
# Due-value parsing and recurrence maths
# ---------------------------------------------------------------------------


def _safe_ymd(year: int, month: int, day: int) -> dt.date | None:
    try:
        return dt.date(year, month, day)
    except ValueError:
        return None


def _safe_clock(text: str) -> tuple[int, int] | None:
    """First sane ``HH:MM`` inside a due string; None when there is no time."""
    for match in re.finditer(r"(?:^|[T\s])(\d{1,2}):(\d{2})(?::\d{2})?", text):
        hour, minute = int(match.group(1)), int(match.group(2))
        if hour <= 23 and minute <= 59:
            return hour, minute
    match = _CJK_CLOCK_RE.search(text)
    if match:
        hour = int(match.group(1))
        minute = int(match.group(2) or 0)
        if hour <= 23 and minute <= 59:
            return hour, minute
    return None


def parse_when(text: str, reference: dt.date | None = None) -> tuple[dt.date | None, dt.datetime | None]:
    """Parse a due value into ``(date, datetime|None)``; ``(None, None)`` when unusable.

    Accepts ISO dates/datetimes (``2026-10-12``, ``2026-10-12T09:00``,
    ``2026-10-12 09:00:00+08:00``) plus the CJK forms an LLM tends to echo
    from the material (``2026年10月12日 09:00``, ``10月12日``). A year-less
    value resolves to its next occurrence at or after ``reference``. The
    ``datetime`` element is only set when the value carries a clock time, which
    is what makes conflict detection time-aware instead of day-granular.
    """
    raw = str(text or "").strip()
    if not raw:
        return None, None
    due_date: dt.date | None = None
    match = _ISO_DUE_RE.match(raw)
    if match:
        due_date = _safe_ymd(int(match.group(1)), int(match.group(2)), int(match.group(3)))
    if due_date is None and (match := _CJK_DATE_RE.search(raw)):
        due_date = _safe_ymd(int(match.group(1)), int(match.group(2)), int(match.group(3)))
    if due_date is None and reference is not None and (match := _CJK_MONTH_DAY_RE.search(raw)):
        month, day_number = int(match.group(1)), int(match.group(2))
        for year in (reference.year, reference.year + 1):
            candidate = _safe_ymd(year, month, day_number)
            if candidate is not None and candidate >= reference:
                due_date = candidate
                break
    if due_date is None:
        return None, None
    clock = _safe_clock(raw)
    if clock is None:
        return due_date, None
    return due_date, dt.datetime.combine(due_date, dt.time(clock[0], clock[1]))


def canonical_due(raw, reference: dt.date | None = None) -> str:
    """Normalize a due value to ``YYYY-MM-DD`` or ``YYYY-MM-DDTHH:MM``.

    An unparsable value is kept as a short raw hint (the registry entry then
    simply stays ``unscheduled`` for the clock scan) so the user's wording is
    never silently dropped.
    """
    text = str(raw or "").strip()
    if not text:
        return ""
    due_date, due_dt = parse_when(text, reference)
    if due_date is None:
        return text[:40]
    if due_dt is not None:
        return f"{due_date.isoformat()}T{due_dt.strftime('%H:%M')}"
    return due_date.isoformat()


def _add_months(base: dt.date, months: int) -> dt.date:
    total = base.month - 1 + max(int(months), 0)
    year, month = base.year + total // 12, total % 12 + 1
    return _safe_ymd(year, month, min(base.day, calendar.monthrange(year, month)[1])) or base


# pylint: disable=too-many-return-statements
def next_occurrence(due: dt.date, recurrence: str, reference: dt.date) -> dt.date:
    """First occurrence of a (possibly recurring) due date at or after ``reference``."""
    rule = str(recurrence or "").strip().lower()
    if rule not in _RECURRING or due >= reference:
        return due
    elapsed = (reference - due).days
    if rule == "daily":
        return due + dt.timedelta(days=elapsed)
    if rule == "weekly":
        return due + dt.timedelta(weeks=-(-elapsed // 7))
    if rule == "monthly":
        months = (reference.year - due.year) * 12 + (reference.month - due.month)
        for step in (months, months + 1, months + 2):
            candidate = _add_months(due, step)
            if candidate >= reference:
                return candidate
        return _add_months(due, months + 3)
    years = reference.year - due.year
    for step in (years, years + 1, years + 2):
        candidate = _add_months(due, max(step, 0) * 12)
        if candidate >= reference:
            return candidate
    return _add_months(due, (years + 3) * 12)


def _schedule_window(schedule: ProactiveSchedule, reference: dt.date) -> tuple[dt.datetime, dt.datetime, bool] | None:
    """``(start, end, timed)`` of the next occurrence; None when undated.

    A dated-only schedule spans its whole day (``timed=False``); a schedule
    with a clock time is a point unless ``due_end_at`` widens it. Conflicts
    need at least one timed side, so two all-day items on the same date are a
    busy day rather than a clash.
    """
    due_date, due_dt = parse_when(str(schedule.due_at or ""), reference)
    if due_date is None:
        return None
    rolled = next_occurrence(due_date, schedule.recurrence, reference)
    shift = dt.timedelta(days=(rolled - due_date).days)
    timed = due_dt is not None
    start = (due_dt + shift) if timed else dt.datetime.combine(rolled, dt.time.min)
    end_date, end_dt = parse_when(str(schedule.due_end_at or ""), reference)
    if end_date is None:
        end = start if timed else dt.datetime.combine(rolled, dt.time.max)
    else:
        end = (end_dt + shift) if end_dt is not None else dt.datetime.combine(end_date + shift, dt.time.max)
    return start, max(end, start), timed


def _as_schedule(entry) -> ProactiveSchedule:
    return entry if isinstance(entry, ProactiveSchedule) else ProactiveSchedule.model_validate(entry or {})


# ---------------------------------------------------------------------------
# Registry rendering
# ---------------------------------------------------------------------------


def dump_schedule(schedule) -> dict:
    """Render one registry entry as an ordered dict (state file + prompts)."""
    get = (lambda k: getattr(schedule, k)) if not isinstance(schedule, dict) else schedule.get
    return {
        "id": str(get("id") or ""),
        "title": str(get("title") or ""),
        "detail": str(get("detail") or ""),
        "due_at": str(get("due_at") or ""),
        "due_end_at": str(get("due_end_at") or ""),
        "recurrence": str(get("recurrence") or ""),
        "lead_days": int(get("lead_days") or 0),
        "status": str(get("status") or "open"),
        "confidence": clamp_confidence(get("confidence")),
        "first_seen": str(get("first_seen") or ""),
        "last_seen_at": str(get("last_seen_at") or ""),
        "evidence": str(get("evidence") or "")[:120],
        "paths": [str(p) for p in (get("paths") or [])],
        "history": [dict(item) for item in (get("history") or []) if isinstance(item, dict)],
    }


def slim_schedule(schedule) -> dict:
    """Registry entry without the bulky history/paths/detail (metadata + logs)."""
    dumped = dump_schedule(schedule)
    for key in ("history", "paths", "detail"):
        dumped.pop(key, None)
    return dumped


# ---------------------------------------------------------------------------
# Clock view: due state and conflicts
# ---------------------------------------------------------------------------


def schedule_view(schedules, now_dt, default_lead_days: int = 3) -> list[dict]:
    """Clock-relative view of the registry: due state plus conflicts per entry.

    ``trigger`` is one of ``overdue`` / ``due_today`` / ``conflict`` /
    ``due_soon`` / ``later`` / ``unscheduled`` / ``closed``; the first four are
    the ones that deserve a reminder (``DUE_TRIGGERS``).
    """
    reference = now_dt.date() if isinstance(now_dt, dt.datetime) else now_dt
    lead_default = max(int(default_lead_days), 0)
    entries: list[dict] = []
    for raw in schedules or []:
        try:
            schedule = _as_schedule(raw)
        except Exception:  # noqa: BLE001 - a malformed entry never breaks the scan
            continue
        window = _schedule_window(schedule, reference)
        due_in_days = (window[0].date() - reference).days if window is not None else None
        lead = int(schedule.lead_days or 0) or lead_default
        if schedule.status != "open":
            trigger = "closed"
        elif window is None:
            trigger = "unscheduled"
        elif due_in_days < 0:
            trigger = "overdue"
        elif due_in_days == 0:
            trigger = "due_today"
        elif due_in_days <= lead:
            trigger = "due_soon"
        else:
            trigger = "later"
        entries.append({"schedule": schedule, "window": window, "due_in_days": due_in_days, "trigger": trigger})

    attention = [e for e in entries if e["trigger"] in ("overdue", "due_today", "due_soon")]
    for entry in attention:
        start, end, timed = entry["window"]
        conflicts: list[str] = []
        for other in entries:
            if other is entry or other["schedule"].status != "open" or other["window"] is None:
                continue
            other_start, other_end, other_timed = other["window"]
            if not (timed or other_timed):
                continue
            if start <= other_end and other_start <= end:
                title = str(other["schedule"].title or "").strip()
                if title and title not in conflicts:
                    conflicts.append(title)
        entry["conflict_with"] = conflicts

    # A clash outranks plain proximity (it needs coordination, not just
    # preparation), but one clash is announced once: the earliest occurrence
    # becomes the ``conflict`` reminder and the other sides stay ``due_soon``
    # while still naming the overlap, so a busy day does not repeat itself.
    announced: set[str] = set()
    for entry in sorted(attention, key=lambda e: (e["window"][0], str(e["schedule"].id or ""))):
        conflicts = [str(title) for title in (entry.get("conflict_with") or [])]
        if conflicts and entry["trigger"] == "due_soon" and not all(title in announced for title in conflicts):
            entry["trigger"] = "conflict"
        if entry["trigger"] == "conflict":
            announced.add(str(entry["schedule"].title or ""))
            announced.update(conflicts)

    view: list[dict] = []
    for entry in entries:
        schedule = entry["schedule"]
        dumped = dump_schedule(schedule)
        window = entry["window"]
        if window is None:
            dumped["next_due"] = ""
        elif window[2]:  # timed: keep the clock time so same-day reminders sort by hour
            dumped["next_due"] = window[0].strftime("%Y-%m-%dT%H:%M")
        else:
            dumped["next_due"] = window[0].date().isoformat()
        dumped["due_in_days"] = entry["due_in_days"]
        dumped["trigger"] = entry["trigger"]
        dumped["conflict_with"] = entry.get("conflict_with") or []
        view.append(dumped)
    return view


SCHEDULE_PROMPT_KEYS = (
    "id",
    "title",
    "detail",
    "status",
    "due_at",
    "due_end_at",
    "recurrence",
    "lead_days",
    "confidence",
    "next_due",
    "due_in_days",
    "trigger",
    "conflict_with",
    "first_seen",
    "last_seen_at",
    "evidence",
)


def schedule_prompt_entries(view: list[dict], closed_lookback_days: int = 7, reference: str = "") -> list[dict]:
    """Slim registry view for prompts: every open entry plus recent closures.

    Recently closed entries stay visible so the model does not re-add a
    schedule it already marked done/cancelled.
    """
    cutoff = ""
    if base := safe_date(reference):
        cutoff = (base - dt.timedelta(days=max(int(closed_lookback_days), 0))).isoformat()
    entries: list[dict] = []
    for item in view:
        status = str(item.get("status") or "open")
        if status != "open":
            if not cutoff or str(item.get("last_seen_at") or "") < cutoff:
                continue
        entries.append({key: item.get(key) for key in SCHEDULE_PROMPT_KEYS})
    return entries


# ---------------------------------------------------------------------------
# Clock-driven reminder candidates
# ---------------------------------------------------------------------------


def _reminder_reason(entry: dict) -> str:
    """Human-readable why-now text for a clock-driven reminder candidate."""
    title = str(entry.get("title") or "")
    trigger = str(entry.get("trigger") or "")
    due = str(entry.get("next_due") or entry.get("due_at") or "")
    days = entry.get("due_in_days")
    days_text = "" if days is None else str(abs(int(days)))
    if trigger == "overdue":
        text = f"『{title}』原定 {due}，已过去 {days_text} 天仍未看到结果，需要确认是已完成还是要改期"
    elif trigger == "due_today":
        text = f"『{title}』就在今天（{due}），需要提前留出时间并确认准备事项"
    elif trigger == "conflict":
        text = f"『{title}』安排在 {due}（还有 {days_text} 天），与其它日程时间重叠，需要先协调"
    else:
        text = f"『{title}』安排在 {due}（还有 {days_text} 天），可以提前准备"
    conflicts = [str(item) for item in (entry.get("conflict_with") or []) if str(item).strip()]
    if conflicts and trigger != "conflict":
        text += "；与" + "、".join(f"『{c}』" for c in conflicts[:3]) + "时间重叠"
    if detail := str(entry.get("detail") or "").strip():
        text += f"；备注：{detail[:80]}"
    return text


def reminder_candidates_from_view(view: list[dict], day: str) -> list[dict]:
    """Clock-driven reminder candidates: deterministic, zero LLM calls.

    Guarantees the reminder branch still fires when the LLM is absent, times
    out, or simply misses a due schedule. Candidates cite the registry entry
    (``schedule_id``) and its historical paths, because their evidence is the
    clock rather than this round's changed material.
    """
    out: list[dict] = []
    for entry in view:
        trigger = str(entry.get("trigger") or "")
        if trigger not in DUE_TRIGGERS:
            continue
        title = str(entry.get("title") or "").strip()
        if not title:
            continue
        confidence = clamp_confidence(entry.get("confidence"))
        floor = 0.8 if trigger in ("overdue", "due_today", "conflict") else 0.6
        paths = [str(p) for p in (entry.get("paths") or []) if str(p)]
        evidence = str(entry.get("evidence") or "")[:120] or (paths[0][:120] if paths else "")
        out.append(
            {
                "id": topic_id(title),
                "title": title,
                "reason": _reminder_reason(entry),
                "kind": "reminder",
                "confidence": min(1.0, max(confidence, floor)),
                "first_seen": day,
                "last_evidence_at": day,
                "evidence": evidence,
                "paths": paths,
                "schedule_id": str(entry.get("id") or "") or topic_id(title),
                "due_at": str(entry.get("due_at") or ""),
                # Next occurrence (recurring schedules roll forward): the value
                # a reader shows and the pin order sorts by.
                "next_due": str(entry.get("next_due") or ""),
                "trigger": trigger,
                "due_in_days": entry.get("due_in_days"),
                "conflict_with": [str(c) for c in (entry.get("conflict_with") or [])][:5],
            },
        )
    return sort_reminders(out)


def _reminder_key(candidate: dict) -> str:
    """Merge key for a reminder candidate: its schedule, else its topic id."""
    return str(candidate.get("schedule_id") or candidate.get("id") or "")


def merge_reminders(seeded: list[dict], emitted: list[dict]) -> list[dict]:
    """Merge clock-driven reminders with model-emitted ones (one per schedule).

    The registry anchor stays authoritative: ``due_at``/``next_due``/
    ``schedule_id`` come from the clock scan, because moving a due date is the
    ``schedule_updates`` channel's job (it is evidence-anchored) and must not
    happen as a side effect of wording a reminder. The model's ``reason`` wins
    when it read fresh material - it can say why *now* better than the
    template - and confidence keeps the stronger of the two signals.
    """
    merged: dict[str, dict] = {}
    for candidate in seeded or []:
        if key := _reminder_key(candidate):
            merged[key] = dict(candidate)
    for candidate in emitted or []:
        key = _reminder_key(candidate)
        if not key:
            continue
        base = merged.get(key)
        if base is None:
            merged[key] = dict(candidate)
            continue
        base["reason"] = str(candidate.get("reason") or base.get("reason") or "")
        base["confidence"] = max(
            clamp_confidence(base.get("confidence")),
            clamp_confidence(candidate.get("confidence")),
        )
        base["trigger"] = str(candidate.get("trigger") or base.get("trigger") or "")
        base["due_at"] = str(base.get("due_at") or candidate.get("due_at") or "")
        base["next_due"] = str(base.get("next_due") or candidate.get("next_due") or "")
        known = [str(p) for p in (base.get("paths") or [])]
        base["paths"] = known + [str(p) for p in (candidate.get("paths") or []) if str(p) not in known]
        if not base.get("evidence"):
            base["evidence"] = str(candidate.get("evidence") or "")[:120]
        clashes = [str(c) for c in (base.get("conflict_with") or [])]
        base["conflict_with"] = (
            clashes + [str(c) for c in (candidate.get("conflict_with") or []) if str(c) not in clashes]
        )[:5]
    return sort_reminders(list(merged.values()))


# ---------------------------------------------------------------------------
# Topic bridging (reminder candidates -> truth-source topics)
# ---------------------------------------------------------------------------


def effective_due(candidate) -> str:
    """Due value a consumer should show: next occurrence, else the raw anchor."""
    get = (lambda k: getattr(candidate, k)) if not isinstance(candidate, dict) else candidate.get
    return str(get("next_due") or get("due_at") or "")


def sort_reminders(candidates: list) -> list:
    """Nearest effective due date first, then confidence desc, then id.

    This is the agenda pin order, so a recurring schedule sorts by its *next*
    occurrence (``next_due``) rather than by its original anchor date.
    """

    def get(candidate, key):
        return getattr(candidate, key) if not isinstance(candidate, dict) else candidate.get(key)

    def due_key(candidate) -> str:
        due_date, due_dt = parse_when(effective_due(candidate))
        if due_dt is not None:
            return due_dt.isoformat(timespec="minutes")
        return due_date.isoformat() if due_date is not None else "9999-12-31"

    out = sorted(candidates, key=lambda c: str(get(c, "id") or ""))
    out.sort(key=lambda c: clamp_confidence(get(c, "confidence")), reverse=True)
    out.sort(key=due_key)
    return out


def cap_notice_reminders(candidates: list[dict], cap: int) -> tuple[list[dict], int]:
    """Bound the dateless notice/key-event reminders; clock-anchored ones are free.

    A reminder carrying a due value is a clock fact about the user's own
    calendar and stays uncapped (it pins today's agenda). A dateless one is a
    judgement call about an external notice or key event, so at most ``cap`` of
    them survive a round; returns ``(kept, dropped_over_cap)``.
    """
    if cap <= 0:
        return list(candidates), 0
    kept: list[dict] = []
    notices = dropped = 0
    for item in candidates:
        if str(item.get("next_due") or item.get("due_at") or "").strip():
            kept.append(item)
            continue
        if notices >= cap:
            dropped += 1
            continue
        notices += 1
        kept.append(item)
    return kept, dropped


def topic_from_candidate(candidate: dict) -> ProactiveTopic:
    """Truth-source topic for a cleaned candidate.

    Reminders publish the *effective* due date on the topic (what a reader or
    the agenda wants to show) while the registry keeps the original anchor plus
    its recurrence rule, so a weekly schedule never gets rewritten by its own
    reminder.
    """
    data = dict(candidate)
    if str(data.get("kind") or "") == "reminder":
        data["due_at"] = effective_due(data)
    return ProactiveTopic.model_validate(data)


# ---------------------------------------------------------------------------
# Registry maintenance
# ---------------------------------------------------------------------------


def _history_entry(
    day: str,
    action: str,
    from_value: str = "",
    to_value: str = "",
    reason: str = "",
    evidence: str = "",
) -> dict:
    """One registry change-log record (kept short; capped by ``trim_schedules``)."""
    entry: dict = {"at": day, "action": action}
    if from_value or to_value:
        entry["from"] = str(from_value or "")
        entry["to"] = str(to_value or "")
    if reason:
        entry["reason"] = str(reason)[:200]
    if evidence:
        entry["evidence"] = str(evidence)[:120]
    return entry


def apply_schedule_updates(
    state_file: ProactiveStateFile,
    updates,
    day: str,
    new_material: set[str] | None = None,
) -> tuple[dict[str, int], set[str]]:
    """Maintain the registry from LLM ``schedule_updates``.

    Returns ``(counts, touched)``. ``touched`` holds every schedule id an
    accepted update acted on, so ``upsert_schedule_from_reminder`` can stand
    down for those: moving a due date belongs to this evidence-anchored
    channel, and a reminder candidate may still carry the value the clock scan
    read *before* this round's update landed - applying it would revert the
    change the material actually reported.

    State-changing actions must anchor this round's material (same hard rule as
    topic ``updates``), otherwise they are skipped with a log: a re-read of an
    old note must not be able to rewrite a due date.
    """
    counts = {"added": 0, "updated": 0, "closed": 0, "skipped": 0}
    touched: set[str] = set()
    anchored = {norm_path(p) for p in (new_material or set())}
    by_id = {s.id: s for s in state_file.schedules if s.id}
    reference = safe_date(day)
    for raw in updates or []:
        if not isinstance(raw, dict):
            counts["skipped"] += 1
            continue
        action = str(raw.get("action") or "").strip().lower()
        title = str(raw.get("title") or "").strip()
        sid = str(raw.get("id") or "").strip() or (topic_id(title) if title else "")
        if action not in SCHEDULE_ACTIONS or not sid:
            counts["skipped"] += 1
            continue
        evidence_raw = str(raw.get("evidence") or "").strip()
        evidence_path = norm_path(evidence_raw.split("#", 1)[0])
        schedule = by_id.get(sid)
        if schedule is None and action != "add":
            logger.info(f"schedule update skipped: unknown id {sid!r} for action={action}")
            counts["skipped"] += 1
            continue
        if action == "add" and schedule is not None:
            action = "reschedule" if str(raw.get("due_at") or "").strip() else "confirm"
        if evidence_path not in anchored:
            logger.info(
                f"schedule update skipped: evidence {evidence_path or '<empty>'!r} for {sid!r} "
                f"is not in this round's new material (action={action})",
            )
            counts["skipped"] += 1
            continue
        if action == "add":
            if not title:
                counts["skipped"] += 1
                continue
            created = ProactiveSchedule(
                id=sid,
                title=title,
                detail=str(raw.get("detail") or "")[:200],
                due_at=canonical_due(raw.get("due_at"), reference),
                due_end_at=canonical_due(raw.get("due_end_at"), reference),
                recurrence=str(raw.get("recurrence") or "").strip().lower(),
                lead_days=int(raw.get("lead_days") or 0),
                status="open",
                confidence=clamp_confidence(raw.get("confidence")),
                first_seen=day,
                last_seen_at=day,
                evidence=evidence_raw[:120],
                paths=[str(p) for p in (raw.get("paths") or []) if str(p)],
                history=[
                    _history_entry(
                        day,
                        "add",
                        "",
                        str(raw.get("due_at") or ""),
                        str(raw.get("reason") or ""),
                        evidence_raw,
                    ),
                ],
            )
            state_file.schedules.append(created)
            by_id[sid] = created
            counts["added"] += 1
            touched.add(sid)
            continue
        changed = _mutate_schedule(schedule, action, raw, day, reference, evidence_raw)
        touched.add(sid)
        if action in ("done", "cancel"):
            counts["closed"] += 1
        elif changed:
            counts["updated"] += 1
    return counts, touched


def _mutate_schedule(
    schedule: ProactiveSchedule,
    action: str,
    raw: dict,
    day: str,
    reference,
    evidence_raw: str,
) -> bool:
    """Apply one non-``add`` registry action; returns whether anything changed."""
    schedule.last_seen_at = day
    reason = str(raw.get("reason") or "")[:200]
    changed = False
    if action in ("reschedule", "confirm"):
        new_due = canonical_due(raw.get("due_at"), reference)
        if new_due and new_due != schedule.due_at:
            schedule.history.append(_history_entry(day, "reschedule", schedule.due_at, new_due, reason, evidence_raw))
            schedule.due_at = new_due
            changed = True
        new_end = canonical_due(raw.get("due_end_at"), reference)
        if new_end and new_end != schedule.due_end_at:
            schedule.due_end_at = new_end
            changed = True
        recurrence = str(raw.get("recurrence") or "").strip().lower()
        if recurrence in RECURRENCES and recurrence != schedule.recurrence:
            schedule.recurrence = recurrence
            changed = True
        if raw.get("lead_days") is not None:
            lead = int(raw.get("lead_days") or 0)
            if lead != schedule.lead_days:
                schedule.lead_days = max(0, min(lead, 365))
                changed = True
        if raw.get("detail"):
            schedule.detail = str(raw["detail"])[:200]
        if raw.get("confidence") is not None:
            schedule.confidence = clamp_confidence(raw.get("confidence"))
        if evidence_raw:
            schedule.evidence = evidence_raw[:120]
        if raw.get("paths"):
            schedule.paths = [str(p) for p in raw["paths"] if str(p)]
        if not changed:
            schedule.history.append(_history_entry(day, "confirm", "", "", reason, evidence_raw))
        return changed
    status = "done" if action == "done" else "cancelled"
    if schedule.status != status:
        schedule.history.append(_history_entry(day, action, schedule.status, status, reason, evidence_raw))
        schedule.status = status
        changed = True
    return changed


def refresh_reminder_dues(reminders: list[dict], state_file: ProactiveStateFile, touched: set[str], now_dt) -> int:
    """Re-anchor reminders on the due value this round's updates just moved.

    The clock scan reads the registry *before* ``schedule_updates`` land, so a
    seeded reminder still carries the pre-update value while the registry has
    moved on. A reader sorts and notifies by that field, so it is re-read from
    the registry - through the same clock view, which also rolls a recurring
    schedule to its next occurrence - instead of publishing a time that no
    longer holds. Returns how many reminders were re-anchored.
    """
    if not touched:
        return 0
    moved = [s for s in state_file.schedules if s.id in touched]
    entries = {str(entry.get("id") or ""): entry for entry in schedule_view(moved, now_dt)}
    refreshed = 0
    for candidate in reminders or []:
        entry = entries.get(str(candidate.get("schedule_id") or ""))
        if entry is None:
            continue
        due = str(entry.get("due_at") or "")
        next_due = str(entry.get("next_due") or "")
        if not due:
            continue
        if str(candidate.get("due_at") or "") == due and str(candidate.get("next_due") or "") == next_due:
            continue
        candidate["due_at"] = due
        candidate["next_due"] = next_due
        refreshed += 1
    return refreshed


def upsert_schedule_from_reminder(
    state_file: ProactiveStateFile,
    candidate: dict,
    day: str,
    protected: set[str] | None = None,
) -> str:
    """Register/refresh the schedule behind a reminder candidate.

    Safety net for the registry: a reminder that names a due date keeps the
    schedule alive even when the model forgot the matching
    ``schedule_updates`` entry. Returns ``added``/``updated``/``adopted``/````.

    ``protected`` lists the ids an explicit ``schedule_updates`` entry already
    acted on this round. For those the reschedule branch stands down, so this
    derived channel can never revert the authoritative one; registration of a
    brand-new schedule is unaffected - except that it yields to an explicit
    entry that already registered the very same matter (see ``_registered_this_round``).
    """
    title = str(candidate.get("title") or "").strip()
    if not title:
        return ""
    sid = str(candidate.get("schedule_id") or "").strip() or topic_id(title)
    due_raw = str(candidate.get("due_at") or "").strip()
    reference = safe_date(day)
    existing = next((s for s in state_file.schedules if s.id == sid), None)
    if existing is None:
        if not due_raw:
            return ""
        canonical = canonical_due(due_raw, reference)
        adopted = _registered_this_round(state_file, protected or set(), candidate, canonical)
        if adopted:
            # Publish the authoritative id so the topic, the agenda card and any
            # later reader point at a schedule that really exists.
            candidate["schedule_id"] = adopted
            return "adopted"
        state_file.schedules.append(
            ProactiveSchedule(
                id=sid,
                title=title,
                detail=str(candidate.get("reason") or "")[:200],
                due_at=canonical,
                status="open",
                confidence=clamp_confidence(candidate.get("confidence")),
                first_seen=day,
                last_seen_at=day,
                evidence=str(candidate.get("evidence") or "")[:120],
                paths=[str(p) for p in (candidate.get("paths") or []) if str(p)],
                history=[_history_entry(day, "add", "", due_raw, "registered from a reminder candidate", "")],
            ),
        )
        return "added"
    existing.last_seen_at = day
    canonical = canonical_due(due_raw, reference)
    if canonical and canonical != existing.due_at and sid not in (protected or set()):
        existing.history.append(
            _history_entry(
                day,
                "reschedule",
                existing.due_at,
                canonical,
                "reminder candidate carries a new due value",
                "",
            ),
        )
        existing.due_at = canonical
        return "updated"
    return ""


def _registered_this_round(state_file: ProactiveStateFile, protected: set[str], candidate: dict, canonical: str) -> str:
    """Id of the schedule an explicit update already registered for this matter.

    The two registration channels name a matter independently, so one notice
    routinely arrives under two titles - and therefore two ids. Same evidence
    *and* same due value is one matter, and registering it twice would pin two
    agenda cards for it while the clock scan reported a schedule clashing with
    itself. A notice carrying two genuinely distinct matters keeps both, since
    their due values differ. Only ids this round's authoritative channel acted
    on are considered, so schedules recorded in earlier rounds are never merged
    away by this derived one.
    """
    evidence = str(candidate.get("evidence") or "").strip()
    if not (protected and evidence and canonical):
        return ""
    for schedule in state_file.schedules:
        if schedule.id not in protected or schedule.status != "open":
            continue
        if str(schedule.evidence or "").strip() == evidence and str(schedule.due_at or "") == canonical:
            return schedule.id
    return ""


def trim_schedules(
    state_file: ProactiveStateFile,
    day: str,
    retention_days: int = 180,
    max_schedules: int = 80,
    max_history: int = 8,
) -> int:
    """Registry lifecycle: expire hopeless overdues, drop aged entries, cap size.

    Deliberately much longer-lived than ``trim_state_file`` (topics): a
    schedule mentioned months ago must still fire on its due date. Entries are
    dropped only once they are closed (or unscheduled and unevidenced) for
    longer than ``retention_days``, and open entries outrank closed ones under
    the ``max_schedules`` cap.
    """
    base = safe_date(day)
    if base is None:
        return 0
    retention = max(int(retention_days), 1)
    cutoff = base - dt.timedelta(days=retention)
    for schedule in state_file.schedules:
        if 0 < max_history < len(schedule.history):
            schedule.history = schedule.history[-max_history:]
        due_date, _ = parse_when(str(schedule.due_at or ""), base)
        if schedule.status == "open" and due_date is not None and due_date < cutoff:
            schedule.status = "cancelled"
            schedule.last_seen_at = day
            schedule.history.append(
                _history_entry(
                    day,
                    "expire",
                    schedule.due_at,
                    "",
                    f"overdue by more than {retention} days; auto-closed",
                    "",
                ),
            )
    kept: list[ProactiveSchedule] = []
    dropped = 0
    for schedule in state_file.schedules:
        due_date, _ = parse_when(str(schedule.due_at or ""), base)
        anchor = safe_date(schedule.last_seen_at) or due_date
        if schedule.status != "open":
            if anchor is not None and anchor < cutoff:
                dropped += 1
                continue
        elif due_date is None and anchor is not None and anchor < cutoff:
            dropped += 1
            continue
        kept.append(schedule)
    limit = max(int(max_schedules), 0)
    if limit and len(kept) > limit:
        # Open entries (nearest due first) always outrank closed ones, which are
        # kept newest-first, so the cap drops the least useful history.
        open_entries = sorted((s for s in kept if s.status == "open"), key=lambda s: (str(s.due_at or "9999"), s.id))
        closed_entries = sorted(
            (s for s in kept if s.status != "open"),
            key=lambda s: str(s.last_seen_at or ""),
            reverse=True,
        )
        ordered = (open_entries + closed_entries)[:limit]
        dropped += len(kept) - len(ordered)
        kept = ordered
    state_file.schedules = kept
    return dropped


# ---------------------------------------------------------------------------
# Candidate cleaning (extract side, branch C)
# ---------------------------------------------------------------------------


def clean_reminder(
    raw,
    allowed_paths: set[str],
    day: str,
    schedules_by_id: dict[str, dict] | None = None,
    notice_paths: set[str] | None = None,
) -> dict:
    """Clean one LLM reminder candidate (branch C).

    Traceability differs from the other kinds on purpose: a reminder about a
    schedule mentioned weeks ago has no path in this round's material, so a
    registry-linked reminder may cite the schedule's own historical paths. A
    reminder that opens a *new* schedule must still be anchored in
    ``changed_paths`` and must carry a due value.

    A reminder anchored on an external notice - a path from ``notice_paths``,
    i.e. the ``resource/`` layer - may be dateless, because a key event worth
    telling the user about need not have a due date. ``cap_notice_reminders``
    bounds that softer sub-type, so the uncapped, agenda-pinning reminder
    channel cannot be flooded with dateless judgement calls.
    """
    if not isinstance(raw, dict):
        return {}
    title = str(raw.get("title") or "").strip()
    reason = str(raw.get("reason") or "").strip()
    if not title or not reason:
        return {}
    registry = schedules_by_id or {}
    schedule_id_value = str(raw.get("schedule_id") or "").strip()
    schedule = registry.get(schedule_id_value) or registry.get(topic_id(title)) or {}
    reference = safe_date(day)
    due_at = canonical_due(raw.get("due_at"), reference) or str(schedule.get("due_at") or "")
    paths = clean_paths(raw.get("paths"), allowed_paths)
    for path in [str(p) for p in (schedule.get("paths") or []) if str(p)]:
        if path not in paths:
            paths.append(path)
    notice_anchored = bool(set(paths) & set(notice_paths or ()))
    if not schedule and (not paths or (not due_at and not notice_anchored)):
        return {}
    trigger = str(raw.get("trigger") or "").strip().lower()
    if trigger not in REMINDER_TRIGGERS:
        trigger = ""
    conflicts = raw.get("conflict_with")
    evidence = str(raw.get("evidence") or "").strip() or str(schedule.get("evidence") or "")
    return {
        "id": topic_id(title),
        "title": title,
        "reason": reason,
        "kind": "reminder",
        "confidence": clamp_confidence(raw.get("confidence")),
        "first_seen": day,
        "last_evidence_at": day,
        "evidence": evidence[:120] or (paths[0][:120] if paths else ""),
        "paths": paths,
        "schedule_id": schedule_id_value or str(schedule.get("id") or "") or topic_id(title),
        "due_at": due_at,
        "next_due": due_at,
        "trigger": trigger,
        "conflict_with": (
            [str(item).strip()[:80] for item in conflicts if str(item).strip()][:5]
            if isinstance(conflicts, list)
            else []
        ),
    }


def clean_schedule_update(raw, allowed_paths: set[str], day: str) -> dict:
    """Normalize one LLM ``schedule_updates`` entry (registry maintenance)."""
    if not isinstance(raw, dict):
        return {}
    action = str(raw.get("action") or "").strip().lower()
    if action not in SCHEDULE_ACTIONS:
        return {}
    title = str(raw.get("title") or "").strip()
    sid = str(raw.get("id") or "").strip() or (topic_id(title) if title else "")
    if not sid:
        return {}
    reference = safe_date(day)
    update: dict = {
        "id": sid,
        "action": action,
        "title": title,
        "reason": str(raw.get("reason") or "").strip()[:200],
        "evidence": str(raw.get("evidence") or "").strip(),
        "paths": clean_paths(raw.get("paths"), allowed_paths),
    }
    if str(raw.get("due_at") or "").strip():
        update["due_at"] = canonical_due(raw.get("due_at"), reference)
    if str(raw.get("due_end_at") or "").strip():
        update["due_end_at"] = canonical_due(raw.get("due_end_at"), reference)
    recurrence = str(raw.get("recurrence") or "").strip().lower()
    if recurrence in RECURRENCES:
        update["recurrence"] = recurrence
    if raw.get("lead_days") is not None:
        try:
            update["lead_days"] = max(0, min(int(raw.get("lead_days")), 365))
        except (TypeError, ValueError):
            pass
    if str(raw.get("detail") or "").strip():
        update["detail"] = str(raw["detail"]).strip()[:200]
    if raw.get("confidence") is not None:
        update["confidence"] = clamp_confidence(raw.get("confidence"))
    return update
