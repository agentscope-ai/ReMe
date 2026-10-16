"""Proactive refresh schemas.

Defines the topic model (v2), the chain-shared context state, and the
``daily/_proactive.yaml`` truth-source file model. The LLM reply contract is
validated structurally by ``parse_extract_reply`` instead of a model here.
See ``PROACTIVE_SPEC.md`` sections F1/A2 for the full contracts.
"""

import datetime as dt

from pydantic import BaseModel, ConfigDict, Field, field_validator

# Topic kinds, ordered by proactive priority: ``reminder`` outranks the rest
# and pins today's agenda, ``interest_extend`` is the lowest-priority
# discovery kind. ``KIND_PRIORITY`` is the single ordering source used by
# ``sort_topics`` and by the plan/agenda steps.
TOPIC_KINDS = ("reminder", "follow_up", "new_idea", "interest_extend")
KIND_PRIORITY = {kind: index for index, kind in enumerate(TOPIC_KINDS)}
DEFAULT_KIND = "interest_extend"

# Reminder registry (``daily/_proactive.yaml: schedules``) vocabularies. The
# registry outlives ``open_topics`` because a schedule mentioned weeks ago
# still has to fire on its due date.
SCHEDULE_ACTIONS = ("add", "reschedule", "confirm", "done", "cancel")
SCHEDULE_STATUSES = ("open", "done", "cancelled")
RECURRENCES = ("", "daily", "weekly", "monthly", "yearly")
REMINDER_TRIGGERS = ("overdue", "due_today", "due_soon", "conflict")

# Delivery feedback (``daily/_proactive.yaml: deliveries``) vocabularies.
# A host agent reports back which agenda items it already surfaced or ran, so
# the reader and the agenda step can stop repeating them. Reporting is
# optional: a framework that never calls ``proactive_feedback`` leaves the
# section empty and the whole chain behaves exactly as before.
DELIVERY_STATUSES = ("shown", "executed", "dismissed", "failed")
DELIVERY_CHANNELS = ("in_conversation", "inbox", "notification", "other")
DEFAULT_DELIVERY_STATUS = "shown"
DEFAULT_DELIVERY_CHANNEL = "other"
# Statuses that retire an agenda item for the whole cooldown window. ``failed``
# is deliberately absent: a failed run should be retryable on the next round.
TERMINAL_DELIVERY_STATUSES = ("executed", "dismissed")
# ``outcome`` is a host-written summary; cap it so a chatty consumer cannot
# inflate the truth source without bound.
MAX_DELIVERY_OUTCOME_CHARS = 500

# Keys inside the untyped ``history`` audit trail that hold a date/datetime.
HISTORY_WHEN_KEYS = ("at", "from", "to")


def clamp_confidence(value) -> float:
    """Coerce confidence into [0, 1]; any conversion failure falls back to 0.5."""
    try:
        return min(1.0, max(0.0, float(value)))
    except (TypeError, ValueError):
        return 0.5


def iso_text(value) -> str:
    """Coerce a YAML-native scalar into the ISO text the string fields expect.

    ``yaml.safe_load`` reads an unquoted ``2026-10-11`` as ``datetime.date`` and
    ``2026-10-11T20:00`` as ``datetime.datetime``. The truth source is a
    hand-editable file and the schedule registry is meant to outlive topics by
    months, so a bare date must never invalidate an entry: date and time
    scalars are rendered back to ISO text, anything else falls back to ``str``.
    """
    if value is None:
        return ""
    if isinstance(value, (dt.datetime, dt.date, dt.time)):
        return value.isoformat()
    if isinstance(value, str):
        return value
    return str(value)


def kind_priority(kind) -> int:
    """Priority rank of a topic kind; unknown kinds sort last."""
    return KIND_PRIORITY.get(str(kind or "").strip(), len(TOPIC_KINDS))


class ProactiveTopic(BaseModel):
    """One proactive topic; every field has a default so v1 files parse seamlessly.

    Fallback rules (A2): invalid ``kind`` -> ``interest_extend``; unparseable
    ``confidence`` -> 0.5. Missing ``id``, ``first_seen`` and
    ``last_evidence_at`` are context-dependent and therefore resolved by the
    loaders, not here.
    """

    id: str = ""
    title: str = ""
    reason: str = ""
    kind: str = DEFAULT_KIND
    confidence: float = 0.5
    first_seen: str = ""
    last_evidence_at: str = ""
    evidence: str = ""
    paths: list[str] = Field(default_factory=list)
    # Reminder-only links: ``schedule_id`` points at the long-lived registry
    # entry, ``due_at`` is the ISO date/datetime the reminder is about.
    schedule_id: str = ""
    due_at: str = ""

    @field_validator("kind", mode="before")
    @classmethod
    def _fallback_kind(cls, value):
        text = str(value or "").strip()
        return text if text in TOPIC_KINDS else DEFAULT_KIND

    @field_validator("confidence", mode="before")
    @classmethod
    def _fallback_confidence(cls, value):
        return clamp_confidence(value)

    @field_validator("first_seen", "last_evidence_at", "due_at", mode="before")
    @classmethod
    def _iso_when(cls, value):
        return iso_text(value)

    @field_validator("paths", mode="before")
    @classmethod
    def _clean_str_list(cls, value):
        if not isinstance(value, list):
            return []
        return [str(item).strip() for item in value if str(item).strip()]


class ProactiveSchedule(BaseModel):
    """One personal schedule/appointment in the long-lived reminder registry.

    The registry is a separate truth-source section from ``open_topics``: a
    schedule survives topic trimming (``carry_forward_days``) until its due
    date has passed by ``schedule_retention_days``, and every change the
    pipeline observes is appended to ``history`` so a reschedule or a
    cancellation stays auditable long after the originating daily note left
    the extract scan window.
    """

    model_config = ConfigDict(extra="ignore")

    id: str = ""
    title: str = ""
    detail: str = ""
    due_at: str = ""
    due_end_at: str = ""
    recurrence: str = ""
    # 0 = "not specified": the reminder scan then applies the step-level
    # ``reminder_lead_days`` default instead of a per-schedule value.
    lead_days: int = 0
    status: str = "open"
    confidence: float = 0.5
    first_seen: str = ""
    last_seen_at: str = ""
    evidence: str = ""
    paths: list[str] = Field(default_factory=list)
    history: list[dict] = Field(default_factory=list)

    @field_validator("status", mode="before")
    @classmethod
    def _fallback_status(cls, value):
        text = str(value or "").strip()
        return text if text in SCHEDULE_STATUSES else "open"

    @field_validator("recurrence", mode="before")
    @classmethod
    def _fallback_recurrence(cls, value):
        text = str(value or "").strip().lower()
        return text if text in RECURRENCES else ""

    @field_validator("due_at", "due_end_at", "first_seen", "last_seen_at", mode="before")
    @classmethod
    def _iso_when(cls, value):
        return iso_text(value)

    @field_validator("confidence", mode="before")
    @classmethod
    def _fallback_confidence(cls, value):
        return clamp_confidence(value)

    @field_validator("lead_days", mode="before")
    @classmethod
    def _fallback_lead_days(cls, value):
        try:
            return max(0, min(int(value), 365))
        except (TypeError, ValueError):
            return 0

    @field_validator("paths", mode="before")
    @classmethod
    def _clean_str_list(cls, value):
        if not isinstance(value, list):
            return []
        return [str(item).strip() for item in value if str(item).strip()]

    @field_validator("history", mode="before")
    @classmethod
    def _clean_history(cls, value):
        if not isinstance(value, list):
            return []
        cleaned = []
        for item in value:
            if not isinstance(item, dict):
                continue
            cleaned.append({k: iso_text(v) if k in HISTORY_WHEN_KEYS else v for k, v in item.items()})
        return cleaned


class ProactiveDelivery(BaseModel):
    """One host-side report that an agenda item was surfaced or executed.

    The identity of a record is ``(topic_id, external_ref)`` when the host
    supplies a reference (an inbox event id, a session id), and
    ``(topic_id, agenda_date, channel)`` otherwise, so repeated reports of the
    same delivery update in place instead of piling up. ``status`` only moves
    forward: ``shown`` may become ``executed``/``dismissed``/``failed``, but a
    terminal record is never downgraded back to ``shown``.
    """

    model_config = ConfigDict(extra="ignore")

    topic_id: str = ""
    agenda_date: str = ""
    delivered_at: str = ""
    channel: str = DEFAULT_DELIVERY_CHANNEL
    consumer: str = ""
    status: str = DEFAULT_DELIVERY_STATUS
    outcome: str = ""
    external_ref: str = ""

    @field_validator("status", mode="before")
    @classmethod
    def _fallback_status(cls, value):
        text = str(value or "").strip().lower()
        return text if text in DELIVERY_STATUSES else DEFAULT_DELIVERY_STATUS

    @field_validator("channel", mode="before")
    @classmethod
    def _fallback_channel(cls, value):
        text = str(value or "").strip().lower()
        return text if text in DELIVERY_CHANNELS else DEFAULT_DELIVERY_CHANNEL

    @field_validator("agenda_date", "delivered_at", mode="before")
    @classmethod
    def _iso_when(cls, value):
        return iso_text(value)

    @field_validator("outcome", mode="before")
    @classmethod
    def _truncate_outcome(cls, value):
        return str(value or "")[:MAX_DELIVERY_OUTCOME_CHARS]

    def identity(self) -> tuple[str, str]:
        """Idempotency key: explicit host reference wins over the natural key."""
        if self.external_ref:
            return (self.topic_id, f"ref:{self.external_ref}")
        return (self.topic_id, f"day:{self.agenda_date}|{self.channel}")


class ProactiveState(BaseModel):
    """Chain-shared proactive context state (``context['proactive']``).

    Extract fills change-detection/carry-forward/LLM output fields; topics fills the
    filtering fields plus ``push_candidates`` (today's pushable topics); plan
    expands candidates into ``scenario_cards``; agenda selects the ordered
    ``agenda`` and records ``suppressed`` candidates with reasons; finish
    records the catalog checkpoint. ``plan_llm_calls`` counts plan+agenda LLM
    calls separately from extract's ``llm_calls``.
    ``file_skip_reason`` is metadata/log only and never persisted to
    interests.yaml (v5 simplification R7).
    """

    date: str = ""
    daily_dir: str = "daily"
    workspace: str = ""
    scan_days: int = 2
    carry_forward_days: int = 14
    changed_paths: list[str] = Field(default_factory=list)
    changed_mtimes: dict[str, float] = Field(default_factory=dict)
    # External source records (``resource/``) scanned for user-facing notices.
    # They feed the reminder branch only, but share ``changed_mtimes`` and the
    # proactive catalog watermark with the daily material set, so a notice that
    # lands on a day without any conversation still triggers a round.
    resource_scanned: int = 0
    resource_skipped: int = 0
    resource_changed: list[str] = Field(default_factory=list)
    carry_forward_count: int = 0
    carry_forward_prompt: list[ProactiveTopic] = Field(default_factory=list)
    llm_calls: int = 0
    follow_ups: list[dict] = Field(default_factory=list)
    extends: list[dict] = Field(default_factory=list)
    reminders: list[dict] = Field(default_factory=list)
    new_ideas: list[dict] = Field(default_factory=list)
    updates: list[dict] = Field(default_factory=list)
    schedule_updates: list[dict] = Field(default_factory=list)
    # Stage-2 adjudication (precision pass over stage-1 candidates). Every
    # failure is fail-open, so ``refine_degraded`` counts the kinds whose
    # stage-1 candidates were kept untouched.
    refine_calls: int = 0
    refine_degraded: int = 0
    refine_dropped: list[dict] = Field(default_factory=list)
    refine_moved: list[dict] = Field(default_factory=list)
    early_exit: str = ""
    # ``reminder_only`` marks a clock-driven round with no new material: the
    # chain stays alive (no ``early_exit``) but made zero LLM calls.
    reminder_only: bool = False
    due_schedules: list[dict] = Field(default_factory=list)
    updates_applied: int = 0
    updates_resolved: int = 0
    schedules_added: int = 0
    schedules_updated: int = 0
    schedules_closed: int = 0
    # Reminders whose published due value was re-read from the registry because
    # this round's ``schedule_updates`` moved it after the clock scan.
    reminders_reanchored: int = 0
    schedules_out: list[dict] = Field(default_factory=list)
    candidates_in: int = 0
    candidates: list[dict] = Field(default_factory=list)
    dropped_missing: int = 0
    dropped_duplicate: int = 0
    dropped_known: int = 0
    dropped_over_cap: int = 0
    topics_out: list[dict] = Field(default_factory=list)
    push_candidates: list[dict] = Field(default_factory=list)
    scenario_cards: list[dict] = Field(default_factory=list)
    agenda: list[dict] = Field(default_factory=list)
    suppressed: list[dict] = Field(default_factory=list)
    # Candidates withheld because the host already reported them delivered.
    suppressed_delivered: int = 0
    plan_llm_calls: int = 0
    push: bool = False
    file_skip_reason: str = ""
    interests_path: str = ""
    interests_written: bool = False
    checkpoint_paths: list[str] = Field(default_factory=list)
    duration_ms: int = 0
    # Delivery-feedback round counters (metadata/log only).
    deliveries_in: int = 0
    deliveries_added: int = 0
    deliveries_updated: int = 0
    deliveries_dropped: int = 0


class ProactiveStateFile(BaseModel):
    """On-disk truth-source ``daily/_proactive.yaml`` (F1.3, v5: 3 sections).

    ``resolved`` tombstones carry ``first_seen`` so a resurrected topic can
    keep its original age anchor (F2.4 reopen channel).
    ``schedules`` is the long-lived reminder registry; it is optional so
    files written before the reminder branch existed still load.
    ``deliveries`` is the optional host-feedback log; it is likewise absent
    from files written before delivery reporting existed.
    """

    model_config = ConfigDict(extra="ignore")

    version: int = 1
    open_topics: list[ProactiveTopic] = Field(default_factory=list)
    resolved: list[dict] = Field(default_factory=list)
    schedules: list[ProactiveSchedule] = Field(default_factory=list)
    deliveries: list[ProactiveDelivery] = Field(default_factory=list)


class ProactiveResult(BaseModel):
    """Result of reading daily interest topics (F5)."""

    date: str = ""
    path: str = ""
    topics: list[dict] = Field(default_factory=list)
    content: str = ""
    skipped: bool = False
    error: str = ""
    summary: str = ""
    push: bool | None = None
    generated_at: str = ""
    agenda: list[dict] = Field(default_factory=list)
    # Agenda topic_ids withheld because the host already reported them as
    # delivered; kept so a caller can see the suppression instead of guessing.
    delivered_ids: list[str] = Field(default_factory=list)
