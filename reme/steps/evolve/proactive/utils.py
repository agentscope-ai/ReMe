"""Shared proactive helpers: frozen topic identity, truth-source state, rendering.

Implements the A7 skeleton from PROACTIVE_SPEC.md. ``normalize_topic`` is a
frozen contract (INV-4): any change to it drifts every historical topic id.
"""

import contextlib
import datetime as dt
import hashlib
import os
import re
import tempfile
import time
import unicodedata
from pathlib import Path

import yaml

from ....enumeration import ComponentEnum
from ....schema import ProactiveDelivery, ProactiveSchedule, ProactiveStateFile, ProactiveTopic
from ....schema.proactive import (
    DEFAULT_DELIVERY_CHANNEL,
    DEFAULT_DELIVERY_STATUS,
    DEFAULT_KIND,
    TERMINAL_DELIVERY_STATUSES,
    TOPIC_KINDS,
    clamp_confidence,
    kind_priority,
)
from ....utils import get_logger
from ...file_io._file_io import get_path_lock
from .._evolve import now
from ..dream.utils import clean_paths, recent_dates, scan_day_files

logger = get_logger(log_to_file=False)


PROACTIVE_STATE_NAME = "_proactive.yaml"


INTERESTS_NAME = "interests.yaml"


EXTRACT_SECTIONS = ("follow_ups", "extends", "reminders", "new_ideas", "updates", "schedule_updates")

# Registry triggers that deserve a reminder; ``later``/``unscheduled``/``closed``
# stay in the prompt view but never become candidates.


def load_yaml_topics(path: Path, *, strict: bool = False) -> list[dict]:
    """Load legacy or current interests YAML topics."""
    if not path.is_file():
        return []
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except Exception as exc:
        if strict:
            raise ValueError(f"Invalid interests YAML at {path}: {exc}") from exc
        return []
    if data is None:
        if strict:
            raise ValueError(f"Invalid interests YAML at {path}: expected an object")
        return []
    topics = data.get("topics") if isinstance(data, dict) else None
    if not isinstance(topics, list):
        if strict:
            raise ValueError(f"Invalid interests YAML at {path}: topics must be a list")
        return []
    cleaned_topics = []
    for index, topic in enumerate(topics):
        if strict:
            _validate_topic(topic, path, index)
        if isinstance(topic, dict) and (cleaned := _clean_legacy_topic(topic)):
            cleaned_topics.append(cleaned)
    return cleaned_topics


def _validate_topic(topic: object, path: Path, index: int) -> None:
    """Reject topic data that would otherwise be silently discarded or coerced."""
    prefix = f"Invalid interests YAML at {path}: topics[{index}]"
    if not isinstance(topic, dict):
        raise ValueError(f"{prefix} must be an object")

    allowed = {"title", "reason", "evidence", "keywords", "paths"}
    if unknown := sorted(set(topic) - allowed):
        raise ValueError(f"{prefix} has unknown field(s): {', '.join(str(key) for key in unknown)}")

    for field in ("title", "reason"):
        value = topic.get(field)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{prefix}.{field} must be a non-empty string")
    if "evidence" in topic and not isinstance(topic["evidence"], str):
        raise ValueError(f"{prefix}.evidence must be a string")
    for field in ("keywords", "paths"):
        if field not in topic:
            continue
        values = topic[field]
        if not isinstance(values, list) or any(not isinstance(value, str) or not value.strip() for value in values):
            raise ValueError(f"{prefix}.{field} must be a list of non-empty strings")


def _clean_legacy_topic(raw: dict) -> dict:
    """Normalize the v1 topic shape returned by the compatibility reader."""
    title = str(raw.get("title") or "").strip()
    reason = str(raw.get("reason") or "").strip()
    if not title or not reason:
        return {}
    keywords = raw.get("keywords") or []
    paths = raw.get("paths") or []
    return {
        "title": title,
        "reason": reason,
        "evidence": str(raw.get("evidence") or "").strip(),
        "keywords": ([str(k).strip() for k in keywords if str(k).strip()] if isinstance(keywords, list) else []),
        "paths": ([str(p).strip() for p in paths if str(p).strip()] if isinstance(paths, list) else []),
    }


# ---------------------------------------------------------------------------


def normalize_topic(title: str) -> str:
    """NFKC -> casefold -> keep only chars whose category starts with L/N.

    Frozen contract (INV-4): removing all whitespace/punctuation means any
    modification would drift every historical topic id.
    """
    text = unicodedata.normalize("NFKC", title or "").casefold()
    return "".join(ch for ch in text if unicodedata.category(ch)[0] in ("L", "N"))


def topic_id(title: str) -> str:
    """Stable topic identity: ``sha1(normalize_topic(title))[:12]``."""
    return hashlib.sha1(normalize_topic(title).encode("utf-8")).hexdigest()[:12]


# ---------------------------------------------------------------------------


def state_file_path(ws: Path, daily: str = "daily") -> Path:
    """Truth-source path ``daily/_proactive.yaml``."""
    return ws / daily / PROACTIVE_STATE_NAME


def interests_path_for(ws: Path, daily: str, day: str) -> Path:
    """Exposure-product path ``daily/<day>/interests.yaml``."""
    return ws / daily / day / INTERESTS_NAME


def norm_path(rel) -> str:
    """Normalize a workspace-relative path (posix, no leading ./)."""
    text = str(rel or "").strip().replace("\\", "/")
    while text.startswith("./"):
        text = text[2:]
    return text


def scan_material_daily(ws: Path, day: str, daily: str, scan_days: int) -> list[str]:
    """M_daily: chunk notes in the scan window, minus day indexes and ``_*`` files (INV-11)."""
    out: list[str] = []
    for scan_day in recent_dates(day, scan_days):
        day_index = f"{daily}/{scan_day}.md"
        for rel in scan_day_files(ws, scan_day, daily):
            rel = norm_path(rel)
            base = rel.rsplit("/", 1)[-1]
            if rel == day_index or base.startswith("_"):
                continue
            if rel not in out:
                out.append(rel)
    return sorted(out)


# External source records the reminder branch may read directly. Binary sources
# are deliberately absent: ``auto_resource`` interprets them into daily cards,
# which already reach extraction through ``scan_material_daily``.
RESOURCE_NOTICE_SUFFIXES = (
    ".md",
    ".markdown",
    ".txt",
    ".json",
    ".yaml",
    ".yml",
    ".csv",
    ".tsv",
    ".html",
    ".eml",
    ".ics",
)


def _mtime_day(path: Path) -> str:
    """Local date a file last changed, or ``""`` when it cannot be stat'ed."""
    try:
        return dt.date.fromtimestamp(path.stat().st_mtime).isoformat()
    except OSError:
        return ""


def scan_material_resource(
    ws: Path,
    day: str,
    scan_days: int,
    resource: str = "resource",
) -> tuple[list[str], int]:
    """M_resource: external source records that may carry notices for the user.

    Mirrors the ``resource/`` entry-point semantics of Auto Resource: a dated
    directory ``resource/<YYYY-MM-DD>/...`` belongs to that date, while a file
    directly under ``resource/`` (or under a non-dated folder) belongs to the
    day it landed, i.e. its mtime. Returns ``(paths, skipped_binary)``; only
    text-readable suffixes are returned, because a binary source has to be
    interpreted by ``auto_resource`` before anyone can quote it.

    This is the reserved hook for future external source interfaces: whatever
    writes into ``resource/`` becomes eligible for the reminder review without
    any further change to the proactive chain.
    """
    window = set(recent_dates(day, scan_days))
    root = ws / resource
    if not window or not root.is_dir():
        return [], 0
    out: list[str] = []
    skipped = 0
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        rel = norm_path(path.relative_to(ws).as_posix())
        parts = rel.split("/")
        if any(part.startswith("_") or part.startswith(".") for part in parts[1:]):
            continue
        owner = parts[1] if len(parts) > 2 and safe_date(parts[1]) else _mtime_day(path)
        if owner not in window:
            continue
        if path.suffix.lower() not in RESOURCE_NOTICE_SUFFIXES:
            skipped += 1
            continue
        if rel not in out:
            out.append(rel)
    return sorted(out), skipped


# ---------------------------------------------------------------------------


def load_state(ws: Path, daily: str = "daily") -> tuple[ProactiveStateFile, bool]:
    """Load the truth source; returns ``(state_file, needs_bootstrap)``.

    A missing file means first run (fresh workspace or upgrade) and triggers
    the one-time F1.4 bootstrap from interests.yaml history. A corrupt or
    invalid file rebuilds empty WITHOUT bootstrap (spec F1.3/A2/A5), as does
    an existing file that already carries the ``open_topics`` key (an empty
    list is a normal state, not a trigger).
    """
    path = state_file_path(ws, daily)
    if not path.is_file():
        logger.info(f"proactive state file missing, first-run bootstrap scheduled: {path}")
        return ProactiveStateFile(), True
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError("state file is not a mapping")
    except Exception as e:  # noqa: BLE001
        logger.warning(f"proactive state file corrupt, rebuilding empty: {path} ({e})")
        return ProactiveStateFile(), False
    needs_bootstrap = "open_topics" not in data
    try:
        state = ProactiveStateFile.model_validate(data)
    except Exception as e:  # noqa: BLE001
        # Salvage entry by entry instead of discarding the file: the schedule
        # registry is designed to outlive topics by months, so one malformed
        # hand-edited entry must not silently delete every long-lived reminder.
        # Still no bootstrap - the file exists, so this is not a first run.
        state, dropped = _salvage_state(data)
        logger.warning(
            f"proactive state file invalid, salvaged entry by entry: {path} "
            f"(dropped {dropped} unreadable entr(ies); kept open_topics={len(state.open_topics)} "
            f"schedules={len(state.schedules)} deliveries={len(state.deliveries)}; {e})",
        )
        return state, False
    return state, needs_bootstrap


# Truth-source sections whose entries are validated one by one when the whole
# file fails to parse. ``resolved`` is a plain dict list, so it only needs the
# non-mapping items filtered out.
_STATE_SECTIONS = (("open_topics", ProactiveTopic), ("schedules", ProactiveSchedule), ("deliveries", ProactiveDelivery))


def _salvage_state(data: dict) -> tuple[ProactiveStateFile, int]:
    """Rebuild a partly-unreadable truth source, dropping only the bad entries.

    Returns the salvaged model and the number of entries that had to be
    dropped. Never raises: a section that is not even a list is treated as
    absent, exactly like a file written before that section existed.
    """
    sections: dict[str, list] = {}
    dropped = 0
    for key, model in _STATE_SECTIONS:
        raw = data.get(key)
        raw = raw if isinstance(raw, list) else []
        kept = []
        for item in raw:
            try:
                kept.append(model.model_validate(item))
            except Exception:  # noqa: BLE001
                dropped += 1
        sections[key] = kept
    raw_resolved = data.get("resolved")
    resolved = [item for item in (raw_resolved if isinstance(raw_resolved, list) else []) if isinstance(item, dict)]
    version = data.get("version")
    state = ProactiveStateFile(
        version=version if isinstance(version, int) else 1,
        resolved=resolved,
        **sections,
    )
    return state, dropped


async def save_state(ws: Path, state_file: ProactiveStateFile, daily: str = "daily") -> None:
    """Atomically persist the truth source (path lock + tmp file + os.replace)."""
    path = state_file_path(ws, daily)
    lock = await get_path_lock(path)
    async with lock:
        path.parent.mkdir(parents=True, exist_ok=True)
        rendered = yaml.safe_dump(state_file.model_dump(), allow_unicode=True, sort_keys=False)
        fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(rendered if rendered.endswith("\n") else f"{rendered}\n")
            os.replace(tmp, path)
        except Exception:
            with contextlib.suppress(OSError):
                os.unlink(tmp)
            raise


def safe_date(text: str) -> dt.date | None:
    """Parse an ISO ``YYYY-MM-DD`` string, returning None when it is not one."""
    try:
        return dt.date.fromisoformat(str(text or "").strip())
    except ValueError:
        return None


async def load_carry_forward(
    ws: Path,
    state_file: ProactiveStateFile,
    day: str,
    days: int,
    top_k: int,
    daily: str = "daily",
    needs_bootstrap: bool = False,
) -> tuple[list[ProactiveTopic], list[ProactiveTopic]]:
    """Return ``(carry_forward_all, carry_forward_prompt)`` sorted per A4 rule 1.

    Bootstraps the truth source from interests.yaml history exactly once on
    first run (missing state file) or when an existing file lacks the
    ``open_topics`` key (F1.4). Over-age topics are dropped here with a log;
    resolved ids are suppressed.
    """
    if needs_bootstrap:
        state_file.open_topics = _bootstrap_from_history(ws, day, days, daily)
        await save_state(ws, state_file, daily)
    resolved_ids = {str(r.get("id") or "") for r in state_file.resolved if isinstance(r, dict)}
    base = safe_date(day)
    open_topics: list[ProactiveTopic] = []
    expired = 0
    for topic in state_file.open_topics:
        if topic.id and topic.id in resolved_ids:
            continue
        first_seen = safe_date(topic.first_seen)
        # Boundary aligned with trim_state_file/_expiry_cutoff: age >= days is
        # over-age everywhere, so a topic never enters the prompt in the same
        # round it gets pruned from the truth source (audit item 9).
        if base is not None and first_seen is not None and (base - first_seen).days >= int(days):
            expired += 1
            continue
        open_topics.append(topic)
    if expired:
        logger.info(f"proactive carry-forward dropped {expired} over-age topic(s) (window={days}d)")
    ordered = sort_topics(open_topics)
    return ordered, ordered[: max(int(top_k), 0)]


def _bootstrap_from_history(ws: Path, day: str, days: int, daily: str) -> list[ProactiveTopic]:
    """One-time bootstrap: newest record per id wins, first_seen takes the min (F1.4)."""
    if safe_date(day) is None:
        return []
    records: dict[str, ProactiveTopic] = {}
    first_seen: dict[str, str] = {}
    for file_date in reversed(recent_dates(day, days)):  # newest -> oldest
        data = read_interests_data(interests_path_for(ws, daily, file_date))
        if not data:
            continue
        topics, _is_v1, _push = parse_interests_topics(data, file_date)
        for topic in topics:
            anchor = topic.first_seen or file_date
            if topic.id not in first_seen or anchor < first_seen[topic.id]:
                first_seen[topic.id] = anchor
            records.setdefault(topic.id, topic)
    out: list[ProactiveTopic] = []
    for tid, topic in records.items():
        topic.first_seen = first_seen.get(tid) or topic.first_seen or day
        out.append(topic)
    logger.info(f"proactive bootstrap built {len(out)} open topic(s) from interests.yaml history")
    return out


def trim_state_file(state_file: ProactiveStateFile, day: str, days: int) -> None:
    """Prune budget/exposure/resolved windows and over-age/resolved open topics."""
    base = safe_date(day)
    if base is None:
        return
    cutoff = (base - dt.timedelta(days=max(int(days), 0))).isoformat()
    state_file.resolved = [
        r for r in state_file.resolved if isinstance(r, dict) and str(r.get("resolved_at") or "") > cutoff
    ]
    resolved_ids = {str(r.get("id") or "") for r in state_file.resolved}
    kept: list[ProactiveTopic] = []
    for topic in state_file.open_topics:
        if topic.id and topic.id in resolved_ids:
            continue
        first_seen = safe_date(topic.first_seen)
        if first_seen is not None and first_seen.isoformat() <= cutoff:
            continue
        kept.append(topic)
    state_file.open_topics = kept


# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Delivery feedback (host -> ReMe). Optional by design: a framework that never
# reports leaves ``deliveries`` empty and every helper below degrades to a
# no-op, so the chain behaves exactly as it did before feedback existed.
# ---------------------------------------------------------------------------


def _delivery_date(text: str) -> str:
    """``YYYY-MM-DD`` prefix of an ISO timestamp; '' when unparseable."""
    raw = str(text or "").strip()
    parsed = safe_date(raw[:10])
    return parsed.isoformat() if parsed is not None else ""


def normalize_delivery(raw, day: str, now_dt: dt.datetime) -> ProactiveDelivery | None:
    """Coerce one host-reported item into a ``ProactiveDelivery``.

    ``topic_id`` is the only mandatory field; a record without one can never
    match an agenda item, so it is dropped instead of being stored. Missing
    ``agenda_date`` defaults to the round's day and missing ``delivered_at``
    to the business clock, which keeps a minimal host report
    (``{"topic_id": ..., "status": ...}``) sufficient.
    """
    if not isinstance(raw, dict):
        return None
    topic = str(raw.get("topic_id") or "").strip()
    if not topic:
        return None
    stamp = str(raw.get("delivered_at") or "").strip()
    if not _delivery_date(stamp):
        stamp = now_dt.isoformat(timespec="seconds")
    return ProactiveDelivery(
        topic_id=topic,
        agenda_date=str(raw.get("agenda_date") or day or "")[:10],
        delivered_at=stamp,
        channel=str(raw.get("channel") or DEFAULT_DELIVERY_CHANNEL),
        consumer=str(raw.get("consumer") or ""),
        status=str(raw.get("status") or DEFAULT_DELIVERY_STATUS),
        outcome=str(raw.get("outcome") or ""),
        external_ref=str(raw.get("external_ref") or ""),
    )


def delivery_index(state_file: ProactiveStateFile) -> dict[str, list[ProactiveDelivery]]:
    """Group delivery records by topic_id, newest first.

    The list is kept (not collapsed to one record) because a topic can be
    surfaced through several channels; suppression only needs the most recent
    relevant record, which is the head after sorting.
    """
    grouped: dict[str, list[ProactiveDelivery]] = {}
    for record in state_file.deliveries:
        if not record.topic_id:
            continue
        grouped.setdefault(record.topic_id, []).append(record)
    for records in grouped.values():
        records.sort(key=lambda r: (str(r.delivered_at or ""), str(r.external_ref or "")), reverse=True)
    return grouped


def is_delivery_suppressed(record: ProactiveDelivery | None, day: str, cooldown_days: int) -> bool:
    """Whether a delivery record still blocks re-proposing its topic.

    ``failed`` never blocks: a run that did not complete should be retryable
    on the next round. Every other status blocks while it is younger than
    ``cooldown_days``, so a surfaced-but-ignored item can come back later
    while an executed or dismissed one does not repeat tomorrow.
    """
    if record is None:
        return False
    if str(record.status or "") == "failed":
        return False
    base = safe_date(day)
    delivered = safe_date(_delivery_date(record.delivered_at))
    if base is None or delivered is None:
        return False
    window = max(int(cooldown_days), 0)
    return (base - delivered).days <= window


def suppressed_delivery_ids(
    state_file: ProactiveStateFile,
    day: str,
    cooldown_days: int,
) -> dict[str, ProactiveDelivery]:
    """topic_id -> blocking record, for the reader and the agenda step."""
    out: dict[str, ProactiveDelivery] = {}
    for item_id, records in delivery_index(state_file).items():
        for record in records:
            if is_delivery_suppressed(record, day, cooldown_days):
                out[item_id] = record
                break
    return out


def upsert_deliveries(
    state_file: ProactiveStateFile,
    items: list,
    day: str,
    now_dt: dt.datetime,
) -> dict[str, int]:
    """Merge host-reported deliveries into the truth source (idempotent).

    Identity is ``ProactiveDelivery.identity()``: an explicit ``external_ref``
    wins, otherwise ``(topic_id, agenda_date, channel)``. Status never moves
    backwards out of a terminal value, so a late ``shown`` report cannot undo
    an ``executed`` one.
    """
    counts = {"added": 0, "updated": 0, "dropped": 0}
    by_identity = {record.identity(): record for record in state_file.deliveries}
    for raw in items:
        record = normalize_delivery(raw, day, now_dt)
        if record is None:
            counts["dropped"] += 1
            continue
        key = record.identity()
        existing = by_identity.get(key)
        if existing is None:
            state_file.deliveries.append(record)
            by_identity[key] = record
            counts["added"] += 1
            continue
        if existing.status in TERMINAL_DELIVERY_STATUSES and record.status not in TERMINAL_DELIVERY_STATUSES:
            logger.info(f"delivery for {existing.topic_id} kept terminal status {existing.status!r}")
            counts["dropped"] += 1
            continue
        existing.agenda_date = record.agenda_date or existing.agenda_date
        existing.delivered_at = record.delivered_at
        existing.channel = record.channel
        existing.consumer = record.consumer or existing.consumer
        existing.status = record.status
        if record.outcome:
            existing.outcome = record.outcome
        counts["updated"] += 1
    return counts


def trim_deliveries(state_file: ProactiveStateFile, day: str, retention_days: int, max_deliveries: int) -> int:
    """Prune the delivery log by age then by size; returns the dropped count."""
    base = safe_date(day)
    before = len(state_file.deliveries)
    if base is not None:
        cutoff = (base - dt.timedelta(days=max(int(retention_days), 0))).isoformat()
        state_file.deliveries = [r for r in state_file.deliveries if _delivery_date(r.delivered_at) >= cutoff]
    cap = max(int(max_deliveries), 0)
    if len(state_file.deliveries) > cap:
        state_file.deliveries.sort(key=lambda r: str(r.delivered_at or ""), reverse=True)
        del state_file.deliveries[cap:]
    return before - len(state_file.deliveries)


def quarantine_interests(path: Path, error: Exception) -> None:
    """Rename a corrupt interests.yaml aside (A2): ``interests.corrupt-<ts>.yaml``."""
    stamp = int(time.time())
    corrupt = path.with_name(f"interests.corrupt-{stamp}.yaml")
    try:
        path.rename(corrupt)
        logger.warning(f"quarantined corrupt interests file {path} -> {corrupt.name}: {error}")
    except OSError:
        logger.warning(f"corrupt interests file {path}: {error}")


def read_interests_file(path: Path) -> tuple[str, dict] | None:
    """Read interests.yaml returning ``(raw_text, data)``; quarantine corrupt files (A2)."""
    if not path.is_file():
        return None
    try:
        raw_text = path.read_text(encoding="utf-8")
        data = yaml.safe_load(raw_text)
        if not isinstance(data, dict):
            raise ValueError("interests.yaml is not a mapping")
        return raw_text, data
    except Exception as e:  # noqa: BLE001
        quarantine_interests(path, e)
        return None


def read_interests_data(path: Path) -> dict | None:
    """Parse interests.yaml; quarantine corrupt files (A2) and return None."""
    loaded = read_interests_file(path)
    return loaded[1] if loaded else None


def parse_interests_topics(data: dict, file_date: str) -> tuple[list[ProactiveTopic], bool, bool]:
    """Return ``(topics, is_v1, push)`` with A2 fallbacks applied.

    Missing ``first_seen``/``last_evidence_at`` fall back to the file date
    (not today); missing ids are derived from the frozen title hash.
    """
    is_v1 = data.get("version") is None
    push = data.get("push", True)
    if not isinstance(push, bool):
        push = True
    raw_topics = data.get("topics") or []
    topics: list[ProactiveTopic] = []
    for raw in raw_topics if isinstance(raw_topics, list) else []:
        if not isinstance(raw, dict):
            continue
        title = str(raw.get("title") or "").strip()
        reason = str(raw.get("reason") or "").strip()
        if not title or not reason:
            continue
        topics.append(
            ProactiveTopic(
                id=str(raw.get("id") or "").strip() or topic_id(title),
                title=title,
                reason=reason,
                kind=raw.get("kind", DEFAULT_KIND),
                confidence=raw.get("confidence", 0.5),
                first_seen=str(raw.get("first_seen") or "").strip() or file_date,
                last_evidence_at=str(raw.get("last_evidence_at") or "").strip() or file_date,
                evidence=str(raw.get("evidence") or "").strip()[:120],
                paths=raw.get("paths") or [],
                schedule_id=str(raw.get("schedule_id") or "").strip(),
                due_at=str(raw.get("due_at") or "").strip(),
            ),
        )
    return topics, is_v1, push


def sort_topics(topics: list) -> list:
    """Order: last_evidence_at desc -> kind priority -> confidence desc -> id asc.

    Freshness is the primary key (v5 aging fix): stale topics sink below newly
    evidenced ones of any kind, so long-lived follow_ups cannot permanently
    crowd out new discoveries.
    Kind priority is ``KIND_PRIORITY`` (reminder > follow_up > new_idea >
    interest_extend), so a reminder always outranks discovery kinds inside the
    same freshness band and survives the ``max_topics`` render cap.
    """

    def get(topic, key):
        return getattr(topic, key) if not isinstance(topic, dict) else topic.get(key)

    out = sorted(topics, key=lambda t: str(get(t, "id") or ""))
    out.sort(key=lambda t: clamp_confidence(get(t, "confidence")), reverse=True)
    out.sort(key=lambda t: kind_priority(get(t, "kind")))
    out.sort(key=lambda t: str(get(t, "last_evidence_at") or ""), reverse=True)
    return out


def dump_topic(topic) -> dict:
    """Render one topic as an A2-ordered dict for interests.yaml v2."""
    get = (lambda k: getattr(topic, k)) if not isinstance(topic, dict) else topic.get
    dumped = {
        "id": str(get("id") or ""),
        "title": str(get("title") or ""),
        "reason": str(get("reason") or ""),
        "kind": str(get("kind") or DEFAULT_KIND),
        "confidence": clamp_confidence(get("confidence")),
        "first_seen": str(get("first_seen") or ""),
        "last_evidence_at": str(get("last_evidence_at") or ""),
        "evidence": str(get("evidence") or "")[:120],
        "paths": [str(p) for p in (get("paths") or [])],
    }
    # Reminder links are emitted only when present so the v2 topic shape of
    # every other kind stays byte-identical.
    if schedule_id := str(get("schedule_id") or ""):
        dumped["schedule_id"] = schedule_id
    if due_at := str(get("due_at") or ""):
        dumped["due_at"] = due_at
    return dumped


def render_interests(
    day: str,
    topics: list,
    push: bool,
    now_dt: dt.datetime,
    agenda: list | None = None,
    suppressed: list | None = None,
) -> dict:
    """Render the full v2 file content from the truth source (INV-6).

    v5: ``skip_reason`` is no longer persisted (no consumer); it survives as
    structured log/metadata on ``ProactiveState.file_skip_reason`` (R7).
    The plan/agenda enrichment keys (``agenda``/``suppressed``) are only
    present when the agenda step rendered the file; the topics-only renderer
    passes ``None`` so its file shape stays unchanged.
    """
    rendered = {
        "version": 2,
        "date": day,
        "generated_at": now_dt.isoformat(timespec="seconds"),
        "push": bool(push),
        "topics": [dump_topic(t) for t in topics],
    }
    if agenda is not None:
        rendered["agenda"] = list(agenda)
    if suppressed is not None:
        rendered["suppressed"] = list(suppressed)
    return rendered


def write_interests_if_changed(ws: Path, path: Path, rendered: dict) -> bool:  # pylint: disable=unused-argument
    """Apply A4 render-write rules (idempotent skip + atomic replace).

    v5 (R1): the "push=false never overwrites nightly v1" special case is
    gone; ``push`` is derived from the cumulative truth source, so re-renders
    are monotonic and need no guard.
    """
    existing: dict | None = None
    if path.is_file():
        existing = read_interests_data(path)
    if existing is not None:
        existing_push = existing.get("push", True)
        if not isinstance(existing_push, bool):
            existing_push = True
        if (
            existing_push == bool(rendered.get("push"))
            and existing.get("topics") == rendered.get("topics")
            and (existing.get("agenda") or []) == (rendered.get("agenda") or [])
            and (existing.get("suppressed") or []) == (rendered.get("suppressed") or [])
        ):
            return False
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = yaml.safe_dump(rendered, allow_unicode=True, sort_keys=False)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(payload if payload.endswith("\n") else f"{payload}\n")
        os.replace(tmp, path)
    except Exception:
        with contextlib.suppress(OSError):
            os.unlink(tmp)
        raise
    return True


# ---------------------------------------------------------------------------


def clean_candidate(raw, allowed_paths: set[str], kind: str, day: str) -> dict:
    """Clean one LLM candidate; drop entries whose paths escape M (INV-8)."""
    if not isinstance(raw, dict):
        return {}
    title = str(raw.get("title") or "").strip()
    reason = str(raw.get("reason") or "").strip()
    paths = clean_paths(raw.get("paths"), allowed_paths)
    if not title or not reason or not paths:
        return {}
    return {
        "id": topic_id(title),
        "title": title,
        "reason": reason,
        "kind": kind if kind in TOPIC_KINDS else DEFAULT_KIND,
        "confidence": clamp_confidence(raw.get("confidence")),
        "first_seen": day,
        "last_evidence_at": day,
        # Derived, not LLM-emitted: the first whitelisted path anchors the
        # candidate, so the output contract stays one field smaller.
        "evidence": paths[0][:120],
        "paths": paths,
    }


def current_now(step) -> dt.datetime:
    """Business-time access per INV-3 (timezone-aware; never datetime.now())."""
    tz = step.app_context.app_config.timezone if step.app_context is not None else None
    return now(tz)


def parse_extract_reply(text: str) -> dict:
    """Parse the A3 fenced YAML/JSON output; fenced blocks take priority.

    A reply only counts as parsed when at least one contract section
    (``EXTRACT_SECTIONS``) is present as a list; a non-empty mapping with
    misspelled or missing section names is a parse failure, so
    ``_extract_with_retry`` retries instead of checkpointing changed files
    on output that can never yield topics.
    Unlike the dream parser there is no scalar-mapping fallback: proactive
    output is sectioned lists, and a partial fallback would corrupt updates.
    """
    candidates = [m.group(1).strip() for m in re.finditer(r"```(?:json|ya?ml)?\s*(.*?)```", text, re.S | re.I)]
    candidates.append((text or "").strip())
    for raw in candidates:
        if not raw:
            continue
        try:
            data = yaml.safe_load(raw)
        except yaml.YAMLError:
            continue
        if isinstance(data, dict) and data and any(isinstance(data.get(key), list) for key in EXTRACT_SECTIONS):
            return data
    return {}


def strip_frontmatter(text: str) -> tuple[dict, str]:
    """Split leading ``---`` YAML frontmatter blocks from the body.

    Handles files with several consecutive frontmatter blocks; returns
    ``(merged_meta, body)``. Parse failures degrade to empty meta.
    """
    meta: dict = {}
    body = text or ""
    while True:
        stripped = body.lstrip()
        if not stripped.startswith("---"):
            break
        rest = stripped[3:]
        end = rest.find("\n---")
        if end < 0:
            break
        block = rest[:end]
        body = rest[end + 4 :]
        try:
            data = yaml.safe_load(block)
        except yaml.YAMLError:
            data = None
        if isinstance(data, dict):
            for key, value in data.items():
                meta.setdefault(str(key), value)
    return meta, body.strip()


def _contained_workspace_path(ws: Path, rel_path: str) -> Path | None:
    """Resolve ``rel_path`` strictly inside the workspace (audit item 7).

    Rejects absolute paths, home-relative paths and ``..`` traversal so the
    profile fallback can never read (and then feed into prompts) files that
    live outside the workspace.
    """
    candidate = Path(rel_path)
    if candidate.is_absolute() or candidate.anchor or rel_path.startswith("~") or ".." in candidate.parts:
        return None
    workspace = ws.resolve()
    resolved = (workspace / candidate).resolve()
    try:
        resolved.relative_to(workspace)
    except ValueError:
        return None
    return resolved


def load_personal_profile_block(
    ws: Path,
    digest_dir: str,
    max_chars: int,
    fallback_rel_path: str = "",
) -> str:
    """Build the user profile/preference block used to personalize proactive inference.

    Primary source is ``<digest_dir>/personal/*.md`` (one file per profile
    facet: identity, preferences, constraints). Each file contributes its
    frontmatter ``description`` plus a body excerpt under an equal per-file
    budget. Falls back to a single profile file (legacy ``profile.md``) when
    the personal directory is absent, and to a sentinel when nothing exists.
    """
    max_chars = int(max_chars)
    if max_chars <= 0:
        return "(no user profile)"
    personal_dir = ws / digest_dir / "personal"
    if personal_dir.is_dir():
        files = sorted(path for path in personal_dir.glob("*.md") if path.is_file())
        if files:
            per_file = max(max_chars // len(files), 200)
            sections: list[str] = []
            for path in files:
                try:
                    raw = path.read_text(encoding="utf-8")
                except OSError:
                    continue
                meta, body = strip_frontmatter(raw)
                title = str(meta.get("name") or path.stem)
                description = str(meta.get("description") or "").strip()
                excerpt = body[:per_file].strip()
                section = f"### {title}\n"
                if description:
                    section += f"{description}\n"
                if excerpt:
                    section += excerpt
                sections.append(section.rstrip())
            if sections:
                return "\n\n".join(sections)[:max_chars]
    if fallback_rel_path:
        fallback = _contained_workspace_path(ws, fallback_rel_path)
        if fallback is not None and fallback.is_file():
            try:
                text = fallback.read_text(encoding="utf-8")[:max_chars]
            except OSError:
                return "(no user profile)"
            return text.strip() or "(no user profile)"
    return "(no user profile)"


CAPABILITY_SECTIONS = (
    ("procedure", "procedure (reusable experience, skills, tool/API know-how the agent already has)"),
    ("wiki", "wiki (existing knowledge nodes of the personal knowledge base)"),
)


def load_capability_block(ws: Path, digest_dir: str, max_chars: int) -> str:
    """Inventory of what the agent can already do, used by the ``new_ideas`` branch.

    Reads only rebuildable digest nodes: ``<digest_dir>/procedure/*.md``
    (accumulated experience, skills, tool/API know-how) and
    ``<digest_dir>/wiki/*.md`` (knowledge nodes an interest can extend).
    Each node contributes its frontmatter ``description`` so proposals stay
    grounded in resources that actually exist instead of invented capabilities.
    """
    max_chars = int(max_chars)
    if max_chars <= 0:
        return "(no capability inventory)"
    per_section = max(max_chars // max(len(CAPABILITY_SECTIONS), 1), 200)
    sections: list[str] = []
    for subdirectory, heading in CAPABILITY_SECTIONS:
        lines = _digest_inventory_lines(ws / digest_dir / subdirectory, per_section)
        if lines:
            sections.append(f"### {heading}\n" + "\n".join(lines))
    if not sections:
        return "(no capability inventory)"
    return "\n\n".join(sections)[:max_chars]


def _digest_inventory_lines(directory: Path, budget: int) -> list[str]:
    """``- <title>: <description>`` lines for one digest subdirectory, budget-bounded."""
    if not directory.is_dir():
        return []
    lines: list[str] = []
    used = 0
    for path in sorted(p for p in directory.glob("*.md") if p.is_file()):
        try:
            raw = path.read_text(encoding="utf-8")
        except OSError:
            continue
        meta, _body = strip_frontmatter(raw)
        title = str(meta.get("name") or path.stem)
        description = " ".join(str(meta.get("description") or "").split())
        line = f"- {title}" + (f": {description[:160]}" if description else "")
        if used + len(line) > budget:
            break
        lines.append(line)
        used += len(line) + 1
    return lines


def parse_fenced_yaml(text: str) -> dict:
    """Parse the first fenced YAML/JSON block (or the bare text) into a dict.

    Shared by the plan/agenda reply parsers; returns ``{}`` when nothing
    parses to a mapping.
    """
    candidates = [m.group(1).strip() for m in re.finditer(r"```(?:json|ya?ml)?\s*(.*?)```", text, re.S | re.I)]
    candidates.append((text or "").strip())
    for raw in candidates:
        if not raw:
            continue
        try:
            data = yaml.safe_load(raw)
        except yaml.YAMLError:
            continue
        if isinstance(data, dict):
            return data
    return {}


def parse_plan_reply(text: str) -> list[dict]:
    """Parse the plan step reply; requires a ``cards`` list of mappings."""
    data = parse_fenced_yaml(text)
    cards = data.get("cards")
    if not isinstance(cards, list):
        return []
    return [card for card in cards if isinstance(card, dict)]


def parse_agenda_reply(text: str) -> tuple[list[dict], list[dict]]:
    """Parse the agenda step reply into ``(agenda, suppressed)`` lists.

    ``agenda`` must be present as a list; ``suppressed`` defaults to empty.
    """
    data = parse_fenced_yaml(text)
    agenda = data.get("agenda")
    if not isinstance(agenda, list):
        return [], []
    suppressed = data.get("suppressed")
    if not isinstance(suppressed, list):
        suppressed = []
    return [item for item in agenda if isinstance(item, dict)], [item for item in suppressed if isinstance(item, dict)]


def resolve_agent_wrapper(step):
    """Return the step's agent_wrapper, falling back to the app default (F4.4)."""
    wrapper = step.agent_wrapper
    if wrapper is not None:
        return wrapper
    if step.app_context is not None:
        fallback = step.app_context.components.get(ComponentEnum.AGENT_WRAPPER, {}).get("default")
        if fallback is not None:
            configured = step.kwargs.get("agent_wrapper", "default")
            step.logger.warning(f"[{step.name}] agent_wrapper '{configured}' missing; using default")
            return fallback
    return None
