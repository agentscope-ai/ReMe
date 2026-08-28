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
from ....schema import ProactiveStateFile, ProactiveTopic
from ....schema.proactive import clamp_confidence
from ....utils import get_logger
from ...file_io._file_io import get_path_lock
from .._evolve import now
from ..dream.utils import clean_paths, recent_dates, scan_day_files

logger = get_logger(log_to_file=False)

PROACTIVE_STATE_NAME = "_proactive.yaml"
INTERESTS_NAME = "interests.yaml"


# ---------------------------------------------------------------------------
# Frozen identity contract (A7 / INV-4)
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
# Paths and material set M (F2.0)
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
        for rel in scan_day_files(ws, scan_day, daily, INTERESTS_NAME):
            rel = norm_path(rel)
            base = rel.rsplit("/", 1)[-1]
            if rel == day_index or base.startswith("_"):
                continue
            if rel not in out:
                out.append(rel)
    return sorted(out)


# ---------------------------------------------------------------------------
# Truth-source state file daily/_proactive.yaml (F1.3)
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
        logger.warning(f"proactive state file invalid, rebuilding empty: {path} ({e})")
        return ProactiveStateFile(), False
    return state, needs_bootstrap


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


def _safe_date(text: str) -> dt.date | None:
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
    base = _safe_date(day)
    open_topics: list[ProactiveTopic] = []
    expired = 0
    for topic in state_file.open_topics:
        if topic.id and topic.id in resolved_ids:
            continue
        first_seen = _safe_date(topic.first_seen)
        if base is not None and first_seen is not None and (base - first_seen).days > int(days):
            expired += 1
            continue
        open_topics.append(topic)
    if expired:
        logger.info(f"proactive carry-forward dropped {expired} over-age topic(s) (window={days}d)")
    ordered = sort_topics(open_topics)
    return ordered, ordered[: max(int(top_k), 0)]


def _bootstrap_from_history(ws: Path, day: str, days: int, daily: str) -> list[ProactiveTopic]:
    """One-time bootstrap: newest record per id wins, first_seen takes the min (F1.4)."""
    if _safe_date(day) is None:
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
    base = _safe_date(day)
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
        first_seen = _safe_date(topic.first_seen)
        if first_seen is not None and first_seen.isoformat() <= cutoff:
            continue
        kept.append(topic)
    state_file.open_topics = kept


# ---------------------------------------------------------------------------
# interests.yaml read/render (F1.2 / A2 / A4)
# ---------------------------------------------------------------------------


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
                kind=raw.get("kind", "interest_extend"),
                confidence=raw.get("confidence", 0.5),
                first_seen=str(raw.get("first_seen") or "").strip() or file_date,
                last_evidence_at=str(raw.get("last_evidence_at") or "").strip() or file_date,
                evidence=str(raw.get("evidence") or "").strip()[:120],
                keywords=raw.get("keywords") or [],
                paths=raw.get("paths") or [],
            ),
        )
    return topics, is_v1, push


def sort_topics(topics: list) -> list:
    """Order: last_evidence_at desc -> follow_up first -> confidence desc -> id asc.

    Freshness is the primary key (v5 aging fix): stale topics sink below newly
    evidenced ones of any kind, so long-lived follow_ups cannot permanently
    crowd out new discoveries.
    """

    def get(topic, key):
        return getattr(topic, key) if not isinstance(topic, dict) else topic.get(key)

    out = sorted(topics, key=lambda t: str(get(t, "id") or ""))
    out.sort(key=lambda t: clamp_confidence(get(t, "confidence")), reverse=True)
    out.sort(key=lambda t: 0 if get(t, "kind") == "follow_up" else 1)
    out.sort(key=lambda t: str(get(t, "last_evidence_at") or ""), reverse=True)
    return out


def dump_topic(topic) -> dict:
    """Render one topic as an A2-ordered dict for interests.yaml v2."""
    get = (lambda k: getattr(topic, k)) if not isinstance(topic, dict) else topic.get
    return {
        "id": str(get("id") or ""),
        "title": str(get("title") or ""),
        "reason": str(get("reason") or ""),
        "kind": str(get("kind") or "interest_extend"),
        "confidence": clamp_confidence(get("confidence")),
        "first_seen": str(get("first_seen") or ""),
        "last_evidence_at": str(get("last_evidence_at") or ""),
        "evidence": str(get("evidence") or "")[:120],
        "keywords": [str(k) for k in (get("keywords") or [])],
        "paths": [str(p) for p in (get("paths") or [])],
    }


def render_interests(day: str, topics: list, push: bool, now_dt: dt.datetime) -> dict:
    """Render the full v2 file content from the truth source (INV-6).

    v5: ``skip_reason`` is no longer persisted (no consumer); it survives as
    structured log/metadata on ``ProactiveState.file_skip_reason`` (R7).
    """
    return {
        "version": 2,
        "date": day,
        "generated_at": now_dt.isoformat(timespec="seconds"),
        "push": bool(push),
        "topics": [dump_topic(t) for t in topics],
    }


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
        if existing_push == bool(rendered.get("push")) and existing.get("topics") == rendered.get("topics"):
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
# Candidate cleaning (extract side, A3)
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
    keywords = raw.get("keywords") or []
    keywords = [str(k).strip() for k in keywords if str(k).strip()] if isinstance(keywords, list) else []
    return {
        "id": topic_id(title),
        "title": title,
        "reason": reason,
        "kind": kind,
        "confidence": clamp_confidence(raw.get("confidence")),
        "first_seen": day,
        "last_evidence_at": day,
        "evidence": str(raw.get("evidence") or "").strip()[:120],
        "keywords": keywords,
        "paths": paths,
    }


def current_now(step) -> dt.datetime:
    """Business-time access per INV-3 (timezone-aware; never datetime.now())."""
    tz = step.app_context.app_config.timezone if step.app_context is not None else None
    return now(tz)


def parse_extract_reply(text: str) -> dict:
    """Parse the A3 fenced YAML/JSON output; fenced blocks take priority.

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
        if isinstance(data, dict) and data:
            return data
    return {}


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
