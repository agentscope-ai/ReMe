"""Stage-2 per-kind adjudication for proactive extraction (precision pass).

Stage 1 owns recall: one batched reply over the changed material produces the
candidates of all four kinds. Stage 2 owns precision: every enabled kind that
actually produced candidates gets one small, material-free call that re-checks
those candidates against the kind's own contract plus the only context slice
that kind needs (registry and clock for reminders, carry-forward topics for
follow-ups, capability/wiki inventory for ideas and extends).

A pass may keep, drop or move a candidate and rescore its confidence. It never
rewrites wording - stage 1 owns the user-facing text - and it never invents a
candidate. Every failure mode is fail-open: a timeout or an unusable reply keeps
the stage-1 candidates untouched, because losing a real finding costs more than
keeping a borderline one.
"""

from ....schema.proactive import TOPIC_KINDS, clamp_confidence
from .utils import parse_fenced_yaml

# Refine order = proactive priority order, which keeps the audit log readable:
# the highest-priority kind is adjudicated first.
REFINE_ORDER = ("reminder", "follow_up", "new_idea", "interest_extend")
DECISIONS = ("keep", "drop", "move")

# Fields only a reminder candidate carries. A move out of ``reminder`` has to
# drop them, otherwise the receiving kind publishes a ``schedule_id`` pointing
# at a registry entry that was never created plus a due date nothing fires on.
REMINDER_ONLY_FIELDS = ("schedule_id", "due_at", "next_due", "trigger", "conflict_with")


def _notice_anchored(item: dict, notice_paths: set[str]) -> bool:
    """True when the candidate cites the external source (``resource/``) layer."""
    return bool({str(p) for p in (item.get("paths") or [])} & notice_paths)


def _reminder_contract_met(item: dict, notice_paths: set[str]) -> bool:
    """``clean_reminder``'s traceability rule: a due value or a notice anchor."""
    return bool(str(item.get("due_at") or "").strip()) or _notice_anchored(item, notice_paths)


def candidates_by_kind(state) -> list[tuple[str, list[dict]]]:
    """``(kind, bucket)`` pairs in refine order, skipping empty buckets."""
    buckets = kind_buckets(state)
    return [(kind, buckets[kind]) for kind in REFINE_ORDER if buckets[kind]]


def kind_buckets(state) -> dict[str, list[dict]]:
    """Map each kind to the ``ProactiveState`` bucket holding its candidates."""
    return {
        "reminder": list(state.reminders),
        "follow_up": list(state.follow_ups),
        "new_idea": list(state.new_ideas),
        "interest_extend": list(state.extends),
    }


def candidate_payload(bucket: list[dict]) -> list[dict]:
    """Candidate fields worth adjudicating; no material, paths trimmed."""
    out = []
    for item in bucket:
        payload = {
            "id": str(item.get("id") or ""),
            "title": str(item.get("title") or ""),
            "reason": str(item.get("reason") or ""),
            "confidence": clamp_confidence(item.get("confidence")),
            "paths": [str(p) for p in (item.get("paths") or [])][:3],
        }
        for key in ("due_at", "schedule_id", "trigger"):
            if str(item.get(key) or "").strip():
                payload[key] = str(item.get(key))
        conflicts = item.get("conflict_with")
        if isinstance(conflicts, list) and conflicts:
            payload["conflict_with"] = [str(c) for c in conflicts][:3]
        out.append(payload)
    return out


def parse_refine_reply(text: str) -> list[dict] | None:
    """Verdict list from a refine reply; ``None`` when the reply is unusable.

    An empty list is a valid reply ("keep everything"); only a missing or
    non-list ``verdicts`` key counts as unusable, which lets the caller keep
    the stage-1 candidates instead of silently dropping a whole kind.
    """
    payload = parse_fenced_yaml(text)
    if "verdicts" not in payload:
        return None
    verdicts = payload["verdicts"]
    if verdicts is None:
        # `verdicts:` with nothing beneath it is how YAML spells the empty list,
        # and an unmentioned candidate is kept - so this is the documented
        # "nothing to change" reply, not a broken one. Counting it as unusable
        # would flag the most common clean round as a degrade and hide a real
        # stage-2 failure behind the noise.
        return []
    if not isinstance(verdicts, list):
        return None
    out = []
    for raw in verdicts:
        if not isinstance(raw, dict):
            continue
        cid = str(raw.get("id") or "").strip()
        if not cid:
            continue
        decision = str(raw.get("decision") or "keep").strip().lower()
        verdict = {
            "id": cid,
            "decision": decision if decision in DECISIONS else "keep",
            "to_kind": str(raw.get("to_kind") or "").strip(),
            "reason": str(raw.get("reason") or "").strip()[:200],
        }
        if raw.get("confidence") is not None:
            verdict["confidence"] = clamp_confidence(raw.get("confidence"))
        out.append(verdict)
    return out


def apply_verdicts(
    state,
    verdicts: list[dict],
    caps: dict[str, int],
    enabled: set[str],
    notice_paths: set[str] | None = None,
) -> None:
    """Rebuild the four buckets from the stage-2 verdicts, in place on ``state``.

    A missing verdict means keep (fail-open), an unknown id is ignored, and a
    move to an unknown or disabled kind is ignored rather than fatal. Caps are
    re-applied afterwards in stage-1 order, so a move can never push a kind over
    its ceiling.

    Two moves are refused rather than applied, because stage 1 owns the
    path/kind invariants and stage 2 must not break them:

    * out of ``reminder`` for a candidate anchored on the external notice
      layer - only the reminder branch may cite ``resource/``, so demoting it
      would publish an external upload under a kind that may not quote one;
    * into ``reminder`` for a candidate with neither a due value nor a notice
      anchor - stage 2 cannot attach a date, so it could never satisfy
      ``clean_reminder``'s contract for a dateless reminder.

    Any other successful move out of ``reminder`` strips the reminder-only
    fields so no dangling registry link survives the reclassification.
    """
    notices = {str(p) for p in (notice_paths or set())}
    buckets = kind_buckets(state)
    index: dict[str, tuple[str, dict]] = {}
    for kind, bucket in buckets.items():
        for item in bucket:
            index.setdefault(str(item.get("id") or ""), (kind, item))
    for verdict in verdicts:
        entry = index.get(verdict["id"])
        if entry is None:
            continue
        kind, item = entry
        if verdict["decision"] == "drop":
            buckets[kind] = [c for c in buckets[kind] if c is not item]
            state.refine_dropped.append(_audit(verdict, kind, item))
        elif verdict["decision"] == "move":
            target = verdict["to_kind"]
            if target not in TOPIC_KINDS or target not in enabled or target == kind:
                continue
            if kind == "reminder" and _notice_anchored(item, notices):
                continue
            if target == "reminder" and not _reminder_contract_met(item, notices):
                continue
            buckets[kind] = [c for c in buckets[kind] if c is not item]
            item["kind"] = target
            if kind == "reminder":
                for field in REMINDER_ONLY_FIELDS:
                    item.pop(field, None)
            buckets[target].append(item)
            index[verdict["id"]] = (target, item)
            state.refine_moved.append({**_audit(verdict, kind, item), "to_kind": target})
        if "confidence" in verdict:
            item["confidence"] = verdict["confidence"]
    for kind, bucket in buckets.items():
        cap = int(caps.get(kind, 0))
        if cap and len(bucket) > cap:
            state.dropped_over_cap += len(bucket) - cap
            buckets[kind] = bucket[:cap]
    state.reminders = buckets["reminder"]
    state.follow_ups = buckets["follow_up"]
    state.new_ideas = buckets["new_idea"]
    state.extends = buckets["interest_extend"]


def _audit(verdict: dict, kind: str, item: dict) -> dict:
    """One audit record so a refine decision stays diagnosable in metadata."""
    return {
        "id": verdict["id"],
        "kind": kind,
        "title": str(item.get("title") or "")[:80],
        "decision": verdict["decision"],
        "reason": verdict["reason"],
    }
