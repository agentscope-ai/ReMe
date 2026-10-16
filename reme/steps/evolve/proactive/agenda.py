"""Proactive agenda step: list-wise generative agenda + auditable silence (F2.6).

Replaces "sort and truncate" selection with generation: the LLM sees every
scenario card plus today's conversation context and an optional user profile,
then decides BY ITSELF how many items deserve today's agenda (bounded only by
the ``max_agenda_items`` safety cap) and must explicitly account for every
candidate it does not schedule (suppressed with a reason - silence is
auditable). This lets it merge candidates that are different entrances to the
same matter, keep narrative coherence, and avoid clashing with the user's
current focus.

Deterministic guarantees: exactly one candidate -> auto-agenda with no LLM
call; LLM absent/timed out/unusable -> freshness/confidence order fallback;
after validation the agenda is never empty while candidates exist, so the push
semantics computed by the topics step never flip. The step also owns the final
interests.yaml render, extending the v2 shape with ``agenda``/``suppressed``.

Reminder priority is enforced here, not negotiated: every reminder card is
pinned to the front of the agenda (nearest due date first) before the editor
LLM sees anything, and only the slots the reminders leave are up for
selection. Pinned items are never suppressed and never counted as "silence".

Delivery feedback is enforced even earlier: a candidate the host already
reported as surfaced/executed inside ``redelivery_cooldown_days`` leaves the
candidate set before pinning and before the editor call, with an auditable
``already delivered ...`` suppression reason. This is the write-side twin of
the reader's filter, so a proposal cannot reappear tomorrow merely because
the host reported late. Hosts that never report keep an empty ``deliveries``
section and see no change.
"""

import asyncio
import json
import time

from ...base_step import BaseStep
from .._evolve import agent_reply_result_text, passthrough_response
from ....components import R
from ....schema import ProactiveState
from ..dream.utils import daily_dir, workspace_dir
from .utils import (
    current_now,
    interests_path_for,
    load_personal_profile_block,
    load_state,
    norm_path,
    parse_agenda_reply,
    render_interests,
    resolve_agent_wrapper,
    suppressed_delivery_ids,
    write_interests_if_changed,
)

FALLBACK_ORDER_REASON = "deterministic fallback: freshness/confidence order"
PINNED_REMINDER_REASONS = {
    "due": "reminder pinned: a due schedule always enters today's agenda",
    "new": "reminder pinned: a new reminder always enters today's agenda",
    "open": "reminder pinned: an open reminder always enters today's agenda",
}


def _pin_reason(candidate: dict, day: str = "") -> str:
    """Order reason for a pinned reminder, naming the due date when known.

    Every reminder is pinned, but not for the same reason: the audit trail
    distinguishes a schedule the clock shows as due/overdue from one this round
    merely found (whose due date may still be weeks away), so a reader can tell
    a calendar fact from a freshness rule.
    """
    due = str(candidate.get("due_at") or "").strip()
    due_day, day = due[:10], str(day or "").strip()[:10]
    first_seen = str(candidate.get("first_seen") or "").strip()[:10]
    if due_day and day and due_day <= day:
        cause = "due"
    elif first_seen and day and first_seen >= day:
        cause = "new"
    else:
        cause = "open"
    base = PINNED_REMINDER_REASONS[cause]
    return f"{base} (due {due})" if due else base


def _delivered_reason(record) -> str:
    """Suppression reason naming when, where and how the item was already used."""
    if record is None:
        return "already delivered (within the redelivery cooldown)"
    when = str(record.agenda_date or "") or str(record.delivered_at or "")[:10]
    where = f" via {record.channel}" if record.channel else ""
    return f"already delivered on {when}{where} ({record.status})"


@R.register("proactive_agenda_step")
class ProactiveAgendaStep(BaseStep):
    """Generate today's proactive agenda and render the enriched interests.yaml.

    Skips silently (leaving the topics step's push=false render in place) when
    there are no push candidates. Otherwise writes the final v2 file with the
    ``agenda`` and ``suppressed`` keys.
    """

    def __init__(
        self,
        max_agenda_items: int = 6,
        min_discovery_slots: int = 2,
        llm_timeout_seconds: float = 120,
        profile_rel_path: str = "profile.md",
        profile_max_chars: int = 2500,
        redelivery_cooldown_days: int = 3,
        skip_key: str = "proactive_skip",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.max_agenda_items = max(int(max_agenda_items), 1)
        self.min_discovery_slots = max(int(min_discovery_slots), 0)
        self.llm_timeout_seconds = float(llm_timeout_seconds)
        self.profile_rel_path = str(profile_rel_path or "")
        self.profile_max_chars = max(int(profile_max_chars), 0)
        self.redelivery_cooldown_days = max(int(redelivery_cooldown_days), 0)
        self.skip_key = skip_key

    async def execute(self):
        assert self.context is not None
        if self.context.get(self.skip_key):
            return passthrough_response(self, self.skip_key)
        started = time.monotonic()
        raw_state = self.context.get("proactive")
        if not raw_state:
            self.context.response.success = True
            self.context.response.answer = "Skipped agenda: no proactive extract state in context"
            return self.context.response
        state = ProactiveState.model_validate(raw_state)
        if state.early_exit:
            self.context.response.success = True
            self.context.response.answer = f"Skipped agenda: {state.early_exit}"
            return self.context.response

        candidates = [c for c in state.push_candidates if isinstance(c, dict) and c.get("id")]
        if not candidates:
            self.context.response.success = True
            self.context.response.answer = "Agenda: no push candidates; interests.yaml left to topics render"
            self.logger.info(f"[{self.name}] no push candidates; nothing to agenda")
            return self.context.response

        ws = workspace_dir(self)
        cards_by_id = {str(c.get("topic_id") or ""): c for c in state.scenario_cards if isinstance(c, dict)}
        carded = [c for c in candidates if str(c.get("id")) in cards_by_id]
        over_budget = [c for c in candidates if str(c.get("id")) not in cards_by_id]
        # Delivery feedback outranks every other selection rule: drop what the
        # host already surfaced before pinning reminders or calling the editor,
        # so the cooldown cannot be argued away by narrative coherence.
        daily = state.daily_dir or daily_dir(self)
        blocked = self._suppressed_deliveries(ws, daily, state.date)
        # Reminders are the one exemption: a due schedule has to be announced
        # every time it comes due, and reminder ids derive from a stable title,
        # so blocking them would silence a recurring reminder for the whole
        # cooldown window after a single delivery.
        reminder_ids = {str(c.get("id")) for c in candidates if str(c.get("kind")) == "reminder"}
        blocked = {
            blocked_id: record
            for blocked_id, record in blocked.items()
            if blocked_id not in reminder_ids
        }
        delivered = [c for c in candidates if str(c.get("id")) in blocked]
        if delivered:
            carded = [c for c in carded if str(c.get("id")) not in blocked]
            over_budget = [c for c in over_budget if str(c.get("id")) not in blocked]
        state.suppressed_delivered = len(delivered)
        # Reminders are pinned: every reminder - a due schedule or a freshly
        # found one - always reaches today's agenda and is never traded away for
        # narrative coherence. What the editor LLM gets
        # to choose is the remaining (discovery) part of the agenda, inside the
        # slots the pinned reminders leave - but never fewer than
        # ``min_discovery_slots``, so a reminder-heavy day still surfaces the
        # discovery kinds instead of collapsing into a plain calendar. On such a
        # day the agenda may exceed ``max_agenda_items``: pinned reminders are
        # additive by design, and dropping either side would break a guarantee.
        pinned = [c for c in carded if str(c.get("kind")) == "reminder"]
        optional = [c for c in carded if str(c.get("kind")) != "reminder"]
        pinned_ids = [str(c.get("id")) for c in pinned]
        slots = max(self.max_agenda_items - len(pinned_ids), self.min_discovery_slots)
        self.logger.info(
            f"[{self.name}] start candidates={len(candidates)} carded={len(carded)} "
            f"over_budget={len(over_budget)} already_delivered={len(delivered)} "
            f"pinned_reminders={len(pinned_ids)} slots={slots} "
            f"max_agenda_items={self.max_agenda_items}",
        )

        if not optional:
            selected_ids, order_reasons, suppressed_reasons = [], {}, {}
        elif slots == 0:
            selected_ids, order_reasons = [], {}
            suppressed_reasons = {
                str(c.get("id")): f"agenda capacity reserved for {len(pinned_ids)} reminder(s)" for c in optional
            }
        elif len(optional) == 1:
            selected_ids = [str(optional[0].get("id"))]
            order_reasons = {selected_ids[0]: "single push candidate; auto-agenda without LLM"}
            suppressed_reasons = {}
        else:
            selected_ids, order_reasons, suppressed_reasons = await self._llm_agenda(
                state,
                ws,
                optional,
                cards_by_id,
                slots,
            )
        for cid in pinned_ids:
            order_reasons.setdefault(
                cid,
                _pin_reason(next((c for c in pinned if str(c.get("id")) == cid), {}), state.date),
            )
        agenda_ids = pinned_ids + [cid for cid in selected_ids if cid not in pinned_ids]

        suppressed: list[dict] = []
        for candidate in delivered:
            record = blocked.get(str(candidate.get("id")))
            suppressed.append(
                {
                    "topic_id": str(candidate.get("id")),
                    "title": str(candidate.get("title") or ""),
                    "reason": _delivered_reason(record),
                },
            )
        for candidate in over_budget:
            is_reminder = str(candidate.get("kind")) == "reminder"
            suppressed.append(
                {
                    "topic_id": str(candidate.get("id")),
                    "title": str(candidate.get("title") or ""),
                    "reason": (
                        "over reminder card budget: not expanded into a scenario card"
                        if is_reminder
                        else "over plan budget: not expanded into a scenario card"
                    ),
                },
            )
        for candidate in carded:
            cid = str(candidate.get("id"))
            if cid in agenda_ids:
                continue
            suppressed.append(
                {
                    "topic_id": cid,
                    "title": str(candidate.get("title") or ""),
                    "reason": suppressed_reasons.get(cid) or "not selected for today's agenda",
                },
            )

        agenda: list[dict] = []
        for cid in agenda_ids:
            card = cards_by_id.get(cid)
            candidate = next((c for c in carded if str(c.get("id")) == cid), None)
            if card is None or candidate is None:
                continue
            agenda.append(
                {
                    "topic_id": cid,
                    "title": str(candidate.get("title") or ""),
                    "kind": str(candidate.get("kind") or ""),
                    "scenario_type": str(card.get("scenario_type") or ""),
                    "opener": str(card.get("opener") or ""),
                    "next_action": str(card.get("next_action") or ""),
                    "preconditions": list(card.get("preconditions") or []),
                    "delivery": str(card.get("delivery") or "in_conversation"),
                    "linked_memory": list(card.get("linked_memory") or []),
                    **({"due_at": str(candidate.get("due_at"))} if candidate.get("due_at") else {}),
                    **({"schedule_id": str(candidate.get("schedule_id"))} if candidate.get("schedule_id") else {}),
                    "order_reason": order_reasons.get(cid) or "",
                },
            )

        state.agenda = agenda
        state.suppressed = suppressed
        state.push = bool(agenda)

        day = state.date
        daily = state.daily_dir or daily_dir(self)
        rendered = render_interests(
            day,
            state.topics_out,
            state.push,
            current_now(self),
            agenda=agenda,
            suppressed=suppressed,
        )
        interests_path = interests_path_for(ws, daily, day)
        written = write_interests_if_changed(ws, interests_path, rendered)
        rel_path = norm_path(interests_path.relative_to(ws).as_posix())
        state.interests_path = rel_path
        state.interests_written = bool(state.interests_written or written)
        state.duration_ms = state.duration_ms + int((time.monotonic() - started) * 1000)
        self._store(state)
        answer = (
            f"Agenda: {len(agenda)} item(s), suppressed={len(suppressed)} "
            f"(already delivered={len(delivered)}), push={state.push}, written={written} to {rel_path}"
        )
        self.context.response.success = True
        self.context.response.answer = answer
        self.logger.info(f"[{self.name}] finish {answer}")
        return self.context.response

    async def _llm_agenda(
        self,
        state: ProactiveState,
        ws,
        carded: list[dict],
        cards_by_id: dict[str, dict],
        slots: int,
    ) -> tuple[list[str], dict[str, str], dict[str, str]]:
        """One list-wise LLM call over the non-pinned cards.

        Returns ``(agenda ids, order reasons, suppressed reasons)``. ``slots``
        is what the pinned reminders left of ``max_agenda_items``, and it is
        both the cap told to the model and the hard truncation limit.
        """
        carded_ids = [str(c.get("id")) for c in carded]
        wrapper = resolve_agent_wrapper(self)
        if wrapper is None:
            self.logger.warning(f"[{self.name}] no agent_wrapper available; deterministic fallback agenda")
            return self._fallback_agenda(carded_ids, slots)

        cards_json = json.dumps(
            [
                {
                    "topic_id": str(c.get("id")),
                    "title": c.get("title"),
                    "kind": c.get("kind"),
                    "scenario_type": cards_by_id[str(c.get("id"))].get("scenario_type"),
                    "opener": cards_by_id[str(c.get("id"))].get("opener"),
                    "next_action": cards_by_id[str(c.get("id"))].get("next_action"),
                    "delivery": cards_by_id[str(c.get("id"))].get("delivery"),
                }
                for c in carded
            ],
            ensure_ascii=False,
        )
        user_message = self.prompt_format(
            "agenda_user_message",
            max_agenda_items=slots,
            cards_json=cards_json,
            changed_files_json=json.dumps(list(dict.fromkeys(state.changed_paths)), ensure_ascii=False),
            profile_block=self._profile_block(ws),
        )
        system_prompt = self.prompt_format("agenda_system_prompt", max_agenda_items=slots)
        state.plan_llm_calls += 1
        try:
            result = await asyncio.wait_for(
                wrapper.reply(user_message, system_prompt=system_prompt),
                timeout=self.llm_timeout_seconds,
            )
        except asyncio.TimeoutError:
            self.logger.warning(f"[{self.name}] LLM reply timed out after {self.llm_timeout_seconds}s")
            return self._fallback_agenda(carded_ids, slots)
        except Exception as e:  # noqa: BLE001 - network/provider errors degrade to fallback agenda
            self.logger.warning(
                f"[{self.name}] LLM reply failed ({type(e).__name__}: {e}); using fallback agenda",
            )
            return self._fallback_agenda(carded_ids, slots)
        raw = agent_reply_result_text(result)
        agenda_raw, suppressed_raw = parse_agenda_reply(raw)

        allowed = set(carded_ids)
        agenda_ids: list[str] = []
        for item in agenda_raw:
            cid = str(item.get("topic_id") or "").strip()
            if cid in allowed and cid not in agenda_ids:
                agenda_ids.append(cid)
            if len(agenda_ids) >= slots:
                break
        order_reasons = {
            str(item.get("topic_id") or "").strip(): str(item.get("order_reason") or "").strip()
            for item in agenda_raw
            if str(item.get("topic_id") or "").strip() in allowed
        }
        suppressed_reasons = {
            str(item.get("topic_id") or "").strip(): str(item.get("reason") or "").strip()
            for item in suppressed_raw
            if str(item.get("topic_id") or "").strip() in allowed
        }
        if not agenda_ids:
            self.logger.warning(f"[{self.name}] agenda reply unusable; raw={raw[:200]!r}")
            return self._fallback_agenda(carded_ids, slots)
        return agenda_ids, order_reasons, suppressed_reasons

    def _fallback_agenda(self, carded_ids: list[str], slots: int) -> tuple[list[str], dict[str, str], dict[str, str]]:
        """Freshness/confidence head when generation is unavailable (cards keep sort order)."""
        agenda_ids = carded_ids[: max(int(slots), 0)]
        order_reasons = {cid: FALLBACK_ORDER_REASON for cid in agenda_ids}
        suppressed_reasons = {cid: "capacity (deterministic fallback)" for cid in carded_ids if cid not in agenda_ids}
        return agenda_ids, order_reasons, suppressed_reasons

    def _suppressed_deliveries(self, ws, daily: str, day: str) -> dict:
        """topic_id -> blocking delivery record reported by the host."""
        state_file, _needs_bootstrap = load_state(ws, daily)
        if not state_file.deliveries:
            return {}
        return suppressed_delivery_ids(state_file, day, self.redelivery_cooldown_days)

    def _profile_block(self, ws) -> str:
        """Digest-personal profile first, legacy single profile file as fallback."""
        digest_dir = str(self.config_value("digest_dir"))
        return load_personal_profile_block(
            ws,
            digest_dir,
            self.profile_max_chars,
            self.profile_rel_path,
        )

    def _store(self, state: ProactiveState) -> None:
        assert self.context is not None
        data = state.model_dump()
        self.context["proactive"] = data
        self.context.response.metadata["proactive"] = data
