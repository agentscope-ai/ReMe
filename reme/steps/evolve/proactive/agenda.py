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
    norm_path,
    parse_agenda_reply,
    render_interests,
    resolve_agent_wrapper,
    write_interests_if_changed,
)

FALLBACK_ORDER_REASON = "deterministic fallback: freshness/confidence order"


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
        llm_timeout_seconds: float = 120,
        profile_rel_path: str = "profile.md",
        profile_max_chars: int = 2500,
        skip_key: str = "proactive_skip",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.max_agenda_items = max(int(max_agenda_items), 1)
        self.llm_timeout_seconds = float(llm_timeout_seconds)
        self.profile_rel_path = str(profile_rel_path or "")
        self.profile_max_chars = max(int(profile_max_chars), 0)
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
        self.logger.info(
            f"[{self.name}] start candidates={len(candidates)} carded={len(carded)} "
            f"over_budget={len(over_budget)} max_agenda_items={self.max_agenda_items}",
        )

        if not carded:
            agenda_ids, order_reasons, suppressed_reasons = [], {}, {}
        elif len(carded) == 1:
            agenda_ids = [str(carded[0].get("id"))]
            order_reasons = {agenda_ids[0]: "single push candidate; auto-agenda without LLM"}
            suppressed_reasons = {}
        else:
            agenda_ids, order_reasons, suppressed_reasons = await self._llm_agenda(state, ws, carded, cards_by_id)

        suppressed: list[dict] = []
        for candidate in over_budget:
            suppressed.append(
                {
                    "topic_id": str(candidate.get("id")),
                    "title": str(candidate.get("title") or ""),
                    "reason": "over plan budget: not expanded into a scenario card",
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
                    "scenario_type": str(card.get("scenario_type") or ""),
                    "opener": str(card.get("opener") or ""),
                    "next_action": str(card.get("next_action") or ""),
                    "preconditions": list(card.get("preconditions") or []),
                    "delivery": str(card.get("delivery") or "in_conversation"),
                    "linked_memory": list(card.get("linked_memory") or []),
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
            f"Agenda: {len(agenda)} item(s), suppressed={len(suppressed)}, push={state.push}, "
            f"written={written} to {rel_path}"
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
    ) -> tuple[list[str], dict[str, str], dict[str, str]]:
        """One list-wise LLM call; returns (agenda ids, order reasons, suppressed reasons)."""
        carded_ids = [str(c.get("id")) for c in carded]
        wrapper = resolve_agent_wrapper(self)
        if wrapper is None:
            self.logger.warning(f"[{self.name}] no agent_wrapper available; deterministic fallback agenda")
            return self._fallback_agenda(carded_ids)

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
            max_agenda_items=self.max_agenda_items,
            cards_json=cards_json,
            changed_files_json=json.dumps(list(dict.fromkeys(state.changed_paths)), ensure_ascii=False),
            profile_block=self._profile_block(ws),
        )
        system_prompt = self.prompt_format("agenda_system_prompt", max_agenda_items=self.max_agenda_items)
        state.plan_llm_calls += 1
        try:
            result = await asyncio.wait_for(
                wrapper.reply(user_message, system_prompt=system_prompt),
                timeout=self.llm_timeout_seconds,
            )
        except asyncio.TimeoutError:
            self.logger.warning(f"[{self.name}] LLM reply timed out after {self.llm_timeout_seconds}s")
            return self._fallback_agenda(carded_ids)
        raw = agent_reply_result_text(result)
        agenda_raw, suppressed_raw = parse_agenda_reply(raw)

        allowed = set(carded_ids)
        agenda_ids: list[str] = []
        for item in agenda_raw:
            cid = str(item.get("topic_id") or "").strip()
            if cid in allowed and cid not in agenda_ids:
                agenda_ids.append(cid)
            if len(agenda_ids) >= self.max_agenda_items:
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
            return self._fallback_agenda(carded_ids)
        return agenda_ids, order_reasons, suppressed_reasons

    def _fallback_agenda(self, carded_ids: list[str]) -> tuple[list[str], dict[str, str], dict[str, str]]:
        """Freshness/confidence head when generation is unavailable (cards keep sort order)."""
        agenda_ids = carded_ids[: self.max_agenda_items]
        order_reasons = {cid: FALLBACK_ORDER_REASON for cid in agenda_ids}
        suppressed_reasons = {
            cid: "capacity (deterministic fallback)" for cid in carded_ids if cid not in agenda_ids
        }
        return agenda_ids, order_reasons, suppressed_reasons

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
