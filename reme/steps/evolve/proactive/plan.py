"""Proactive plan step: expand push candidates into scenario cards (F2.5).

Runs between the topics and agenda steps. Only topics that qualify for today's
push (``first_seen == today`` and ``confidence >= min_push_confidence``,
computed by the topics step into ``state.push_candidates``) are expanded, so
the LLM cost stays proportional to what will actually be surfaced. One batched
LLM call per round; on any LLM failure every selected candidate still gets a
deterministic fallback card so the agenda step never sees an empty input.

Card contract: ``scenario_type`` (resume_task | answer_pending |
explore_interest | prepare_upcoming), ``opener`` (a casual, friend-like
conversation opener that already names the minimal next action - never a
"based on your records" notification tone), ``next_action``, ``preconditions``
and ``delivery`` (in_conversation | notification | agenda_item).
``linked_memory`` is derived from the topic's paths/evidence by code, never by
the LLM.
"""

import asyncio
import json
import time

from ...base_step import BaseStep
from .._evolve import agent_reply_result_text, passthrough_response
from ....components import R
from ....schema import ProactiveState
from ..dream.utils import workspace_dir
from .utils import load_personal_profile_block, parse_plan_reply, resolve_agent_wrapper

SCENARIO_TYPES = ("resume_task", "answer_pending", "explore_interest", "prepare_upcoming")
DELIVERY_MODES = ("in_conversation", "notification", "agenda_item")
DEFAULT_SCENARIO_BY_KIND = {"follow_up": "resume_task", "interest_extend": "explore_interest"}
MAX_OPENER_CHARS = 300
MAX_NEXT_ACTION_CHARS = 200
MAX_PRECONDITIONS = 5
MAX_PRECONDITION_CHARS = 80


def linked_memory(candidate: dict) -> list[str]:
    """Workspace paths related to a topic, derived from paths + evidence."""
    out: list[str] = []
    raw = list(candidate.get("paths") or [])
    evidence = str(candidate.get("evidence") or "").strip()
    if evidence:
        raw.append(evidence)
    for path in raw:
        rel = str(path or "").strip().split("#", 1)[0]
        if rel and rel not in out:
            out.append(rel)
    return out


def fallback_card(candidate: dict) -> dict:
    """Deterministic card used when the LLM is absent or its output is unusable."""
    kind = str(candidate.get("kind") or "interest_extend")
    title = str(candidate.get("title") or "")
    scenario = DEFAULT_SCENARIO_BY_KIND.get(kind, "explore_interest")
    if kind == "follow_up":
        opener = f"上次聊到「{title}」，好像还没收尾——要不要趁现在花几分钟往前推一步？"
        next_action = "先回顾相关记录，确认卡点，然后给出下一步动作"
    else:
        opener = f"感觉你最近可能会想看看「{title}」，有空的话可以先从相关材料扫一眼。"
        next_action = "快速浏览相关材料，判断值不值得深入"
    memory = linked_memory(candidate)
    if memory:
        next_action = f"{next_action}（材料：{memory[0]}）"
    return {
        "scenario_type": scenario,
        "opener": opener,
        "next_action": next_action,
        "preconditions": [],
        "delivery": "in_conversation",
    }


def _clean_card(raw: dict, candidate: dict) -> dict:
    """Validate one LLM card; per-field fallback keeps the card always usable."""
    base = fallback_card(candidate)
    scenario = str(raw.get("scenario_type") or "").strip()
    if scenario not in SCENARIO_TYPES:
        scenario = base["scenario_type"]
    opener = str(raw.get("opener") or "").strip()[:MAX_OPENER_CHARS]
    if not opener:
        opener = base["opener"]
    next_action = str(raw.get("next_action") or "").strip()[:MAX_NEXT_ACTION_CHARS]
    if not next_action:
        next_action = base["next_action"]
    preconditions = raw.get("preconditions")
    if not isinstance(preconditions, list):
        preconditions = []
    preconditions = [str(item).strip()[:MAX_PRECONDITION_CHARS] for item in preconditions if str(item).strip()]
    delivery = str(raw.get("delivery") or "").strip()
    if delivery not in DELIVERY_MODES:
        delivery = base["delivery"]
    return {
        "scenario_type": scenario,
        "opener": opener,
        "next_action": next_action,
        "preconditions": preconditions[:MAX_PRECONDITIONS],
        "delivery": delivery,
    }


@R.register("proactive_plan_step")
class ProactivePlanStep(BaseStep):
    """Expand today's push candidates into scenario cards.

    Zero LLM calls when there are no push candidates. At most
    ``max_plan_topics`` candidates are expanded; the overflow stays card-less
    and the agenda step suppresses it with an explicit reason.
    """

    def __init__(
        self,
        max_plan_topics: int = 6,
        llm_timeout_seconds: float = 120,
        profile_max_chars: int = 800,
        skip_key: str = "proactive_skip",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.max_plan_topics = max(int(max_plan_topics), 1)
        self.llm_timeout_seconds = float(llm_timeout_seconds)
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
            self.context.response.answer = "Skipped plan: no proactive extract state in context"
            return self.context.response
        state = ProactiveState.model_validate(raw_state)
        if state.early_exit:
            self.context.response.success = True
            self.context.response.answer = f"Skipped plan: {state.early_exit}"
            return self.context.response

        candidates = [c for c in state.push_candidates if isinstance(c, dict) and c.get("id")]
        if not candidates:
            state.scenario_cards = []
            self._store(state)
            self.context.response.success = True
            self.context.response.answer = "Plan: no push candidates, 0 cards, 0 LLM calls"
            self.logger.info(f"[{self.name}] no push candidates; skipping")
            return self.context.response

        selected = candidates[: self.max_plan_topics]
        self.logger.info(
            f"[{self.name}] start candidates={len(candidates)} selected={len(selected)} "
            f"max_plan_topics={self.max_plan_topics}",
        )
        cards_by_id = await self._llm_cards(state, selected)

        cards: list[dict] = []
        fallbacks = 0
        for candidate in selected:
            cid = str(candidate.get("id"))
            raw_card = cards_by_id.get(cid)
            if raw_card is None:
                fallbacks += 1
            card = _clean_card(raw_card, candidate) if raw_card is not None else fallback_card(candidate)
            card = {
                "topic_id": cid,
                "title": str(candidate.get("title") or ""),
                "kind": str(candidate.get("kind") or "interest_extend"),
                **card,
                "linked_memory": linked_memory(candidate),
            }
            cards.append(card)

        state.scenario_cards = cards
        state.duration_ms = state.duration_ms + int((time.monotonic() - started) * 1000)
        self._store(state)
        answer = f"Plan: {len(cards)} scenario card(s), fallback={fallbacks}, llm_calls={state.plan_llm_calls}"
        self.context.response.success = True
        self.context.response.answer = answer
        self.logger.info(f"[{self.name}] finish {answer}")
        return self.context.response

    async def _llm_cards(self, state: ProactiveState, selected: list[dict]) -> dict[str, dict]:
        """One batched LLM call; returns candidate-id -> raw card dict."""
        wrapper = resolve_agent_wrapper(self)
        if wrapper is None:
            self.logger.warning(f"[{self.name}] no agent_wrapper available; using fallback cards")
            return {}
        selected_ids = {str(c.get("id")) for c in selected}
        ws = workspace_dir(self)
        user_message = self.prompt_format(
            "plan_user_message",
            profile_block=self._profile_block(ws),
            candidates_json=json.dumps(
                [
                    {
                        "id": c.get("id"),
                        "title": c.get("title"),
                        "kind": c.get("kind"),
                        "reason": c.get("reason"),
                        "confidence": c.get("confidence"),
                        "first_seen": c.get("first_seen"),
                        "evidence": c.get("evidence"),
                    }
                    for c in selected
                ],
                ensure_ascii=False,
            ),
        )
        system_prompt = self.prompt_format("plan_system_prompt")
        state.plan_llm_calls += 1
        try:
            result = await asyncio.wait_for(
                wrapper.reply(user_message, system_prompt=system_prompt),
                timeout=self.llm_timeout_seconds,
            )
        except asyncio.TimeoutError:
            self.logger.warning(f"[{self.name}] LLM reply timed out after {self.llm_timeout_seconds}s")
            return {}
        except Exception as e:  # noqa: BLE001 - network/provider errors degrade to fallback cards
            self.logger.warning(
                f"[{self.name}] LLM reply failed ({type(e).__name__}: {e}); using fallback cards",
            )
            return {}
        raw = agent_reply_result_text(result)
        cards_by_id: dict[str, dict] = {}
        for card in parse_plan_reply(raw):
            cid = str(card.get("topic_id") or "").strip()
            if cid in selected_ids and cid not in cards_by_id:
                cards_by_id[cid] = card
        if not cards_by_id:
            self.logger.warning(f"[{self.name}] plan reply unusable; raw={raw[:200]!r}")
        return cards_by_id

    def _profile_block(self, ws) -> str:
        """Digest-personal profile sketch so openers respect user preferences."""
        digest_dir = str(self.config_value("digest_dir"))
        return load_personal_profile_block(ws, digest_dir, self.profile_max_chars)

    def _store(self, state: ProactiveState) -> None:
        assert self.context is not None
        data = state.model_dump()
        self.context["proactive"] = data
        self.context.response.metadata["proactive"] = data
