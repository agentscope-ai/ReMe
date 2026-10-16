"""Proactive plan step: expand push candidates into scenario cards (F2.5).

Runs between the topics and agenda steps. Only topics that qualify for today's
push (computed by the topics step into ``state.push_candidates``) are expanded,
so the LLM cost stays proportional to what will actually be surfaced. One
batched LLM call per round; on any LLM failure every selected candidate still
gets a deterministic fallback card so the agenda step never sees an empty input.

Card contract: ``scenario_type`` (resume_task | answer_pending |
explore_interest | prepare_upcoming | propose_delegation), ``opener`` (a casual,
friend-like conversation opener that already names the minimal next action -
never a "based on your records" notification tone, and never a stock lead-in
worn into a template), ``next_action``,
``preconditions`` and ``delivery`` (in_conversation | notification |
agenda_item). ``linked_memory`` is derived from the topic's paths/evidence by
code, never by the LLM.

Reminders are expanded first and are never squeezed out by ``max_plan_topics``
(they are bounded by ``max_reminder_cards`` instead), because the agenda step
pins every reminder card it receives.
"""

import asyncio
import hashlib
import json
import time

from ...base_step import BaseStep
from .._evolve import agent_reply_result_text, passthrough_response
from ....components import R
from ....schema import ProactiveState
from ....schema.proactive import DEFAULT_KIND
from ..dream.utils import workspace_dir
from .utils import load_personal_profile_block, parse_plan_reply, resolve_agent_wrapper

SCENARIO_TYPES = ("resume_task", "answer_pending", "explore_interest", "prepare_upcoming", "propose_delegation")
DELIVERY_MODES = ("in_conversation", "notification", "agenda_item")
DEFAULT_SCENARIO_BY_KIND = {
    "follow_up": "resume_task",
    "interest_extend": "explore_interest",
    "reminder": "prepare_upcoming",
    "new_idea": "propose_delegation",
}
MAX_OPENER_CHARS = 300
MAX_NEXT_ACTION_CHARS = 200
MAX_PRECONDITIONS = 5
MAX_PRECONDITION_CHARS = 80

# Deterministic openers, several per kind, so a batch of fallback cards does not
# open every line the same way - one repeated lead-in reads as a script. ``{when}``
# is the parenthesised due date (empty when unknown) and only reminder variants
# carry it, because a reminder may not hide when it is due.
FALLBACK_OPENERS = {
    "reminder": (
        "「{title}」{when}就快到了，要不要现在把准备事项先过一遍？",
        "时间快到了：「{title}」{when}，先确认下会不会和别的安排撞上？",
        "有件事排在前面了：「{title}」{when}，现在动手准备最从容。",
    ),
    "new_idea": (
        "有件活儿我觉得可以替你办：「{title}」，要不要我先出个方案你看看？",
        "「{title}」这件事其实可以直接交给我，要不要我先出一版草案给你审？",
        "这活儿不必你自己来：「{title}」，我先跑个初稿你看看值不值？",
    ),
    "follow_up": (
        "上次聊到「{title}」，好像还没收尾——要不要趁现在花几分钟往前推一步？",
        "「{title}」还挂着，要不要我先去查一下最新进展，再回来跟你说下一步？",
        "那件「{title}」好像只差临门一脚了，现在顺手推一下？",
    ),
    "interest_extend": (
        "感觉你最近可能会想看看「{title}」，有空的话可以先从相关材料扫一眼。",
        "「{title}」这块还能再往深挖一层，要不要我挑两处给你展开讲讲？",
        "顺着「{title}」又翻到些延伸内容，值得的话我带你过一遍？",
    ),
}


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
    """Deterministic card used when the LLM is absent or its output is unusable.

    Its opener is picked per topic out of ``FALLBACK_OPENERS``, so the
    no-LLM path honours the same "no repeated lead-in" rule the prompt states.
    """
    kind = str(candidate.get("kind") or DEFAULT_KIND)
    title = str(candidate.get("title") or "")
    scenario = DEFAULT_SCENARIO_BY_KIND.get(kind, DEFAULT_SCENARIO_BY_KIND[DEFAULT_KIND])
    opener = _fallback_opener(kind, candidate, title)
    if kind == "reminder":
        next_action = "确认时间、所需材料与是否和其它安排冲突，需要改期就现在说"
    elif kind == "new_idea":
        next_action = "确认值不值得委托；值得的话由 agent 先产出一版草案供你审阅"
    elif kind == "follow_up":
        next_action = "先回顾相关记录，确认卡点，然后给出下一步动作"
    else:
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


def _fallback_opener(kind: str, candidate: dict, title: str) -> str:
    """Pick one opener variant for this kind, seeded by the topic identity.

    The seed is the topic id, so one topic keeps one opener across rounds
    instead of flip-flopping, while different topics in the same round land on
    different variants and the agenda does not read as one repeated sentence.
    """
    variants = FALLBACK_OPENERS.get(kind) or FALLBACK_OPENERS[DEFAULT_KIND]
    seed = str(candidate.get("id") or title)
    picked = variants[int(hashlib.sha1(seed.encode("utf-8")).hexdigest(), 16) % len(variants)]
    due = str(candidate.get("due_at") or "").strip()
    return picked.format(title=title, when=f"（{due}）" if due else "")


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

    Zero LLM calls when there are no push candidates. The two budgets are
    independent: at most ``max_reminder_cards`` reminders plus
    ``max_plan_topics`` discovery candidates are expanded. A reminder-heavy day
    therefore cannot starve the discovery kinds of cards, because reminders are
    pinned into the agenda later and would otherwise take every slot with them.
    The overflow stays card-less and the agenda step suppresses it with an
    explicit reason.
    """

    def __init__(
        self,
        max_plan_topics: int = 6,
        max_reminder_cards: int = 8,
        llm_timeout_seconds: float = 120,
        profile_max_chars: int = 800,
        skip_key: str = "proactive_skip",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.max_plan_topics = max(int(max_plan_topics), 1)
        self.max_reminder_cards = max(int(max_reminder_cards), 1)
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

        # Reminders first and outside the discovery budget: they pin today's
        # agenda, so starving them here would silently break that guarantee.
        reminders = [c for c in candidates if str(c.get("kind")) == "reminder"][: self.max_reminder_cards]
        others = [c for c in candidates if str(c.get("kind")) != "reminder"]
        selected = reminders + others[: self.max_plan_topics]
        self.logger.info(
            f"[{self.name}] start candidates={len(candidates)} selected={len(selected)} "
            f"reminders={len(reminders)} max_plan_topics={self.max_plan_topics} "
            f"max_reminder_cards={self.max_reminder_cards}",
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
                "kind": str(candidate.get("kind") or DEFAULT_KIND),
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
                        **({"due_at": c.get("due_at")} if c.get("due_at") else {}),
                        **({"schedule_id": c.get("schedule_id")} if c.get("schedule_id") else {}),
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
