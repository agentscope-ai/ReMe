"""Proactive refresh extract step: material scan + follow_ups/extends/updates (F2.0-F2.3)."""

import asyncio
import json
import time

from ...base_step import BaseStep
from .._evolve import agent_reply_result_text, passthrough_response
from ....components import R
from ....schema import ProactiveState
from ..dream.utils import daily_dir, pack_paths, today, workspace_dir
from .utils import (
    clean_candidate,
    load_carry_forward,
    load_personal_profile_block,
    load_state,
    parse_extract_reply,
    resolve_agent_wrapper,
    scan_material_daily,
)


@R.register("proactive_extract_step")
class ProactiveExtractStep(BaseStep):
    """Scan the proactive material set and extract follow-ups, extends and updates.

    Owns the ``proactive`` catalog watermark (never touches the dream catalog)
    and the daily LLM budget. Short-circuits via ``context[skip_key]`` on
    busy/budget/timeout per F4.3; early-exits with zero LLM calls when no new
    evidence exists (F2.0).
    """

    def __init__(
        self,
        scan_days: int = 2,
        carry_forward_days: int = 14,
        max_carry_forward_topics: int = 20,
        llm_timeout_seconds: float = 300,
        max_chars_per_file: int = 60000,
        max_total_chars: int = 300000,
        profile_max_chars: int = 1500,
        extends_enabled: bool = True,
        skip_key: str = "proactive_skip",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.scan_days = max(int(scan_days), 1)
        self.carry_forward_days = max(int(carry_forward_days), 1)
        self.max_carry_forward_topics = max(int(max_carry_forward_topics), 0)
        self.llm_timeout_seconds = float(llm_timeout_seconds)
        self.max_chars_per_file = max(int(max_chars_per_file), 1000)
        self.max_total_chars = max(int(max_total_chars), 0)
        self.profile_max_chars = max(int(profile_max_chars), 0)
        self.extends_enabled = bool(extends_enabled)
        self.skip_key = skip_key

    async def execute(self):
        assert self.context is not None
        if self.context.get(self.skip_key):
            return passthrough_response(self, self.skip_key)
        started = time.monotonic()
        day = today(self, str(self.context.get("date", "") or ""))
        ws = workspace_dir(self)
        daily = daily_dir(self)
        if self.file_catalog is None:
            raise RuntimeError("proactive_extract_step requires file_catalog")
        state = ProactiveState(
            date=day,
            daily_dir=daily,
            workspace=str(ws),
            scan_days=self.scan_days,
            carry_forward_days=self.carry_forward_days,
        )
        self.logger.info(f"[{self.name}] start date={day} scan_days={self.scan_days} extends={self.extends_enabled}")

        # 1) Material set M (F2.0) - all cheap, before any LLM work.
        m_daily = scan_material_daily(ws, day, daily, self.scan_days)

        # 2) Truth source + carry-forward (F1.3/F1.4).
        state_file, needs_bootstrap = load_state(ws, daily)
        carry_all, carry_prompt = await load_carry_forward(
            ws,
            state_file,
            day,
            self.carry_forward_days,
            self.max_carry_forward_topics,
            daily,
            needs_bootstrap,
        )
        state.carry_forward_count = len(carry_all)
        state.carry_forward_prompt = carry_prompt

        # 3) Change detection against the proactive catalog watermark.
        # Only daily notes are material: fresh resource uploads already flow
        # into daily notes via auto_resource, so no separate resource scan.
        existing = {}
        for rel in m_daily:
            try:
                existing[rel] = (ws / rel).stat().st_mtime
            except OSError as e:
                self.logger.error(f"[{self.name}] stat failed on {rel}: {e}")
        nodes = await self.file_catalog.get_nodes()
        indexed = {n.path: n.st_mtime for n in nodes}
        state.changed_paths = [rel for rel, mt in existing.items() if indexed.get(rel) != mt]
        self.logger.info(
            f"[{self.name}] material daily={len(m_daily)} "
            f"changed={len(state.changed_paths)} carry_forward={len(carry_all)}",
        )

        # 4) Zero-consumption early exit (A4 row 1).
        if not state.changed_paths:
            state.early_exit = "no_new_evidence"
            return self._finish(state, started, "Skipped: no new evidence; 0 LLM calls")

        # 5) LLM channel (F4.4): structurally one reply per round plus at most
        # one parse-failure retry, so no persistent budget is needed (v5 R5).
        wrapper = resolve_agent_wrapper(self)
        if wrapper is None:
            state.early_exit = "no_agent_wrapper"
            self.logger.warning(f"[{self.name}] no agent_wrapper available; skipping round")
            return self._finish(state, started, "Skipped: no agent_wrapper configured")

        # 6) One LLM call with sectioned output (A3); retry once on parse failure.
        meta = await self._extract_with_retry(wrapper, state, ws, day, daily)
        if meta is None:
            self.context[self.skip_key] = {"reason": "llm_timeout"}
            self._store(state)
            return passthrough_response(self, self.skip_key)
        self._clean_output(state, meta, set(m_daily), day)
        answer = (
            f"Extracted {len(state.follow_ups)} follow_up(s), {len(state.extends)} extend(s), "
            f"{len(state.updates)} update(s) from {len(state.changed_paths)} changed file(s)"
        )
        return self._finish(state, started, answer)

    def _build_messages(self, state: ProactiveState, ws, day: str, daily: str, material_blob: str) -> tuple[str, str]:
        carry_forward_json = json.dumps(
            [
                {
                    "id": t.id,
                    "title": t.title,
                    "kind": t.kind,
                    "confidence": t.confidence,
                    "reason": t.reason,
                    "last_evidence_at": t.last_evidence_at,
                    "evidence": t.evidence,
                }
                for t in state.carry_forward_prompt
            ],
            ensure_ascii=False,
        )
        changed = list(dict.fromkeys(state.changed_paths))
        user_message = self.prompt_format(
            "extract_user_message",
            date=day,
            changed_paths_json=json.dumps(changed, ensure_ascii=False),
            carry_forward_json=carry_forward_json,
            material_blob=material_blob,
            profile_block=self._profile_block(ws),
            extends=self.extends_enabled,
        )
        system_prompt = self.prompt_format(
            "extract_system_prompt",
            workspace_dir=str(ws),
            daily_dir=daily,
            extends=self.extends_enabled,
        )
        return user_message, system_prompt

    def _profile_block(self, ws) -> str:
        """Digest-personal profile sketch; background knowledge, never material."""
        digest_dir = str(self.config_value("digest_dir"))
        return load_personal_profile_block(ws, digest_dir, self.profile_max_chars)

    async def _extract_with_retry(self, wrapper, state, ws, day: str, daily: str) -> dict | None:
        """Returns parsed meta; None means LLM timeout."""
        # Newest daily material first: when the total budget bites, the
        # freshest evidence survives and the oldest files are omitted.
        ordered = list(dict.fromkeys(state.changed_paths))[::-1]
        material_blob = pack_paths(
            ws,
            ordered,
            limit_per_file=self.max_chars_per_file,
            max_total_chars=self.max_total_chars or None,
        )
        user_message, system_prompt = self._build_messages(state, ws, day, daily, material_blob)
        for attempt in (1, 2):
            raw = await self._reply(wrapper, state, user_message, system_prompt)
            if raw is None:
                return None
            meta = parse_extract_reply(raw)
            if meta:
                return meta
            self.logger.warning(f"[{self.name}] parse failed on attempt {attempt}; raw={raw[:200]!r}")
        return {}

    async def _reply(self, wrapper, state, user_message, system_prompt):
        """One timeout-wrapped reply; returns raw text or None on timeout."""
        state.llm_calls += 1
        try:
            result = await asyncio.wait_for(
                wrapper.reply(user_message, system_prompt=system_prompt),
                timeout=self.llm_timeout_seconds,
            )
        except asyncio.TimeoutError:
            self.logger.warning(f"[{self.name}] LLM reply timed out after {self.llm_timeout_seconds}s")
            return None
        return agent_reply_result_text(result)

    def _clean_output(self, state: ProactiveState, meta: dict, allowed: set[str], day: str) -> None:
        for raw in meta.get("follow_ups") or []:
            if candidate := clean_candidate(raw, allowed, "follow_up", day):
                state.follow_ups.append(candidate)
            else:
                state.dropped_missing += 1
        if self.extends_enabled:
            for raw in meta.get("extends") or []:
                if candidate := clean_candidate(raw, allowed, "interest_extend", day):
                    state.extends.append(candidate)
                else:
                    state.dropped_missing += 1
        for raw in meta.get("updates") or []:
            if not isinstance(raw, dict):
                continue
            topic_id = str(raw.get("id") or "").strip()
            if not topic_id:
                continue
            action = str(raw.get("action") or "keep").strip()
            if action not in ("keep", "update", "resolve"):
                action = "keep"
            state.updates.append(
                {
                    "id": topic_id,
                    "action": action,
                    "evidence": str(raw.get("evidence") or "").strip()[:120],
                    "reason": str(raw.get("reason") or "").strip(),
                    "confidence": raw.get("confidence"),
                },
            )

    def _store(self, state: ProactiveState) -> None:
        assert self.context is not None
        data = state.model_dump()
        self.context["proactive"] = data
        self.context.response.metadata["proactive"] = data

    def _finish(self, state: ProactiveState, started: float, answer: str):
        state.duration_ms = int((time.monotonic() - started) * 1000)
        self._store(state)
        self.context.response.success = True
        self.context.response.answer = answer
        self.logger.info(f"[{self.name}] finish answer={answer!r}")
        return self.context.response
