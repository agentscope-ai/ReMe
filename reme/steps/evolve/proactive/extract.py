"""Proactive refresh extract step: material scan + four proposal kinds (F2.0-F2.3).

The step owns discovery. It reads the changed daily notes plus three cheap
read-only views - the carry-forward topics, the long-lived schedule registry
and the digest capability inventory - together with the current business
clock, and asks for one sectioned reply covering four proposal kinds:

``follow_ups``       open loops: propose fetching the latest status and reporting;
``extends``          knowledge-base recommendations: propose a wiki node to expand;
``reminders``        personal schedule review: propose due/conflicting reminders,
                     plus must-tell notices found in changed ``resource/`` records;
``new_ideas``        delegatable work the user has not thought of: propose only.

Only the reminder branch reads the ``resource/`` layer, which is the reserved
hook for external source interfaces: whatever lands there becomes eligible for
the schedule review without any further change to the chain.

Reminders additionally have a deterministic, clock-driven path
(``reminder_candidates_from_view``) so a due schedule still reaches today's
agenda when there is no new material, no agent wrapper, or an unusable reply.
Nothing here executes a proposal; the output stays a proposal end to end.
"""

import asyncio
import json
import time

from ...base_step import BaseStep
from .._evolve import agent_reply_result_text, passthrough_response
from ....components import R
from ....schema import ProactiveState
from ..dream.utils import daily_dir, pack_paths, today, workspace_dir
from .refine import apply_verdicts, candidate_payload, candidates_by_kind, parse_refine_reply
from .schedules import (
    DUE_TRIGGERS,
    cap_notice_reminders,
    clean_reminder,
    clean_schedule_update,
    merge_reminders,
    reminder_candidates_from_view,
    schedule_prompt_entries,
    schedule_view,
)
from .utils import (
    clean_candidate,
    current_now,
    load_capability_block,
    load_carry_forward,
    load_personal_profile_block,
    load_state,
    parse_extract_reply,
    resolve_agent_wrapper,
    scan_material_daily,
    scan_material_resource,
)

DUE_SCHEDULE_KEYS = (
    "id",
    "title",
    "status",
    "due_at",
    "next_due",
    "due_in_days",
    "trigger",
    "conflict_with",
)


@R.register("proactive_extract_step")
class ProactiveExtractStep(BaseStep):
    """Scan the proactive material set and extract the four proposal kinds.

    Owns the ``proactive`` catalog watermark (never touches the dream catalog)
    and the daily LLM budget. Short-circuits via ``context[skip_key]`` on
    budget/timeout per F4.3; early-exits with zero LLM calls when no new
    evidence exists (F2.0) - unless the clock produced due reminders, which
    are delivered without any LLM call.

    Per-kind ceilings keep discovery proportional (``max_follow_ups``,
    ``max_extends``, ``max_new_ideas``); reminders are deliberately uncapped
    because they are bounded by the registry itself and pin today's agenda.
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
        capability_max_chars: int = 1200,
        extends_enabled: bool = True,
        reminders_enabled: bool = True,
        new_ideas_enabled: bool = True,
        max_follow_ups: int = 5,
        max_extends: int = 5,
        max_new_ideas: int = 3,
        reminder_lead_days: int = 3,
        schedule_closed_lookback_days: int = 7,
        resources_enabled: bool = True,
        resource_scan_days: int = 0,
        resource_max_chars: int = 40000,
        max_notice_reminders: int = 3,
        refine_enabled: bool = True,
        refine_timeout_seconds: float = 90,
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
        self.capability_max_chars = max(int(capability_max_chars), 0)
        self.extends_enabled = bool(extends_enabled)
        self.reminders_enabled = bool(reminders_enabled)
        self.new_ideas_enabled = bool(new_ideas_enabled)
        self.max_follow_ups = max(int(max_follow_ups), 0)
        self.max_extends = max(int(max_extends), 0)
        self.max_new_ideas = max(int(max_new_ideas), 0)
        self.reminder_lead_days = max(int(reminder_lead_days), 0)
        self.schedule_closed_lookback_days = max(int(schedule_closed_lookback_days), 0)
        self.resources_enabled = bool(resources_enabled)
        # 0 inherits ``scan_days``: the notice window normally matches the
        # conversation window, but can be widened on its own.
        self.resource_scan_days = max(int(resource_scan_days), 0)
        self.resource_max_chars = max(int(resource_max_chars), 0)
        self.max_notice_reminders = max(int(max_notice_reminders), 0)
        self.refine_enabled = bool(refine_enabled)
        self.refine_timeout_seconds = float(refine_timeout_seconds)
        self.skip_key = skip_key

    # pylint: disable=too-many-return-statements
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
        self.logger.info(
            f"[{self.name}] start date={day} scan_days={self.scan_days} extends={self.extends_enabled} "
            f"reminders={self.reminders_enabled} new_ideas={self.new_ideas_enabled}",
        )

        # 1) Material set M (F2.0) - all cheap, before any LLM work. Daily
        # notes feed every branch. The ``resource/`` layer feeds the reminder
        # branch only (external notices the user must be told about), so it is
        # scanned only when reminders are active.
        m_daily = scan_material_daily(ws, day, daily, self.scan_days)
        notices = self.reminders_enabled and self.resources_enabled
        if notices:
            m_resource, resource_skipped = scan_material_resource(
                ws,
                day,
                self.resource_scan_days or self.scan_days,
                str(self.config_value("resource_dir")),
            )
        else:
            m_resource, resource_skipped = [], 0
        state.resource_scanned, state.resource_skipped = len(m_resource), resource_skipped

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

        # 3) Change detection against the proactive catalog watermark. Both
        # material sets share one watermark, so an external notice arriving on
        # a day without any conversation still triggers a round.
        existing = {}
        for rel in (*m_daily, *m_resource):
            try:
                existing[rel] = (ws / rel).stat().st_mtime
            except OSError as e:
                self.logger.error(f"[{self.name}] stat failed on {rel}: {e}")
        nodes = await self.file_catalog.get_nodes()
        indexed = {n.path: n.st_mtime for n in nodes}
        changed = [rel for rel, mt in existing.items() if indexed.get(rel) != mt]
        daily_set = set(m_daily)
        state.changed_paths = [rel for rel in changed if rel in daily_set]
        state.resource_changed = [rel for rel in changed if rel not in daily_set]
        # mtime snapshot of what this round actually read; finish only
        # checkpoints paths whose mtime still matches, so files modified while
        # the LLM calls run stay "changed" for the next round (audit item 3).
        state.changed_mtimes = {rel: existing[rel] for rel in changed}
        self.logger.info(
            f"[{self.name}] material daily={len(m_daily)} resource={len(m_resource)} "
            f"resource_skipped={resource_skipped} changed={len(state.changed_paths)} "
            f"notices_changed={len(state.resource_changed)} carry_forward={len(carry_all)}",
        )

        # 4) Clock + schedule registry scan (branch C). Deterministic and free,
        # so it runs before any LLM decision: a due schedule must reach today's
        # agenda even when there is no new material, no agent wrapper, or an
        # unusable reply.
        now_dt = current_now(self)
        view = schedule_view(state_file.schedules, now_dt, self.reminder_lead_days) if self.reminders_enabled else []
        schedules_by_id = {str(entry.get("id") or ""): entry for entry in view if entry.get("id")}
        seeded = reminder_candidates_from_view(view, day)
        state.due_schedules = [
            {key: entry.get(key) for key in DUE_SCHEDULE_KEYS} for entry in view if entry.get("trigger") in DUE_TRIGGERS
        ]
        if seeded:
            self.logger.info(
                f"[{self.name}] clock scan: registry={len(state_file.schedules)} "
                f"due={len(state.due_schedules)} seeded_reminders={len(seeded)} now={now_dt.isoformat()}",
            )

        # 5) Zero-consumption early exit (A4 row 1) - unless the clock alone
        # already produced reminders, which need neither material nor an LLM.
        if not state.changed_paths and not state.resource_changed:
            if seeded:
                return self._clock_only(state, started, seeded, "Skipped LLM: no new evidence")
            state.early_exit = "no_new_evidence"
            return self._finish(state, started, "Skipped: no new evidence; 0 LLM calls")

        # 6) LLM channel (F4.4): structurally one reply per round plus at most
        # one parse-failure retry, so no persistent budget is needed (v5 R5).
        wrapper = resolve_agent_wrapper(self)
        if wrapper is None:
            if seeded:
                return self._clock_only(state, started, seeded, "Degraded: no agent_wrapper configured")
            state.early_exit = "no_agent_wrapper"
            self.logger.warning(f"[{self.name}] no agent_wrapper available; skipping round")
            return self._finish(state, started, "Skipped: no agent_wrapper configured")

        # 7) One LLM call with sectioned output (A3); retry once on parse failure.
        # meta is None when this round produced no usable output (timeout, or
        # two unusable replies): the material stays "changed" for the next
        # round instead of being silently consumed on output that can never
        # yield topics (audit #1).
        meta, failure = await self._extract_with_retry(wrapper, state, ws, day, daily, now_dt, view)
        if meta is None:
            if seeded:
                return self._clock_only(state, started, seeded, f"Degraded: {failure}")
            self.context[self.skip_key] = {"reason": failure}
            self._store(state)
            return passthrough_response(self, self.skip_key)
        self._clean_output(state, meta, set(m_daily), day, schedules_by_id, set(m_resource))
        if self.refine_enabled:
            await self._refine_stage(state, wrapper, ws, day, now_dt, view, set(m_resource))
        state.reminders = merge_reminders(seeded, state.reminders)
        answer = (
            f"Extracted {len(state.follow_ups)} follow_up(s), {len(state.extends)} extend(s), "
            f"{len(state.reminders)} reminder(s), {len(state.new_ideas)} idea(s), "
            f"{len(state.updates)} update(s), {len(state.schedule_updates)} schedule update(s) "
            f"from {len(state.changed_paths)} changed file(s) "
            f"and {len(state.resource_changed)} external notice file(s)"
        )
        if state.refine_calls:
            answer += (
                f"; refine: {state.refine_calls} pass(es), dropped {len(state.refine_dropped)}, "
                f"moved {len(state.refine_moved)}, degraded {state.refine_degraded}"
            )
        return self._finish(state, started, answer)

    def _clock_only(self, state: ProactiveState, started: float, seeded: list[dict], reason: str):
        """Deliver clock-driven reminders without consuming this round's material.

        ``changed_paths`` is cleared so the finish step checkpoints nothing and
        the unread material stays "changed" for the next round; the chain still
        runs, so today's agenda gets its reminders.
        """
        state.changed_paths, state.resource_changed, state.changed_mtimes = [], [], {}
        state.reminders = seeded
        state.reminder_only = True
        self.logger.warning(f"[{self.name}] {reason}; delivering {len(seeded)} clock-driven reminder(s)")
        return self._finish(state, started, f"{reason}; {len(seeded)} clock-driven reminder(s), 0 LLM output used")

    def _build_messages(
        self,
        state: ProactiveState,
        ws,
        day: str,
        daily: str,
        material_blob: str,
        notice_blob: str,
        now_dt,
        view: list[dict],
    ) -> tuple[str, str]:
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
        notices = list(dict.fromkeys(state.resource_changed))
        schedules_json = json.dumps(
            schedule_prompt_entries(view, self.schedule_closed_lookback_days, day),
            ensure_ascii=False,
        )
        user_message = self.prompt_format(
            "extract_user_message",
            date=day,
            clock=f"{now_dt.isoformat(timespec='minutes')} ({now_dt.strftime('%A')})",
            resource_dir=str(self.config_value("resource_dir")),
            changed_paths_json=json.dumps(changed, ensure_ascii=False),
            carry_forward_json=carry_forward_json,
            schedules_json=schedules_json,
            material_blob=material_blob,
            profile_block=self._profile_block(ws),
            capability_block=self._capability_block(ws),
            resource_paths_json=json.dumps(notices, ensure_ascii=False),
            resource_block=notice_blob or "(no external notices in this round)",
            extends=self.extends_enabled,
            reminders=self.reminders_enabled,
            ideas=self.new_ideas_enabled,
            resources=self.reminders_enabled and self.resources_enabled,
        )
        system_prompt = self.prompt_format(
            "extract_system_prompt",
            workspace_dir=str(ws),
            daily_dir=daily,
            resource_dir=str(self.config_value("resource_dir")),
            output_shape=self._output_shape(),
            max_follow_ups=self.max_follow_ups,
            max_extends=self.max_extends,
            max_new_ideas=self.max_new_ideas,
            max_notice_reminders=self.max_notice_reminders,
            extends=self.extends_enabled,
            reminders=self.reminders_enabled,
            ideas=self.new_ideas_enabled,
            resources=self.reminders_enabled and self.resources_enabled,
        )
        return user_message, system_prompt

    def _profile_block(self, ws) -> str:
        """Digest-personal profile sketch; background knowledge, never material."""
        digest_dir = str(self.config_value("digest_dir"))
        return load_personal_profile_block(ws, digest_dir, self.profile_max_chars)

    def _capability_block(self, ws) -> str:
        """Digest procedure/wiki inventory; grounds the ``new_ideas`` branch."""
        digest_dir = str(self.config_value("digest_dir"))
        return load_capability_block(ws, digest_dir, self.capability_max_chars)

    def _output_shape(self) -> str:
        """YAML sample listing exactly the active sections.

        Composed from per-section snippets rather than inline ``[flag]`` lines:
        the flag filter left-strips kept lines, which would flatten the
        indentation of a nested YAML example.
        """
        names = ["follow_ups"]
        if self.extends_enabled:
            names.append("extends")
        if self.reminders_enabled:
            names.append("reminders")
        if self.new_ideas_enabled:
            names.append("new_ideas")
        names.append("updates")
        if self.reminders_enabled:
            names.append("schedule_updates")
        return "\n".join(self.get_prompt(f"extract_shape_{name}") for name in names)

    async def _extract_with_retry(
        self,
        wrapper,
        state,
        ws,
        day: str,
        daily: str,
        now_dt,
        view,
    ) -> tuple[dict | None, str]:
        """Returns ``(meta, failure_reason)``; meta None means no usable output."""
        # Newest daily material first: when the total budget bites, the
        # freshest evidence survives and the oldest files are omitted.
        ordered = list(dict.fromkeys(state.changed_paths))[::-1]
        material_blob = pack_paths(
            ws,
            ordered,
            limit_per_file=self.max_chars_per_file,
            max_total_chars=self.max_total_chars or None,
        )
        # External notices are packed under their own, smaller budget so a
        # large upload can neither crowd out the conversation material nor be
        # crowded out by it.
        notice_blob = pack_paths(
            ws,
            list(dict.fromkeys(state.resource_changed))[::-1],
            limit_per_file=self.max_chars_per_file,
            max_total_chars=self.resource_max_chars or None,
        )
        user_message, system_prompt = self._build_messages(
            state,
            ws,
            day,
            daily,
            material_blob,
            notice_blob,
            now_dt,
            view,
        )
        raw = ""
        for attempt in (1, 2):
            raw = await self._reply(wrapper, state, user_message, system_prompt)
            if raw is None:
                return None, "llm_timeout"
            meta = parse_extract_reply(raw)
            if meta:
                return meta, ""
            self.logger.warning(f"[{self.name}] parse failed on attempt {attempt}; raw={raw[:200]!r}")
        self.logger.error(
            f"[{self.name}] reply unusable after 2 attempts; keeping material un-checkpointed; raw={raw[:200]!r}",
        )
        return None, "extract_parse_failed"

    async def _reply(self, wrapper, state, user_message, system_prompt, timeout: float | None = None):
        """One timeout-wrapped reply; returns raw text or None on timeout."""
        state.llm_calls += 1
        limit = self.llm_timeout_seconds if timeout is None else float(timeout)
        try:
            result = await asyncio.wait_for(
                wrapper.reply(user_message, system_prompt=system_prompt),
                timeout=limit,
            )
        except asyncio.TimeoutError:
            self.logger.warning(f"[{self.name}] LLM reply timed out after {limit}s")
            return None
        return agent_reply_result_text(result)

    def _clean_output(
        self,
        state: ProactiveState,
        meta: dict,
        allowed: set[str],
        day: str,
        schedules_by_id: dict,
        notice_allowed: set[str] | None = None,
    ) -> None:
        # Only the reminder branch may cite the ``resource/`` layer: the other
        # three kinds reason about the user's own conversations, and letting
        # them quote an external upload would blur that boundary.
        reminder_allowed = set(allowed) | set(notice_allowed or ())
        self._append_kind(
            state.follow_ups,
            meta.get("follow_ups"),
            allowed,
            "follow_up",
            day,
            self.max_follow_ups,
            state,
        )
        if self.extends_enabled:
            self._append_kind(
                state.extends,
                meta.get("extends"),
                allowed,
                "interest_extend",
                day,
                self.max_extends,
                state,
            )
        if self.reminders_enabled:
            cleaned: list[dict] = []
            for raw in meta.get("reminders") or []:
                if candidate := clean_reminder(raw, reminder_allowed, day, schedules_by_id, notice_allowed):
                    cleaned.append(candidate)
                else:
                    state.dropped_missing += 1
            # Dateless notice/key-event reminders are bounded; clock-anchored
            # ones are not (they are facts, and they pin today's agenda).
            state.reminders, over_cap = cap_notice_reminders(cleaned, self.max_notice_reminders)
            state.dropped_over_cap += over_cap
            for raw in meta.get("schedule_updates") or []:
                if update := clean_schedule_update(raw, reminder_allowed, day):
                    state.schedule_updates.append(update)
                else:
                    state.dropped_missing += 1
        if self.new_ideas_enabled:
            self._append_kind(
                state.new_ideas,
                meta.get("new_ideas"),
                allowed,
                "new_idea",
                day,
                self.max_new_ideas,
                state,
            )
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

    def _append_kind(
        self,
        bucket: list[dict],
        raw_items,
        allowed: set[str],
        kind: str,
        day: str,
        cap: int,
        state: ProactiveState,
    ) -> None:
        """Clean one discovery section into ``bucket``, honouring its ceiling.

        ``cap <= 0`` means unlimited. Over-cap entries are counted separately
        from invalid ones so a noisy round stays diagnosable in metadata.
        """
        for raw in raw_items or []:
            if cap and len(bucket) >= cap:
                state.dropped_over_cap += 1
                continue
            if candidate := clean_candidate(raw, allowed, kind, day):
                bucket.append(candidate)
            else:
                state.dropped_missing += 1

    # ------------------------------------------------------------------
    # Stage 2: per-kind adjudication (precision pass)
    # ------------------------------------------------------------------

    def _enabled_kinds(self) -> set[str]:
        """Kinds this round may hold candidates in (the legal ``move`` targets)."""
        return {
            kind
            for kind, active in (
                ("follow_up", True),
                ("interest_extend", self.extends_enabled),
                ("reminder", self.reminders_enabled),
                ("new_idea", self.new_ideas_enabled),
            )
            if active
        }

    def _kind_caps(self) -> dict[str, int]:
        """Per-kind ceilings re-applied after stage-2 moves (0 = uncapped)."""
        return {
            "follow_up": self.max_follow_ups,
            "interest_extend": self.max_extends,
            "new_idea": self.max_new_ideas,
            "reminder": 0,
        }

    def _cap_label(self, kind: str) -> str:
        """Ceiling as the refine prompt should state it."""
        if kind != "reminder":
            return str(self._kind_caps()[kind])
        return f"uncapped, but at most {self.max_notice_reminders} dateless notice/key-event reminder(s)"

    async def _refine_stage(self, state, wrapper, ws, day: str, now_dt, view, notice_paths: set[str]) -> None:
        """One small adjudication call per kind that produced candidates.

        Stage 2 never sees the material blob: it receives that kind's candidates
        plus the only context slice the kind needs, which keeps the pass cheap
        and its judgement focused. Each pass is independently fail-open, so a
        timeout or an unusable reply keeps the stage-1 candidates of that kind.
        """
        verdicts: list[dict] = []
        for kind, bucket in candidates_by_kind(state):
            known = {str(candidate.get("id") or "") for candidate in bucket}
            user_message = self.prompt_format(
                "refine_user_message",
                kind=kind,
                cap=self._cap_label(kind),
                checks=self.get_prompt(f"refine_checks_{kind}"),
                candidates_json=json.dumps(candidate_payload(bucket), ensure_ascii=False),
                context_block=self._refine_context(kind, state, ws, view, day, now_dt),
            )
            system_prompt = self.prompt_format("refine_system_prompt", kind=kind)
            raw = await self._reply(wrapper, state, user_message, system_prompt, self.refine_timeout_seconds)
            state.refine_calls += 1
            if raw is None:
                state.refine_degraded += 1
                self.logger.warning(f"[{self.name}] refine({kind}) timed out; keeping stage-1 candidates")
                continue
            parsed = parse_refine_reply(raw)
            if parsed is None:
                state.refine_degraded += 1
                self.logger.warning(f"[{self.name}] refine({kind}) reply unusable; raw={raw[:160]!r}")
                continue
            # A pass may not reach into another kind's bucket: ids outside it
            # are ignored rather than trusted.
            verdicts.extend(verdict for verdict in parsed if verdict["id"] in known)
        if not verdicts:
            # Nothing to adjudicate: every pass answered "keep everything".
            # The caps only need re-applying when a move could have crossed a
            # kind boundary, but the summary is worth logging either way.
            self.logger.info(
                f"[{self.name}] refine passes={state.refine_calls} dropped=0 moved=0 "
                f"degraded={state.refine_degraded} no verdicts; kept: "
                f"reminders={len(state.reminders)} follow_ups={len(state.follow_ups)} "
                f"ideas={len(state.new_ideas)} extends={len(state.extends)}",
            )
            return
        apply_verdicts(state, verdicts, self._kind_caps(), self._enabled_kinds(), notice_paths)
        state.reminders, over_cap = cap_notice_reminders(state.reminders, self.max_notice_reminders)
        state.dropped_over_cap += over_cap
        self.logger.info(
            f"[{self.name}] refine passes={state.refine_calls} dropped={len(state.refine_dropped)} "
            f"moved={len(state.refine_moved)} degraded={state.refine_degraded} kept: "
            f"reminders={len(state.reminders)} follow_ups={len(state.follow_ups)} "
            f"ideas={len(state.new_ideas)} extends={len(state.extends)}",
        )

    def _refine_context(self, kind: str, state, ws, view, day: str, now_dt) -> str:
        """The single context slice this kind's adjudication needs (no material)."""
        if kind == "reminder":
            entries = schedule_prompt_entries(view, self.schedule_closed_lookback_days, day)
            return (
                f"now: {now_dt.isoformat(timespec='minutes')} ({now_dt.strftime('%A')})\n"
                "schedules (registry, triggers already computed from the clock):\n"
                f"{json.dumps(entries, ensure_ascii=False)}"
            )
        if kind == "follow_up":
            carried = [
                {"id": t.id, "title": t.title, "kind": t.kind, "last_evidence_at": t.last_evidence_at}
                for t in state.carry_forward_prompt
            ]
            return (
                "carry_forward_topics (already tracked; a candidate that restates one of these "
                f"must be dropped):\n{json.dumps(carried, ensure_ascii=False)}"
            )
        if kind == "new_idea":
            return f"capability_inventory (what the agent can actually do):\n{self._capability_block(ws)}"
        return (
            "knowledge_base inventory (existing wiki nodes; a candidate that merely restates one "
            f"of these must be dropped):\n{self._capability_block(ws)}"
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
