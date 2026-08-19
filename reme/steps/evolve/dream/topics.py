"""Daily interests.yaml step."""

import json
from pathlib import Path

from ...base_step import BaseStep
from ...file_io import refresh_day_index
from .._evolve import agent_reply_result_text
from ....components import R
from .utils import (
    load_yaml_topics,
    llm_available,
    normalize_topic,
    parse_structured_reply,
    previous_dates,
    state_from_context,
    store_state,
    workspace_dir,
    write_yaml,
)


@R.register("dream_topics_step")
class DreamTopicsStep(BaseStep):
    """Write ``daily/<date>/interests.yaml`` with same-day and recent de-dup."""

    def __init__(self, topic_count: int = 3, topic_diversity_days: int = 7, **kwargs):
        super().__init__(**kwargs)
        self.topic_count = topic_count
        self.topic_diversity_days = topic_diversity_days

    async def execute(self):
        assert self.context is not None
        state = state_from_context(self)
        topic_count = int(self.context.get("topic_count", self.topic_count) or self.topic_count)
        raw_days = self.context.get("topic_diversity_days", self.topic_diversity_days)
        diversity_days = int(raw_days or self.topic_diversity_days)
        workspace = Path(state.workspace).resolve() if state.workspace else workspace_dir(self)
        target_day = state.date or ((state.dates or [""])[-1])
        self.logger.info(
            f"[{self.name}] start target_day={target_day!r} candidates={len(state.topics)} "
            f"topic_count={topic_count} diversity_days={diversity_days}",
        )

        if not state.topics:
            existing_paths = []
            if target_day and self._abs_path(workspace, state.daily_dir, target_day).is_file():
                existing_paths = [self._rel_path(state.daily_dir, target_day)]
            state.interests_paths = existing_paths
            state.interests_path = existing_paths[-1] if existing_paths else ""
            state.topics_written = (
                len(load_yaml_topics(self._abs_path(workspace, state.daily_dir, target_day))) if target_day else 0
            )
            answer = (
                f"Kept existing interest topic(s) at {', '.join(existing_paths)}"
                if existing_paths
                else "Skipped interests.yaml write: no new topic candidates"
            )
            self.logger.info(f"[{self.name}] skip no candidates existing_paths={len(existing_paths)}")
            return self._finish(state, True, answer)

        try:
            if not target_day:
                state.interests_paths = []
                state.interests_path = ""
                state.topics_written = 0
                self.logger.info(f"[{self.name}] skip no target date")
                return self._finish(state, True, "Skipped interests.yaml write: no target date")

            rel_path = self._rel_path(state.daily_dir, target_day)
            abs_path = self._abs_path(workspace, state.daily_dir, target_day)
            same_day = load_yaml_topics(abs_path, strict=True)
            recent = [
                topic
                for previous_day in previous_dates(target_day, diversity_days)
                for topic in load_yaml_topics(self._abs_path(workspace, state.daily_dir, previous_day))
            ]
            self.logger.info(
                f"[{self.name}] loaded context same_day={len(same_day)} recent={len(recent)} target={rel_path}",
            )
            topics, _used_llm = await self._select_topics(
                target_day,
                state.topics,
                same_day,
                recent,
                topic_count,
                diversity_days,
                state,
            )
            self.logger.info(f"[{self.name}] selected topics={len(topics)} used_llm={_used_llm}")
            payload = {
                "date": target_day,
                "topic_count": topic_count,
                "diversity_days": diversity_days,
                "topics": topics,
            }
            before_content = abs_path.read_bytes() if abs_path.is_file() else None
            self.logger.info(f"[{self.name}] write yaml start path={rel_path}")
            write_yaml(abs_path, payload)
            self.logger.info(f"[{self.name}] write yaml done path={rel_path}")
            if before_content != abs_path.read_bytes() and rel_path not in state.modified_paths:
                state.modified_paths.append(rel_path)
            self.logger.info(f"[{self.name}] refresh index start date={target_day} daily_dir={state.daily_dir}")
            await refresh_day_index(self.file_store, target_day, state.daily_dir)
            self.logger.info(f"[{self.name}] refresh index done date={target_day}")
            state.interests_paths = [rel_path]
            state.interests_path = rel_path
            state.topics_written = len(topics)
            answer = f"Wrote {len(topics)} interest topic(s) to {rel_path}"
            return self._finish(state, True, answer)
        except Exception as e:  # noqa: BLE001
            state.topic_error = f"{type(e).__name__}: {e}"
            state.errors.append(state.topic_error)
            self.logger.error(f"[{self.name}] failed: {state.topic_error}")
            return self._finish(state, False, f"Error: {state.topic_error}")

    async def _select_topics(
        self,
        day: str,
        candidates: list[dict],
        same_day: list[dict],
        recent: list[dict],
        count: int,
        days: int,
        state,
    ):
        if not candidates:
            return self._dedupe([], same_day, recent, count), False
        if not llm_available(self):
            self.logger.info(f"[{self.name}] select topics without llm candidates={len(candidates)}")
            return self._dedupe(candidates, same_day, recent, count), False
        self.logger.info(
            f"[{self.name}] topics agent start candidates={len(candidates)} "
            f"same_day={len(same_day)} recent={len(recent)}",
        )
        message = self.prompt_format(
            "topics_user_message",
            date=day,
            topic_count=count,
            diversity_days=days,
            candidates_json=json.dumps(candidates, ensure_ascii=False, indent=2),
            same_day_json=json.dumps(same_day, ensure_ascii=False, indent=2),
            recent_topics_json=json.dumps(recent, ensure_ascii=False, indent=2),
        )
        try:
            result = await self.agent_wrapper.reply(
                message,
                system_prompt=self.prompt_format("topics_system_prompt"),
            )
            self.logger.info(f"[{self.name}] topics agent done has_result={bool(result.get('result'))}")
            raw_result = agent_reply_result_text(result)
            meta = parse_structured_reply(raw_result)
        except Exception as e:  # noqa: BLE001
            warning = f"topic selection agent unavailable; used deterministic fallback ({type(e).__name__})"
            state.warnings.append(warning)
            self.logger.warning(f"[{self.name}] {warning}: {e}")
            return self._dedupe(candidates, same_day, recent, count), False
        allowed_paths = {
            str(path).strip() for candidate in candidates for path in candidate.get("paths") or [] if str(path).strip()
        }
        selected = [self._clean_topic(t, allowed_paths) for t in meta.get("topics") or []]
        if not any(selected):
            if not isinstance(meta.get("topics"), list):
                warning = "topic selection skipped unusable agent receipt; used deterministic fallback"
                state.warnings.append(warning)
                self.logger.warning(f"[{self.name}] {warning}")
            self.logger.info(f"[{self.name}] topics agent produced no usable topics; fallback to candidates")
            selected = candidates
        return self._dedupe(selected, same_day, recent, count), True

    @staticmethod
    def _rel_path(daily_dir: str, day: str) -> str:
        return f"{daily_dir}/{day}/interests.yaml"

    @staticmethod
    def _abs_path(workspace: Path, daily_dir: str, day: str) -> Path:
        return workspace / daily_dir / day / "interests.yaml"

    @staticmethod
    def _clean_topic(raw, allowed_paths: set[str] | None = None) -> dict:
        if not isinstance(raw, dict):
            return {}
        title, reason = (
            str(raw.get("title") or "").strip(),
            str(raw.get("reason") or "").strip(),
        )
        if not title or not reason:
            return {}
        keywords, paths = raw.get("keywords") or [], raw.get("paths") or []
        cleaned_keywords = [str(k).strip() for k in keywords if str(k).strip()] if isinstance(keywords, list) else []
        cleaned_paths = (
            [
                str(p).strip()
                for p in paths
                if str(p).strip() and (allowed_paths is None or str(p).strip() in allowed_paths)
            ]
            if isinstance(paths, list)
            else []
        )
        return {
            "title": title,
            "reason": reason,
            "evidence": str(raw.get("evidence") or "").strip(),
            "keywords": cleaned_keywords,
            "paths": cleaned_paths,
        }

    @staticmethod
    def _dedupe(topics: list[dict], same_day: list[dict], recent: list[dict], count: int) -> list[dict]:
        recent_norm = {normalize_topic(t.get("title", "")) for t in recent}
        seen = {normalize_topic(t.get("title", "")) for t in same_day}
        out = list(same_day)
        for topic in [t for t in topics if t]:
            title_norm = normalize_topic(topic.get("title", ""))
            if title_norm and title_norm not in seen and title_norm not in recent_norm:
                seen.add(title_norm)
                out.append(topic)
            if len(out) >= count:
                break
        return out[:count]

    def _finish(self, state, success: bool, answer: str):
        assert self.context is not None
        state.summary = answer
        store_state(self, state)
        self.context.response.success = success
        self.context.response.answer = answer
        self.logger.info(f"[{self.name}] finish success={success} answer={answer!r}")
        return self.context.response
