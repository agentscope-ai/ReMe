"""Daily interest-topic aggregation for auto_dream."""

import datetime as dt
import json
import re
from pathlib import Path

import frontmatter
from pydantic import BaseModel, Field

from ._evolve import now
from ..base_step import BaseStep
from ..file_io._daily_index import refresh_day_index, validate_session_id
from ..file_io._file_io import write_file_safe
from ...components import R


class DailyTopic(BaseModel):
    """One final daily topic selected from dream candidates."""

    title: str = Field(description="Specific topic title.")
    reason: str = Field(description="Why this topic is likely interesting to the user.")
    evidence: str = Field(description="Supporting source path or concise evidence pointer.")
    keywords: list[str] = Field(default_factory=list, description="Keywords for future de-duplication.")


class DailyTopicsOutput(BaseModel):
    """Structured output for the final daily topic selector."""

    topics: list[DailyTopic] = Field(default_factory=list)


class DailyTopicsResult(BaseModel):
    """Result of one daily topic aggregation."""

    date: str = ""
    path: str = ""
    topics: list[dict] = Field(default_factory=list)
    recent_days: int = 0
    candidates_seen: int = 0
    used_llm: bool = False
    skipped: bool = False
    error: str = ""
    summary: str = ""


def _normalize_topic(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def _clean_candidates(raw_candidates) -> list[dict]:
    out: list[dict] = []
    if not isinstance(raw_candidates, list):
        return out
    for raw in raw_candidates:
        if not isinstance(raw, dict):
            continue
        title = str(raw.get("title") or "").strip()
        reason = str(raw.get("reason") or "").strip()
        if not title or not reason:
            continue
        evidence = str(raw.get("evidence") or "").strip()
        source_path = str(raw.get("source_path") or "").strip()
        if source_path and source_path not in evidence:
            evidence = f"{evidence} ({source_path})" if evidence else source_path
        keywords_raw = raw.get("keywords") or []
        keywords = [str(k).strip() for k in keywords_raw if str(k).strip()] if isinstance(keywords_raw, list) else []
        out.append({"title": title, "reason": reason, "evidence": evidence, "keywords": keywords[:8]})
    return out


def _previous_dates(day: str, n_days: int) -> list[str]:
    try:
        base = dt.date.fromisoformat(day)
    except ValueError:
        return []
    return [(base - dt.timedelta(days=i)).isoformat() for i in range(1, max(n_days, 0) + 1)]


def _render_topics_note(day: str, topics: list[dict], topic_count: int, diversity_days: int) -> str:
    body_lines = ["# Interested Topics", ""]
    if not topics:
        body_lines.append("(none)")
    for i, topic in enumerate(topics, start=1):
        keywords = ", ".join(topic.get("keywords") or [])
        body_lines.extend(
            [
                f"{i}. **{topic.get('title', '').strip()}**",
                f"   - reason: {topic.get('reason', '').strip()}",
                f"   - evidence: {topic.get('evidence', '').strip()}",
            ],
        )
        if keywords:
            body_lines.append(f"   - keywords: {keywords}")
        body_lines.append("")

    post = frontmatter.Post(
        "\n".join(body_lines).rstrip() + "\n",
        name="interests",
        description=f"{len(topics)} interest topic(s) inferred for {day}.",
        date=day,
        topic_count=topic_count,
        diversity_days=diversity_days,
    )
    return frontmatter.dumps(post)


@R.register("daily_topics_step")
class DailyTopicsStep(BaseStep):
    """Select and write de-duplicated daily interest topics."""

    def __init__(
        self,
        topic_count: int = 3,
        diversity_days: int = 7,
        session_id: str = "interests",
        persist_index: bool = True,
        preserve_response: bool = False,
        metadata_key: str = "",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.topic_count = topic_count
        self.diversity_days = diversity_days
        self.session_id = session_id
        self.persist_index = persist_index
        self.preserve_response = preserve_response
        self.metadata_key = metadata_key

    def _daily_dir(self) -> str:
        cfg = self.app_context.app_config if self.app_context is not None else None
        return (cfg.daily_dir if cfg else "") or "daily"

    def _vault_dir(self) -> Path:
        vr = getattr(self.file_store, "vault_path", None)
        return Path(vr).resolve() if vr else Path.cwd().resolve()

    def _llm_available(self) -> bool:
        try:
            return self.as_llm is not None and self.agent_wrapper is not None
        except Exception:
            return False

    def _recent_topic_notes(self, day: str, daily_dir: str, session_id: str, n_days: int) -> list[dict]:
        vault = self._vault_dir()
        out: list[dict] = []
        for previous in _previous_dates(day, n_days):
            rel = f"{daily_dir}/{previous}/session_agent_{session_id}.md"
            abs_path = vault / rel
            if not abs_path.is_file():
                continue
            try:
                text = abs_path.read_text(encoding="utf-8")
            except OSError:
                continue
            out.append({"date": previous, "path": rel, "content": text[:12000]})
        return out

    @staticmethod
    def _fallback_select(candidates: list[dict], recent_notes: list[dict], topic_count: int) -> list[dict]:
        return DailyTopicsStep._dedupe_against_recent(candidates, recent_notes, topic_count)

    @staticmethod
    def _dedupe_against_recent(topics: list[dict], recent_notes: list[dict], topic_count: int) -> list[dict]:
        recent_text = "\n".join(note["content"] for note in recent_notes)
        recent_norm = _normalize_topic(recent_text)
        seen: set[str] = set()
        out: list[dict] = []
        for candidate in topics:
            title_norm = _normalize_topic(candidate.get("title", ""))
            if not title_norm or title_norm in seen:
                continue
            if title_norm and title_norm in recent_norm:
                continue
            seen.add(title_norm)
            out.append(candidate)
            if len(out) >= topic_count:
                break
        return out

    async def _select_topics(
        self,
        day: str,
        candidates: list[dict],
        recent_notes: list[dict],
        topic_count: int,
        diversity_days: int,
    ) -> tuple[list[dict], bool]:
        if not candidates:
            return [], False
        if not self._llm_available():
            return self._fallback_select(candidates, recent_notes, topic_count), False

        user_message = self.prompt_format(
            "select_user_message",
            date=day,
            topic_count=topic_count,
            diversity_days=diversity_days,
            candidates_json=json.dumps(candidates, ensure_ascii=False, indent=2),
            recent_topics_json=json.dumps(recent_notes, ensure_ascii=False, indent=2),
        )
        result = await self.agent_wrapper.reply(
            user_message,
            system_prompt=self.prompt_format("select_system_prompt"),
            output_schema=DailyTopicsOutput,
        )
        meta = result.get("structured_output") if isinstance(result.get("structured_output"), dict) else {}
        topics = []
        for raw in meta.get("topics") or []:
            if not isinstance(raw, dict):
                continue
            title = str(raw.get("title") or "").strip()
            reason = str(raw.get("reason") or "").strip()
            if not title or not reason:
                continue
            evidence = str(raw.get("evidence") or "").strip()
            keywords_raw = raw.get("keywords") or []
            keywords = (
                [str(k).strip() for k in keywords_raw if str(k).strip()] if isinstance(keywords_raw, list) else []
            )
            topics.append({"title": title, "reason": reason, "evidence": evidence, "keywords": keywords[:8]})
            if len(topics) >= topic_count:
                break
        return self._dedupe_against_recent(topics, recent_notes, topic_count), True

    async def execute(self):
        assert self.context is not None
        previous_success = self.context.response.success
        previous_answer = self.context.response.answer
        tz = self.app_context.app_config.timezone if self.app_context is not None else None
        day = (self.context.get("date", "") or "").strip() or now(tz).strftime("%Y-%m-%d")
        raw_candidates = self.context.get("candidates") or self.context.response.metadata.get("topic_candidates") or []
        topic_count = int(self.context.get("topic_count", self.topic_count) or self.topic_count)
        diversity_days = int(self.context.get("diversity_days", self.diversity_days) or self.diversity_days)
        session_id = (self.context.get("session_id", self.session_id) or self.session_id).strip()

        def finish_success(result: DailyTopicsResult):
            if not self.preserve_response:
                self.context.response.success = True
                self.context.response.answer = result.summary
            else:
                self.context.response.success = previous_success
                self.context.response.answer = previous_answer
            if self.metadata_key:
                self.context.response.metadata[self.metadata_key] = result.model_dump()
            else:
                self.context.response.metadata.update(result.model_dump())
            return self.context.response

        result = DailyTopicsResult(date=day, recent_days=diversity_days)
        err = validate_session_id(session_id)
        if err:
            result.error = err
            self.context.response.success = False
            self.context.response.answer = f"Error: {err}"
            self.context.response.metadata.update(result.model_dump())
            return self.context.response

        candidates = _clean_candidates(raw_candidates)
        result.candidates_seen = len(candidates)
        if not candidates:
            result.skipped = True
            result.summary = "Skipped: no topic candidates"
            return finish_success(result)

        daily_dir = self._daily_dir()
        recent_notes = self._recent_topic_notes(day, daily_dir, session_id, diversity_days)
        topics, used_llm = await self._select_topics(day, candidates, recent_notes, topic_count, diversity_days)
        result.used_llm = used_llm
        result.topics = topics

        rel_path = f"{daily_dir}/{day}/session_agent_{session_id}.md"
        result.path = rel_path
        abs_path = self._vault_dir() / rel_path
        await write_file_safe(abs_path, _render_topics_note(day, topics, topic_count, diversity_days), encoding="utf-8")

        if self.persist_index:
            await refresh_day_index(self.file_store, day, daily_dir)

        result.summary = f"Wrote {len(topics)} interest topic(s) to {rel_path}"
        return finish_success(result)
