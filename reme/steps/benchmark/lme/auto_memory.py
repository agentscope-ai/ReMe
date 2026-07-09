"""lme_auto_memory — turn every LongMemEval session into a search-friendly note.

For a workspace such as ``datasets/longmemeval/1`` this step walks each raw
session under ``resource_dir`` (files named ``<date>_(...)_<time>@<session_id>.json``
with ``haystack_date`` / ``haystack_session_id`` / ``messages``) and, one per
session, asks an agent to *completely* extract its content — entities, times,
numbers, preferences, events, causal links — into a daily note optimized for
both BM25 and vector retrieval.

Each note is written to ``<daily_dir>/<YYYY-MM-DD>/<name>.md`` via the shared
``daily_write`` job, so the frontmatter carries ``session_id`` for progressive
expansion (the agentic-answer flow pivots from a search hit back to the raw
session through this id). Filenames are LLM-generated topic stems; same-day
collisions are disambiguated by appending the session id.
"""

import asyncio
import json
import re
import time
from datetime import datetime
from pathlib import Path

import frontmatter

from ...base_step import BaseStep
from ...file_io import extract_daily_date
from ....components import R

START_INTERVAL_SECONDS = 1.0
MAX_CONCURRENCY = 60
RETRY_INITIAL_SECONDS = 5.0
RETRY_MAX_SECONDS = 300.0
_LME_DATETIME_RE = re.compile(r"(\d{4})/(\d{2})/(\d{2}).*?(\d{2}):(\d{2})")

# Structured extraction the memory agent must return per session.
_MEMORY_SCHEMA = {
    "type": "object",
    "properties": {
        "name": {
            "type": "string",
            "description": "Concise, stable topic/event filename stem (kebab-case, no date, no slash or "
            "reserved characters). E.g. 'daily-commute-details' or 'leather-boot-care'.",
        },
        "description": {
            "type": "string",
            "description": "Thorough one-paragraph summary of the note body — specific enough that this "
            "description alone conveys all key facts. Used as a search-friendly abstract.",
        },
        "body": {
            "type": "string",
            "description": "Complete markdown extraction of every core fact in the session, written for "
            "retrieval (natural-language statements, explicit entities, dates and numbers verbatim).",
        },
    },
    "required": ["name", "description", "body"],
}


@R.register("lme_auto_memory_step")
class LmeAutoMemoryStep(BaseStep):
    """Extract each LME session into a daily note via a per-session agent."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._reserve_lock = asyncio.Lock()
        self._reserved: dict[tuple[str, str], str] = {}

    def _resource_dir_name(self) -> str:
        return self.app_context.app_config.resource_dir if self.app_context is not None else "session"

    def _session_dir(self) -> Path:
        return self.workspace_path / self._resource_dir_name()

    @staticmethod
    def _parse_lme_datetime(raw_date: str) -> datetime | None:
        """Parse LongMemEval timestamps like ``2023/05/20 (Sat) 03:29``."""
        match = _LME_DATETIME_RE.search(raw_date.strip())
        if match is None:
            return None
        try:
            year, month, day, hour, minute = (int(part) for part in match.groups())
            return datetime(year, month, day, hour, minute)
        except ValueError:
            return None

    @staticmethod
    def _parse_day(raw_date: str) -> str | None:
        """Parse a LongMemEval ``haystack_date`` (e.g. '2023/05/20 (Sat) 03:29') to YYYY-MM-DD."""
        head = raw_date.strip()[:10].replace("/", "-")
        return extract_daily_date(head)

    @staticmethod
    def _load_json(path: Path) -> dict:
        with path.open(encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            raise ValueError("session file is not a JSON object")
        return data

    @staticmethod
    def _format_messages(messages: list) -> str:
        lines: list[str] = []
        for msg in messages:
            if not isinstance(msg, dict):
                continue
            role = str(msg.get("role", "")).strip() or "unknown"
            content = msg.get("content", "")
            if not isinstance(content, str):
                content = json.dumps(content, ensure_ascii=False)
            lines.append(f"[{role}]\n{content}")
        return "\n\n".join(lines)

    async def _existing_session_id(self, rel_path: str) -> str:
        note = self.workspace_path / rel_path
        if not note.is_file():
            return ""
        try:
            post = frontmatter.loads(note.read_text(encoding="utf-8"))
        except Exception:
            return ""
        return str((post.metadata or {}).get("session_id", "") or "").strip()

    async def _reserve_name(self, daily_dir: str, day: str, name: str, session_id: str) -> str:
        """Pick a collision-free filename stem for this session under ``day``."""
        async with self._reserve_lock:
            for cand in (name, f"{name}-{session_id}"):
                key = (day, cand)
                owner = self._reserved.get(key)
                if owner == session_id:
                    return cand
                if owner is not None:
                    continue
                existing = await self._existing_session_id(f"{daily_dir}/{day}/{cand}.md")
                if existing and existing != session_id:
                    continue
                self._reserved[key] = session_id
                return cand
            # Extremely unlikely fallback (same topic AND same session id twice).
            i = 2
            while True:
                cand = f"{name}-{session_id}-{i}"
                key = (day, cand)
                if key not in self._reserved:
                    self._reserved[key] = session_id
                    return cand
                i += 1

    # pylint: disable=too-many-statements
    async def execute(self):
        assert self.context is not None
        if self.agent_wrapper is None:
            raise ValueError("lme_auto_memory_step requires agent_wrapper")

        query_data = self._load_json(self.workspace_path / "query.json")
        question_date = str(query_data.get("question_date") or "").strip()
        question_dt = self._parse_lme_datetime(question_date)
        if question_dt is None:
            raise ValueError(f"query.json has an invalid 'question_date': {question_date!r}")

        session_dir = self._session_dir()
        if not session_dir.is_dir():
            raise FileNotFoundError(f"Session directory not found: {session_dir}")
        session_files = sorted(p for p in session_dir.iterdir() if p.suffix == ".json")
        sessions: list[tuple[dict, Path, str, str, str]] = []
        filtered_sessions: list[dict] = []
        session_ids_illegal: list[str] = []

        for session_path in session_files:
            try:
                session = self._load_json(session_path)
            except (ValueError, OSError) as exc:
                self.logger.warning(f"[{self.name}] skip {session_path.name}: {exc}")
                continue

            session_id = str(session.get("haystack_session_id") or session_path.stem)
            session_date = str(session.get("haystack_date") or "").strip()
            session_dt = self._parse_lme_datetime(session_date)
            if session_dt is not None and session_dt > question_dt:
                session_ids_illegal.append(session_id)
                filtered_sessions.append(
                    {
                        "session_id": session_id,
                        "session_date": session_date,
                        "session_file": session_path.name,
                        "reason": "session_date_after_question_date",
                    },
                )
                continue
            if session_dt is None:
                self.logger.warning(
                    f"[{self.name}] keep {session_id}: cannot parse haystack_date={session_date!r}",
                )
            day = session_dt.strftime("%Y-%m-%d") if session_dt is not None else (self._parse_day(session_date) or "")
            sessions.append((session, session_path, session_id, session_date, day))

        daily_dir = self.config_value("daily_dir")
        resource_dir = self._resource_dir_name()
        start_interval_seconds = float(self.kwargs.get("start_interval_seconds", START_INTERVAL_SECONDS))
        if start_interval_seconds < 0:
            start_interval_seconds = START_INTERVAL_SECONDS
        concurrency = int(self.kwargs.get("concurrency", MAX_CONCURRENCY))
        if concurrency <= 0:
            concurrency = MAX_CONCURRENCY
        concurrency = min(concurrency, MAX_CONCURRENCY)
        total = len(sessions)
        self.logger.info(
            f"[{self.name}] extracting {total} sessions from {session_dir} "
            f"(filtered {len(session_ids_illegal)} sessions after question_date, "
            f"start_interval={start_interval_seconds}s, concurrency={concurrency})",
        )

        self._reserved.clear()
        failed_extracts: list[dict] = []
        retry_initial_seconds = float(self.kwargs.get("retry_initial_seconds", RETRY_INITIAL_SECONDS))
        retry_max_seconds = float(self.kwargs.get("retry_max_seconds", RETRY_MAX_SECONDS))
        retry_max_attempts_raw = self.kwargs.get("retry_max_attempts")
        retry_max_attempts = int(retry_max_attempts_raw) if retry_max_attempts_raw not in (None, "") else 0
        if retry_initial_seconds <= 0:
            retry_initial_seconds = RETRY_INITIAL_SECONDS
        retry_max_seconds = max(retry_max_seconds, retry_initial_seconds)
        retry_gate = asyncio.Condition()
        retry_sleeping_extract_idxs: set[int] = set()
        submit_lock = asyncio.Lock()
        last_submitted_at = 0.0
        semaphore = asyncio.Semaphore(concurrency)

        def has_prior_retry_sleeping(idx: int) -> bool:
            return any(retry_idx < idx for retry_idx in retry_sleeping_extract_idxs)

        async def wait_for_start_slot() -> None:
            nonlocal last_submitted_at
            async with submit_lock:
                sleep_seconds = last_submitted_at + start_interval_seconds - time.monotonic()
                if sleep_seconds > 0:
                    await asyncio.sleep(sleep_seconds)
                last_submitted_at = time.monotonic()

        async def wait_for_healthy_start_slot(idx: int, session_id: str) -> None:
            while True:
                async with retry_gate:
                    if has_prior_retry_sleeping(idx):
                        self.logger.info(
                            f"[{self.name}] ({idx}/{total}) {session_id} waits for earlier retry sleep",
                        )
                    await retry_gate.wait_for(lambda: not has_prior_retry_sleeping(idx))

                await wait_for_start_slot()

                async with retry_gate:
                    if not has_prior_retry_sleeping(idx):
                        return

        async def mark_retry_sleeping(idx: int) -> None:
            async with retry_gate:
                retry_sleeping_extract_idxs.add(idx)
                retry_gate.notify_all()

        async def mark_retry_awake(idx: int) -> None:
            async with retry_gate:
                retry_sleeping_extract_idxs.discard(idx)
                retry_gate.notify_all()

        async def reply_with_retry(idx: int, user_prompt: str, session_id: str) -> dict:
            attempt = 1
            sleep_seconds = retry_initial_seconds
            while True:
                try:
                    await wait_for_healthy_start_slot(idx, session_id)
                    result = await self.agent_wrapper.reply(
                        user_prompt,
                        system_prompt=self.get_prompt("system_prompt"),
                        output_schema=_MEMORY_SCHEMA,
                    )
                    if not isinstance(result.get("structured_output"), dict):
                        raise ValueError("agent reply missing structured_output")
                    await mark_retry_awake(idx)
                    if attempt > 1:
                        self.logger.info(f"[{self.name}] extract recovered for {session_id} after {attempt} attempts")
                    return result
                except Exception as exc:
                    if 0 < retry_max_attempts <= attempt:
                        await mark_retry_awake(idx)
                        raise
                    await mark_retry_sleeping(idx)
                    next_sleep = min(sleep_seconds, retry_max_seconds)
                    self.logger.warning(
                        f"[{self.name}] extract attempt {attempt} failed for {session_id}: {exc}; "
                        f"retrying in {next_sleep:.1f}s",
                    )
                    await asyncio.sleep(next_sleep)
                    await mark_retry_awake(idx)
                    sleep_seconds = min(sleep_seconds * 2, retry_max_seconds)
                    attempt += 1

        async def extract_one(
            idx: int,
            session: dict,
            session_path: Path,
            session_id: str,
            session_date: str,
            day: str,
        ) -> dict | None:
            if not day:
                self.logger.warning(f"[{self.name}] skip {session_id}: unparseable date {session_date!r}")
                return None
            messages = session.get("messages") or []

            user_prompt = self.prompt_format(
                "user_message",
                session_id=session_id,
                session_date=session_date,
                messages=self._format_messages(messages),
            )
            try:
                result = await reply_with_retry(idx, user_prompt, session_id)
            except Exception as exc:  # noqa: BLE001 — one bad session must not abort the sweep
                self.logger.warning(f"[{self.name}] extract failed for {session_id}: {exc}")
                failed_extracts.append(
                    {
                        "session_id": session_id,
                        "session_date": session_date,
                        "error": str(exc),
                    },
                )
                return None

            extracted = result.get("structured_output")
            name = description = body = ""
            if isinstance(extracted, dict):
                name = str(extracted.get("name") or "").strip()
                description = str(extracted.get("description") or "").strip()
                body = str(extracted.get("body") or "").strip()
            if not isinstance(extracted, dict) or not name or not body:
                if isinstance(extracted, dict):
                    self.logger.info(f"[{self.name}] empty extraction for {session_id}; skipping")
                else:
                    self.logger.warning(f"[{self.name}] no structured output for {session_id}; skipping")
                return None

            unique_name = await self._reserve_name(daily_dir, day, name, session_id)
            rel_path = f"{daily_dir}/{day}/{unique_name}.md"
            post = frontmatter.Post(
                body,
                name=unique_name,
                description=description,
                session_id=session_id,
                session_date=session_date,
                source=f"[[{resource_dir}/{session_path.name}]]",
            )
            abs_path = self.workspace_path / rel_path
            abs_path.parent.mkdir(parents=True, exist_ok=True)
            abs_path.write_text(frontmatter.dumps(post), encoding="utf-8")

            self.logger.info(f"[{self.name}] ({idx}/{total}) {session_id} -> {rel_path}")
            return {"session_id": session_id, "date": day, "path": rel_path}

        async def extract_one_limited(
            idx: int,
            session: dict,
            session_path: Path,
            session_id: str,
            session_date: str,
            day: str,
        ) -> dict | None:
            async with semaphore:
                return await extract_one(idx, session, session_path, session_id, session_date, day)

        results = await asyncio.gather(
            *(
                extract_one_limited(idx, session, session_path, session_id, session_date, day)
                for idx, (session, session_path, session_id, session_date, day) in enumerate(sessions, start=1)
            ),
        )
        written = [r for r in results if r is not None]

        self.context.response.success = True
        self.context.response.answer = f"wrote {len(written)}/{total} session notes"
        self.context.response.metadata.update(
            {
                "num_sessions": total,
                "num_session_files": len(session_files),
                "num_written": len(written),
                "num_failed_extracts": len(failed_extracts),
                "num_filtered_sessions": len(session_ids_illegal),
                "session_ids_illegal": session_ids_illegal,
                "filtered_sessions": filtered_sessions,
                "failed_extracts": failed_extracts,
                "notes": written,
            },
        )
        return self.context.response
